import os
import sys
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List
from tqdm.notebook import tqdm

# sound
import librosa as lb
import soundfile as sf

# pytorch
import torch
from torch import nn
from  torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50


import bird_recognition

NUM_CLASSES = 397
SR = 32_000
DURATION = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# 提出用
TEST_AUDIO_ROOT = Path("../input/birdclef-2021/test_soundscapes")
SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
TARGET_PATH = None

if not len(list(TEST_AUDIO_ROOT.glob("*.ogg"))): # テスト用の音源がないなら検証用
    TEST_AUDIO_ROOT = Path("../input/birdclef-2021/train_soundscapes")
    SAMPLE_SUB_PATH = None
    # SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
    TARGET_PATH = Path("../input/birdclef-2021/train_soundscape_labels.csv")

def is_submit_mode() -> bool:
    """提出時かどうか"""
    return (TARGET_PATH is None)

class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs

    def __call__(self, y):

        melspec = lb.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, **self.kwargs,
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length):
    if len(y) < length:
        y = np.concatenate([y, length - np.zeros(len(y))])
    elif len(y) > length:
        y = y[:length]
    return y

class BirdCLEFDataset(Dataset):
    def __init__(self, data, sr=SR, n_mels=128, fmin=0, fmax=None, duration=DURATION, step=None, res_type="kaiser_fast", resample=True):

        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        self.npy_save_root = Path("./data")
        
        os.makedirs(self.npy_save_root, exist_ok=True)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        filename = filepath.stem
        npy_path = self.npy_save_root / f"{filename}.npy"
        
        if not os.path.exists(npy_path):
            audio, orig_sr = sf.read(filepath, dtype="float32")

            if self.resample and orig_sr != self.sr:
                audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

            audios = []
            for i in range(self.audio_length, len(audio) + self.step, self.step):
                start = max(0, i - self.audio_length)
                end = start + self.audio_length
                audios.append(audio[start:end])

            if len(audios[-1]) < self.audio_length:
                audios = audios[:-1]

            images = [self.audio_to_image(audio) for audio in audios]
            images = np.stack(images)
            
            np.save(str(npy_path), images)
        return np.load(npy_path)

    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "filepath"])

def load_net(checkpoint_path, num_classes=NUM_CLASSES):
    net = resnest50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net

@torch.no_grad()
def get_thresh_preds(out, thresh=None):
    thresh = thresh or THRESH
    o = (-out).argsort(1)
    npreds = (out > thresh).sum(1)
    preds = []
    for oo, npred in zip(o, npreds):
        preds.append(oo[:npred].cpu().numpy().tolist())
    return preds

def predict(nets, test_data, names=True):
    preds = []
    with torch.no_grad():
        for idx in  tqdm(list(range(len(test_data)))):
            xb = torch.from_numpy(test_data[idx]).to(DEVICE)
            pred = 0.
            for net in nets:
                o = net(xb)
                o = torch.sigmoid(o)
                pred += o
            pred /= len(nets)
            if names:
                pred = bird_recognition.datasets.get_bird_columns(get_thresh_preds(pred))

            preds.append(pred)
    return preds


    
    
def optimize(
    candidate_df:pd.DataFrame,
    prob_df:pd.DataFrame,
    num_kfolds:int,
    weights_filepath_dict:dict, #example: {'lgbm':['filepath1', 'filepath2'], 'xgb':['filepath1', 'filepath2']}
):
    feature_names = bird_recognition.feature_extraction.get_feature_names()
    X = candidate_df[feature_names]
    y_preda_list = []
    for mode in weights_filepath_dict.keys():
        fold_y_preda_list = []
        for kfold_index in range(num_kfolds):
            clf = pickle.load(open(weights_filepath_dict[mode][kfold_index], "rb"))
            if mode=='lgbm':
                y_preda = clf.predict(X, num_iteration=clf.best_iteration)
            elif mode=='lgbm_rank':
                y_preda = clf.predict(X, num_iteration=clf.best_iteration)
            else:
                y_preda = clf.predict_proba(X)[:,1]
            fold_y_preda_list.append(y_preda)
        mean_preda = np.mean(fold_y_preda_list, axis=0)
        if mode=='lgbm_rank': # スケーリング
            mean_preda = 1/(1 + np.exp(-mean_preda))
        y_preda_list.append(mean_preda)
    y_preda = np.mean(y_preda_list, axis=0)
    candidate_df["y_preda"] = y_preda
    
    def f(th):
        _df = candidate_df[y_preda > th]
        if len(_df) == 0:
            return 0
        _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["label"].apply(
            lambda _: " ".join(_)
        ).rename(columns={
            "label": "predictions"
        })
        submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
        submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"
        return submission_df.apply(
            lambda row: bird_recognition.metrics.get_metrics(row["birds"], row["predictions"])["f1"],
            axis=1
        ).mean()
    
    lb, ub = 0, 1
    for k in range(30):
        th1 = (lb * 2 + ub) / 3
        th2 = (lb + ub * 2) / 3
        if f(th1) < f(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    print("-" * 30)
    print("## 下記の閾値をメモして，動作確認時にモデル動作確認用のF1値と一致していることを確認")
    print("📌best threshold: %f" % th)
    print("best F1: %f" % f(th))
    
    # nocall injection
    _df = candidate_df[y_preda > th]
    if len(_df) == 0:
        return 0
    _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
    )["label"].apply(
        lambda _: " ".join(_)
    ).rename(columns={
        "label": "predictions"
    })
    submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
    submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"

    
    _gdf2 = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
    )["y_preda"].sum()
    submission_df = pd.merge(
            submission_df,
            _gdf2,
            how="left",
            on=["audio_id", "seconds"]
        )
    def f_nocall(nocall_th):
        submission_df_with_nocall = submission_df.copy()
        submission_df_with_nocall.loc[(submission_df_with_nocall["y_preda"]<nocall_th) 
                                      & (submission_df_with_nocall["predictions"]!="nocall")
                                       & (submission_df_with_nocall["birds"].str.split().str.len()==1), "predictions"] += " nocall"
        return submission_df_with_nocall.apply(
            lambda row: bird_recognition.metrics.get_metrics(row["birds"], row["predictions"])["f1"],
            axis=1
        ).mean()
    lb, ub = 0, 1
    for k in range(30):
        th1 = (lb * 2 + ub) / 3
        th2 = (lb + ub * 2) / 3
        if f_nocall(th1) < f_nocall(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    print("-" * 30)
    print("## nocall injection")
    print("📌best nocall threshold: %f" % th)
    print("best F1: %f" % f_nocall(th))
    
    

def make_submission(
    candidate_df:pd.DataFrame,
    prob_df:pd.DataFrame,
    num_kfolds:int,
    th:float,
    nocall_th:float,
    weights_filepath_dict:dict,
    max_distance:int
):
    feature_names = bird_recognition.feature_extraction.get_feature_names()
    X = candidate_df[feature_names]
    y_preda_list = []
    for mode in weights_filepath_dict.keys():
        fold_y_preda_list = []
        for kfold_index in range(num_kfolds):
            clf = pickle.load(open(weights_filepath_dict[mode][kfold_index], "rb"))
            if mode=='lgbm':
                y_preda = clf.predict(X, num_iteration=clf.best_iteration)
            elif mode=='lgbm_rank':
                y_preda = clf.predict(X, num_iteration=clf.best_iteration)
            else:
                y_preda = clf.predict_proba(X)[:,1]
            fold_y_preda_list.append(y_preda)
        mean_preda = np.mean(fold_y_preda_list, axis=0)
        if mode=='lgbm_rank':  # スケーリング
            mean_preda = 1/(1 + np.exp(-mean_preda))
        y_preda_list.append(mean_preda)
    y_preda = np.mean(y_preda_list, axis=0)
    candidate_df["y_preda"] = y_preda
    
    _df = candidate_df[y_preda > th]
    _gdf = _df.groupby(
        ["audio_id", "seconds"],
        as_index=False
    )["label"].apply(
        lambda _: " ".join(_)
    ).rename(columns={
        "label": "predictions"
    })
    submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
    submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"
    if TARGET_PATH:
        score_df = pd.DataFrame(
            submission_df.apply(
                lambda row: bird_recognition.metrics.get_metrics(row["birds"], row["predictions"]),
                axis=1
            ).tolist()
        )
        print("-" * 30)
        print("BEFORE(nocall injection 前)")
        print("図鑑で学習済みモデルでのCVスコア(モデルの動作確認用)")
        print("上のピン📌と一致するか確認")
        print("F1: %.4f" % score_df["f1"].mean())
        print("Recall: %.4f" % score_df["rec"].mean())
        print("Precision: %.4f" % score_df["prec"].mean())
        
    # nocall injection
    _gdf2 = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
    )["y_preda"].sum()
    submission_df = pd.merge(
            submission_df,
            _gdf2,
            how="left",
            on=["audio_id", "seconds"]
        )
    submission_df.loc[(submission_df["y_preda"] < nocall_th) 
                     & (submission_df["predictions"]!="nocall")
                      & (submission_df["birds"].str.split().str.len()==1), "predictions"] += " nocall"
    if TARGET_PATH:
        score_df = pd.DataFrame(
            submission_df.apply(
                lambda row: bird_recognition.metrics.get_metrics(row["birds"], row["predictions"]),
                axis=1
            ).tolist()
        )
        print("-" * 30)
        print("AFTER(nocall injection 後)")
        print("図鑑で学習済みモデルでのCVスコア(モデルの動作確認用)")
        print("上のピン📌と一致するか確認")
        print("F1: %.4f" % score_df["f1"].mean())
        print("Recall: %.4f" % score_df["rec"].mean())
        print("Precision: %.4f" % score_df["prec"].mean())
                
    return submission_df[["row_id", "predictions"]].rename(columns={
        "predictions": "birds"
    })

def get_prob_df(config, audio_paths):
    data = pd.DataFrame(
         [(path.stem, *path.stem.split("_"), path) for path in Path(audio_paths).glob("*.ogg")],
        columns = ["filename", "id", "site", "date", "filepath"]
    )
    LABEL_IDS = bird_recognition.datasets.get_bird_label_to_index()
    INV_LABEL_IDS = bird_recognition.datasets.get_bird_index_to_label()
    test_data = BirdCLEFDataset(data=data)

    for checkpoint_path in config.checkpoint_paths:
        prob_filepath = config.get_prob_filepath_from_checkpoint(checkpoint_path)
        if (not os.path.exists(prob_filepath)) or (TARGET_PATH is None):  # キャッシュがない or 提出時は必ず計算
            nets = [load_net(checkpoint_path.as_posix())]
            pred_probas = predict(nets, test_data, names=False)
            if TARGET_PATH: # 手元                
                df = pd.read_csv(TARGET_PATH, usecols=["row_id", "birds"])
            else: # 提出時
                if str(audio_paths)=="../input/birdclef-2021/train_soundscapes":
                    print(audio_paths)
                    df = pd.read_csv(Path("../input/birdclef-2021/train_soundscape_labels.csv"), usecols=["row_id", "birds"])
                else:
                    print(SAMPLE_SUB_PATH)
                    df = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id", "birds"])
            df["audio_id"] = df["row_id"].apply(lambda _: int(_.split("_")[0]))
            df["site"] = df["row_id"].apply(lambda _: _.split("_")[1])
            df["seconds"] = df["row_id"].apply(lambda _: int(_.split("_")[2]))
            assert len(data) == len(pred_probas)
            n = len(data)
            audio_id_to_date = {}
            audio_id_to_site = {}
            for filepath in audio_paths.glob("*.ogg"):
                audio_id, site, date = os.path.basename(filepath).replace(".ogg", "").split("_")
                audio_id = int(audio_id)
                audio_id_to_date[audio_id] = date
                audio_id_to_site[audio_id] = site
            dfs = []
            for i in range(n):
                row = data.iloc[i]
                audio_id = int(row["id"])
                pred = pred_probas[i]
                _df = pd.DataFrame(pred.to("cpu").numpy())
                _df.columns = [INV_LABEL_IDS[j] for j in range(_df.shape[1])]
                _df["audio_id"] = audio_id
                _df["date"] = audio_id_to_date[audio_id]
                _df["site"] = audio_id_to_site[audio_id]
                _df["seconds"] = [(j+1)*5 for j in range(120)]
                dfs.append(_df)
            prob_df = pd.concat(dfs)
            prob_df = pd.merge(prob_df, df, how="left", on=["site", "audio_id", "seconds"])
            print(f"Save to {prob_filepath}")
            prob_df.to_csv(prob_filepath, index=False)

    # 予測結果のアンサンブル
    prob_df = pd.read_csv(
        config.get_prob_filepath_from_checkpoint(config.checkpoint_paths[0])
    )
    if len(config.checkpoint_paths) > 1:
        columns = bird_recognition.datasets.get_bird_columns()
        for checkpoint_path in config.checkpoint_paths[1:]:
            _df = pd.read_csv(
                config.get_prob_filepath_from_checkpoint(checkpoint_path)
            )
            prob_df[columns] += _df[columns]
        prob_df[columns] /= len(config.checkpoint_paths)

    return prob_df

def run(training_config, config, prob_df, model_dict):
    ####################################################
    # テーブルコンペ部分のモデルの訓練
    # 外部モデルが指定されているならスキップできるとかにする? 
    ####################################################
    
    # short audio
    if training_config.min_rating:
        print("before: %d" % len(prob_df))
        prob_df = prob_df[prob_df["rating"] >= 3.0].reset_index(drop=True)
        print("after: %d" % len(prob_df))

    # 学習する必要のない項目を除外
    # 図鑑の場合
    if not "site" in prob_df.columns:
        prob_df["site"] = prob_df.apply(
            lambda row: bird_recognition.feature_extraction.to_site(
                row,
                max_distance=training_config.max_distance
            ),
            axis=1
        )
        print("[exclude other]before: %d" % len(prob_df))
        prob_df = prob_df[prob_df["site"] != "Other"].reset_index(drop=True)
        print("[exclude other]after: %d" % len(prob_df))


    candidate_df = bird_recognition.candidate_extraction.make_candidates(
        prob_df,
        num_spieces=training_config.num_spieces,
        num_candidates=training_config.num_candidates,
        max_distance=training_config.max_distance
    )
    candidate_df = bird_recognition.feature_extraction.add_features(
        candidate_df,
        prob_df,
        max_distance=training_config.max_distance
    )
   
    # soundscapes
    prob_df_soundscapes = bird_recognition.evaluation.get_prob_df(config,Path("../input/birdclef-2021/train_soundscapes"))
    candidate_df_soundscapes = bird_recognition.candidate_extraction.make_candidates(
        prob_df_soundscapes,
        num_spieces=training_config.num_spieces,
        num_candidates=training_config.num_candidates,
        max_distance=training_config.max_distance
    )
    candidate_df_soundscapes = bird_recognition.feature_extraction.add_features(
        candidate_df_soundscapes,
        prob_df_soundscapes,
        max_distance=training_config.max_distance
    )    
    
    
    for mode in model_dict.keys():
        print(f'training of {mode} is going...')
        bird_recognition.training.train(
            candidate_df,
            prob_df,
            candidate_df_soundscapes,
            prob_df_soundscapes,
            num_kfolds=training_config.num_kfolds,
            num_candidates=training_config.num_candidates,
            weight_rate=training_config.weight_rate,
            verbose=True,
            xgb_params=training_config.xgb_params,
            mode=mode,
        )

    ######################################################
    # 以下提出用
    ######################################################
    prob_df = get_prob_df(config, TEST_AUDIO_ROOT)

    # 候補抽出
    candidate_df = bird_recognition.candidate_extraction.make_candidates(
        prob_df,
        num_spieces=config.num_spieces,
        num_candidates=config.num_candidates,
        max_distance=config.max_distance,
        num_prob=config.num_prob,
        nocall_threshold=config.nocall_threshold
    )
    # 特徴量の追加
    candidate_df = bird_recognition.feature_extraction.add_features(
        candidate_df,
        prob_df,
        max_distance=config.max_distance
    )
    
    if TARGET_PATH:
        optimize(
            candidate_df,
            prob_df,
            num_kfolds=config.num_kfolds,
            weights_filepath_dict=config.weights_filepath_dict,
        )
    if config.check_baseline:
        print("-" * 30)
        print("閾値でバサッと切ったCVスコア(参考値)")
        bird_recognition.baseline.calc_baseline(prob_df)

    submission_df = make_submission(
        candidate_df,
        prob_df,
        num_kfolds=config.num_kfolds,
        th=config.threshold,
        nocall_th=config.nocall_threshold,
        weights_filepath_dict=config.weights_filepath_dict,
        max_distance=config.max_distance
    )
    return submission_df



