import warnings
warnings.filterwarnings('ignore')
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from . import datasets

def get_locations():
    return [{
        "site": "COL",
        "latitude": 5.57,
        "longitude": -75.85
    }, {
        "site": "COR",
        "latitude": 10.12,
        "longitude": -84.51
    }, {
        "site": "SNE",
        "latitude": 38.49,
        "longitude": -119.95
    }, {
        "site": "SSW",
        "latitude": 42.47,
        "longitude": -76.45
    }]

def to_site(row, max_distance:int):
    best = max_distance
    answer = "Other"
    for location in get_locations():
        x = (row["latitude"] - location["latitude"])
        y = (row["longitude"] - location["longitude"])
        dist = (x**2 + y**2) ** 0.5
        if dist < best:
            best = dist
            answer = location["site"]
    return answer

def to_latitude(site:str) -> str:
    for location in get_locations():
        if site == location["site"]:
            return location["latitude"]
    return -10000

def to_longitude(site:str) -> str:
    for location in get_locations():
        if site == location["site"]:
            return location["longitude"]
    return -10000

# datasets.py
def load_metadata():
    meta_df = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
    meta_df["id"] = meta_df.index + 1
    meta_df["year"] = meta_df["date"].apply(lambda _: _.split("-")[0]).astype(int)
    meta_df["month"] = meta_df["date"].apply(lambda _: _.split("-")[1]).astype(int)
    return meta_df

def make_candidates(df:pd.DataFrame, num_spieces:int, num_candidates:int):
    """
    Parameters
    ----------
    df - サンプルごとに各鳥の鳴く確率が入ったデータ
    """
    df["year"] = df["date"].apply(lambda _: int(str(_)[:4]))
    df["month"] = df["date"].apply(lambda _: int(str(_)[4:6]))
    df["latitude"] = df["site"].apply(to_latitude)
    df["longitude"] = df["site"].apply(to_longitude)
    label_to_index = {label:index for index, label in enumerate(datasets.get_bird_columns())}
    label_to_index["nocall"] = -1
    index_to_label = {v:k for k, v in label_to_index.items()}
    records = []
    n = len(df)
    X = df[df.columns[:num_spieces]].values
    scores = []
    prev_row = None
    next_row = None
    for i in range(n):
        row = df.iloc[i]
        S = set(row["birds"].split())
        probs = -np.sort(-X[i])[:num_candidates]
        bird_ids = np.argsort(-X[i])[:num_candidates]
        labels = [index_to_label[index] for index in bird_ids]
        if row["birds"] != "nocall":
            T = set(["nocall"] + labels)
            score = len(S&T) / len(S)
            scores.append(score)
        if i > 0:
            prev_row = df.iloc[i-1]
        if i+1 < n:
            next_row = df.iloc[i+1]
        for prob, bird_id, label in zip(probs, bird_ids, labels):
            record = {
                "site": row["site"],
                "year": row["year"],
                "month": row["month"],
                "prob": prob,
                "bird_id": bird_id,
                "label": label,
                "audio_id": row["audio_id"],
                "seconds": row["seconds"],
                "target": int(label in S)
            }
            if prev_row is not None:
                if prev_row["audio_id"] == row["audio_id"]:
                    record["prev_prob"] = prev_row[label]
            if next_row is not None:
                if next_row["audio_id"] == row["audio_id"]:
                    record["next_prob"] = next_row[label]
            records.append(record)
    candidate_df = pd.DataFrame(records)
    print("候補数: %d, 網羅率: %.4f" % (num_candidates, np.mean(scores)))
    return candidate_df

def add_features(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame,
    max_distance:int,
) -> pd.DataFrame:
    meta_df = load_metadata()
    # 緯度経度
    candidate_df["latitude"] = candidate_df["site"].apply(to_latitude)
    candidate_df["longitude"] = candidate_df["site"].apply(to_longitude)
    # 出現回数
    candidate_df["num_appear"] = candidate_df["label"].map(
        meta_df["primary_label"].value_counts()
    )
    meta_df["site"] = meta_df.apply(
        lambda row: to_site(
            row,
            max_distance=max_distance
        ),
        axis=1
    )

    # 地域ごとの出現回数
    _df = meta_df.groupby(
        ["primary_label", "site"],
        as_index=False
    )["id"].count().rename(
        columns={
            "primary_label": "label",
            "id": "site_num_appear"
        }
    )
    candidate_df = pd.merge(
        candidate_df,
        _df,
        how="left",
        on=["label", "site"]
    )
    candidate_df["site_appear_ratio"] = candidate_df["site_num_appear"] / candidate_df["num_appear"]
    # 季節ごとの統計量
    _df = meta_df.groupby(
        ["primary_label", "month"],
        as_index=False
    )["id"].count().rename(
        columns={
            "primary_label": "label",
            "id": "month_num_appear"
        }
    )
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["label", "month"])
    candidate_df["month_appear_ratio"] = candidate_df["month_num_appear"] / candidate_df["num_appear"]

    candidate_df = add_same_audio_features(candidate_df, df)

    # 確率の補正(全部下がった)
    candidate_df["prob / num_appear"] = candidate_df["prob"] / (candidate_df["num_appear"].fillna(0) + 1)
    candidate_df["prob / site_num_appear"] = candidate_df["prob"] / (candidate_df["site_num_appear"].fillna(0) + 1)
    candidate_df["prob * site_appear_ratio"] = candidate_df["prob"] * (candidate_df["site_appear_ratio"].fillna(0) + 0.001)

    # 前後のフレームとの変化量
    candidate_df["prob_avg"] = candidate_df[["prev_prob", "prob", "next_prob"]].mean(axis=1)
    candidate_df["prob_diff"] = candidate_df["prob"] - candidate_df["prob_avg"]
    candidate_df["prob - prob_max_in_same_audio"] = candidate_df["prob"] - candidate_df["prob_max_in_same_audio"]
    return candidate_df


def to_zscore(row):
    x = row["prob"]
    mu = row["prob_avg_in_same_audio"]
    sigma = row["prob_var_in_same_audio"] ** 0.5
    if sigma < 1e-6:
        return 0
    else:
        return (x - mu) / sigma

def add_same_audio_features(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame
):
    bird_columns = datasets.get_bird_columns()
    # 同一audio内での鳥ごとの平均確率
    _gdf = df.groupby(["audio_id"], as_index=False).mean()[["audio_id"] + bird_columns]
    _df = pd.melt(
        _gdf,
        id_vars=["audio_id"]
    ).rename(columns={
        "variable": "label",
        "value": "prob_avg_in_same_audio"
    })
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
    # 同一audio内での鳥ごとの最大値
    _gdf = df.groupby(["audio_id"], as_index=False).max()[["audio_id"] + bird_columns]
    _df = pd.melt(
        _gdf,
        id_vars=["audio_id"]
    ).rename(columns={
        "variable": "label",
        "value": "prob_max_in_same_audio"
    })
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
    # 同一audio内での鳥ごとの分散
    _gdf = df.groupby(["audio_id"], as_index=False).var()[["audio_id"] + bird_columns]
    _df = pd.melt(
        _gdf,
        id_vars=["audio_id"]
    ).rename(columns={
        "variable": "label",
        "value": "prob_var_in_same_audio"
    })
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
    candidate_df["zscore_in_same_audio"] = candidate_df.apply(to_zscore, axis=1)
    return candidate_df

def get_feature_names() -> List[str]:
    return [
        "year",
        "month",
        "prev_prob",
        "prob",
        "next_prob",
        "latitude",
        "longitude",
        "bird_id", # 📈, +0.013700
        "seconds", # 📈, -0.0050
        "num_appear",
        "site_num_appear",
        "site_appear_ratio",
        # "prob / num_appear", # 📉, -0.005
        # "prob / site_num_appear", # 📉, -0.0102
        # "prob * site_appear_ratio", # 📉, -0.0049
        # "prob_avg", # 📉, -0.0155
        "prob_diff", # 📈, 0.0082
        # "prob_avg_in_same_audio", # 📉, -0.0256
        # "prob_max_in_same_audio", # 📉, -0.0142
        # "prob_var_in_same_audio", # 📉, -0.0304
        # "prob - prob_max_in_same_audio", # 📉, -0.0069
        # "zscore_in_same_audio", # 📉, -0.0110
        # "month_num_appear", # 📉, 0.0164
    ]

def train(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame,
    num_kfolds:int,
    num_candidates:int
):
    feature_names = get_feature_names()
    print("\n".join(feature_names))
    X = candidate_df[feature_names].values
    y = candidate_df["target"].values
    groups = candidate_df["audio_id"]
    kf = StratifiedGroupKFold(n_splits=num_kfolds)
    for kfold_index, (_, valid_index) in enumerate(kf.split(X, y, groups)):
        candidate_df.loc[valid_index, "fold"] = kfold_index
    oof = np.zeros(len(y), dtype=int)
    for kfold_index in range(num_kfolds):
        train_index = candidate_df[candidate_df["fold"] != kfold_index].index
        valid_index = candidate_df[candidate_df["fold"] == kfold_index].index
        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]
        # 正例の重みを weight_rate, 負例を1にする
        weight_rate = 2.5
        sample_weight = np.ones(y_train.shape)
        sample_weight[y_train==1] = weight_rate
        sample_weight_val = np.ones(y_valid.shape)
        sample_weight_val[y_valid==1] = weight_rate
        sample_weight_eval_set = [sample_weight, sample_weight_val]

        clf = xgb.XGBClassifier(
            objective = "binary:logistic",
        )
        clf.fit(
            X_train, y_train,
            eval_set = [
                (X_train, y_train),
                (X_valid, y_valid)
            ],
            eval_metric = "logloss",
            verbose = 10,
            early_stopping_rounds = 20,
            sample_weight = sample_weight,
            sample_weight_eval_set = sample_weight_eval_set,
        )
        oof[valid_index] = clf.predict(X_valid)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    print("Call or No call classirication")
    print("candiates: %d" % num_candidates)
    print("F1: %.4f" % f1_score(y, oof))
    print("gt positive ratio: %.4f" % np.mean(y))
    print("oof positive ratio: %.4f" % np.mean(oof))
    print("Accuracy: %.4f" % accuracy_score(y, oof))
    print("Recall: %.4f" % recall_score(y, oof))
    print("Precision: %.4f" % precision_score(y, oof))

    _df = candidate_df[oof == 1]
    _gdf = _df.groupby(["audio_id", "seconds"], as_index=False)["label"].apply(lambda _: " ".join(_))
    df2 = pd.merge(df[["audio_id", "seconds", "birds"]], _gdf, how="left", on=["audio_id", "seconds"])
    df2.loc[df2["label"].isnull(), "label"] = "nocall"
    print("F1: %.4f" % df2.apply(to_f1_score, axis=1).mean())

def to_f1_score(row):
    S = set(row["birds"].split())
    T = set(row["label"].split())
    return len(S & T) / len(S | T)

def calc_baseline(df, num_spieces):
    """単純に閾値だけでのF1スコアの最適値を算出"""
    columns = datasets.get_bird_columns()
    label_to_index = {label:index for index, label in enumerate(datasets.get_bird_columns())}
    label_to_index["nocall"] = -1
    index_to_label = {v:k for k, v in label_to_index.items()}

    X = df[columns].values
    def f(th):
        n = X.shape[0]
        pred_labels = [[] for i in range(n)]
        I, J = np.where(X > th)
        for i, j in zip(I, J):
            pred_labels[i].append(index_to_label[j])
        scores = []
        for i in range(n):
            if len(pred_labels[i]) == 0:
                S = set(["nocall"])
            else:
                S = set(pred_labels[i])
            T = set(df.iloc[i]["birds"].split())
            scores.append(len(S & T) / len(S | T))
        return np.mean(scores)

    lb, ub = 0, 1
    for k in range(30):
        th1 = (2*lb + ub) / 3
        th2 = (lb + 2*ub) / 3
        if f(th1) < f(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    print("best th: %.4f" % th)
    print("best F1: %.4f" % f(th))
    return th


