from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from . import datasets
from . import feature_extraction

def train(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame,
    num_kfolds:int,
    num_candidates:int
):
    feature_names = feature_extraction.get_feature_names()
    print("features", feature_names)
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


