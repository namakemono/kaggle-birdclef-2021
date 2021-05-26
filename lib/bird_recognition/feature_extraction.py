from typing import List
import numpy as np
import pandas as pd
from . import datasets

def to_site(row, max_distance:int):
    best = max_distance
    answer = "Other"
    for location in datasets.get_locations():
        x = (row["latitude"] - location["latitude"])
        y = (row["longitude"] - location["longitude"])
        dist = (x**2 + y**2) ** 0.5
        if dist < best:
            best = dist
            answer = location["site"]
    return answer

def to_latitude(site:str) -> str:
    for location in datasets.get_locations():
        if site == location["site"]:
            return location["latitude"]
    return -10000

def to_longitude(site:str) -> str:
    for location in datasets.get_locations():
        if site == location["site"]:
            return location["longitude"]
    return -10000

def add_features(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame,
    max_distance:int,
) -> pd.DataFrame:
    meta_df = datasets.load_metadata()
    # 緯度経度
    if not "latitude" in candidate_df.columns:
        candidate_df["latitude"] = candidate_df["site"].apply(to_latitude)
    if not "longitude" in candidate_df.columns:
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

    # 前後フレームの平均

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
        "sum_prob",
        "mean_prob",
        #"std_prob",
        "max_prob",
        #"min_prob",
        #"skew_prob",
        #"kurt_prob",
        "prev6_prob",
        "prev5_prob",
        "prev4_prob",
        "prev3_prob",
        "prev2_prob",
        "prev_prob",
        "prob",
        "next_prob",
        "next2_prob",
        "next3_prob",
        "next4_prob",
        "next5_prob",
        "next6_prob",
        "rank",
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


