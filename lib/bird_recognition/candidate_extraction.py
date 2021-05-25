from typing import List
import numpy as np
import pandas as pd
from . import datasets
from . import feature_extraction

def make_candidates(
    prob_df:pd.DataFrame,
    num_spieces:int,
    num_candidates:int,
    max_distance:int
):
    """
    Parameters
    ----------
    prob_prob_df - サンプルごとに各鳥の鳴く確率が入ったデータ
    """
    if "author" in prob_df.columns: # メタデータ(図鑑/short audio)
        prob_df["audio_id"] = prob_df["filename"].apply(
            lambda _: int(_.replace("XC", "").replace(".ogg", ""))
        )
        prob_df["row_id"] = prob_df.apply(
            lambda row: "%s_%s" % (row["audio_id"], row["seconds"]),
            axis=1
        )
        prob_df["year"] = prob_df["date"].apply(lambda _: int(_.split("-")[0]))
        prob_df["month"] = prob_df["date"].apply(lambda _: int(_.split("-")[1]))
        prob_df["site"] = prob_df.apply(
            lambda row: feature_extraction.to_site(row, max_distance),
            axis=1
        )
    else:
        prob_df["year"] = prob_df["date"].apply(lambda _: int(str(_)[:4]))
        prob_df["month"] = prob_df["date"].apply(lambda _: int(str(_)[4:6]))
        prob_df["latitude"] = prob_df["site"].apply(feature_extraction.to_latitude)
        prob_df["longitude"] = prob_df["site"].apply(feature_extraction.to_longitude)
    label_to_index = datasets.get_bird_label_to_index()
    index_to_label = datasets.get_bird_index_to_label()
    records = []
    n = len(prob_df)
    X = prob_df[datasets.get_bird_columns()].values
    scores = []
    prev_row = None
    next_row = None
    for i in range(n):
        row = prob_df.iloc[i]
        S = set(row["birds"].split())
        probs = -np.sort(-X[i])[:num_candidates]
        bird_ids = np.argsort(-X[i])[:num_candidates]
        labels = [index_to_label[index] for index in bird_ids]
        if row["birds"] != "nocall":
            T = set(["nocall"] + labels)
            score = len(S&T) / len(S)
            scores.append(score)
        if i > 0:
            prev_row = prob_df.iloc[i-1]
        if i+1 < n:
            next_row = prob_df.iloc[i+1]
        for prob, bird_id, label in zip(probs, bird_ids, labels):
            record = {
                "row_id": row["row_id"], # row_id := audio_id + "_" + seconds
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


