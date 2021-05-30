from typing import List
import numpy as np
import pandas as pd
from . import datasets
from . import feature_extraction


def to_birds(row, th:float) -> str:
    if row["call_prob"] < th:
        return "nocall"
    res = [row["primary_label"]] + eval(row["secondary_labels"])
    return " ".join(res)

def make_candidates(
    prob_df:pd.DataFrame,
    num_spieces:int,
    num_candidates:int,
    max_distance:int,
    num_prob:int=6, # 前後の確保するフレームの数(3なら前のフレーム3個, 後のフレーム3個)
    nocall_threshold:float=0.5,
):
    if "author" in prob_df.columns: # メタデータ(図鑑/short audio)
        prob_df["birds"] = prob_df.apply(
            lambda row: to_birds(row, th=nocall_threshold),
            axis=1
        )
        print("Candidate nocall ratio: %.4f" % (prob_df["birds"] == "nocall").mean())
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
        
    sum_prob_list = prob_df[datasets.get_bird_columns()].sum(axis=1).tolist()
    mean_prob_list = prob_df[datasets.get_bird_columns()].mean(axis=1).tolist()
    std_prob_list = prob_df[datasets.get_bird_columns()].std(axis=1).tolist()
    max_prob_list = prob_df[datasets.get_bird_columns()].max(axis=1).tolist()
    min_prob_list = prob_df[datasets.get_bird_columns()].min(axis=1).tolist()
    skew_prob_list = prob_df[datasets.get_bird_columns()].skew(axis=1).tolist()
    kurt_prob_list = prob_df[datasets.get_bird_columns()].kurt(axis=1).tolist()
    
    label_to_index = datasets.get_bird_label_to_index()
    index_to_label = datasets.get_bird_index_to_label()
    bird_columns = datasets.get_bird_columns()
    X = prob_df[bird_columns].values
    bird_ids_list = np.argsort(-X)[:,:num_candidates]
    row_ids = prob_df["row_id"].tolist()
    rows = [i//num_candidates for i in range(len(bird_ids_list.flatten()))]
    cols = bird_ids_list.flatten()
    # 何番目の候補か
    ranks = [i%num_candidates for i in range(len(rows))]
    probs_list = X[rows, cols]
    D = {
        "row_id": [row_ids[i] for i in rows],
        "rank": ranks,
        "bird_id": bird_ids_list.flatten(),
        "prob": probs_list.flatten(),
        "sum_prob": [sum_prob_list[i//num_candidates] for i in range(num_candidates*len(mean_prob_list))],
        "mean_prob": [mean_prob_list[i//num_candidates] for i in range(num_candidates*len(mean_prob_list))],
        "std_prob": [std_prob_list[i//num_candidates] for i in range(num_candidates*len(std_prob_list))],
        "max_prob": [max_prob_list[i//num_candidates] for i in range(num_candidates*len(max_prob_list))],
        "min_prob": [min_prob_list[i//num_candidates] for i in range(num_candidates*len(min_prob_list))],
        "skew_prob": [skew_prob_list[i//num_candidates] for i in range(num_candidates*len(skew_prob_list))],
        "kurt_prob": [kurt_prob_list[i//num_candidates] for i in range(num_candidates*len(kurt_prob_list))],
    }
    audio_ids = prob_df["audio_id"].values[rows]
    for diff in range(-num_prob, num_prob+1):
        if diff == 0:
            continue
        neighbor_audio_ids = prob_df["audio_id"].shift(diff).values[rows]
        Y = prob_df[bird_columns].shift(diff).values
        c = f"next{abs(diff)}_prob" if diff < 0 else f"prev{diff}_prob"
        c = c.replace("1_prob", "_prob") # next1_probをnext_probに修正
        v = Y[rows, cols].flatten()
        v[audio_ids != neighbor_audio_ids] = np.nan
        D[c] = v

    candidate_df = pd.DataFrame(D)
    columns = [
        "row_id",
        "site",
        "year",
        "month",
        "audio_id",
        "seconds",
        "birds",
    ]
    candidate_df = pd.merge(
        candidate_df,
        prob_df[columns],
        how="left",
        on="row_id"
    )
    candidate_df["target"] = candidate_df.apply(
        lambda row: index_to_label[row["bird_id"]] in set(row["birds"].split()),
        axis=1
    )
    candidate_df["label"] = candidate_df["bird_id"].map(index_to_label)
    return candidate_df

