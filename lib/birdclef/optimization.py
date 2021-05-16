import pandas as pd
from pathlib import Path
import torch
from .constants import LABEL_IDS, INV_LABEL_IDS
from .datasets import load_train_data
from .metrics import get_metrics
@torch.no_grad()
def get_thresh_preds(out, thresh=None):
    thresh = thresh or THRESH
    o = (-out).argsort(1)
    npreds = (out > thresh).sum(1)
    preds = []
    for oo, npred in zip(o, npreds):
        preds.append(oo[:npred].cpu().numpy().tolist())
    return preds

def get_bird_names(preds):
    bird_names = []
    for pred in preds:
        if not pred:
            bird_names.append("nocall")
        else:
            bird_names.append(" ".join([INV_LABEL_IDS[bird_id] for bird_id in pred]))
    return bird_names

def preds_as_df(
    data,
    preds,
    submission_filepath:str=None
) -> pd.DataFrame:
    sub = {
        "row_id": [],
        "birds": [],
    }

    for row, pred in zip(data.itertuples(False), preds):
        row_id = [f"{row.id}_{row.site}_{5*i}" for i in range(1, len(pred)+1)]
        sub["birds"] += pred
        sub["row_id"] += row_id

    sub = pd.DataFrame(sub)

    if submission_filepath is not None:
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id"])
        sub = sample_sub.merge(sub, on="row_id", how="left")
        sub["birds"] = sub["birds"].fillna("nocall")
    return sub

def optimize(data, pred_probas):
    lb, ub = 0, 1
    def f(th):
        preds = [get_bird_names(get_thresh_preds(pred, thresh=th)) for pred in pred_probas]
        sub = preds_as_df(data, preds)
        sub_target = load_train_data()
        sub_target = sub_target.merge(sub, how="left", on="row_id")

        assert sub_target["birds_x"].notnull().all()
        assert sub_target["birds_y"].notnull().all()

        df_metrics = pd.DataFrame([
            get_metrics(s_true, s_pred) for s_true, s_pred in zip(sub_target.birds_x, sub_target.birds_y)
        ])
        score = df_metrics["f1"].mean()
        return score

    for k in range(30):
        th1 = (2 * lb + ub) / 3
        th2 = (lb + 2 * ub) / 3
        if f(th1) > f(th2):
            ub = th2
        else:
            lb = th1
    return (lb + ub) / 2
