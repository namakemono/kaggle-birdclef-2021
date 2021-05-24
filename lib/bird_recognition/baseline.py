import numpy as np
import pandas as pd
from . import datasets
from . import metrics

def calc_baseline(prob_df:pd.DataFrame):
    """単純に閾値だけでのF1スコアの最適値を算出"""
    columns = datasets.get_bird_columns()
    index_to_label = datasets.get_bird_index_to_label()
    X = prob_df[columns].values
    def f(th):
        n = X.shape[0]
        pred_labels = [[] for i in range(n)]
        I, J = np.where(X > th)
        for i, j in zip(I, J):
            pred_labels[i].append(index_to_label[j])
        for i in range(n):
            if len(pred_labels[i]) == 0:
                pred_labels[i] = "nocall"
            else:
                pred_labels[i] = " ".join(pred_labels[i])
        prob_df["pred_labels"] = pred_labels
        return prob_df.apply(
            lambda _: metrics.get_metrics(_["birds"], _["pred_labels"])["f1"],
            axis=1
        ).mean()

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


