def f1_score(y_true:str, y_pred:str):
    """文字列形式ののF1スコア
    空白で区切る
    >>> to_f1_score("a b", "b c")
    0.3333333333333333
    """
    S = set(y_true.split())
    T = set(y_pred.split())
    return len(S & T) / len(S | T)

def get_metrics(s_true, s_pred):
    s_true = set(s_true.split())
    s_pred = set(s_pred.split())
    n, n_true, n_pred = len(s_true.intersection(s_pred)), len(s_true), len(s_pred)
    prec = n/n_pred
    rec = n/n_true
    f1 = 2*prec*rec/(prec + rec) if prec + rec else 0
    return {
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "n_true": n_true,
        "n_pred": n_pred,
        "n": n
    }
