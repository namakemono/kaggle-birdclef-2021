def f1_score(y_true:str, y_pred:str):
    """文字列形式ののF1スコア
    空白で区切る
    >>> to_f1_score("a b", "b c")
    0.3333333333333333
    """
    S = set(y_true.split())
    T = set(y_pred.split())
    return len(S & T) / len(S | T)


