import pandas as pd
from .datasets import load_metadata
from .feature_extraction import to_site

def filter_by_rules(
    submission_df:pd.DataFrame,
    max_distance:int,
) -> pd.DataFrame:
    """
    ルールベースで起こりづらそうな地域x時期を予測している候補を除外
    submission_dfにはmonth, site, predictionsが含まれている必要あり
    """
    # 時期と地域でフィルタリング
    meta_df = pd.read_csv(
        "../input/metadata-probability-v0525-2100/birdclef_resnest50_fold1_epoch_34_f1_val_04757_20210524185455.csv"
    )
    meta_df["month"] = meta_df["date"].apply(lambda _: _.split("-")[1]).astype(int)
    # meta_df = load_metadata()
    meta_df["site"] = meta_df.apply(
        lambda row: to_site(row, max_distance=max_distance),
        axis=1
    )
    # バグった日付のデータは除外して，site x monthでprimary labelの集合を作る
    _gdf = meta_df[meta_df["month"].isin(list(range(1,13)))].groupby(
        ["site", "month"],
        as_index=False
    ).agg({
        "primary_label": set 
    })
    site_month_to_spieces = {}
    for idx, row in _gdf.iterrows():
        for d in [-1,0,1]:
            month = (row["month"] + d) % 12 + 1
            key = (row["site"], month)
            site_month_to_spieces[key] = site_month_to_spieces.get(key, set()) | row["primary_label"]
    def filter_by_site_and_month(row):
        if row["predictions"] == "nocall":
            return "nocall"
        else:
            k = (row["site"], row["month"])
            spieces = site_month_to_spieces.get(k, set())
            res = []
            for target in row["predictions"].split():
                if target in spieces:
                    res.append(target)
            if len(res) == 0:
                return "nocall"
            else:
                return " ".join(res) 
    submission_df["predictions"] = submission_df.apply(filter_by_site_and_month, axis=1)
    return submission_df
