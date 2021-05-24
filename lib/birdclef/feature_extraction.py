import pandas as pd
from .datasets import get_locations

def to_site(row, th=15):
    for location in get_locations():
        site = location["site"]
        if row[f"dist_to_{site}"] < th:
            return site
    return "Other"

def add_distances(df:pd.DataFrame):
    for location in get_locations():
        def to_dist(row):
            dy = row["latitude"] - location["latitude"]
            dx = row["longitude"] - location["longitude"]
            return (dx**2 + dy**2) ** 0.5
        df[f"dist_to_{location['site']}"] = df.apply(to_dist, axis=1)
    df["site"] = df.apply(to_site, axis=1)

def add_features(df:pd.DataFrame):
    add_distances(df)


