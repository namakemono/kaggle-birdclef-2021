import os
import sys
import pandas as pd

def get_root_directory() -> str:
    return "/content" if ("google.colab" in sys.modules) else "../input"

def load_external_train_data():
    df = pd.read_csv(
        os.path.join(
            get_root_directory(),
            "birdclef-2021/train_metadata.csv"
        )
    )
    df["filepath"] = df.apply(
        lambda row: os.path.join(
            get_root_directory(),
            "birdclef-2021/train_short_audio",
            row["primary_label"],
            row["filename"]
        ),
        axis=1
    )
    return df

def load_train_data():
    df = pd.read_csv(
        os.path.join(
            get_root_directory(),
            "birdclef-2021/train_soundscape_labels.csv"
        )
    )
    df["filepath"] = df["row_id"].apply(
        lambda _: os.path.join(
            get_root_directory(),
            f"birdclef-2021/train_soundscapes/{_}.oog"
        )
    )
    return df

def load_test_data():
    df = pd.read_csv(
        os.path.join(
            get_root_directory(),
            "birdclef-2021/test.csv"
        )
    )
    df["filepath"] = df["row_id"].apply(
        lambda _: os.path.join(
            get_root_directory(),
            f"birdclef-2021/test_soundscapes/{_}.oog"
        )
    )
    return df
