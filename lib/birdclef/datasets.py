import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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

class BirdClefDataset(Dataset):
    def __init__(
        self,
        audio_image_store,
        meta,
        sr:int,
        is_train:bool,
        num_classes:int,
        duration:int
    ):
        self.audio_image_store = audio_image_store
        self.meta = meta.copy().reset_index(drop=True)
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration*self.sr

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        image = self.audio_image_store[row.filename]

        image = image[np.random.choice(len(image))]
        image = self.normalize(image)

        t = np.zeros(self.num_classes, dtype=np.float32) + 0.0025 # Label smoothing
        t[row.label_id] = 0.995

        return image, t

