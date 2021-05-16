import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from .preprocessing import MelSpecComputer, mono_to_color

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

"""
class BirdClefDataset(Dataset):
    def __init__(
        self,
        audio_image_store,
        meta:pd.DataFrame,
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
"""

class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        data,
        sr:int          = 32000,            # サンプリングレート
        n_mels:int      = 128,              # メルフィルタバンク
        fmin:int        = 0,
        fmax:int        = None,
        duration:int    = 5,
        step:int        = None,
        res_type:str    = "kaiser_fast",
        resample:bool   = True
    ):
        self.data = data
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(
            sr      = self.sr,
            n_mels  = self.n_mels,
            fmin    = self.fmin,
            fmax    = self.fmax
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = []
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])

        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]

        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        return images


    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "filepath"])














