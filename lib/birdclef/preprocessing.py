import os
import sys
import numpy as np
import pandas as pd
import librosa as lb
import joblib
import soundfile as sf

# cf. https://github.com/ryanwongsa/kaggle-birdsong-recognition/blob/4ad1aa4ed99bc097289c7593c55bc09234e0fc59/src/config_params/configs.py
class MelSpecComputer:
    def __init__(
        self,
        sr:int,
        n_mels:int,
        fmin:int,
        fmax:int
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.n_fft = self.n_mels * 20

    def __call__(self, y):
        melspec = lb.feature.melspectrogram(
            y,
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])

        n_repeats = length // len(y)
        epsilon = length % len(y)

        y = np.concatenate([y]*n_repeats + [y[:epsilon]])

    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y

class AudioToImage:
    def __init__(
        self,
        sr:int, # サンプリングレート(今回は32000)
        n_mels:int,
        fmin:int,
        fmax:int,
        duration:int, # 何秒のクリップデータにするか
        step:int,
        res_type:str,
        resample:bool,
        audio_image_directory:str
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample
        self.audio_image_directory = audio_image_directory

        self.mel_spec_computer = MelSpecComputer(
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        return image

    def __call__(self, row, save=True):
        audio, orig_sr = sf.read(row["filepath"], dtype="float32")

        if self.resample and (orig_sr != self.sr):
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = [audio[i:i+self.audio_length] for i in range(0, max(1, len(audio) - self.audio_length + 1), self.step)]
        audios[-1] = crop_or_pad(audios[-1] , length=self.audio_length)
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        if save:
            path = os.path.join(
                self.audio_image_directory,
                row['primary_label'],
                row['filename'].replace(".ogg", ".npy")
            )
            os.makedirs(f"{self.audio_image_directory}/{row['primary_label']}", exist_ok=True)
            np.save(path, images)
        else:
            return  row.filename, images



