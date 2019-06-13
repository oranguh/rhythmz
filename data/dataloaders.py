import os
import random
import logging
import hashlib

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

from utils import common
from data.transforms import MelSpectogram, StdScaler, Compose

log = logging.getLogger(__name__)


def get_dataset(rhythm, split, features):
    if features == "raw":
        transforms = None
        cache = False
    elif features == "ms":
        transforms = Compose([MelSpectogram(8000)])
        cache = True
    log.info("Transforms: {}".format(transforms))
    return LibrivoxDataset(split, rhythm, transforms=transforms, cache=cache)


class LibrivoxDataset(Dataset):
    ROOT_PATH = "./datasets/librivox_splits/"
    ROOT_PATH_RHYTHM = "./datasets/rhythm_librivox_splits/"

    def __init__(self, split, rhythm, transforms=None, padding="wrap", cache=False, cache_dir="./cache",):
        # TODO: repeat
        assert padding in {"wrap"}
        self.transforms = transforms
        root_path = {
            False: self.ROOT_PATH,
            True: self.ROOT_PATH_RHYTHM
        }[rhythm]
        log.info("Data root path: {}".format(root_path))
        self.path = os.path.join(root_path, split)
        self.class_to_idx = {}
        classes = sorted(os.listdir(self.path))
        self.class_to_idx = {cl: idx for (idx, cl) in enumerate(classes)}
        self.n_classes = len(classes)
        self.padding = padding
        self.data = []
        for cl in self.class_to_idx:
            count = 0
            for file in os.listdir(os.path.join(self.path, cl)):
                aud_file_path = os.path.join(self.path, cl, file)
                self.data.append((cl, aud_file_path))
                count += 1
            log.debug("Class: {}. Instances: {}".format(cl, count))

        # caching stuff
        self.cache = cache
        self.cache_dir = cache_dir
        if self.cache:
            self.transforms_str = str(transforms)
            log.info("Caching is enabled! Cache dir is : {}".format(self.cache_dir))
            common.mkdir(self.cache_dir)

    def _load_clip(self, idx):
        _, aud_path = self.data[idx]
        # set sr=None to load the clip with proper sr
        sound, sample_rate = librosa.load(aud_path, sr=None)
        sound = self.pad_audio(sound)

        if self.transforms:
            sound = self.transforms(sound)

        return sound

    def _load(self, idx):
        _, aud_path = self.data[idx]

        if self.cache:
            _id = hashlib.sha1(
                (self.transforms_str + aud_path).encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, _id) + ".npy"
            if os.path.exists(cache_path):
                print("yay for caching")
                sound = np.load(cache_path)
            else:
                sound = self._load_clip(idx)
                np.save(cache_path, sound)
        else:
            sound = self._load_clip(idx)

        return sound

    def pad_audio(self, sound, length=80000):
        if sound.shape[0] == length:
            return sound

        if self.padding == "wrap":
            return np.pad(sound, (0, length-sound.shape[0]), "wrap")

    def __getitem__(self, idx):
        cl, aud_path = self.data[idx]
        cl = self.class_to_idx[cl]
        # reshape to 1 channel for easier conv operations
        return torch.FloatTensor(self._load(idx)), torch.LongTensor([cl])

    def __len__(self):
        return len(self.data)
