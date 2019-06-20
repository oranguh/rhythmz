import os
import json
import random
import logging
import hashlib

import torch
import librosa
import numpy as np
import pandas as pd
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
    METADATA_PATH = "./datasets/librivox_metadata"

    @staticmethod
    def collate_fn(batch):
        data, target, meta = [], [], []
        for d, t, m in batch:
            data.append(d)
            target.append(t)
            meta.append(m)
        return [torch.stack(data).unsqueeze(1), torch.stack(target), meta]

    def __init__(self, split, rhythm, transforms=None, padding="wrap", cache=False, cache_dir="./cache",):
        # TODO: repeat
        assert padding in {"wrap"}
        self.transforms = transforms
        root_path = {
            False: self.ROOT_PATH,
            True: self.ROOT_PATH_RHYTHM
        }[rhythm]
        self.root_path = root_path
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

        self._load_metadata()

    def _load_metadata(self):
        self.metadata = {}
        for cl in self.class_to_idx:
            with open(os.path.join(self.METADATA_PATH, cl + ".json")) as reader:
                meta = json.load(reader)
                # make meta a dict with book_id as key
                self.metadata[cl] = {m["book_id"]: m for m in meta}

        log.info("Loading gender info")
        g = pd.read_csv(os.path.join(self.root_path, "gender_annotations.csv"))
        self.genders = {}
        for _, row in g.iterrows():
            self.genders[row["author_id"]] = row["gender"]

        log.info("Finished loading metadata!")

    def _load_clip(self, idx):
        _, aud_path = self.data[idx]
        # set sr=None to load the clip with proper sr
        sound, sample_rate = librosa.load(aud_path, sr=None)
        sound = self.pad_audio(sound)

        if self.transforms:
            sound = self.transforms(sound)

        return sound

    def _get_meta(self, idx):
        language, aud_path = self.data[idx]
        file_name = os.path.split(aud_path)[1]
        book_id = file_name.split("_")[0]

        m = self.metadata[language][book_id]

        return {
            "book_id": book_id,
            "author_id": m["reader_url"],
            "gender": self.genders[m["reader_url"]],
            "language": language,
            "path": aud_path
        }

    def _load(self, idx):
        _, aud_path = self.data[idx]

        if self.cache:
            _id = hashlib.sha1(
                (self.transforms_str + aud_path).encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, _id) + ".npy"
            if os.path.exists(cache_path):
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
        meta = self._get_meta(idx)
        cl, aud_path = self.data[idx]
        cl = self.class_to_idx[cl]
        return torch.FloatTensor(self._load(idx)), torch.LongTensor([cl]), meta

    def __len__(self):
        return len(self.data)
