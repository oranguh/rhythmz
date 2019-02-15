import os
import random
import logging
import hashlib

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

from data.transforms import MelSpectogram
from utils import common

log = logging.getLogger(__name__)


class AudioDataset(Dataset):

    def __init__(self, root_dir, sample_rate=16000, transforms=None, cache=False, cache_dir="./cache"):
        self.root_dir = root_dir
        self.cache = cache
        self.cache_dir = cache_dir
        if self.cache:
            self.transforms_str = str(transforms)
            log.info("Caching is enabled! Cache dir is : {}".format(self.cache_dir))
            common.mkdir(self.cache_dir)
        self.class_to_idx = {}
        classes = set()
        data_points = []
        for cl in os.listdir(self.root_dir):
            classes.add(cl)
            count = 0
            for file in os.listdir(os.path.join(self.root_dir, cl)):
                aud_file_path = os.path.join(self.root_dir, cl, file)
                data_points.append((cl, aud_file_path))
                count += 1
            log.debug("Class: {}. Instances: {}".format(cl, count))
        random.shuffle(data_points)
        self.data = data_points
        self.class_to_idx = {cl: idx for (
            idx, cl) in enumerate(sorted(classes))}
        self.idx_to_class = {idx: cl for (
            cl, idx) in self.class_to_idx.items()}
        self.n_classes = len(self.class_to_idx)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def one_hot(self, cl):
        c = np.zeros(len(self.class_to_idx))
        c[self.class_to_idx[cl]] = 1
        return c

    def _load(self, idx):
        _, aud_path = self.data[idx]
        sound, sample_rate = librosa.load(aud_path)
        if self.transforms:
            sound = self.transforms(sound)
        return sound

    def __getitem__(self, idx):
        cl, aud_path = self.data[idx]

        if self.cache:
            _id = hashlib.sha1(
                (self.transforms_str + aud_path).encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, _id) + ".npy"
            if os.path.exists(cache_path):
                sound = np.load(cache_path)
            else:
                sound = self._load(idx)
                np.save(cache_path, sound)
            sound = torch.from_numpy(sound)
            return sound, self.one_hot(cl)
        else:
            return torch.from_numpy(self._load(idx)), self.one_hot(cl)
