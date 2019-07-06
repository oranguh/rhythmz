import os
import json
import random
import logging
import hashlib
from collections import defaultdict

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

    def __init__(self, split, rhythm, transforms=None, padding="wrap", cache=False, cache_dir="./cache", filter_single_train=True):
        excluded_langs = {}
        if filter_single_train:
            excluded_langs = {"Danish", "Esperanto", "Hebrew",
                              "Japanese", "Korean", "Tagalog", "Tamil"}

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
        classes = [c for c in classes if c not in excluded_langs]

        log.info(f"{len(classes)} langauges are being loaded")
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

        self.clip_metadata = [None] * len(self)
        for idx in range(len(self)):
            meta = self._get_meta(idx)
            self.clip_metadata[idx] = meta
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

    def sample_authors(self, all_meta, same_lang):
        """
            Returns clips (same lang, diff author) if y=1 or (diff lang, diff speaker) if y=0, 
        """

        batch = []
        for meta, same in zip(all_meta, same_lang):
            clip_id = meta["path"]
            author_id = meta["author_id"]
            lang = meta["language"]
            # print(meta, same)
            allowed_ids = list()
            allowed_meta = list()
            for idx, m in enumerate(self.clip_metadata):
                if m["author_id"] == author_id:
                    continue

                # if same_lang, then only same language is allowed
                if same and m["language"] != lang:
                    continue
                elif not same and m["language"] == lang:
                    # if not same_lang, skip clips where the lang is the same
                    # this assumes that one speaker can have multiple languages
                    continue

                allowed_ids.append(idx)
                allowed_meta.append(m)

            # if same:
            #     assert all(m["author_id"] !=
            #                author_id and m["language"] == lang for m in allowed_meta)
            # else:
            #     assert all(m["author_id"] !=
            #                author_id and m["language"] != lang for m in allowed_meta)

            clip_idx = random.choice(allowed_ids)
            batch.append(self[clip_idx])

        return self.collate_fn(batch)

    def __getitem__(self, idx):
        meta = self._get_meta(idx)
        cl, aud_path = self.data[idx]
        cl = self.class_to_idx[cl]
        return torch.FloatTensor(self._load(idx)), torch.LongTensor([cl]), meta

    def __len__(self):
        return len(self.data)
