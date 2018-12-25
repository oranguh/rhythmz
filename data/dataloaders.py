import os
import random
import logging

import torchaudio
from torchaudio.transforms import Scale
import numpy as np

from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class AudioDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
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
        # TODO: explore more transforms
        self.transforms = Scale()

    def __len__(self):
        return len(self.data)

    def one_hot(self, cl):
        c = np.zeros(len(self.class_to_idx))
        c[self.class_to_idx[cl]] = 1
        return c

    def __getitem__(self, idx):
        cl, aud_path = self.data[idx]
        sound, sample_rate = torchaudio.load(aud_path)
        sound = self.transforms(sound)
        return sound, self.one_hot(cl)
