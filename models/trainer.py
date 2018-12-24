import os
import logging

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from models import evaluate
from models import classifier
from data.dataloaders import AudioDataset


def collate_fn(batch):
    data = [item[0].squeeze() for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):

        sets = {"train", "val", "test"}

        self.datasets = {}
        self.dataloaders = {}

        for s in sets:
            self.datasets[s] = AudioDataset(os.path.join(args.data, s))
            self.dataloaders[s] = DataLoader(
                self.datasets[s], batch_size=args.batch_size, collate_fn=collate_fn)

        self.clf = classifier.AudioClassifier("MoT", 8128, 4069, 8)

        # TODO add resume code

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.clf.parameters())

    def train_epoch(self, epoch, split):

        if split == "train":
            self.clf.train()
        else:
            self.clf.eval()

        cm = evaluate.ConfusionMatrix()
        for batch_idx, (x, y) in enumerate(self.dataloaders[split], 1):
            with torch.set_grad_enabled(split == "train"):
                output = self.clf(x)
                loss = self.criterion(output, y.argmax(1))
                cm.add_many(y.argmax(1), output.argmax(1))

            if split == "train":
                loss.backward()
                self.optimizer.step()

            if batch_idx % 10 == 0:
                log.info("{}: {}".format(batch_idx, loss.item()))

        cm.pprint()

    def train(self):
        self.train_epoch(0, "train")

    def test():
        pass
