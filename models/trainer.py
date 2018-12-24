import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from models import classifier
from data.dataloaders import AudioDataset

def collate_fn(batch):
    data = [item[0].squeeze() for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

class Trainer:
    def __init__(self, args):

        sets = {"train", "val", "test"}

        self.datasets = {}
        self.dataloaders = {}

        for s in sets:
            self.datasets[s] = AudioDataset(os.path.join(args.data, s))
            self.dataloaders[s] = DataLoader(self.datasets[s], batch_size=args.batch_size, collate_fn=collate_fn)


        self.clf = classifier.AudioClassifier("MoT", 8128, 4069, 8)

        # TODO add resume code

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.clf.parameters())


    def train_epoch(self, epoch, split):

        if split == "train":
            self.clf.train()
        else:
            self.clf.eval()

        for (x, y) in self.dataloaders[split]:
            with torch.set_grad_enabled(split == "train"):
                output = self.clf(x)
                loss = self.criterion(output, y.argmax(1))

            if split == "train":
                loss.backward()
                self.optimizer.step()

            print(loss)


    def train(self):
        self.train_epoch(0, "train")

    def test():
        pass
