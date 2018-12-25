import os
import time
import logging

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from models import evaluate
from models import classifier
from utils.common import mkdir
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

        self.clf = classifier.AudioClassifier(
            "MoT", 16256, 4069, self.datasets["train"].n_classes)

        # TODO add resume code

        self.n_epochs = args.epochs
        self.results_path = args.results_path
        mkdir(self.results_path)
        self.model_id = args.model_id
        self.device = torch.device(args.device)

        self.criterion = CrossEntropyLoss().to(self.device)

    def train_epoch(self, epoch, optimizer, split):

        if split == "train":
            self.clf.train()
        else:
            self.clf.eval()

        cm = evaluate.ConfusionMatrix()
        epoch_loss = 0.0
        epoch_samples = 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(self.dataloaders[split], 1):
            x = [_.to(self.device) for _ in x]
            y = y.to(self.device)

            with torch.set_grad_enabled(split == "train"):
                output = self.clf(x)
                loss = self.criterion(output, y.argmax(1))
                cm.add_many(y.argmax(1), output.argmax(1))

            if split == "train":
                loss.backward()
                optimizer.step()

            if batch_idx % 10 == 0:
                log.info("{}: {}".format(batch_idx, loss.item()))

            # get un-normalized loss
            batch_loss = loss.item() * len(x)
            epoch_loss += batch_loss
            epoch_samples += len(x)

        cm.pprint()
        metrics = cm.get_metrics()
        metrics["epoch_loss"] = epoch_loss / epoch_samples
        metrics["epoch"] = epoch
        metrics["time"] = time.time() - start_time

        log.info("Epoch {} complete in {} seconds. Loss: {}".format(
            epoch, metrics["time"], metrics["epoch_loss"]))

        return metrics

    def train(self):

        model_path = os.path.join(self.results_path, self.model_id)
        mkdir(model_path)
        epochs_path = os.path.join(model_path, "epochs")
        mkdir(epochs_path)

        optimizer = Adam(self.clf.parameters())
        self.clf = self.clf.to(self.device)

        best_val_score = 0
        best_model = self.clf.state_dict()

        for epoch in range(self.n_epochs):
            train_metrics = self.train_epoch(epoch, optimizer, "train")
            val_metrics = self.train_epoch(epoch, None, "val")

            print(val_metrics)

            torch.save(val_metrics, os.path.join(
                epochs_path, "{}_val_metrics.pkl".format(epoch)))
            torch.save(train_metrics, os.path.join(
                epochs_path, "{}_train_metrics.pkl".format(epoch)))

            val_score = None  # TODO
            if best_val_score < val_score:
                log.info("Validation Score {} exceeds best score of {}. Saving new best model".format(
                    val_score, best_val_score))
                best_model = self.clf.state_dict()
                torch.save(best_model, os.path.join(
                    model_path, "best_model.pkl"))

    def load(self):
        pass

    def test():
        self.clf = self.clf.to(self.device)
        self.train_epoch(0, "test")
