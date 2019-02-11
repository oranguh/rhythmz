import os
import time
import logging
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from models import evaluate
from models import classifier
from utils.common import mkdir
from data.dataloaders import AudioDataset
from data.transforms import MelSpectogram, StdScaler, Compose


def collate_fn(batch):
    data = [item[0].squeeze() for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def get_dataset(path, features, sample_rate, mean, std):
    if features == "raw":
        if mean and std:
            transforms = StdScaler(
                mean=mean, std=args.data_std)
        else:
            transforms = None
        # no need to cache if we're using raw audio
        cache = False
    elif features == "mel-spectogram":
        if mean and std:
            transforms = Compose([MelSpectogram(sample_rate), StdScaler(
                mean=mean, std=std)])
        else:
            transforms = MelSpectogram(sample_rate)
        # cache if we're using raw audio
        cache = True

    return AudioDataset(path,
                        sample_rate=sample_rate,
                        transforms=transforms,
                        cache=cache)


log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):

        sets = {"train", "val", "test"}

        self.datasets = {}
        self.dataloaders = {}

        if args.features == "raw":
            transforms = StdScaler(
                mean=args.data_mean, std=args.data_std)
        elif args.features == "mel-spectogram":
            transforms = Compose([MelSpectogram(args.sample_rate), StdScaler(
                mean=args.data_mean, std=args.data_std)])

        for s in sets:
            self.datasets[s] = get_dataset(os.path.join(
                args.data, s), args.features, args.sample_rate, args.data_mean, args.data_std)
            self.dataloaders[s] = DataLoader(
                self.datasets[s], batch_size=args.batch_size, collate_fn=collate_fn)

        self.device = torch.device(args.device)

        self.clf = classifier.AudioClassifier(
            args.combine, args.features, args.input_size,
            args.input_stride, self.datasets["train"].n_classes, self.device)

        self.clf = self.clf.to(self.device)

        self.n_epochs = args.epochs
        self.results_path = args.results_path
        mkdir(self.results_path)
        self.model_id = args.model_id

        self.criterion = CrossEntropyLoss().to(self.device)

    def train_epoch(self, epoch, optimizer, split):

        if split == "train":
            self.clf.train()
        else:
            self.clf.eval()

        self.clf = self.clf.to(self.device)

        cm = evaluate.ConfusionMatrix()
        epoch_loss = 0.0
        epoch_samples = 0
        start_time = time.time()

        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(self.dataloaders[split], 1):
            x = [_.to(self.device).float() for _ in x]
            y = y.to(self.device)

            if split == "train":
                optimizer.zero_grad()

            with torch.set_grad_enabled(split == "train"):
                output = self.clf(x)
                loss = self.criterion(output, y.argmax(1))
                cm.add_many(y.argmax(1).detach().cpu().numpy(),
                            output.argmax(1).detach().cpu().numpy())

                correct += (y.argmax(1) == output.argmax(1)).sum().item()
                total += len(x)

            if split == "train":
                loss.backward()
                optimizer.step()

            if batch_idx % 10 == 0:
                log.info("{}: Loss: {}, Accuracy: {}".format(
                    batch_idx, loss.item(), correct/total))
                correct = 0
                total = 0

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

        return metrics, cm.mat

    def train(self):

        model_path = os.path.join(self.results_path, self.model_id)
        mkdir(model_path)
        epochs_path = os.path.join(model_path, "epochs")
        mkdir(epochs_path)
        cm_path = os.path.join(model_path, "confusion_matrix")
        mkdir(cm_path)

        optimizer = Adam(self.clf.parameters())

        best_val_score = 0
        best_model = self.clf.state_dict()

        for epoch in range(self.n_epochs):
            train_metrics, cm_train = self.train_epoch(epoch, optimizer, "train")
            val_metrics, cm_val = self.train_epoch(epoch, None, "val")

            torch.save(val_metrics, os.path.join(
                epochs_path, "{}_val_metrics.pkl".format(epoch)))
            torch.save(train_metrics, os.path.join(
                epochs_path, "{}_train_metrics.pkl".format(epoch)))
            np.save(os.path.join(
                cm_path, "{}_train_metrics".format(epoch)), cm_train)
            np.save(os.path.join(
                cm_path, "{}_val_metrics".format(epoch)), cm_val)

            val_score = val_metrics["micro avg"]["f1-score"]
            if best_val_score < val_score:
                log.info("Validation Score {} exceeds best score of {}. Saving new best model".format(
                    val_score, best_val_score))
                best_val_score = val_score
                best_model = self.clf.state_dict()
                torch.save(best_model, os.path.join(
                    model_path, "best_model.pkl"))

    def load(self):
        model_path = os.path.join(self.results_path, self.model_id)
        self.clf.load_state_dict(torch.load(model_path))

    def test():
        self.clf = self.clf.to(self.device)
        self.train_epoch(0, "test")
