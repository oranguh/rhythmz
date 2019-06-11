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
from data.dataloaders import get_dataset


def collate_fn(batch):
    data = [item[0].squeeze() for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):

        self.features = args.features
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.n_epochs = args.epochs
        self.results_path = args.results_path
        self.model_id = args.model_id

        sets = {"train", "val", "test"}

        self.datasets = {}
        self.dataloaders = {}
        self.dataset_sizes = {}
        for s in sets:
            self.datasets[s] = get_dataset(s, self.features)
            self.dataloaders[s] = DataLoader(
                self.datasets[s], batch_size=self.batch_size, num_workers=self.num_workers)

        self.device = torch.device(args.device)

        self.clf = classifier.LibrivoxAudioClassifier(
            args.features, self.datasets["train"].n_classes, self.device)

        self.clf = self.clf.to(self.device)

        mkdir(self.results_path)

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
            x = x.to(self.device)
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

        optimizer = Adam(self.clf.parameters(), weight_decay=1/(200*9))

        best_val_score = 0
        best_model = self.clf.state_dict()

        for epoch in range(self.n_epochs):
            train_metrics, cm_train = self.train_epoch(
                epoch, optimizer, "train")
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
