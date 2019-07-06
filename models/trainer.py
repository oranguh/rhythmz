import os
import json
import time
import random
import logging
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss

from models import evaluate
from models import classifier
from utils.common import mkdir
from models import diag_classifier
from data.dataloaders import get_dataset, LibrivoxDataset


log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):

        self.features = args.features
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.n_epochs = args.epochs
        self.results_path = args.results_path
        self.model_id = args.model_id
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.batch_norm = args.batch_norm
        self.feature_training_epochs = args.feature_training_epochs

        sets = {"train", "val", "test"}

        self.datasets = {}
        self.dataloaders = {}
        self.dataset_sizes = {}
        for s in sets:
            self.datasets[s] = get_dataset(not args.audio, s, self.features)
            self.dataset_sizes[s] = len(self.datasets[s])
            self.dataloaders[s] = DataLoader(
                self.datasets[s], batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers,
                collate_fn=LibrivoxDataset.collate_fn)

        self.device = torch.device(args.device)

        self.clf = classifier.LibrivoxAudioClassifier2(
            self.features, self.datasets["train"].n_classes, self.device, self.batch_norm)

        self.clf = self.clf.to(self.device)

        mkdir(self.results_path)

    def train_epoch(self, epoch, optimizer, split):

        if split == "train":
            self.clf.train()
        else:
            self.clf.eval()

        self.clf = self.clf.to(self.device)
        criterion = CrossEntropyLoss().to(self.device)

        cm = evaluate.ConfusionMatrix()
        epoch_loss = 0.0
        epoch_samples = 0
        start_time = time.time()

        correct = 0
        total = 0

        total_batches = (self.dataset_sizes[split] // self.batch_size) + 1
        for batch_idx, (x, y, meta) in enumerate(self.dataloaders[split], 1):
            x = x.to(self.device)
            y = y.to(self.device)
            if split == "train":
                optimizer.zero_grad()

            with torch.set_grad_enabled(split == "train"):
                output = self.clf(x)
                loss = criterion(output, y.view(-1))

                y_pred = torch.softmax(output,
                                       dim=1).argmax(1)
                cm.add_many(y.view(-1).cpu().numpy(),
                            y_pred.detach().cpu().numpy())
                correct += (y.view(-1) == y_pred).sum().item()
                total += len(x)

            if split == "train":
                loss.backward()
                optimizer.step()

            if batch_idx % 10 == 0:
                log.info(
                    f"{batch_idx}/{total_batches}: Loss: {loss}, Accuracy: {correct/total}")
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

    def train_epoch_features_only(self, epoch, optimizer):
        split = "train"  # split is always train

        self.clf = self.clf.to(self.device)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        start_time = time.time()

        correct = 0
        total = 0

        criterion = BCEWithLogitsLoss().to(self.device)

        total_batches = (self.dataset_sizes[split] // self.batch_size) + 1
        for batch_idx, (x, y, meta) in enumerate(self.dataloaders[split], 1):
            x = x.to(self.device)
            y = y.to(self.device)

            same = np.array([random.choice([True, False])
                             for _ in range(x.size(0))])
            x_other, y_other, meta_other = self.datasets[split].sample_authors(
                meta, same)

            x_other = x_other.to(self.device)

            # is this correct?
            y_speaker = torch.zeros(x.size(0)).to(self.device)
            y_speaker[np.where(same)] = 1

            # print(sum(same), y_speaker.sum())

            optimizer.zero_grad()

            output = self.clf.forward2(x, x_other)
            loss = criterion(output, y_speaker)

            loss.backward()
            optimizer.step()

            y_speaker_pred = torch.sigmoid(output).detach()
            y_speaker_pred[y_speaker_pred > 0.5] = 1
            y_speaker_pred[y_speaker_pred <= 0.5] = 0

            correct += (y_speaker == y_speaker_pred).sum().item()
            total += len(x)
            epoch_correct += correct

            if batch_idx % 10 == 0:
                log.info(
                    f"{batch_idx}/{total_batches}: Loss: {loss}, Accuracy: {correct/total}")
                correct = 0
                total = 0

            # get un-normalized loss
            batch_loss = loss.item() * len(x)
            epoch_loss += batch_loss
            epoch_samples += len(x)

        metrics = {}
        metrics["epoch_loss"] = epoch_loss / epoch_samples
        metrics["epoch"] = epoch
        metrics["time"] = time.time() - start_time
        metrics["epoch_accuracy"] = epoch_correct / float(epoch_samples)

        log.info("Feature Training Epoch {} complete in {} seconds. Loss: {}".format(
            epoch, round(time.time() - start_time), epoch_loss))

        return metrics

    def train(self):

        model_path = os.path.join(self.results_path, self.model_id)
        mkdir(model_path)
        epochs_path = os.path.join(model_path, "epochs")
        mkdir(epochs_path)
        cm_path = os.path.join(model_path, "confusion_matrix")
        mkdir(cm_path)
        feature_epochs_path = os.path.join(model_path, "feat_epochs")
        mkdir(feature_epochs_path)

        best_val_score = 0
        best_model = self.clf.state_dict()

        # first if the feature_training_epochs is positive, train for that many epochs
        if self.feature_training_epochs > 0:
            assert self.feature_training_epochs < self.n_epochs

            params = []
            params.extend(self.clf.aux_clf.parameters())
            params.extend(self.clf.layers.parameters())

            optimizer = Adam(params, lr=self.learning_rate,
                             weight_decay=self.weight_decay)

            for f_epoch in range(self.feature_training_epochs):
                metrics = self.train_epoch_features_only(f_epoch, optimizer)
                torch.save(metrics, os.path.join(
                    feature_epochs_path, "{}_metrics.pkl".format(f_epoch)))

            # we now freeze the features
            self.clf.aux_clf.requires_grad = False
            self.clf.layers.requires_grad = False
            self.clf.classifier.requires_grad = True

            # learn only the classifier
            optimizer = Adam(self.clf.classifier.parameters(),
                             lr=self.learning_rate,
                             weight_decay=self.weight_decay)

            trained_epochs = self.feature_training_epochs
        else:
            trained_epochs = 0
            optimizer = Adam(self.clf.parameters(),
                             lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(self.n_epochs - trained_epochs):
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

        self.examine("train")
        self.examine("val")

    def examine(self, split):
        n_points = self.dataset_sizes[split]
        meta_keys = {"author_id", "book_id", "gender"}
        diag = diag_classifier.DiagnosticClassifier(
            self.dataloaders[split], self.clf, self.device,
            self.clf.feature_size, meta_keys, n_points)
        diag_results = diag.run()

        path = os.path.join(self.results_path, self.model_id,
                            f"diag_{split}_results.json")
        with open(path, "w") as writer:
            json.dump(diag_results, writer)

    def load(self, map_location="cpu"):
        model_path = os.path.join(
            self.results_path, self.model_id, "best_model.pkl")
        state_dict = torch.load(
            model_path, map_location=map_location)
        self.clf.load_state_dict(state_dict)

    def test(self):
        self.clf = self.clf.to(self.device)
        self.examine("test")
        self.train_epoch(0, None, "val")
