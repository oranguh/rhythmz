import logging
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import classification_report, SCORERS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

log = logging.getLogger(__name__)


class DiagnosticClassifier:
    def __init__(self, dataloader, model, device, feature_size, meta_keys, n_points):
        self.dataloader = dataloader
        self.model = model
        self.meta_keys = meta_keys
        self.feature_size = feature_size
        self.n_points = n_points
        self.device = device

    def build_xy(self):
        log.info("Building X, Y")
        X = []
        Ys = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            print_freq = self.n_points // self.dataloader.batch_size
            for batch_idx, (x, y, meta) in enumerate(self.dataloader):
                x = x.to(self.device)
                x = self.model.get_features(x).detach().cpu().numpy()
                X.extend(x)

                for key in self.meta_keys:
                    Ys[key].extend([m[key] for m in meta])

                if batch_idx % print_freq == 0:
                    log.info("\t{} of {} done".format(len(X), self.n_points))

                if len(X) >= self.n_points:
                    break

        X = np.array(X[:self.n_points])
        for key in Ys:
            Ys[key] = np.array(Ys[key][:self.n_points])
        log.info("\t.. complete")
        return X, Ys

    def run(self):
        X, ys = self.build_xy()

        results = {}
        for key in ys:
            y = ys[key]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)

            log.info("{}: Fitting a logistic regression model (CV=5)".format(key))
            clf = LogisticRegressionCV(scoring="f1_micro", cv=3, max_iter=250)
            clf.fit(X_train, y_train)

            train_results = classification_report(
                y_train, clf.predict(X_train), output_dict=True)
            test_results = classification_report(
                y_test, clf.predict(X_test), output_dict=True)

            results[key] = {
                "train": train_results,
                "test": test_results
            }

        return results
