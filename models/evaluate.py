import logging
from collections import defaultdict

from sklearn import metrics

log = logging.getLogger(__name__)


class ConfusionMatrix(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.mat = defaultdict(int)
        self.instances = []

    def classes(self):
        classes = {k[0] for k in self.mat.keys()}
        classes = classes.union({k[1] for k in self.mat.keys()})
        return classes

    def _get_ys(self):
        y_true = [i[0] for i in self.instances]
        y_pred = [i[1] for i in self.instances]
        return (y_true, y_pred)

    def add(self, true_class, predicted_class):
        self.mat[(true_class, predicted_class)] += 1
        self.instances.append((true_class, predicted_class))

    def add_many(self, true_classes, predicted_classes):
        for (t, p) in zip(true_classes, predicted_classes):
            self.add(t, p)

    def pprint(self):
        classes = self.classes()

        if self.verbose:
            log.info("Confusion Matrix: ")
            log.info("\n\tCorrect")
            for cl in classes:
                log.info("\t{}: {}".format(cl, self.mat[(cl, cl)]))

            log.info("\n\tIncorrect")
            for cl1 in classes:
                for cl2 in classes:
                    if cl1 == cl2:
                        continue
                    log.info("\t(True: {}, Predicted: {}):: {}".format(
                        cl1, cl2, self.mat[(cl1, cl2)]))

        y_true, y_pred = self._get_ys()
        report = metrics.classification_report(y_true, y_pred)
        log.info("Classification Report: \n{}\n".format(report))

    def get_metrics(self):
        y_true, y_pred = self._get_ys()
        m = metrics.classification_report(
            y_true, y_pred, output_dict=True)

        return m
