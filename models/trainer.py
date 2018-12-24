from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from models import classifier
from data.dataloaders import AudioDataset

class Trainer:
    def __init__(self, args):

        sets = {"train", "val", "test"}

        self.datasets = {}
        self.dataloaders = {}

        for s in sets:
            self.datasets[s] = AudioDataset(os.path.join(args.data, s))
            self.dataloaders[s] = DataLoader(self.datasets[s], batch_size=args.batch_size)


        self.clf = classifier.AudioClassifier()

        # TODO add resume code

        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(self.clf.parameters())


    def train():
        pass

    def test():
        pass
