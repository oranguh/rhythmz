from models import classifier
from data.dataloaders import AudioDataLoader

class Trainer:
    def __init__(self, args):
        
        self.clf = classifier.AudioClassifier()