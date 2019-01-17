import os
import logging

from data.dataloaders import AudioDataset
from data.transforms import MelSpectogram

log = logging.getLogger(__name__)

class ComputeMean:
    def __init__(self, args):
        self.data = AudioDataset(os.path.join(args.data, "train"),
                                      sample_rate=args.sample_rate,
                                      transforms=MelSpectogram(args.sample_rate))
        self.save_path = args.save_path

    def compute(self):
        sum = 0.0
        n = 0
        mean = 5.592183756080553
        var = 3105.0013460920486
        std = 55.7225389415
        for idx in range(len(self.data)):
            spec, cl = self.data[idx]
            sum += ((spec - mean) ** 2).sum().sum()
            n += spec.size(0) * spec.size(1)

            if idx % 10 == 0:
                print("{}: Sum: {}, N: {}".format(idx, sum, n))


        print("Mean: {}".format(sum / n))
