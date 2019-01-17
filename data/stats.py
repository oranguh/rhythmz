import os
import logging

from data.dataloaders import AudioDataset
from data.transforms import MelSpectogram

log = logging.getLogger(__name__)


class ComputeMean:
    def __init__(self, args):
        self.data = args.data
        self.sample_rate = args.sample_rate

    def raw_audio(self):
        data = AudioDataset(os.path.join(self.data, "train"),
                            sample_rate=self.sample_rate,
                            transforms=None)

        mean = 0.0
        n = 0

        for idx in range(len(data)):
            if (idx + 1) % 100 == 0:
                log.info("Computing Mean: {}% done : {}".format(
                    (idx / len(data) * 100), mean / n))
            aud, cl = data[idx]
            mean += aud.view(-1).sum()
            n += aud.size(0)

        mean /= n

        log.info("Mean Computed: {}".format(mean))

        std = 0.0
        n = 0
        # second pass to compute std
        for idx in range(len(data)):
            if (idx + 1) % 100 == 0:
                log.info("Computing Std: {}% done : {}".format(
                    (idx / len(data) * 100), std / n))
            aud, cl = data[idx]
            std += ((mean - aud.view(-1)) ** 2).sum()
            n += aud.size(0)

        std /= n

        log.info("Std computed: {}".format(std))
