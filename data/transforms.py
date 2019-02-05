import librosa
import numpy as np


class MelSpectogram:
    def __init__(self, sample_rate, n_mels=128, n_fft=2048, hop_length=512, power=2.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.n_mels = n_mels

    def __call__(self, audio):
        return librosa.power_to_db(librosa.feature.melspectrogram(audio, sr=self.sample_rate, n_mels=self.n_mels,
                                                                  n_fft=self.n_fft, hop_length=self.hop_length), ref=np.max)

    def __str__(self):
        return "MelSpectogram(n_mels={}, n_fft={}, hop_length={}, power={})".format(self.n_mels,
                                                                                    self.n_fft,
                                                                                    self.hop_length, self.power)


class StdScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __str__(self):
        return "StdScaler(mean={}, std={})".format(self.mean, self.std)

    def __call__(self, spectogram):
        return (spectogram - self.mean) / self.std


class Compose(object):
    # from: https://github.com/pytorch/audio/blob/master/torchaudio/transforms.py
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(),
        >>>     transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


if __name__ == '__main__':
    y, sr = librosa.load("../indic_data_processed/train/bengali/ben_0146.wav")
    S = MelSpectogram(sr)
    print(S(y).shape)
