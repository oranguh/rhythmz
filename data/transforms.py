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

class StdScaler:
    def __init__(self, mean=5.592183756080553, std=55.7225389415):
        self.mean = mean
        self.std = std

    def __call__(self, spectogram):
        return (spectogram - self.mean) / self.std

if __name__ == '__main__':
    y, sr = librosa.load("../indic_data_processed/train/bengali/ben_0146.wav")
    S = MelSpectogram(sr)
    print(S(y).shape)
