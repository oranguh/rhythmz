import librosa
from librosa import display
import csv
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import numpy as np
import pylab
from tqdm import tqdm
import matplotlib.image as mpimg
# import eyed3

def main():

    # root_dir = "/media/meow/72A23121A230EAED/librivox/Multilingual"
    # save_dir = "/media/meow/72A23121A230EAED/librivox_Processed"
    #
    # process_librivox_data(root_dir, save_dir)

    rootDir = "/media/meow/72A23121A230EAED/daata/new_dataset"
    rootDir = "D:\\daata\\indic_splits"
    process_audio_data(rootDir)


def top_coder_extractor():
    data_path = "D:/daata/top_coder_challenge/"
    csv_path = data_path + "testingData.csv"
    chrsitan_data = "testingdata/"
    csv_path = "trainingData.csv"
    counter_dict = OrderedDict()
    counter = 0
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            try:
                counter_dict[row[1]] += 1
            except KeyError:
                counter_dict[row[1]] = 1
            #
            # save_name = data_path + "Data_organized_test"+ "\\" + row[1] + "\\" + row[1] + "_" + str(counter_dict[row[1]]) + ".wav"
            # sound.export(save_name, format="wav")


    for key in counter_dict:
        print(key)
        try:
            os.mkdir(data_path + "Data_organized\\"+ key);
        except FileExistsError:
            pass
    print(counter_dict)

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

def process_audio_data(rootDir, savedir = None):
    """
    input any folder. It will make an identical copy containing wav files of desired sampling rate (8000) and numpy arrays
    """
    for dirName, subdirList, fileList in tqdm(os.walk(rootDir)):

        # making new head directories for processed data
        if savedir == None:
            save_dir_wav = rootDir + "_wav"
            save_dir_melspec = rootDir + "_melspec"
        else:
            new_directory_wav = savedir + "_wav"
            new_directory_melspec = savedir + "_melspec"

        if not os.path.isdir(save_dir_wav):
            os.mkdir(save_dir_wav)
        if not os.path.isdir(save_dir_melspec):
            os.mkdir(save_dir_melspec)

        # make copy of each sub directory
        new_directory_wav = dirName.replace(os.path.split(rootDir)[1], os.path.split(save_dir_wav)[1])
        new_directory_melspec = dirName.replace(os.path.split(rootDir)[1], os.path.split(save_dir_melspec)[1])

        if not os.path.isdir(new_directory_wav):
            os.mkdir(new_directory_wav)
        if not os.path.isdir(new_directory_melspec):
            os.mkdir(new_directory_melspec)

        for fname in tqdm(fileList):

            # print('filename {}'.format(new_directory + "\\" + fname))
            image_name = os.path.splitext(fname)[0]
            y, sr = librosa.load(os.path.join(dirName, fname),sr=None)

            # resampled wav
            save_path = os.path.join(new_directory_wav, image_name)
            new_sr = 8000
            y_8k = librosa.resample(y, sr, new_sr)
            librosa.output.write_wav(save_path + "_8K.wav", y_8k, new_sr, norm=False)

            # numpy matrix
            save_path = os.path.join(new_directory_melspec, image_name)
            S = MelSpectogram(sr)
            np.save(os.path.join(new_directory_melspec, image_name), S(y))

# root_dir = root_dir + "/mshortworks_001_1202_librivox"

def audio_splitter_upper(y, length, sr, margin = 20):
    split_audio = []
    # remove start and end for that annoying librivox intro
    y = y[margin*sr:-margin*sr]

    while len(y) > length*sr:
        split_audio.append(y[:length*sr])
        y = y[length*sr:]
    return np.asarray(split_audio)


def get_language_librivox_mp3(audiofile):
    audiofile = eyed3.load(audiofile)
    if audiofile.tag.title[0].isdigit():
        audiofile_language = audiofile.tag.title
        while audiofile_language[0].isdigit():
            audiofile_language = audiofile_language[1:]
            audiofile_language = audiofile_language.strip(" ")
            audiofile_language = audiofile_language.strip("-")

        audiofile_language = audiofile_language.split(":")[0].strip(" ")
        audiofile_language = audiofile_language.split("-")[0].strip(" ")


        print(audiofile_language, "\t")
    else:
        audiofile_language = audiofile.tag.title.split("-")[0].strip(" ")
        print(audiofile_language, "\t")
    return audiofile_language

def process_librivox_data(root_dir, save_dir):
    language_counter_dict = OrderedDict()

    for dirName, subdirList, fileList in tqdm(os.walk(root_dir)):
        for fname in tqdm(fileList):
            if ".zip" in fname:
                continue
            # print(dirName + "/" + fname)



            print("\n",fname, "\t", dirName)

            # big ass strip split thing to get language of audiofile
            audiofile = os.path.join(dirName, fname)
            audiofile_language = get_language_librivox_mp3(audiofile)

            try:
                language_counter_dict[audiofile_language] += 1
            except KeyError:
                language_counter_dict[audiofile_language] = 1

            if not os.path.isdir(os.path.join(save_dir, "numpy_melspec", audiofile_language)):
                os.mkdir(os.path.join(save_dir, "numpy_melspec", audiofile_language))
            if not os.path.isdir(os.path.join(save_dir, "wav_files", audiofile_language)):
                os.mkdir(os.path.join(save_dir, "wav_files", audiofile_language))

            # try:
            #     folder_name = save_dir + "/" + "numpy_melspec" + "/" + audiofile_language
            #     os.mkdir(folder_name);
            # except FileExistsError:
            #     pass
            # try:
            #     folder_name = save_dir + "/" + "wav_files" + "/" + audiofile_language
            #     os.mkdir(folder_name);
            # except FileExistsError:
            #     pass

            y, sr = librosa.load(os.path.join(dirName, fname))
            S = MelSpectogram(sr)

            check_if_exists = os.path.join(save_dir, "numpy_melspec", audiofile_language, audiofile_language + "_" + str(language_counter_dict[audiofile_language]) + "_0.npy")
            print(os.path.isfile(check_if_exists))
            if not os.path.isfile(check_if_exists):
                split_audio = audio_splitter_upper(y, 20, sr, 20)
                print(len(split_audio))
                for i, audio_fragment in enumerate(split_audio):
                    savename = os.path.join(save_dir, "numpy_melspec", audiofile_language, audiofile_language + "_" + str(language_counter_dict[audiofile_language]) + "_" + str(i))
                    print(savename)
                    np.save(savename, S(audio_fragment))
                    savename = savename.replace("numpy_melspec", "wav_files")
                    librosa.output.write_wav(savename, audio_fragment, sr)
    print(language_counter_dict)


if __name__ == "__main__":
    main()
