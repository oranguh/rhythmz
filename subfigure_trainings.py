import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from tqdm import tqdm
from models.classifier import AudioClassifier
import torch

def main():
    all_models = "C:\\Users\\murco.DESKTOP-R324UUU\\Documents\\rhythmz\\results"
    y_trains = []
    y_vals = []
    titles = []

    for dirName, subdirList, fileList in os.walk(all_models):
        print(dirName)
        if os.path.split(dirName)[-1] != "epochs":
            current_dir = os.path.split(dirName)[-1]
            print(current_dir)
        else:
            y_train = []
            y_val = []
            for file in tqdm(fileList):

                if "train" in file:
                    train_data = os.path.join(dirName, file)
                    performance = torch.load(train_data)
                    # print(train_data, performance["weighted avg"]["precision"])
                    y_train.append(performance["weighted avg"]["precision"])
                else:
                    val_data = os.path.join(dirName, file)
                    performance = torch.load(val_data)
                    # print(val_data, performance["weighted avg"]["precision"])
                    y_val.append(performance["weighted avg"]["precision"])
            y_trains.append(y_train)
            y_vals.append(y_val)
            titles.append(current_dir)

    for i, subplots in enumerate(y_trains):
        y_train = y_trains[i]
        y_val = y_vals[i]
        x = list(range(0, len(y_val)))

        plt.subplot(4, 2, i+1).set_title(titles[i])
        plt.plot(x, y_train, 'o-', label="training")
        plt.plot(x, y_val, 'r.-', label="validation")
        plt.ylim(0, 1)
        plt.xlim(0, 10)
        plt.ylabel("precision")
        plt.xlabel("epoch")
    blue = mpatches.Patch(color='blue', label='Training')
    red = mpatches.Patch(color='red', label='Validation')
    plt.legend(handles=[blue, red])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
