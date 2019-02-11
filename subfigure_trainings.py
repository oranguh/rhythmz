import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from models.classifier import AudioClassifier
import torch
import seaborn as sn
import pandas as pd

def main():
    boxplot = True
    confusion = True

    all_models = "C:\\Users\\murco.DESKTOP-R324UUU\\Documents\\rhythmz\\results"
    y_trains = []
    y_vals = []
    titles = []
    figure_count = 0
    for dirName, subdirList, fileList in os.walk(all_models):
        # print(dirName)
        if os.path.split(dirName)[-1] == ("confusion_matrix"):
            for file in tqdm(fileList):
                # continue
                epoch = file.split("_")[0]
                language_list = ["bengali", "hindi", "kannada", "malayalam", "marathi", "tamil", "telegu"]
                confusion_title = os.path.split(os.path.split(dirName)[0])[1]
                confusion_matrix = np.load(os.path.join(dirName, file))
                # print(type(confusion_matrix))
                # print(confusion_matrix[()][(2, 3)])
                class_count = 7
                confusion_real_matrix = np.zeros((class_count, class_count))
                for i in range(class_count):
                    for j in range(class_count):
                        confusion_real_matrix[i,j] = confusion_matrix[()][(i, j)]

                if confusion:
                    df_cm = pd.DataFrame(confusion_real_matrix, language_list,
                      language_list)
                    plt.figure(figure_count)#, figsize = (10,7))
                    plt.title("epoch: {}  ".format(epoch) + confusion_title)
                    figure_count += 1

                    # sn.set(font_scale=1.4)#for label size
                    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')# font size
                    plt.show()

                if boxplot:
                    df = pd.DataFrame(columns=["bengali", "hindi", "kannada", "malayalam", "marathi", "tamil", "telegu"],
                                    data = {"bengali": [i for i in confusion_real_matrix[0]],
                                    "hindi": [i for i in confusion_real_matrix[1]],
                                    "kannada": [i for i in confusion_real_matrix[2]],
                                    "malayalam": [i for i in confusion_real_matrix[3]],
                                    "marathi": [i for i in confusion_real_matrix[4]],
                                    "tamil": [i for i in confusion_real_matrix[5]],
                                    "telegu": [i for i in confusion_real_matrix[6]]},
                                    index=["bengali", "hindi", "kannada", "malayalam", "marathi", "tamil", "telegu"])
                    # plt.figure(figure_count)
                    # figure_count += 1
                    sn.set()
                    df.T.plot(kind='bar', stacked=True)
                    # plt.xlabel("Languages")
                    plt.ylabel("Predicted")
                    plt.title("epoch: {}  ".format(epoch) + confusion_title)
                    # plt.show()

        if os.path.split(dirName)[-1] == ("epochs"):
            y_train = []
            y_val = []
            for file in tqdm(fileList):
                if "train" in file:
                    train_data = os.path.join(dirName, file)
                    performance = torch.load(train_data)
                    # print(train_data, performance["weighted avg"]["precision"])
                    y_train.append(performance["weighted avg"]["precision"])
                elif "val" in file:
                    val_data = os.path.join(dirName, file)
                    performance = torch.load(val_data)
                    # print(val_data, performance["weighted avg"]["precision"])
                    y_val.append(performance["weighted avg"]["precision"])
                else:
                    print("unknown file detected {}".format(file))

            y_trains.append(y_train)
            y_vals.append(y_val)
            titles.append(os.path.split(os.path.split(dirName)[0])[1])


    plt.figure(figure_count)
    plt.cla()
    figure_count+=1

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
