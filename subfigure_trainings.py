import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# from models.classifier import AudioClassifier
import torch
import seaborn as sn
import pandas as pd

def main():
    boxplot = True
    confusion = True

    folder_name = "sam2"
    all_models = "C:\\Users\\murco.DESKTOP-R324UUU\\Documents\\rhythmz\\results\\" + folder_name
    images_path = "C:\\Users\\murco.DESKTOP-R324UUU\\Documents\\rhythmz\\images\\" + folder_name
    try:
        os.mkdir(images_path)
        print("Directory ", images_path, " Created ")
    except FileExistsError:
        print("Directory ", images_path, " already exists")

    dataset = os.path.split(all_models)[1]
    confusion_title = "hello"
    y_trains = []
    y_vals = []
    titles = []
    figure_count = 0
    for dirName, subdirList, fileList in os.walk(all_models):
        # print(os.path.split(dirName)[-1])
        if os.path.split(dirName)[-1] == "confusion_matrix":
            # print("samasnjadn")
            for file in tqdm(fileList):
                # continue
                epoch = file.split("_")[0]
                train_val = file.split("_")[1]
                if not (int(epoch) == 29):
                    continue
                if (train_val == "train"):
                    # continue
                    pass
                if (train_val == "val"):
                    # continue
                    pass
                language_list = ['Chinese', 'Danish', 'Dutch', 'English',
                                 'Esperanto', 'Finnish', 'French', 'German',
                                 'Greek', 'Hebrew', 'Italian', 'Japanese', 'Korean',
                                 'Latin', 'Polish', 'Portuguese',
                                 'Russian', 'Spanish', 'Swedish', 'Tagalog', 'Tamil']

                if dataset == "indic":
                    language_list = ["bengali", "hindi", "kannada", "malayalam", "marathi", "tamil", "telegu"]
                elif dataset == "topCoder":
                    language_list = ["arabic", "dutch", "hindi", "N korean", "polish", "romanian", "thai", "vietnamese"]
                confusion_title = os.path.split(os.path.split(dirName)[0])[1]

                confusion_matrix = np.load(os.path.join(dirName, file))
                # print(type(confusion_matrix))
                # print(confusion_matrix[()][(2, 3)])
                class_count = len(language_list)
                confusion_real_matrix = np.zeros((class_count, class_count))
                for i in range(class_count):
                    for j in range(class_count):
                        confusion_real_matrix[i,j] = confusion_matrix[()][(i, j)]

                if confusion:
                    df_cm = pd.DataFrame(confusion_real_matrix, language_list,
                      language_list)
                    plt.figure(figure_count, figsize=(8, 6))#, figsize = (10,7))
                    plt.title("{}  epoch: {} {}".format(train_val, epoch, confusion_title))
                    figure_count += 1

                    # sn.set(font_scale=1.4)#for label size
                    sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}, fmt='g')# font size
                    savefile = "{}_{}_matrix".format(train_val, confusion_title) + ".png"
                    savefile = os.path.join(images_path, savefile)
                    plt.savefig(savefile, bbox_inches='tight')

                if boxplot:

                    df = pd.DataFrame(columns=language_list,
                                data = confusion_real_matrix.T,
                                index=language_list)

                    # plt.figure(figure_count)
                    # figure_count += 1
                    # color = ['turquoise', 'darkgreen', 'tomato', 'indianred', 'forestgreen', 'sienna', 'darkorange']
                    sn.set()
                    # fig = df.T.plot(legend = False, kind='bar', stacked=True).get_figure()

                    ax = plt.figure(figsize=(10, 6)).add_subplot(111)
                    fig = df.T.plot(ax=ax, kind='bar', stacked=True, legend=True).get_figure()

                    bars = ax.patches
                    # print(len(['-', '+', 'x','/','//','O','o','\\','\\\\']), len(df))
                    hatches = ''.join(h*len(df) for h in ['-', '+', 'x','/','//','O','o','|','*'])#'x/O.')
                    # print(hatches)
                    # print("\n", len(bars), len(hatches), len(language_list), "\n")
                    for bar, hatch in zip(bars, hatches):
                        bar.set_hatch(hatch)

                    # ax.legend(loc='center right', bbox_to_anchor=(1, 1), ncol=4)

                    # plt.xlabel("Languages")
                    plt.ylabel("Predicted")
                    plt.title("{}  epoch: {} {}".format(train_val, epoch, confusion_title))
                    figure_count += 1
                    savefile = "{}_{}_barchart".format(train_val, confusion_title) + ".png"
                    savefile = os.path.join(images_path, savefile)
                    fig.savefig(savefile, bbox_inches='tight')
                # plt.show()
                plt.close('all')

        if os.path.split(dirName)[-1] == ("epochs"):
            y_train = []
            y_val = []
            fileList = sorted(fileList, key=lambda _: int(_.split("_")[0]))
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

    plt.figure(figure_count, figsize=(20,10))
    plt.cla()
    figure_count+=1

    for i, subplots in enumerate(y_trains):
        y_train = y_trains[i]
        y_val = y_vals[i]
        x = list(range(0, len(y_val)))
        epoch_count = len(subplots)

        plt.subplot(len(y_trains), 1, i+1).set_title(titles[i])
        plt.plot(x, y_train, 'o-', label="training")
        plt.plot(x, y_val, 'r.-', label="validation")
        plt.ylim(0, 1.1)
        plt.xlim(0, epoch_count)
        plt.ylabel("precision")
        plt.xlabel("epoch")
        # if (not (i+1 == 8 or i+1 == 7)):
        if (not (i+1 == len(y_trains))):
            plt.xlabel("")
            plt.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False)
    blue = mpatches.Patch(color='blue', label='Training')
    red = mpatches.Patch(color='red', label='Validation')
    plt.legend(handles=[blue, red])
    # plt.tight_layout()
    savefile = "{}_epochs".format(confusion_title) + ".png"
    savefile = os.path.join(images_path, savefile)
    plt.savefig(savefile, bbox_inches='tight')
    # plt.show()
    plt.close('all')



if __name__ == "__main__":
    main()
