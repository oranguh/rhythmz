import re
import os
import glob
import shutil
import logging
import argparse

import numpy as np


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def read_files(folder):
    return [file_name.split("/")[-1] for file_name in list_dir(folder)]


def write_split(path, data):
    classes = {d[0] for d in data}
    for cl in classes:
        mkdir(os.path.join(path, cl))

    for d in data:
        file_name = d[1].split("/")[-1]
        dest = os.path.join(path, d[0], file_name)
        log.info("Src: {}, Dest: {}".format(d[1], dest))
        shutil.copy(d[1], dest)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        "split_dataset", description="splits a given dataset into two")
    parser.add_argument("input_folder", type=str,
                        help="location of input folder")
    parser.add_argument("output_folder", type=str,
                        help="location of output folder")
    parser.add_argument("--train-split", dest="train_split", default=0.6,
                        type=float, help="(0, 1.0) %age of train split")
    parser.add_argument("--val-split", dest="val_split", default=0.2,
                        type=float, help="(0, 1.0) %age of val split")

    args = parser.parse_args()

    train_split = args.train_split
    val_split = args.val_split
    test_split = 1 - (train_split + val_split)

    log.info("Train: {}, Val: {}, Test: {}".format(
        train_split, val_split, test_split))

    mkdir(args.output_folder)
    mkdir(os.path.join(args.output_folder, "train"))
    mkdir(os.path.join(args.output_folder, "test"))
    mkdir(os.path.join(args.output_folder, "val"))

    all_files = []
    for class_name in os.listdir(args.input_folder):
        class_path = os.path.join(args.input_folder, class_name)
        class_files = [os.path.join(class_path, _) for _ in os.listdir(class_path)]
        log.info("Class: {}. Instances found: {}".format(
            class_name, len(class_files)))
        all_files.extend([(class_name, cf) for cf in class_files])

    train_idx = int(train_split * len(all_files))
    val_idx = int(val_split * len(all_files)) + train_idx

    all_files_sorted = np.random.permutation(all_files)
    train_split = all_files_sorted[:train_idx]
    val_split = all_files_sorted[train_idx: val_idx]
    test_split = all_files_sorted[val_idx:]

    log.info("Train Size: {}, Val Size: {}, Test Size: {}".format(
        len(train_split), len(val_split), len(test_split)))

    write_split(os.path.join(args.output_folder, "train"), train_split)
    write_split(os.path.join(args.output_folder, "val"), val_split)
    write_split(os.path.join(args.output_folder, "test"), test_split)
