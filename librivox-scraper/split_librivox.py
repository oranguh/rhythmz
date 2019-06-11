import re
import os
import json
import glob
import shutil
import logging
import argparse
from collections import defaultdict, OrderedDict

import numpy as np


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def read_files(folder):
    return [os.path.split(file_name)[-1] for file_name in list_dir(folder)]


def write_split(src_folder, dest_folder, clips):
    for c in clips:
        src_path = os.path.join(src_folder, c)
        dest_path = os.path.join(dest_folder, c)
        log.info("Src: {}, Dest: {}".format(src_path, dest_path))
        shutil.copy(src_path, dest_path)


def read_split(path, split, prefix):
    path = os.path.join(path, split)
    files = []
    for cl in os.listdir(path):
        for file in os.listdir(os.path.join(os.path.join(path, cl))):
            files.append((cl, os.path.join(prefix, cl, file)))
    return files


def split_clips(meta, clips):
    """
    Split the clips into a train, validation and test set. 
    Due to the nature of the experiment, the test set should contain a unique author,
    that isn't present in the training set. This author may be present in the validation set, 
    but not ideally i.e we want to have a zero-shot learning settting for the test set. 

    The splits are made as follows:
        TODO
    """
    book_ids = [c.split("_")[0] for c in clips]
    author_meta = {m["book_id"]: m["reader_url"] for m in meta}

    authors = list(set(author_meta.values()))

    authors_count = defaultdict(int)
    for b in book_ids:
        authors_count[author_meta[b]] += 1
    authors_count = sorted(authors_count.items(), key=lambda _: -_[1])

    authors = [a[0] for a in authors_count]
    print(authors_count)
    log.info("\t{} authors found".format(len(authors)))

    # we want a val/test split with authors not in the train
    val_authors = set()
    test_authors = set()
    train_authors = set()

    test_authors = test_authors.union(authors)
    val_authors = val_authors.union(authors)
    if len(authors_count) == 2:
        train_authors.add(authors[0])
    elif len(authors_count) == 3:
        train_authors = train_authors.union(authors[:2])
    else:
        train_authors = train_authors.union(authors[:len(authors) - 1])

    log.info("Train Authors: {}, Val Authors: {}, Test Authors:{}".format(
        len(train_authors), len(val_authors), len(test_authors)))

    train_clips, val_clips, test_clips = [], [], []

    train_prob = args.train_split
    val_prob = args.val_split
    test_prob = (1 - train_prob - val_prob)

    for clip in clips:
        book_id = clip.split("_")[0]
        author_id = author_meta[book_id]

        # print(author_id)
        # print(author_id in test_authors)
        # print(author_id in val_authors)
        # print(author_id in train_authors)
        # print(a)
        if author_id in test_authors and author_id in val_authors and author_id in train_authors:
            selected = np.random.choice(
                ["train", "val", "test"],
                1,
                p=[train_prob, val_prob, test_prob]
            )[0]
            selected = {
                "train": train_clips,
                "val": val_clips,
                "test": test_clips,
            }[selected]

            selected.append(clip)
        elif author_id in test_authors and author_id in val_authors:
            selected = np.random.choice(
                ["val", "test"],
                1,
                p=[val_prob / (val_prob + test_prob),
                   test_prob / (val_prob + test_prob)]
            )[0]
            selected = {
                "val": val_clips,
                "test": test_clips,
            }[selected]

            selected.append(clip)
        else:
            raise ValueError("Um")

    total = float(len(train_clips) + len(val_clips) + len(test_clips))
    assert total == len(clips)

    log.info("Percentages:\n\tTrain: {:.2f}\n\tVal: {:.2f}\n\tTest:{:.2f}".format(
        len(train_clips) / total,
        len(val_clips) / total,
        len(test_clips) / total,))

    return train_clips, val_clips, test_clips


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        "split_dataset", description="splits a given dataset into two")
    parser.add_argument("--input", type=str, required=True,
                        help="location of librivox data")
    parser.add_argument("--meta", type=str, required=True,
                        help="location of the librivox metadata")
    parser.add_argument("--output", type=str, required=True,
                        help="location of output folder")
    parser.add_argument("--seed", type=int, help="seed for RNG", required=True)
    parser.add_argument("--train-split", dest="train_split", default=0.6,
                        type=float, help="(0, 1.0) %age of train split")
    parser.add_argument("--val-split", dest="val_split", default=0.2,
                        type=float, help="(0, 1.0) %age of val split")

    args = parser.parse_args()

    np.random.seed(args.seed)

    mkdir(args.output)
    mkdir(os.path.join(args.output, "train"))
    mkdir(os.path.join(args.output, "val"))
    mkdir(os.path.join(args.output, "test"))

    # load languages
    langs = os.listdir(args.meta)
    for lang in langs:
        # remove the `.json` part in the string
        lang = lang.split(".")[0]

        with open(os.path.join(args.meta, lang + ".json")) as reader:
            meta = json.load(reader)

        clips = os.listdir(os.path.join(args.input, lang))

        if len(clips) == 0:
            log.info("No clips for language {} found. Skipping".format(lang))
            continue

        log.info("Language: {}".format(lang))
        log.info("\t{} clips found".format(len(clips)))

        train_clips, val_clips, test_clips = split_clips(meta, clips)

        # save / move clips to the proper position
        train_dir = os.path.join(args.output, "train", lang)
        val_dir = os.path.join(args.output, "val", lang)
        test_dir = os.path.join(args.output, "test",  lang)

        # make directories
        mkdir(train_dir)
        mkdir(val_dir)
        mkdir(test_dir)

        # move clips
        write_split(os.path.join(args.input, lang), train_dir, train_clips)
        write_split(os.path.join(args.input, lang), val_dir, val_clips)
        write_split(os.path.join(args.input, lang), test_dir, test_clips)
