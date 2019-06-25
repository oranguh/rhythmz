import os
import pickle as pkl


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def load_obj(path):
    with open(path, "rb") as reader:
        return pkl.load(reader)


def save_obj(obj, path):
    with open(path, "wb") as writer:
        pkl.dump(writer)
