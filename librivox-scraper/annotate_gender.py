# this script is used to create a single audio file
# per author, so it's easy to annotate
import os
import json
import shutil
import random
from collections import defaultdict

import pandas as pd

if __name__ == '__main__':
    data_root = "../datasets/librivox_splits"
    meta_root = "../datasets/librivox_metadata"
    n_clips = 3
    annotate_location = "../datasets/temp_gender_annotate"

    random.seed(42)

    if not os.path.exists(annotate_location):
        os.mkdir(annotate_location)

    author_clips = defaultdict(list)
    splits = os.listdir(data_root)
    for split in splits:
        split_path = os.path.join(data_root, split)
        languages = os.listdir(split_path)
        metadata = {}
        for cl in languages:
            with open(os.path.join(meta_root, cl + ".json")) as reader:
                meta = json.load(reader)
                # make meta a dict with book_id as key
                metadata[cl] = {m["book_id"]: m for m in meta}

        for language in languages:
            for clip in os.listdir(os.path.join(split_path, language)):
                book_id = clip.split("_")[0]
                author_id = metadata[language][book_id]["reader_url"]

                author_clips[author_id].append(
                    os.path.join(split_path, language, clip))

    rows = []
    for author_id, clips in author_clips.items():
        choices = random.choices(clips, k=n_clips)
        for choice in choices:
            _, name = os.path.split(choice)
            dest = os.path.join(annotate_location, str(
                len(rows)).zfill(5)) + ".wav"
            print("Src: {}, Dest:{} ".format(
                choice, dest))

            shutil.copy(choice, dest)

            rows.append({
                "file_name": dest,
                "author_id": author_id
            })

    pd.DataFrame(rows).to_csv(os.path.join(
        annotate_location, "temp.csv"), index=False)
