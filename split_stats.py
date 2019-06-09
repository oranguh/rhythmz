import os
import json
import logging
from datetime import timedelta
from collections import defaultdict

import librosa
import pandas as pd
from librosa.core import get_duration

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

if __name__ == '__main__':
    path = "datasets/librivox_splits"
    meta_path = "datasets/librivox_metadata"
    assigned_sample_rate = 8000
    times = []
    author_times = []
    book_times = []

    splits = os.listdir(path)

    # key: (split, lang, author, book_id)
    data = defaultdict(lambda: timedelta(
        hours=0, minutes=0, seconds=0))

    for split in splits:
        log.info("Split: {}".format(split))
        split_path = os.path.join(path, split)
        langs = os.listdir(split_path)

        for lang in langs:
            log.info("\tLanguage: {}".format(lang))
            lang_path = os.path.join(split_path, lang)

            with open(os.path.join(meta_path, lang) + ".json") as reader:
                meta_data = json.load(reader)

            meta_data = {d["book_id"]: d for d in meta_data}
            clips = os.listdir(lang_path)
            print_freq = len(clips) // 10
            for idx, clip in enumerate(clips, 1):
                file_path = os.path.join(lang_path, clip)
                sound, sample_rate = librosa.load(file_path, sr=None)
                assert sample_rate == assigned_sample_rate
                clip_len = timedelta(
                    seconds=get_duration(sound, sr=sample_rate))
                book_id = clip.split("_")[0]

                key = (split, lang, meta_data[book_id]["reader_url"], book_id)

                data[key] += clip_len

                if idx % print_freq == 0:
                    log.info("\t\t{} of {} done".format(idx, len(clips)))

            break

        break

    row_data = []
    for (split, lang, author_id, book_id), td in data.items():
        row_data.append({
            "split": split,
            "language": lang,
            "author_id": author_id,
            "book_id": book_id,
            "seconds": td.seconds
        })

    pd.DataFrame(row_data).to_csv("librivox_splits_times.csv", index=False)
