import os
import json
from datetime import timedelta
from collections import defaultdict

import librosa
import pandas as pd
from librosa.core import get_duration

if __name__ == '__main__':
    path = "datasets/librivox"
    meta_path = "datasets/librivox_metadata"
    assigned_sample_rate = 8000
    times = []
    author_times = []
    book_times = []

    for cl in os.listdir(path):
        cl_path = os.path.join(path, cl)
        current_time = timedelta(hours=0, minutes=0, seconds=0)
        all_files = os.listdir(cl_path)
        print("Reading: {}".format(cl))

        with open(os.path.join(meta_path, cl) + ".json") as reader:
            meta_data = json.load(reader)

        meta_data = {d["book_id"]: d for d in meta_data}

        book_ids = defaultdict(lambda: timedelta(
            hours=0, minutes=0, seconds=0))
        author_ids = defaultdict(
            lambda: timedelta(hours=0, minutes=0, seconds=0))

        for idx, file_id in enumerate(all_files):
            file_path = os.path.join(cl_path, file_id)
            sound, sample_rate = librosa.load(file_path, sr=None)
            assert sample_rate == assigned_sample_rate
            current_time += timedelta(
                seconds=get_duration(sound, sr=sample_rate))

            book_id = file_id.split("_")[0]
            book_ids[book_id] += timedelta(
                seconds=get_duration(sound, sr=sample_rate))
            author_ids[meta_data[book_id]["reader_url"]] += timedelta(
                seconds=get_duration(sound, sr=sample_rate))

            if idx % 100 == 0:
                print("\t{} of {}".format(idx, len(all_files)))

        print("{} Books".format(cl))
        for book_id in book_ids:
            print("\t{}: Time: {}".format(book_id, book_ids[book_id]))
            book_times.append({
                "language": cl,
                "book_id": book_id,
                "time": str(book_ids[book_id])
            })

        print("{} Authors".format(cl))
        for author_id in author_ids:
            print("\t{}: Time: {}".format(author_id, author_ids[author_id]))
            author_times.append({
                "language": cl,
                "author_id": author_id,
                "time": str(author_ids[author_id])
            })

        print("Class: {}, Time: {}".format(cl, current_time))

        times.append({
            "language": cl,
            "time": str(current_time)
        })

    pd.DataFrame(times).to_csv("librivox_times.csv", index=False)
    pd.DataFrame(book_times).to_csv("librivox_book_times.csv", index=False)
    pd.DataFrame(author_times).to_csv("librivox_author_times.csv", index=False)
