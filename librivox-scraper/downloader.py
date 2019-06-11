import os
import sys
import json
import codecs
import shutil
import random
import zipfile
import logging
import hashlib
import argparse
from datetime import timedelta
from collections import defaultdict

import bs4
import librosa
import requests
import numpy as np
from librosa.core import get_duration

from scraper import mkdir

MAX_META = 1500


def get_argparser():
    parser = argparse.ArgumentParser(
        "librivox-downloader", description="Downloads the files")
    parser.add_argument(
        "--input", help="folder containing files with lists of URLS", required=True)
    parser.add_argument("--output", help="output folder", required=True)
    parser.add_argument(
        "--metadata", help="location to store metadata per clip", required=True)
    parser.add_argument("--temp-dir", dest="temp_dir", default="./temp",
                        help="location of the temporary directory where intermediate files are downloaded")
    parser.add_argument(
        "--limit", help="max time to download", default="00:01:00")
    parser.add_argument("--max-clip-len", dest="max_clip_len", type=int,
                        help="maximum clip length (seconds)", default=10)
    parser.add_argument("--sample-rate", dest="sample_rate", default=8000,
                        help="target sample rate")
    parser.add_argument(
        "--min-authors", dest="min_authors", type=int,
        help="the minimum number of authors to download (otherwise that language is ignored)", default=2)
    parser.add_argument("--max-authors", dest="max_authors", type=int,
                        help="max number of authors to download", default=6)
    parser.add_argument("--max-per-author", dest="max_per_author", default=2,
                        type=int, help="max # books to download per author")
    return parser


def audio_splitter_upper(y, length, sr, margin=20):
    split_audio = []
    # remove start and end for that annoying librivox intro
    y = y[margin*sr:-margin*sr]

    while len(y) > length*sr:
        split_audio.append(y[:length*sr])
        y = y[length*sr:]
    return np.asarray(split_audio)


def download_file(url, destination):
    # https://stackoverflow.com/a/39217788
    r = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        shutil.copyfileobj(r.raw, f)


def download_meta(language, url_list):
    meta = []

    # first, download the metadata for all the books
    for idx, url in enumerate(url_list):
        if idx % 10 == 0:
            log.info("\t{} of {}".format(idx + 1, len(url_list)))
        if len(meta) >= MAX_META:
            log.info("Maximum metadata length achieved. Quitting")
            break
        try:
            page = bs4.BeautifulSoup(requests.get(url).text, "html.parser")

            divs = page.find_all("div", class_="book-page-sidebar")
            meta_data_div = [div for div in divs if div.h4.text ==
                             "Production details"][0]

            book_id = hashlib.md5(url.encode()).hexdigest()

            meta_data = {
                "url": url,
                "language": language,
                "book_id": book_id
            }

            key_tags = meta_data_div.find_all("dt")
            value_tags = meta_data_div.find_all("dd")

            for key_tag, value_tag in zip(key_tags, value_tags):
                value = value_tag.text.strip()
                key = key_tag.text.strip().strip(":")
                if key == "Read by":
                    reader_url = value_tag.find("a")["href"]
                    meta_data["reader_url"] = reader_url
                meta_data[key] = value

            download_btns = page.find_all("a", class_="book-download-btn")
            download_btn = [
                btn for btn in download_btns if btn.text == "Download"][0]
            zip_link = download_btn["href"]

            meta_data["zip_link"] = zip_link

            meta.append(meta_data)

        except KeyboardInterrupt:
            log.exception("Keyboard Int. Quitting!")
            sys.exit(-1)
        except:
            log.exception("Some exception occurred. Skipping!")

    log.info("Downloaded metadata for {} books".format(len(meta)))
    return meta


def get_next_clip(all_files):
    # select only clips that are available
    all_files = [a for a in all_files if a["elapsed"] is False]
    # if no more files are available, exit
    if len(all_files) == 0:
        return None

    # randomly pick a file
    selected_idx = np.random.randint(0, len(all_files))
    selected_file = all_files[selected_idx]
    file_path = selected_file["file_path"]

    # load only part of the clip

    clip, sr = librosa.load(
        file_path, sr=args.sample_rate,
        offset=selected_file["time_elapsed"].seconds,
        duration=args.max_clip_len)

    assert sr == args.sample_rate

    clip_len = timedelta(
        seconds=get_duration(clip, sr=sr))

    if clip_len.seconds == 0:
        selected_file["elapsed"] = True
        return timedelta(seconds=0, hours=0, minutes=0)

    elapsed_time = selected_file["time_elapsed"].seconds
    # print("Clip ID: ", elapsed_time, "Clip Len:", clip_len,
    #       "Time Elapsed: ", selected_file["time_elapsed"])
    selected_file["time_elapsed"] += clip_len

    split_clip_id = "{}_{}_{}".format(
        selected_file["book_id"], selected_file["clip_id"], elapsed_time)
    dest_path = os.path.join(
        output_dir, "{}.wav".format(split_clip_id))

    log.debug("Saving clip to {}".format(
        dest_path))

    librosa.output.write_wav(dest_path, clip, args.sample_rate)

    return clip_len


def download_books(language, output_dir, temp_dir, url_list, max_time, args):
    log.info("Language: {}".format(language))

    meta_path = os.path.join(args.metadata, "{}.json".format(language))

    if os.path.exists(meta_path):
        log.info("Metadata exists for {}, skipping download".format(language))
        with codecs.open(meta_path, "r", "utf-8") as reader:
            metadata = json.load(reader)
    else:
        log.info("Downloading metadata for {}".format(language))
        metadata = download_meta(language, url_list)
        # download the meta data for all the files
        log.info("Saving metadata to {}".format(meta_path))
        with codecs.open(meta_path, "w", "utf-8") as writer:
            json.dump(metadata, writer)

    authors = set(meta["reader_url"] for meta in metadata)
    log.info("Language {} has {} authors".format(language, len(authors)))
    if len(authors) < args.min_authors:
        log.info(
            "Ignoring language {} because only {} authors are available. Min: {}".format(language, len(authors), args.min_authors))
        return
    if len(authors) > args.max_authors:
        log.info("Too many authors, so selecting {} authors only".format(
            args.max_authors))
        authors = sorted(list(authors))[:args.max_authors]

    log.info("Sampling from {} authors: {}".format(len(authors), authors))

    metadata_all = [m for m in metadata if m["reader_url"] in authors]

    metadata_all = sorted(metadata_all, key=lambda _: _["reader_url"])

    metadata = []
    author_book_count = defaultdict(int)
    for m in metadata_all:
        author_id = m["reader_url"]
        if author_book_count[author_id] >= args.max_per_author:
            continue
        author_book_count[author_id] += 1
        metadata.append(m)

    log.info("Attempting to download {} books".format(len(metadata)))

    all_files = []

    metadata.sort(key=lambda _: _["book_id"])

    # first, download all the files and read in the file names
    # we do this because we want many authors in the final audio clips
    for meta in metadata:
        book_id = meta["book_id"]
        zip_link = meta["zip_link"]
        author_id = meta["reader_url"]

        # see if the file exists in the temp dir - if it exists, don't download it
        download_loc = os.path.join(temp_dir, book_id + ".zip")
        folder_loc = os.path.join(temp_dir, book_id)

        # download zip file
        if not os.path.exists(download_loc):
            log.info("Downloading {} to {}".format(zip_link, download_loc))
            download_file(zip_link, download_loc)

        log.info("Extracting {} to {}".format(download_loc, folder_loc))
        # extract zip file
        with zipfile.ZipFile(download_loc, "r") as zip_ref:
            zip_ref.extractall(folder_loc)

        # # read time, and add to list
        for clip_id, file_name in enumerate(os.listdir(folder_loc)):
            file_path = os.path.join(folder_loc, file_name)

            all_files.append({
                "book_id": book_id,
                "reader_url": meta["reader_url"],
                "author_id": author_id,
                "clip_id": clip_id,
                "file_path": file_path,
                "time_elapsed": timedelta(hours=0, minutes=0, seconds=0),
                "elapsed": False
            })

    current_time = timedelta(hours=0, minutes=0, seconds=0)

    # keep going until we reach the time we want
    while current_time < max_time:
        time_saved = get_next_clip(all_files)
        if time_saved is None:
            break
        else:
            current_time += time_saved

    log.info("Total length: {}".format(current_time))


def read_urls(folder):
    urls = {}
    for lang in os.listdir(folder):
        path = os.path.join(folder, lang)
        lang = lang.split(".")[0]

        with codecs.open(path, "r", "utf-8") as reader:
            url = reader.readlines()
            url = [u.strip() for u in url]

        if len(url) > 0:
            log.info("Language: {} ({})".format(lang, len(url)))
            urls[lang] = url
        else:
            log.info("No URLS found for {}".format(lang))

    return urls


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)

    log = logging.getLogger("librivox-downloader")

    args = get_argparser().parse_args()

    log.info("Args: {}".format(args))

    mkdir(args.output)
    mkdir(args.temp_dir)
    mkdir(args.metadata)

    urls = read_urls(args.input)

    hours, minutes, seconds = args.limit.split(":")
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    max_time = timedelta(hours=hours, minutes=minutes, seconds=seconds)

    log.info("Max Time: {}".format(max_time))

    # skip_langs = ["Arabic", "Balinese", "Bulgarian", "Chinese",
    #               "Danish", "Dutch", "English", "Esperanto", "Finnish", "French",
    #               "German", "Greek", "Hebrew", "Hungarian", "Indonesian", "Italian",
    #               "Japanese", "Korean", "Latin", "Latvian", "Multilingual", "Portuguese", "Russian",
    #               "Spanish", "Sudanese", "Swedish"]
    for lang in urls:
        if lang in skip_langs:
            log.info("Skipping lang: {}".format(lang))
            continue
        output_dir = os.path.join(args.output, lang)
        mkdir(output_dir)
        download_books(lang, output_dir, args.temp_dir,
                       urls[lang], max_time, args)

    # clear_dir(temp_dir)
