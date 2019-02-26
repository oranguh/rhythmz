import os
import codecs
import shutil
import zipfile
import logging
import hashlib
import argparse
from datetime import timedelta

import bs4
import requests
import librosa
from librosa.core import get_duration

from scraper import mkdir


def get_argparser():
    parser = argparse.ArgumentParser(
        "librivox-downloader", description="Downloads the files")
    parser.add_argument(
        "--input", help="folder containing files with lists of URLS", required=True)
    parser.add_argument("--output", help="output folder", required=True)
    parser.add_argument("--temp-dir", dest="temp_dir", default="./temp",
                        help="location of the temporary directory where intermediate files are downloaded")
    parser.add_argument(
        "--limit", help="max time to download", default="00:10:00")
    return parser


def download_file(url, destination):
    # https://stackoverflow.com/a/39217788
    r = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        shutil.copyfileobj(r.raw, f)


def download_books(language, output_dir, temp_dir, url_list, max_time):
    current_time = timedelta(hours=0, minutes=0, seconds=0)

    to_move = []

    meta = []
    for url in url_list:
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

        # read time, and add to list
        for clip_id, file_name in enumerate(os.listdir(folder_loc)):
            file_path = os.path.join(folder_loc, file_name)
            clip, sr = librosa.load(file_path)
            clip_len = timedelta(seconds=get_duration(clip, sr=sr))
            log.info("Loading clip: {}, Time: {}".format(file_path, clip_len))
            to_move.append({
                "path": file_path,
                "book_id": book_id,
                "clip_id": clip_id
            })
            log.debug("Current length: {}".format(current_time))
            current_time += clip_len

            if current_time >= max_time:
                break

        if current_time >= max_time:
            break

    log.info("Total length: {}, Starting copy".format(current_time))

    for file_data in to_move:
        # TODO split the files into the given length
        src_file_path = file_data["path"]
        src_ext = os.path.split(src_file_path)[1].split(".")[1]
        dest_path = os.path.join(output_dir, "{}_{}.{}".format(
            file_data["book_id"], file_data["clip_id"], src_ext))
        log.debug("Copying {} to {}".format(src_file_path, dest_path))
        shutil.copyfile(file_path, dest_path)


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
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)

    log = logging.getLogger("librivox-downloader")

    args = get_argparser().parse_args()

    log.info("Args: {}".format(args))

    mkdir(args.output)
    mkdir(args.temp_dir)

    urls = read_urls(args.input)

    hours, minutes, seconds = args.limit.split(":")
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    max_time = timedelta(hours=hours, minutes=minutes, seconds=seconds)

    log.info("Max Time: {}".format(max_time))

    for lang in urls:
        output_dir = os.path.join(args.output, lang)
        mkdir(output_dir)
        download_books(lang, output_dir, args.temp_dir, urls[lang], max_time)

    clear_dir(temp_dir)
