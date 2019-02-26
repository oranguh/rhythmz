import os
import codecs
import logging
import argparse

import bs4
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


MAIN_URL = "https://librivox.org/search?title=&author=&reader=&keywords=&genre_id=0&status=complete&project_type=solo&recorded_language={language_id}&sort_order=catalog_date&search_page={search_page}&search_form=advanced"


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def wait_till_class(driver, class_id, timeout=10):
    element_present = EC.presence_of_element_located((By.CLASS_NAME, class_id))
    WebDriverWait(driver, timeout).until(element_present)


def get_languages():
    url = "https://librivox.org/search"
    html = requests.get(url).text
    bs = bs4.BeautifulSoup(html, "html.parser")
    langs = []
    for opt in bs.find(id="recorded_language").find_all("option"):
        langs.append((opt["value"], opt.text.strip()))
    return langs


def download_lang(args, lang, languages):
    log.info("Downloading language: {}".format(lang))
    lang_id = [l[0] for l in languages if l[1] == lang][0]
    log.info("Language ID: {}".format(lang_id))
    # search page starts from 1
    next_search_page = 1
    driver = webdriver.Firefox(executable_path="./geckodriver")
    url_set = set()
    while next_search_page:
        log.info("Getting books from page {}. Links so far: {}".format(
            next_search_page, len(url_set)))
        url = MAIN_URL.format(
            language_id=lang_id, search_page=next_search_page)

        driver.get(url)
        wait_till_class(driver, "catalog-result")
        page = bs4.BeautifulSoup(driver.page_source, "html.parser")

        results = page.find_all("li", class_="catalog-result")
        for res in results:
            url = res.find("h3").a["href"]
            url_set.add(url)

        book_count = len(url_set)

        next_page_avail = page.find("div", class_="page-number-nav")

        # page-number-nav is not displayed if there's only one page
        if next_page_avail is None:
            next_search_page = None
        else:
            page_nos = [a.text for a in next_page_avail.find_all("a")]
            page_nos = [int(pg) for pg in page_nos if pg.isdigit()]

            max_page = max(page_nos)
            log.debug("Max page: {}".format(max_page))
            # if we've reached the end i.e the current search_page is == max
            # then quit!
            if next_search_page == max_page:
                next_search_page = None
            else:
                next_search_page += 1

        if len(url_set) >= args.limit:
            log.info("Reached limit. Quitting!")
            break

    driver.close()

    log.info("Scraped links for {} books".format(book_count))

    write_path = "{}.txt".format(lang)
    with codecs.open(write_path, "w", "utf-8") as writer:
        for url in url_set:
            writer.write("{}\n".format(url))

    log.info("Wrote list of URLS to {}".format(write_path))


def get_argparser():
    parser = argparse.ArgumentParser(
        "librivox-scraper", description="Scrapes libribox")
    parser.add_argument("--output", help="output folder", required=True)
    parser.add_argument("--selected-languages", dest="langs",
                        help="csv of languages that are downloaded. if this is not specified, the program lists all available langs")
    parser.add_argument("--temp-dir", dest="temp_dir", default="./temp",
                        help="location of the temporary directory where intermediate files are downloaded")
    parser.add_argument(
        "--limit", help="# books to download (max)", default=100000)
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)

    log = logging.getLogger("librivox-scraper")

    args = get_argparser().parse_args()

    log.info("Args: {}".format(args))

    mkdir(args.temp_dir)

    languages = get_languages()
    if not args.langs:
        log.info("No languages selected. Printing options: ")
        for lang in languages:
            log.debug("\tFound language: {}".format(lang[1]))
    else:
        sel_langs = args.langs.split(",")
        log.info("Selected languages: {}".format(sel_langs))
        for lang in sel_langs:
            download_lang(args, lang, languages)
