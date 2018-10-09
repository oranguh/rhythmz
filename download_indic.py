import os
import sys
import argparse
import subprocess

import wget

URLS = [
    "http://www.festvox.org/databases/iiit_voices/iiit_kan_lp.tar.gz",  # kannada
    "http://www.festvox.org/databases/iiit_voices/iiit_ben_ant.tar.gz",  # bengali
    "http://www.festvox.org/databases/iiit_voices/iiit_hin_sukh.tar.gz",  # hindi
    "http://www.festvox.org/databases/iiit_voices/iiit_mal_abi.tar.gz",  # malayalam
    "http://www.festvox.org/databases/iiit_voices/iiit_mar_ash.tar.gz",  # marathi
    "http://www.festvox.org/databases/iiit_voices/iiit_tam_moh.tar.gz",  # tamil
    "http://www.festvox.org/databases/iiit_voices/iiit_tel_baji.tar.gz",  # telugu
]

LANG = [
    "kannada",
    "bengali",
    "hindi",
    "malayalam",
    "marathi",
    "tamil",
    "telugu"
]

TEMP_FOLDER = "./temp"

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "download_indic", description="downloads the indic database and creates directory structure")
    parser.add_argument("output_folder", type=str,
                        help="location of root folder")

    args = parser.parse_args()

    mkdir(args.output_folder)
    mkdir(TEMP_FOLDER)

    for i, url in enumerate(URLS):
        file_name = url.split("/")[-1]
        folder_name = file_name.split(".")[0]
        dest_folder = os.path.join(args.output_folder, folder_name)
        download_location = os.path.join(args.output_folder, file_name)

        mkdir(dest_folder)

        print("Downloading {} to {}".format(url, dest_folder))

        if os.path.exists(download_location):
            print("File already exists. Skipping download")
        else:
            wget.download(url, out=download_location)

        # extract to folder
        print("Extracting data")
        with open(os.devnull, 'w') as devnull:
            subprocess.run(["tar", "-C", dest_folder, "-xvf", download_location],
                           stdout=devnull,  check=True)
        print("Done!")

        print("Saving relevant files and discard the rest!")
        final_folder = os.path.join(args.output_folder, LANG[i])
        with open(os.devnull, 'w') as devnull:
            subprocess.run(["mv", os.path.join(dest_folder, folder_name, "wav"), final_folder],
                           stdout=devnull,  check=True)


        # perform cleanup
        with open(os.devnull, 'w') as devnull:
            subprocess.run(["mv", download_location, os.path.join(TEMP_FOLDER, file_name)],
                           stdout=devnull,  check=True)
            
            subprocess.run(["rm", "-rf", dest_folder],
                           stdout=devnull)