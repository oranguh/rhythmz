import os
import logging
from process.praat_util import PraatUtil

log = logging.getLogger(__name__)


class Process(object):
    def __init__(self, args):
        self.root_folder = args.root_folder
        self.output_folder = args.output_folder
        self.praat_util = PraatUtil(args.praat_path)

    def _mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def _process_file(self, source_file, target_file):
        log.debug("Processing file: {}".format(source_file))
        self.praat_util.stop_hann_band_filter(
            source_file, target_file, 500, 99999, 100)
        log.debug("Output destination: {}".format(target_file))

    def process(self):
        log.info("Traversing directory: {}".format(self.root_folder))

        self._mkdir(self.output_folder)
        for folder_name in os.listdir(self.root_folder):
            folder = os.path.join(self.root_folder, folder_name)
            files = os.listdir(folder)
            log.info("Found folder: {} with {} files".format(folder, len(files)))
            target_folder = os.path.join(self.output_folder, folder_name)
            log.info("Creating folder: {}".format(target_folder))
            self._mkdir(target_folder)

            for file_name in files:
                source_file = os.path.join(folder, file_name)
                target_file = os.path.join(target_folder, file_name)

                self._process_file(source_file, target_file)
