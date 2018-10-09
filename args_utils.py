import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--verbose-off', dest='verbose_off', required=False,
                        default=False, action='store_true', help="set to true for verbose output")
    parser.add_argument('--seed', dest="seed", required=False,
                        default=42, help="random seed")

    subparsers = parser.add_subparsers(dest='module', help='module to run')

    # subparaser for building permit-desc classifiers
    parser_process = subparsers.add_parser(
        "process", help="process audio files")
    parser_process.add_argument("--root-folder", dest="root_folder",
                                type=str, required=True, help="location of input data")
    parser_process.add_argument("--output-folder", dest="output_folder",
                                type=str, required=True, help="location of the output folder")
    return parser
