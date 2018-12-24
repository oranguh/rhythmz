import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--verbose', dest='verbose', default=False,
                        action='store_true', help="set to true for verbose output")
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
    parser_process.add_argument("--praat-path", dest="praat_path", type=str,
                                default="./praat", help="location of the praat executable")
    parser_process.add_argument("--filter-type", dest="filter_type",
                                type=str, default="stop", choices={"stop", "pass"},
                                help="whether to use a stop/pass filter")
    parser_process.add_argument("--freq-from", dest="freq_from",
                                type=int, default=500, help="frequency range start for filter")
    parser_process.add_argument("--freq-to", dest="freq_to", type=int,
                                default=999999, help="frequency range end for filter")
    parser_process.add_argument("--freq-smooth", dest="freq_smooth", type=int,
                                default=100, help="frequency for smoothing")

    parser_train = subparsers.add_parser("train", help="train the model")
    parser_train.add_argument("--data", dest="data",
                                type=str, required=True, help="location of input data")
    parser_train.add_argument("--batch-size", dest="batch_size",
                                type=int, default=2, help="batch size for training")
    return parser
