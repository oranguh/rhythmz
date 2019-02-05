import argparse


def add_train_args(parser):
    parser.add_argument("--features", type=str,
                        choices=["mel-spectogram", "raw"],
                        required=True,
                        help="the features to use for training the network")
    parser.add_argument("--combine", type=str,
                        choices=["MoT", "LSTM"],
                        required=True,
                        help="the method to use for aggregating features learnt over time")
    parser.add_argument("--batch-size", dest="batch_size",
                        type=int, default=2, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=1,
                        help="batch size for training")
    parser.add_argument("--results-path", dest="results_path", type=str,
                        default="results", help="location to store results")
    parser.add_argument("--model-id", dest="model_id",
                        type=str, required=True, help="id of the experiment")
    parser.add_argument("--device", type=str, default="cpu",
                        help="device to train model on")
    parser.add_argument("--sample-rate", required=True,
                        help="sample rate of the audio", type=int, dest="sample_rate")
    parser.add_argument("--data-mean", type=float, dest="data_mean",
                        help="data mean. used to standardize the data", required=True)
    parser.add_argument("--data-std", type=float, dest="data_std",
                        help="data std. used to standardize the data", required=True)

    parser.add_argument("--input-size", type=int, dest="input_size",
                        help="size of the sliding window", required=True)
    parser.add_argument("--input-stride", type=int, dest="input_stride",
                        help="size of the hop / stride (time dimension)", required=True)


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
    parser_train.add_argument(
        "--test", action="store_true", help="flag to test the model")
    add_train_args(parser_train)

    parser_stats = subparsers.add_parser("compute-mean", help="compute mean")
    parser_stats.add_argument("--data", dest="data",
                              type=str, required=True, help="location of input data")
    parser_stats.add_argument("--sample-rate", required=True,
                              help="sample rate of the audio", type=int, dest="sample_rate")
    parser_stats.add_argument("--features", type=str,
                              choices=["mel-spectogram", "raw"],
                              required=True,
                              help="the features to use for training the network")

    return parser
