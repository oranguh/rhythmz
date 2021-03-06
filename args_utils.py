import argparse


def add_train_args(parser):
    n_epochs_default = 30
    parser.add_argument("--features", type=str,
                        choices=["ms", "raw"],
                        required=True,
                        help="the features to use for training the network")
    parser.add_argument("--audio", default=False, action="store_true",
                        help="if set, use raw audio instead of rhythm data")
    parser.add_argument("--batch-size", dest="batch_size",
                        type=int, default=32, help="batch size for training")
    parser.add_argument("--learning-rate", dest="learning_rate",
                        type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch-norm", dest="batch_norm",
                        action="store_true", default=False, help="(flag) use BN in features")
    parser.add_argument("--epochs", type=int, default=n_epochs_default,
                        help="batch size for training")
    parser.add_argument("--results-path", dest="results_path", type=str,
                        default="results", help="location to store results")
    parser.add_argument("--model-id", dest="model_id",
                        type=str, required=True, help="id of the experiment")
    parser.add_argument("--device", type=str, default="cpu",
                        help="device to train model on")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="num workers to load data")
    parser.add_argument("--weight-decay", dest="weight_decay", default=0,
                        type=float, help="weight decay for Adam")
    parser.add_argument("--feature-training-epochs", dest="feature_training_epochs", default=-1, type=int,
                        help="(n): if n positive, mode 2 classifier is trained for that n epochs and the features are then frozen")


def get_argparser():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--verbose', dest='verbose', default=False,
                        action='store_true', help="set to true for verbose output")
    parser.add_argument('--seed', dest="seed", required=False, type=int,
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
    parser_train.add_argument(
        "--test", action="store_true", help="flag to test the model")
    add_train_args(parser_train)

    return parser
