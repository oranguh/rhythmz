import logging
from logging.handlers import RotatingFileHandler


import torch
import numpy as np

from process import process
from models import trainer
from args_utils import get_argparser
from data.stats import ComputeMean


def configure_logging(module, verbose):
    handlers = [
        RotatingFileHandler(
            "logs/{}.log".format(module), maxBytes=1048576*5, backupCount=7),
        logging.StreamHandler()
    ]
    log_format = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    if verbose:
        logging.basicConfig(level=logging.DEBUG,
                            handlers=handlers, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=handlers, format=log_format)


if __name__ == '__main__':

    args = get_argparser().parse_args()

    log = logging.getLogger("rhythm")

    if args.module is None:
        raise ValueError("Provide a module argument (see --help)")

    configure_logging(args.module, args.verbose)

    # seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if hasattr(args, "device") and args.device != "cpu":
        torch.cuda.manual_seed_all(args.seed)

    log.info("Arguments: {}".format(args))
    if args.module == "process":
        data_process = process.Process(args)
        data_process.process()
    elif args.module == "compute-mean":
        mean_compute = ComputeMean(args)
        mean_compute.compute()
    elif args.module == "train":
        trainer = trainer.Trainer(args)
        if args.test:
            trainer.load()
            trainer.test()
        else:
            trainer.train()
    else:
        raise ValueError("Unknown module: {}".format(args.module))
