import logging


import torch
import numpy as np

from process import process
from models import trainer
from args_utils import get_argparser
from data.stats import ComputeMean


if __name__ == '__main__':


    log = logging.getLogger("rhythm")

    args = get_argparser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    if args.module is None:
        raise ValueError("Provide a module argument (see --help)")

    # seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.module == "process":
        data_process = process.Process(args)
        data_process.process()
    elif args.module == "compute-mean":
        mean_compute = ComputeMean(args)
        mean_compute.compute()
    elif args.module == "train":
        trainer = trainer.Trainer(args)
        trainer.train()
    else:
        raise ValueError("Unknown module: {}".format(args.module))
