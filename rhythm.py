import logging

from process import process
from data import dataloaders
from args_utils import get_argparser


if __name__ == '__main__':

    
    log = logging.getLogger("rhythm")

    args = get_argparser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    if args.module is None:
        raise ValueError("Provide a module argument (see --help)")

    if args.module == "process":
        data_process = process.Process(args)
        data_process.process()
    elif args.module == "train":
        dl = dataloaders.AudioDataLoader("indic_data/")
        print(dl[0])
    else:
        raise ValueError("Unknown module: {}".format(args.module))
