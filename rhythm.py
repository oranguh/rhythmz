import logging

from process import process
from args_utils import get_argparser


if __name__ == '__main__':

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log = logging.getLogger("rhythm")

    args = get_argparser().parse_args()

    if args.module is None:
        raise ValueError("Provide a module argument (see --help)")

    if args.module == "process":
        data_process = process.Process(args)
        data_process.process()
    else:
        raise ValueError("Unknown module: {}".format(args.module))
