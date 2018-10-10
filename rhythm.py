import logging

from process import process
from args_utils import get_argparser


if __name__ == '__main__':

    # todo verbosity
    logging.basicConfig(level=logging.DEBUG)

    log = logging.getLogger(__name__)

    args = get_argparser().parse_args()

    if args.module == "process":
        data_process = process.Process(args)
        data_process.process()
    else:
        raise ValueError("Unknown module: {}".format(args.module))
