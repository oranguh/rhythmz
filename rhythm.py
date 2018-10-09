import logging as log

from process import process
from args_utils import get_argparser

if __name__ == '__main__':

    log.basicConfig(level=log.DEBUG)

    args = get_argparser().parse_args()

    if args.module == "process":
        data_process = process.Process(args)
        data_process.process()
    else:
        raise ValueError("Unknown module: {}".format(args.module))
