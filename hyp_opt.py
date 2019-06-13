import os
import json
import time
import logging
import argparse
import itertools
import subprocess
from multiprocessing import Pool
from collections import OrderedDict
from logging.handlers import RotatingFileHandler

import numpy as np

# PROC_PREFIX_DAS4 = "srun -p fatq --gres=gpu:1 --time=99:00:00".split()
# PROC_PREFIX_DAS4_CPU = "srun --time=99:00:00".split()
PROC_PREFIX_CPU = []
PROC_FILE = "python rhythm.py train".split()


log = logging.getLogger("hyp_opt")


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


def _is_trained(run_name, model_id):
    # try:
    #     args, monitor = read_monitor_args(run_name, model_id)
    #     # we can say a model is trained if the tracked epochs == number of epochs in the run
    #     return args["epochs"] == len(monitor["epoch_val_loss"])
    # except FileNotFoundError:
    return False


def generate_cmd(hyperparameters):
    hyperparameters = OrderedDict(
        sorted(hyperparameters.items(), key=lambda _: _[0]))
    keys = list(hyperparameters.keys())
    indices = [len(values) for (arg, values) in hyperparameters.items()]
    arg_choices = {arg: len(values)
                   for (arg, values) in hyperparameters.items()}
    choices = []
    for idx_choice in np.ndindex(*indices):
        one_choice = {}
        for arg_idx, (arg, val_idx) in enumerate(zip(keys, idx_choice)):
            one_choice[arg] = hyperparameters[arg][val_idx]
        choices.append(one_choice)

    log.info("Found {} possible choices".format(len(choices)))

    for choice_idx, choice in enumerate(choices, 1):
        log.debug("Constructing command for: {}".format(choice))

        cmd = []
        cmd.extend(PROC_PREFIX)
        cmd.extend(PROC_FILE)

        model_id = []
        run_name = None
        for arg, value in choice.items():
            cmd.append("--{}".format(arg))
            cmd.append(str(value))
            # only include the file name in the model id
            if arg == "prior":
                value = os.path.split(value)[1].split(".")[0]
            # adding run name is redundant
            if arg == "run":
                run_name = value
                continue

            # don't add to model-id if there's only a single choice to be made
            if arg_choices[arg] <= 1:
                continue

            model_id.append("{}{}".format(arg, str(value)))

        model_id = "_".join(model_id)
        cmd.append("--model-id")
        cmd.append(model_id)

        yield cmd, _is_trained(run_name, model_id)


def execute_cmd(cmd, suppress_output=False):
    start_time = time.time()
    log.info("Executing command: {}".format(" ".join(cmd)))
    stdout, stderr = None, None
    if suppress_output:
        stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
    return_code = subprocess.call(cmd, stderr=stderr,
                                  stdout=stdout)
    time_elapsed = time.time() - start_time
    log.info("Finished executing command: {} , Took {:.0f}m {:.0f}s".format(" ".join(cmd),
                                                                            time_elapsed // 60, time_elapsed % 60))
    return return_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "hyp_opt", description="Searches over hyperparameters")
    parser.add_argument(
        "--run-loc", dest="run_location",
        default="cpu", choices={"cpu"},
        help="location to run processes")
    parser.add_argument("--n-jobs", dest="n_jobs", type=int,
                        help="number of jobs to run in parallel", required=True)
    parser.add_argument(
        "hyp_file", help="location of the JSON file specifying the hyperparameters")

    args = parser.parse_args()

    configure_logging("hyp_opt", True)

    PROC_PREFIX = {
        "cpu": PROC_PREFIX_CPU
    }[args.run_location]

    log.info("PROC_PREFIX: {}".format(PROC_PREFIX))

    with open(args.hyp_file, "r") as reader:
        hyperparameters = json.load(reader)

    cmds_to_execute = []
    for cmd, is_trained in generate_cmd(hyperparameters):
        if is_trained:
            log.info("Model is already trained, skipping")
            continue

        # we don't want to see a wall of text, so supress the output
        cmds_to_execute.append((cmd, True))

    log.info("Total of {} jobs to execute. Starting a Pool of {} processes".format(
        len(cmds_to_execute), args.n_jobs))

    start_time = time.time()

    pool = Pool(processes=args.n_jobs)

    return_codes = pool.starmap(execute_cmd, cmds_to_execute)

    for return_code, cmd in zip(return_codes, cmds_to_execute):
        if return_code == 0:
            continue
        log.info("Possible failure because of a non zero return code for command: {}".format(
            " ".join(cmd[0])))

    time_elapsed = time.time() - start_time
    log.info("Took {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
