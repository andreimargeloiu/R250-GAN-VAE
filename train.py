"""
Usage:
    train.py [options]

Options:
    -h
    --debug                         Enable debug routines. [default: False]

    --log-file=NAME                 Path to the log file
"""
import json
import logging

import git
from docopt import docopt
from dpu_utils.utils import run_and_debug


def initialize_logger(args):
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m-%d %H:%M:%S',
                        filename=args['--log-file'],
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))

    # Attach the console to the root logger
    logging.getLogger('').addHandler(console)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    logging.info(f"\n\n---Started Training from commit {sha}---")
    logging.info(json.dumps(args, ensure_ascii=True, indent=2, sort_keys=True))


def run(args):
    print("I'm inside run")


if __name__ == "__main__":
    args = docopt(__doc__)
    initialize_logger(args)

    run_and_debug(lambda: run(args), args["--debug"])






