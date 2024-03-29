import argparse
import time
import os
from pathlib import Path

from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='config file path')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--log_dir', default=None, type=str, help='log file save path')
    parser.add_argument('--tag', default='base', type=str, help='experiment tag')
    parser.add_argument('--seed', default=8, type=int, help='random seed')

    return parser.parse_args()


def main(kargs):
    import logging
    import numpy as np
    import random
    # import torch
    import mindspore as ms
    from runners import MainRunner

    seed = kargs.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    ms.set_seed(seed + 2)

    if kargs.log_dir:
        Path(kargs.log_dir).mkdir(parents=True, exist_ok=True)
        log_filename = time.strftime("%Y-%m-%d_%H-%M-%S.log", time.localtime())
        log_filename = os.path.join(kargs.log_dir, "{}_{}".format(kargs.tag, log_filename))
    else:
        log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    args = load_json(kargs.config_path)
    args['train']['model_saved_path'] = os.path.join(args['train']['model_saved_path'], kargs.tag)

    logging.info(str(args))

    runner = MainRunner(args)

    if kargs.resume:
        runner._load_model(kargs.resume)

    if kargs.eval:
        runner._eval_model()
        return
    runner.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
