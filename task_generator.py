import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

from src import TASKS

CONFIGS = {
    "train": {
        "task": TASKS['nlp'] + TASKS['vis'], "seed": [i for i in range(5)]
    },
    "mask": {
        "task": TASKS['nlp'] + TASKS['vis'], "seed": [i for i in range(5)]
    },
    "retrain": {
        "task": TASKS['nlp'] + TASKS['vis'], "seed": [i for i in range(5)]
    }
}


def train(task, seed):
    return f"python train.py {task} {seed}"


def mask(task, seed):
    return f"python mask.py {task} {seed}"


def retrain(task, seed):
    return f"python retrain.py {task} {seed}"


def main(args):

    for config in args.configs:

        options = CONFIGS[config]
        keys, values = zip(*options.items())
        tasks = [dict(zip(keys, v)) for v in itertools.product(*values)]
        tasks = [globals()[config](**inputs) for inputs in tasks]
        tasks = [t for t in tasks if t is not None]

        os.makedirs(args.out_dir, exist_ok=True)
        tasks_file = os.path.join(args.out_dir, f"{config}.tasks")
        with open(tasks_file, 'w') as f:
            for task in tasks:
                f.write(f"{task}\n")
        print(f"{config} - {len(tasks)} tasks generated to {tasks_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('--configs', nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument('--out-dir', default='tasks/')
    args = parser.parse_args()
    main(args)