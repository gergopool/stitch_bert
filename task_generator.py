"""
This script generates tasks for the experiments.
The script generates a bash prompt for each task that needs to be run,
without the specification of which GPU to use. Then, another script
processes the bash prompts and distributes and runs the tasks on the GPUs.
"""

import argparse
import itertools
import os

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
    },
    "compare_in_tasks": {
        "task1": TASKS['nlp'] + TASKS['vis'],
        "seed1": [i for i in range(5)],
        "task2": TASKS['nlp'] + TASKS['vis'],
        "seed2": [i for i in range(5)],
    },
    "compare_across_tasks": {
        "task1": TASKS['nlp'] + TASKS['vis'],
        "seed1": [i for i in range(5)],
        "task2": TASKS['nlp'] + TASKS['vis'],
        "seed2": [i for i in range(5)],
    },
    "evaluate": {
        "task": TASKS['nlp'] + TASKS['vis'],
        "seed": [i for i in range(5)],
    }
}


def train(task: str, seed: int) -> str:
    return f"python train.py {task} {seed}"


def mask(task: str, seed) -> str:
    return f"python mask.py {task} {seed}"


def retrain(task: str, seed) -> str:
    return f"python retrain.py {task} {seed}"


def evaluate(task: str, seed) -> str:
    return f"python evaluate.py {task} {seed}"


def compare_in_tasks(task1: str, seed1: int, task2: str, seed2: int) -> str:
    stop_criteria1 = not (task1 == task2)
    stop_criteria2 = ((seed1 + 1) % 5) != (seed2 % 5)
    if stop_criteria1 or stop_criteria2:
        return None
    return [
        f"python compare.py {task1} {seed1} {task2} {seed2} --stitch-type linear",
        f"python compare.py {task1} {seed1} {task2} {seed2} --stitch-type linearbn"
    ]


def compare_across_tasks(task1: str, seed1: int, task2: str, seed2: int) -> str:
    stop_criteria1 = not ((task1 in TASKS['vis']) == (task2 in TASKS['vis']))
    stop_criteria2 = task1 == task2
    stop_criteria3 = (seed1 != 0) or (seed2 != 0)
    if stop_criteria1 or stop_criteria2 or stop_criteria3:
        return None
    return [
        f"python compare.py {task1} {seed1} {task2} {seed2} --stitch-type linear",
        f"python compare.py {task1} {seed1} {task2} {seed2} --stitch-type linearbn"
    ]


def flatten(lst):
    def flat_gen(l):
        for x in l:
            if isinstance(x, list):
                yield from flat_gen(x)
            else:
                yield x
    return list(flat_gen(lst))


def main(args):

    for config in args.configs:

        options = CONFIGS[config]
        keys, values = zip(*options.items())
        tasks = [dict(zip(keys, v)) for v in itertools.product(*values)]
        tasks = [globals()[config](**inputs) for inputs in tasks]
        tasks = flatten([t for t in tasks if t is not None])

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