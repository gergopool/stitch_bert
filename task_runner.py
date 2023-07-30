"""
This file grabs up a text file with a list of tasks and distributes them to the available GPUs.
It is used to run the experiments in parallel.
"""
import os
import sys
import subprocess
import threading
import argparse
import unicodedata
import re
from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('in_file', type=str, help='Path to the input file')
    parser.add_argument('-g', '--gpus', nargs="+", help='E.g. 0 0 1 1', required=True)
    parser.add_argument('-o',
                        '--out-file',
                        default=None,
                        help='If None generated automatically',
                        required=False)
    parser.add_argument('-l',
                        '--log-dir',
                        default=None,
                        help='If None no log wil be created',
                        required=False)
    return parser.parse_args(args)


def read_tasks(in_file: str) -> set:
    """
    Read tasks from the input file.

    Args:
        in_file: Path to the input file.

    Returns:
        A set of tasks.
    """
    if not os.path.isfile(in_file):
        raise FileNotFoundError(f"{in_file} not found")

    with open(in_file, 'r') as f:
        tasks = set(f.read().splitlines())

    return tasks


def remove_tasks_done(tasks: set, out_file: str) -> set:
    """
    Remove the tasks that are already done.

    Args:
        tasks: A set of tasks.
        out_file: Path to the output file.

    Returns:
        A set of tasks that are not done yet.
    """
    if not os.path.isfile(out_file):
        return tasks

    with open(out_file, 'r') as f:
        tasks_done = set(f.read().splitlines())

    return tasks.difference(tasks_done)


def slugify(value: str, allow_unicode=False) -> str:
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py

    Args:
        value: The string to be converted.
        allow_unicode: If False, the function will convert the string to ASCII.

    Returns:
        The slugified version of the input string.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def process_commands(tasks: set, gpu: int, out_file: str, log_dir: str, pbar: tqdm):
    """
    Process the tasks by running each one as a separate process.

    Args:
        tasks: A set of tasks to be processed.
        gpu: The GPU to be used for the tasks.
        out_file: Path to the output file.
        log_dir: Directory for the log files.
        pbar: Progress bar object.
    """
    while len(tasks):
        try:
            task = tasks.pop()
            orig_task = task
            if log_dir is not None:
                file_name = slugify(task)
                file_name = os.path.join(log_dir, file_name)
                redirect_str = f"> {file_name}.out 2>&1"
            else:
                redirect_str = "> /dev/null 2>&1"
            task = f"CUDA_VISIBLE_DEVICES={gpu} {task} {redirect_str}"
            process = subprocess.Popen(task, shell=True, stdout=subprocess.DEVNULL)
            process.wait()
            if process.returncode == 0:
                with open(out_file, 'a') as f:
                    f.write(f"{orig_task}\n")
            else:
                print(f"Warning: Task ran into error: {task}")
            pbar.update(1)
        except IndexError:
            print("No task left")


def main(args=None):
    """
    The main function that executes the task processing.

    Args:
        args: The command line arguments.
    """
    if args is None:
        args = sys.argv[1:]
    conf = parse_args(args)
    tasks = read_tasks(conf.in_file)
    if conf.out_file is not None:
        out_file = conf.out_file
    else:
        name, _ = os.path.splitext(conf.in_file)
        out_file = f"{name}.done"
    tasks = remove_tasks_done(tasks, out_file)

    if not len(tasks):
        print('Every task is done already.')
        return

    if conf.log_dir is not None:
        os.makedirs(conf.log_dir, exist_ok=True)

    pbar = tqdm(total=len(tasks))
    threads = []
    for gpu_id in conf.gpus:
        t = threading.Thread(target=process_commands,
                             args=(tasks, gpu_id, out_file, conf.log_dir, pbar))
        t.start()
        threads.append(t)


if __name__ == '__main__':
    main()
