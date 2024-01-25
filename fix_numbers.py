import argparse
import torch
from torch import nn
import os
import pickle
from tqdm import tqdm
import glob
import time
from typing import Tuple, Dict, Union, List
from torch.utils.data import DataLoader

from src.utils import set_seed, set_memory_limit
from src.static import Logger, GlobalState, TASKS
from src.compare import cka, jaccard_similarity, functional_similarity, calculate_embeddings
from src.models import load_model
from src.metrics import get_metric_for, Metric
from src.evaluator import evaluate
from src.data import load_data_from_args

from compare import (
    randomize_mask,
    load_mask,
    load_data_loaders,
    load_models_and_masks,
    calculate_layer_similarities,
    calculate_similarities,
    save_results
)


def main(args):

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    last_folder = os.path.basename(os.path.normpath(args.folder))
    args.folder = os.path.join(args.folder, 'linear')

    # Current time
    current_time = time.time()

    # Time one day ago
    one_day_ago = current_time - 60 * 60 * 24  # 60 seconds * 60 minutes * 24 hours

    for filepath in tqdm(glob.glob(os.path.join(args.folder, "*.pkl"))):

        last_modified_time = os.path.getmtime(filepath)
        if last_modified_time > one_day_ago:
            continue
        print(filepath)

        task1, seed1, task2, seed2 = os.path.basename(filepath).replace(".pkl", "").split("_")
        args.task1 = task1
        args.seed1 = int(seed1)
        args.task2 = task2
        args.seed2 = int(seed2)
        args.randomize_m1 = last_folder == "randomize_model1"
        args.randomize_m2 = last_folder == "randomize_model2"
        args.shuffle_mask1 = last_folder == "shuffle_mask1"
        args.shuffle_mask2 = last_folder == "shuffle_mask2"

        set_seed(args.run_seed)
        set_memory_limit(50)
        _, val_loader = load_data_loaders(args)
        model_info = load_models_and_masks(args, device)
        is_vis = args.task2 in TASKS['vis']
        metric = get_metric_for(args.task2)
        model2_performance = evaluate(model_info['model2'],
                                      val_loader,
                                      metric,
                                      is_vis,
                                      mask=model_info['mask2'])
        wrong_model2_performance = evaluate(model_info['model2'],
                                            val_loader,
                                            metric,
                                            is_vis,
                                            mask=None)
        
        reference_pt = filepath.replace(last_folder, f"save/{last_folder}")
        with open(reference_pt, 'rb') as f:
            reference = pickle.load(f)

        for i in range(len(reference['fs'])):
            reference['fs'][i] *= wrong_model2_performance / model2_performance

        
        with open(filepath, 'wb') as f:
            pickle.dump(reference, f)
        Logger.info(f"Results saved to {filepath}.")


def parse_args(cli_args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("folder", type=str, help='folder to save the results')


    parser.add_argument("--stitch-type",
                        type=str,
                        default='linear',
                        choices=['linear', 'linearbn'],
                        help="How to stitch: with or without batchnorm.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data/',
                        help="Directory where the data is located.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="Maximum sequence length for the tokenized data. Default is 128.")
    parser.add_argument("--n_iterations",
                        type=int,
                        default=2000,
                        help="Number of training iterations. Default is 2000.")
    parser.add_argument("--n_points",
                        type=int,
                        default=2000,
                        help="Number datapoints to take for cka and ps_inv. Default is 2000.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size for training. Default is 128.")
    parser.add_argument("--run_seed",
                        type=str,
                        default=42,
                        help="Seed of the run, for reproducibility. Default is 42.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="If True, overwrite the cached data. Default is False.")
    parser.add_argument("--retrain_dir",
                        type=str,
                        default='results/retrained/',
                        help="Directory which contains the retrained models. " + \
                             "Default is 'results/retrained/'.")
    parser.add_argument("--mask_dir",
                        type=str,
                        default='results/masks/',
                        help="Directory which contains the masks. Default is 'results/masks'.")
    parser.add_argument("--out_dir",
                        type=str,
                        default='results/compare/',
                        help="Directory to save the output. Default is './output'.")
    parser.add_argument("--shuffle_mask1",
                        action='store_true',
                        help="Shuffle active heads in mask1. Default is False.")
    parser.add_argument("--shuffle_mask2",
                        action='store_true',
                        help="Shuffle active heads in mask2. Default is False.")
    parser.add_argument("--randomize_m1",
                        action='store_true',
                        help="Randomize weights of the first model. Default is False.")
    parser.add_argument("--randomize_m2",
                        action='store_true',
                        help="Randomize weights of the second model. Default is False.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")

    # Parse the arguments
    args = parser.parse_args(cli_args)

    return args


if __name__ == "__main__":

    args = parse_args()

    # Execute the main function
    main(args)
