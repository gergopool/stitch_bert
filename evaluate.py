import argparse
import torch
import os
from transformers import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train import load_datasets
from src.data import load_data_from_args
from src.static import Logger, GlobalState, TASKS
from src.models import load_model
from src.glue_metrics import get_metric_for, Metric
from src.evaluator import evaluate


def load_mask(mask_dir: str, task: str, seed: int, device: torch.device) -> torch.Tensor:
    mask_path = os.path.join(mask_dir, f"{task}_{seed}.pt")
    head_mask = torch.load(mask_path, map_location=device)
    Logger.info(f"Loaded mask from file {mask_path}.")
    return head_mask


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    test_loader = load_data_from_args(args, dev=True)

    # Load model and mask
    trained_model = load_model(args.train_dir, args.task, args.seed, device)
    retrained_model = load_model(args.retrain_dir, args.task, args.seed, device)
    head_mask = load_mask(args.mask_dir, args.task, args.seed, device)

    # Calculate mask sparstiy
    mask_sparsity = head_mask.sum().item() / head_mask.numel() * 100

    # Load metric
    metric = get_metric_for(args.task)

    # Get benchmark performance
    is_vis = args.task in TASKS['vis']
    full_trained_perf = evaluate(trained_model, test_loader, metric, is_vis, mask=None)
    masked_trained_perf = evaluate(trained_model, test_loader, metric, is_vis, mask=head_mask)
    masked_retrained_perf = evaluate(retrained_model, test_loader, metric, is_vis, mask=head_mask)

    Logger.info(f"Mask sparsity                           : {mask_sparsity:.2f}%")
    Logger.info(f"Original finetuned performance          : {full_trained_perf:.4f}")
    Logger.info(f"Masked, without re-training performance : {masked_trained_perf:.4f}")
    Logger.info(f"Masked, with retraining performance     : {masked_retrained_perf:.4f}")


def parse_args():

    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument("task",
                        type=str,
                        choices=TASKS['vis'] + TASKS['nlp'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed", type=int, help="The seed of the run, for reproducibility.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data',
                        help="Directory where the data is located.")
    parser.add_argument("--model_type",
                        type=str,
                        default='bert-base-uncased',
                        help="Type of the model. Default is 'bert-base-uncased'.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="Maximum sequence length for the tokenized data. Default is 128.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size. Default is 128.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="If True, overwrite the cached data. Default is False.")
    parser.add_argument("--train_dir",
                        type=str,
                        default='results/finetune/',
                        help="Directory which contains the finetuned models. " + \
                             "Default is 'results/finetune/'.")
    parser.add_argument("--retrain_dir",
                        type=str,
                        default='results/retrained/',
                        help="Directory which contains the retrained models. " + \
                             "Default is 'results/retrained/'.")
    parser.add_argument("--mask_dir",
                        type=str,
                        default='results/masks/',
                        help="Directory which contains the masks. Default is 'results/masks'.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    # Execute the main function
    main(args)
