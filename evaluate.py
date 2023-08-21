import argparse
import torch
import os
from transformers import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train import load_datasets
from src.utils import set_seed
from src.static import Logger, GlobalState
from src.models import build_pretrained_transformer
from src.glue_metrics import get_metric_for, Metric
from src.evaluator import evaluate


def load_model(model_root: str, model_type: str, task: str, seed: int,
               device: torch.device) -> nn.Module:
    model_path = os.path.join(model_root, f"{task}_{seed}.pt")
    model = build_pretrained_transformer(model_type, task)
    model.load_state_dict(torch.load(model_path, map_location=device))
    Logger.info(f"Model loaded from file {model_path}.")
    model.eval()
    return model


def load_mask(mask_dir: str, task: str, seed: int, device: torch.device) -> torch.Tensor:
    mask_path = os.path.join(mask_dir, f"{task}_{seed}.pt")
    head_mask = torch.load(mask_path, map_location=device)
    Logger.info(f"Loaded mask from file {mask_path}.")
    return head_mask


def get_model_performance(val_dataset: Dataset,
                          model: nn.Module,
                          head_mask: torch.Tensor,
                          metric: Metric) -> float:
    data_loader = DataLoader(val_dataset,
                             batch_size=32,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)
    return evaluate(model, data_loader, metric, head_mask)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False)
    Logger.info("Tokenizer initialized.")

    # Load model and mask
    model = load_model(args.retrain_dir, args.model_type, args.task, args.seed, device)
    head_mask = load_mask(args.mask_dir, args.task, args.seed, device)

    # Load dataset & metric
    _, val_dataset = load_datasets(args, tokenizer)
    metric = get_metric_for(args.task)

    # Get benchmark performance
    performance = get_model_performance(val_dataset, model, head_mask, metric)
    Logger.info(f"Model metric: {performance:.4f}")


def parse_args():

    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument(
        "task",
        type=str,
        choices=['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli'],
        help="Name of the task (dataset).")
    parser.add_argument("seed", type=int, help="The seed of the run, for reproducibility.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data/glue_data',
                        help="Directory where the data is located.")
    parser.add_argument("--model_type",
                        type=str,
                        default='bert-base-uncased',
                        help="Type of the model. Default is 'bert-base-uncased'.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="Maximum sequence length for the tokenized data. Default is 128.")
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
