import os
import argparse
import torch
from transformers import AutoTokenizer

# Import custom modules
from src import Logger, GlobalState
from src.models import build_pretrained_transformer
from src.utils import set_seed
from src.glue_metrics import get_metric_for
from src.trainer import train
from train import load_datasets


def main(args):

    # Set the random seed for reproducibility
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False)
    Logger.info("Tokenizer initialized.")

    # Load the mask
    mask_path = os.path.join(args.mask_dir, f"{args.task}_{args.seed}.pt")
    head_mask = torch.load(mask_path, map_location=device)
    Logger.info(f"Loaded mask from file {mask_path}.")

    # Initialize the pre-trained transformer (original BERT initialization)
    model = build_pretrained_transformer(args.model_type, args.task)
    model.to(device)
    Logger.info("Pre-trained transformer initialized.")

    # Create dataloader
    train_dataset, val_dataset = load_datasets(args, tokenizer)

    # Retrain the model
    Logger.info(f"Retraining starts with the applied mask..")
    metric = get_metric_for(args.task)
    model, _ = train(model,
                     train_dataset,
                     val_dataset,
                     metric,
                     is_vis=args.task in TASKS['vis'],
                     n_iters=args.n_iterations,
                     batch_size=args.batch_size,
                     head_mask=head_mask)
    Logger.info(f"Retraining finished..")

    # Save the retrained model
    if not GlobalState.debug:
        save_path = os.path.join(args.out_dir, f"{args.task}_{args.seed}.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        Logger.info(f"Retrained model saved to {save_path}")
    else:
        Logger.info(f"Retrained model not saved due to debug mode.")


def parse_args():
    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument(
        "task",
        type=str,
        choices=['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli'],
        help="Name of the task (dataset).")
    parser.add_argument("seed", type=int, help="The random seed of the run, for reproducibility.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data/glue_data',
                        help="Directory where the data is located.")
    parser.add_argument("--model_type",
                        type=str,
                        default='bert-base-uncased',
                        help="Type of the model. Default is 'bert-base-uncased'.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="If True, overwrite the cached data. Default is False.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="Maximum sequence length for the tokenized data. Default is 128.")
    parser.add_argument("--n_iterations",
                        type=int,
                        default=2000,
                        help="Number of training iterations. Default is 2000.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size for training. Default is 128.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default='results/retrained/',
        help="Directory to save the retrained model output. Default is 'results/retrained/'.")
    parser.add_argument("--mask_dir",
                        type=str,
                        default='results/masks/',
                        help="Directory which contains the masks. Default is 'results/masks'.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    # Execute the main function
    main(args)
