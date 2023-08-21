import os
import argparse
import torch
from transformers import AutoTokenizer

# Import custom modules
from src.data import load_glue_data_from_args
from src.models import build_pretrained_transformer
from src.utils import set_seed
from src.glue_metrics import get_metric_for
from src.trainer import train
from src import Logger, GlobalState


def load_datasets(args, tokenizer):
    train_dataset = load_glue_data_from_args(args, tokenizer, dev=False)
    val_dataset = load_glue_data_from_args(args, tokenizer, dev=True)
    Logger.info(f"Training dataset is loaded with {len(train_dataset)} datapoints.")
    Logger.info(f"Validation dataset is loaded with {len(val_dataset)} datapoints.")
    return train_dataset, val_dataset


def main(args):

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False)
    Logger.info("Tokenizer initialized.")

    # Initialize the pre-trained transformer
    model = build_pretrained_transformer(args.model_type, args.task)
    Logger.info("Pre-trained transformer initialized.")

    # Load the dataset
    train_dataset, val_dataset = load_datasets(args, tokenizer)

    # Train the model
    Logger.info(f"Training starts..")
    metric = get_metric_for(args.task)
    model, _ = train(model,
                     train_dataset,
                     val_dataset,
                     metric,
                     n_iters=args.n_iterations,
                     batch_size=args.batch_size)
    Logger.info(f"Training finished..")

    # Save the trained model
    if not GlobalState.debug:
        save_path = os.path.join(args.out_dir, f"{args.task}_{args.seed}.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        Logger.info(f"Finetuned model saved to {save_path}")
    else:
        Logger.info(f"Finetuned model not saved due to debug mode.")


def parse_args():

    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument(
        "task",
        type=str,
        choices=['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli'],
        help="Name of the task (dataset).")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data/glue_data',
                        help="Directory where the data is located.")
    parser.add_argument("--model_type",
                        type=str,
                        default='bert-base-uncased',
                        help="Type of the model. Default is 'bert-base-uncased'.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="The random seed of the run, for reproducibility. Default is 0.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="If True, overwrite the cached data. Default is False.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="Maximum sequence length for the tokenized data. Default is 128.")
    parser.add_argument("--n_iterations",
                        type=int,
                        default=10000,
                        help="Number of training iterations. Default is 1000.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size for training. Default is 32.")
    parser.add_argument("--out_dir",
                        type=str,
                        default='results/finetune/',
                        help="Directory to save the output. Default is './output'.")
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
