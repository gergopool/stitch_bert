import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import argparse
import torch

# Import custom modules
from src.data import load_data_from_args
from src.models import build_pretrained_transformer
from src.utils import set_seed, set_memory_limit
from src.metrics import get_metric_for
from src.trainer import train
from src import Logger, GlobalState, TASKS


def load_datasets(args):
    train_loader = load_data_from_args(args, n_iters=args.n_iterations, dev=False)
    val_loader = load_data_from_args(args, dev=True)
    Logger.info(f"Training dataset is loaded with {len(train_loader.dataset)} datapoints.")
    Logger.info(f"Validation dataset is loaded with {len(val_loader.dataset)} datapoints.")
    return train_loader, val_loader


def main(args):

    set_memory_limit(50)

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Load data
    train_loader, val_loader = load_datasets(args)

    # Initialize the pre-trained transformer
    model = build_pretrained_transformer(args.task)
    Logger.info("Pre-trained transformer initialized.")

    # Train the model
    Logger.info(f"Training starts..")
    metric = get_metric_for(args.task)
    model, _ = train(model, train_loader, val_loader, metric, is_vis=args.task in TASKS['vis'])
    Logger.info(f"Training finished..")

    # Save the trained model
    if not GlobalState.debug:
        save_path = os.path.join(args.out_dir, f"{args.task}_{args.seed}.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        Logger.info(f"Finetuned model saved to {save_path}")
    else:
        Logger.info(f"Finetuned model not saved due to debug mode.")


def parse_args(cli_args=None):

    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument("task",
                        type=str,
                        choices=TASKS['nlp'] + TASKS['vis'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed", type=int, help="The random seed of the run, for reproducibility.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data/',
                        help="Directory where the data is located.")
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
                        help="Maximum number of training iterations. Default is 10000.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size for training. Default is 128.")
    parser.add_argument("--out_dir",
                        type=str,
                        default='results/finetune/',
                        help="Directory to save the output. Default is './results/finetune'.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")

    # Parse the arguments
    args = parser.parse_args(cli_args)

    return args


if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Execute the main function
    main(args)
