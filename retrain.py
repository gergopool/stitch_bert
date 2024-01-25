import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import argparse
import torch

# Import custom modules
from src import Logger, GlobalState
from src.models import build_pretrained_transformer
from src.utils import set_seed, set_memory_limit
from src.metrics import get_metric_for
from src.trainer import train
from train import load_datasets

from src.static import TASKS


def main(args):

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    # Set the random seed for reproducibility
    set_seed(args.seed)
    set_memory_limit(50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the mask
    mask_path = os.path.join(args.mask_dir, f"{args.task}_{args.seed}.pt")
    head_mask = torch.load(mask_path, map_location=device)
    Logger.info(f"Loaded mask from file {mask_path}.")

    # Load data
    train_loader, val_loader = load_datasets(args)

    # Initialize the pre-trained transformer
    model = build_pretrained_transformer(args.task).to(device)
    Logger.info("Pre-trained transformer initialized.")

    # Retrain the model
    Logger.info(f"Retraining starts with the applied mask..")
    metric = get_metric_for(args.task)
    model, _ = train(model,
                     train_loader,
                     val_loader,
                     metric,
                     is_vis=args.task in TASKS['vis'],
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


def parse_args(cli_args=None):
    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument("task",
                        type=str,
                        choices=TASKS['vis'] + TASKS['nlp'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed", type=int, help="The random seed of the run, for reproducibility.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data',
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
    args = parser.parse_args(cli_args)

    return args


if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Execute the main function
    main(args)
