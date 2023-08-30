import os
import argparse
import torch
from torch.utils.data import DataLoader

# Import custom modules
from src import Logger, GlobalState
from src.data import load_data_from_args
from src.models import load_model
from src.metrics import get_metric_for
from src.mask_utils import mask_heads, magnitude_masks
from src.static import TASKS


def main(args):

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    # Load data
    test_loader = load_data_from_args(args, dev=True)

    # Initialize the pre-trained transformer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.finetune_dir, args.task, args.seed, device)

    # Get the metric
    metric = get_metric_for(args.task)

    # Get mask
    is_vis = args.task in TASKS['vis']
    
    postfix_for_pruning = ''
    if args.pruning_method=='structured':
        mask = mask_heads(model, test_loader, metric, args.stop_threshold, args.drop_ratio, is_vis)
    elif args.pruning_method=='magnitude_uniform' or args.pruning_method=='magnitude_all':
        mask = magnitude_masks(model, test_loader, metric, args.pruning_method, args.stop_threshold, args.drop_ratio, is_vis)
        postfix_for_pruning = f'_{args.pruning_method}'

    Logger.info(f"Mask generation is done.")

    # Save mask
    if not GlobalState.debug:
        os.makedirs(args.out_dir, exist_ok=True)
        save_path = os.path.join(args.out_dir, f"{args.task}_{args.seed}{postfix_for_pruning}.pt")
        torch.save(mask, save_path)
        Logger.info(f"Mask saved to {save_path}.")
    else:
        Logger.info(f"Mask is not saved in debug mode.")


def parse_args(cli_args=None):

    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument("task",
                        type=str,
                        choices=TASKS['vis'] + TASKS['nlp'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed",
                        type=int,
                        help="Seed of the model. Note: masking is determinsitic, " + \
                             "it has no effect on the results.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data/',
                        help="Directory where the data is located.")
    parser.add_argument("--stop_threshold",
                        type=float,
                        default=0.9,
                        help="Threshold of permitted performance drop. Default is 0.9.")
    parser.add_argument("--drop_ratio",
                        type=float,
                        default=0.1,
                        help="Ratio of number of parameters to be dropped per step. " + \
                             "Default is 0.1.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="If True, overwrite the cached data. Default is False.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="Maximum sequence length for the tokenized data. Default is 128.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size for training. Default is 32.")
    parser.add_argument("--out_dir",
                        type=str,
                        default='results/masks/',
                        help="Directory to save the output. Default is 'results/masks/'.")
    parser.add_argument("--finetune_dir",
                        type=str,
                        default='results/finetune/',
                        help="Directory which contains the finetuned models. " + \
                             "Default is 'results/finetune'")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")

    # Parse the arguments
    args = parser.parse_args(cli_args)

    return args


if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Execute the main function
    main(args)
