import argparse
import torch
import os
import pickle

from src.data import load_data_from_args
from src.static import Logger, GlobalState, TASKS
from src.models import load_model
from src.metrics import get_metric_for
from src.evaluator import evaluate


def load_mask(mask_dir: str, task: str, seed: int, device: torch.device) -> torch.Tensor:
    mask_path = os.path.join(mask_dir, f"{task}_{seed}.pt")
    head_mask = torch.load(mask_path, map_location=device)
    Logger.info(f"Loaded mask from file {mask_path}.")
    return head_mask


def main(args):

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    test_loader = load_data_from_args(args, dev=True)

    # Load model and mask
    trained_model = load_model(args.train_dir, args.task, args.seed, device)
    retrained_model = load_model(args.retrain_dir, args.task, args.seed, device)
    head_mask = load_mask(args.mask_dir, args.task, args.seed, device)

    # Load metric
    metric = get_metric_for(args.task)

    # Get benchmark performance
    is_vis = args.task in TASKS['vis']
    results = {}
    results['mask_sparsity'] = head_mask.sum().item() / head_mask.numel()
    results['orig'] = evaluate(trained_model, test_loader, metric, is_vis, mask=None)
    results['masked'] = evaluate(trained_model, test_loader, metric, is_vis, mask=head_mask)
    results['retrained'] = evaluate(retrained_model, test_loader, metric, is_vis, mask=head_mask)

    Logger.info(f"Mask sparsity                           : {results['mask_sparsity']*100:.2f}%")
    Logger.info(f"Original finetuned performance          : {results['orig']:.4f}")
    Logger.info(f"Masked, without re-training performance : {results['masked']:.4f}")
    Logger.info(f"Masked, with retraining performance     : {results['retrained']:.4f}")

    if not GlobalState.debug:
        save_path = os.path.join(args.out_dir, f"{args.task}_{args.seed}.pkl")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        Logger.info(f"Results saved to {save_path}")
    else:
        Logger.info(f"Results not saved due to debug mode.")


def parse_args(cli_args=None):

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
    parser.add_argument("--out_dir",
                        type=str,
                        default='results/evaluate/',
                        help="Directory to save the output. Default is './results/evaluate'.")
    parser.add_argument("--mask_dir",
                        type=str,
                        default='results/masks/',
                        help="Directory which contains the masks. Default is 'results/masks'.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")

    # Parse the arguments
    args = parser.parse_args(cli_args)

    return args


if __name__ == "__main__":

    args = parse_args()

    # Execute the main function
    main(args)
