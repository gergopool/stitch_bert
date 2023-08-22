import argparse
import torch
import os
import pickle
from transformers import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train import load_datasets
from src.utils import set_seed
from src.static import Logger, GlobalState, TASKS
from src.compare import cka, jaccard_similarity, functional_similarity, calculate_embeddings
from src.models import load_model
from src.glue_metrics import get_metric_for, Metric
from src.evaluator import evaluate
from src.data import load_data_from_args


def load_mask(mask_dir: str, task: str, seed: int, device: torch.device) -> torch.Tensor:
    mask_path = os.path.join(mask_dir, f"{task}_{seed}.pt")
    head_mask = torch.load(mask_path, map_location=device)
    Logger.info(f"Loaded mask from file {mask_path}.")
    return head_mask


def main(args):

    # Set the random seed for reproducibility
    set_seed(args.run_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataloaders
    train_loader = load_data_from_args(args, n_iters=args.n_iterations, dev=False)
    val_loader = load_data_from_args(args, dev=True)
    Logger.info(f"Training dataset is loaded with {len(train_loader.dataset)} datapoints.")
    Logger.info(f"Validation dataset is loaded with {len(val_loader.dataset)} datapoints.")

    # Load masks, models and metric
    model_info = {
        "model1": load_model(args.retrain_dir, args.task1, args.seed1, device),
        "model2": load_model(args.retrain_dir, args.task2, args.seed2, device),
        "mask1": load_mask(args.mask_dir, args.task1, args.seed1, device),
        "mask2": load_mask(args.mask_dir, args.task2, args.seed2, device)
    }

    # Load datasets & metric
    metric = get_metric_for(args.task2)

    # Get benchmark performance
    is_vis = args.task2 in TASKS['vis']
    model2_performance = evaluate(model_info['model2'], val_loader, metric, is_vis, mask=None)
    Logger.info(f"Model2 metric: {model2_performance:.4f}")

    # Dictionary used to store results for jaccard, cka, and functional similarity
    results = {"jaccard": [], "cka": [], "fs": []}

    # Calculate CKA and functional similarity
    n_layers = model_info['model1'].config.num_hidden_layers
    for layer_i in range(n_layers):

        # Get jaccard similarity
        current_jaccard = jaccard_similarity(model_info['mask1'][:layer_i + 1],
                                             model_info["mask2"][:layer_i + 1])
        results['jaccard'].append(current_jaccard)
        Logger.info(f"Jaccard @ {layer_i+1}/{n_layers} : {current_jaccard:.3f}.")

        # Calculate embeddings first for both CKA and pseudo-inverse init
        embeddings = calculate_embeddings(**model_info,
                                          layer_i=layer_i,
                                          data_loader=val_loader,
                                          n_points=args.n_points,
                                          is_vis=is_vis,
                                          device=device)
        Logger.info(f"{len(embeddings['x1'])} embeddings calculated for layer {layer_i+1}.")

        # CKA
        current_cka = cka(**embeddings)
        results['cka'].append(current_cka)
        Logger.info(f"CKA @ {layer_i+1}/{n_layers}     : {current_cka:.3f}.")

        # Functional similarity
        current_fs = functional_similarity(**model_info,
                                           layer_i=layer_i,
                                           embeddings=embeddings,
                                           train_loader=train_loader,
                                           val_loader=val_loader,
                                           model2_performance=model2_performance,
                                           metric=metric,
                                           is_vis=is_vis,
                                           device=device)
        results['fs'].append(current_fs)
        Logger.info(f"FS @ {layer_i+1}/{n_layers}      : {current_fs:.3f}.")

    # Save results
    if not GlobalState.debug:
        os.makedirs(args.out_dir, exist_ok=True)
        save_path = os.path.join(args.out_dir,
                                 f"{args.task1}_{args.seed1}_{args.task2}_{args.seed2}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        Logger.info(f"Results saved to {save_path}.")
    else:
        Logger.info(f"Results are not saved in debug mode.")


def parse_args():

    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument("task1",
                        type=str,
                        choices=TASKS['vis'] + TASKS['nlp'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed1", type=int, help="The seed of the first run, for reproducibility.")
    parser.add_argument("task2",
                        type=str,
                        choices=TASKS['vis'] + TASKS['nlp'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed2", type=int, help="The seed of the second run, for reproducibility.")
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
                        default=1000,
                        help="Number of training iterations. Default is 200.")
    parser.add_argument("--n_points",
                        type=int,
                        default=1000,
                        help="Number datapoints to take for cka and ps_inv. Default is 10000.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size for training. Default is 32.")
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