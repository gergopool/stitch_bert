import argparse
import torch
import os
import pickle
from transformers import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train import load_datasets
from src.utils import set_seed
from src.static import Logger, GlobalState
from src.compare import cka, jaccard_similarity, functional_similarity, calculate_embeddings
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


def get_model_performance(val_dataset: Dataset, model_info: dict, metric: Metric) -> float:
    data_loader = DataLoader(val_dataset,
                             batch_size=32,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)
    return evaluate(model_info['model2'], data_loader, metric, model_info['mask2'])


def main(args):

    # Set the random seed for reproducibility
    set_seed(args.run_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False)
    Logger.info("Tokenizer initialized.")

    # Load masks, models and metric
    model_info = {
        "model1": load_model(args.retrain_dir, args.model_type, args.task1, args.seed1, device),
        "model2": load_model(args.retrain_dir, args.model_type, args.task2, args.seed2, device),
        "mask1": load_mask(args.mask_dir, args.task1, args.seed1, device),
        "mask2": load_mask(args.mask_dir, args.task2, args.seed2, device)
    }

    # Load datasets & metric
    train_dataset, val_dataset = load_datasets(args, tokenizer)
    metric = get_metric_for(args.task2)

    # Get benchmark performance
    model2_performance = get_model_performance(val_dataset, model_info, metric)
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
                                          dataset=train_dataset,
                                          n_points=args.n_points,
                                          batch_size=args.batch_size,
                                          device=device)
        # CKA
        current_cka = cka(**embeddings)
        results['cka'].append(current_cka)
        Logger.info(f"CKA @ {layer_i+1}/{n_layers}     : {current_cka:.3f}.")

        # Functional similarity
        current_fs = functional_similarity(**model_info,
                                           layer_i=layer_i,
                                           embeddings=embeddings,
                                           train_dataset=train_dataset,
                                           val_dataset=val_dataset,
                                           model2_performance=model2_performance,
                                           metric=metric,
                                           n_iters=args.n_iterations,
                                           batch_size=args.batch_size,
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
                        choices=['cola', 'mnli', 'mrpc', 'qnli', \
                                 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed1", type=int, help="The seed of the first run, for reproducibility.")
    parser.add_argument("task2",
                        type=str,
                        choices=['cola', 'mnli', 'mrpc', 'qnli', \
                                 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli'],
                        help="Name of the task (dataset).")
    parser.add_argument("seed2", type=int, help="The seed of the second run, for reproducibility.")
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
    parser.add_argument("--n_iterations",
                        type=int,
                        default=1000,
                        help="Number of training iterations. Default is 200.")
    parser.add_argument("--n_points",
                        type=int,
                        default=10000,
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