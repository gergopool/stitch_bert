import argparse
import torch
from torch import nn
import os
import pickle
from typing import Tuple, Dict, Union, List
from torch.utils.data import DataLoader

from src.utils import set_seed
from src.static import Logger, GlobalState, TASKS
from src.compare import cka, jaccard_similarity, functional_similarity, calculate_embeddings
from src.compare.jaccard_for_magnitude_pruning import jaccard_for_magnitude_pruning
from src.models import load_model
from src.metrics import get_metric_for, Metric
from src.evaluator import evaluate
from src.data import load_data_from_args
from src.magnitude_pruning import MAGNITUDE_PRUNING_GLOBAL, MAGNITUDE_PRUNING_COMPONENT_WISE, RANDOM_PRUNING

def load_mask(mask_dir: str, task: str, seed: int, device: torch.device, pruning_method: str) -> torch.Tensor:
    postfix_for_pruning =  f'_{pruning_method}' if pruning_method in \
                            [MAGNITUDE_PRUNING_GLOBAL, MAGNITUDE_PRUNING_COMPONENT_WISE, RANDOM_PRUNING] else ''
    mask_path = os.path.join(mask_dir, f"{task}_{seed}{postfix_for_pruning}.pt")
    head_mask = torch.load(mask_path, map_location=device)
    Logger.info(f"Loaded mask from file {mask_path}.")
    return head_mask


def load_data_loaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Load training and validation data loaders.

    Args:
        args: Command-line arguments.

    Returns:
        Tuple containing training and validation data loaders.
    """
    train_loader = load_data_from_args(args, n_iters=args.n_iterations, dev=False)
    val_loader = load_data_from_args(args, dev=True)
    Logger.info(f"Training dataset is loaded with {len(train_loader.dataset)} datapoints.")
    Logger.info(f"Validation dataset is loaded with {len(val_loader.dataset)} datapoints.")
    return train_loader, val_loader


def load_models_and_masks(args, device: torch.device, pruning_method: str) -> Dict[str, Union[nn.Module, torch.Tensor]]:
    """
    Load models and masks based on the provided arguments.

    Args:
        args: Command-line arguments.
        device: PyTorch device.

    Returns:
        Dictionary containing the loaded models and masks.
    """
    model_info = {
        "model1": load_model(args.retrain_dir, args.task1, args.seed1, device, pruning_method),
        "model2": load_model(args.retrain_dir, args.task2, args.seed2, device, pruning_method),
        "mask1": load_mask(args.mask_dir, args.task1, args.seed1, device, pruning_method),
        "mask2": load_mask(args.mask_dir, args.task2, args.seed2, device, pruning_method)
    }
    return model_info


def calculate_layer_similarities(model_info: Dict[str, Union[nn.Module, torch.Tensor]],
                                 layer_i: int,
                                 train_loader: DataLoader,
                                 val_loader: DataLoader,
                                 model2_performance: float,
                                 metric: Metric,
                                 n_points,
                                 is_vis: bool,
                                 device: torch.device,
                                 pruning_method: str) -> Tuple[float, float, float]:
    """
    Calculate similarities for a specific layer.

    Args:
        args: Command-line arguments.
        model_info: Information about the models and masks.
        layer_i: Index of the layer to calculate the similarities for.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        model2_performance: Performance of model2 on the validation set.
        metric: Metric for evaluation.
        n_points: Number of points to use for CKA and PS_inv.
        is_vis: Flag indicating whether it's a visual task.
        device: PyTorch device.

    Returns:
        Jaccard, CKA and functional similarity for the given layer, respectively.
    """
    # Get jaccard similarity
    if pruning_method  in [MAGNITUDE_PRUNING_GLOBAL, MAGNITUDE_PRUNING_COMPONENT_WISE, RANDOM_PRUNING]:
        current_jaccard = jaccard_for_magnitude_pruning(model_info['mask1'], model_info["mask2"], layer_i + 1)
        # they are not head mask so they can be removed 
        model_info['mask1'] = None
        model_info["mask2"] = None
    elif pruning_method == 'structured': 
        current_jaccard = jaccard_similarity(model_info['mask1'][:layer_i + 1],
                                         model_info["mask2"][:layer_i + 1])
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}. Currently, we support magnitude_uniform, magnitude_all and structured.")
    Logger.info(f"Jaccard @ {layer_i+1} : {current_jaccard:.3f}.")

    # Calculate embeddings
    embeddings = calculate_embeddings(**model_info,
                                      layer_i=layer_i,
                                      data_loader=val_loader,
                                      n_points=n_points,
                                      is_vis=is_vis,
                                      device=device)
    Logger.info(f"{len(embeddings['x1'])} embeddings calculated for layer {layer_i+1}.")

    # CKA
    current_cka = cka(**embeddings)
    Logger.info(f"CKA @ {layer_i+1} : {current_cka:.3f}.")

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
    Logger.info(f"FS @ {layer_i+1} : {current_fs:.3f}.")

    return current_jaccard, current_cka, current_fs


def calculate_similarities(args,
                           model_info: Dict,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           device: torch.device,
                           pruning_method: str) -> Dict[str, List[float]]:
    """
    Calculate similarities for all layers.

    Args:
        args: Command-line arguments.
        model_info: Information about the models and masks.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: PyTorch device.

    Returns:
        Dictionary containing the calculated similarities.
    """
    metric = get_metric_for(args.task2)
    is_vis = args.task2 in TASKS['vis']
    model2_performance = evaluate(model_info['model2'], val_loader, metric, is_vis, mask=None)
    Logger.info(f"Model2 metric: {model2_performance:.4f}")

    results = {"jaccard": [], "cka": [], "fs": []}
    n_layers = model_info['model1'].config.num_hidden_layers

    for layer_i in range(n_layers):
        current_jaccard, current_cka, current_fs = calculate_layer_similarities(
            model_info=model_info,
            layer_i=layer_i,
            train_loader=train_loader,
            val_loader=val_loader,
            model2_performance=model2_performance,
            metric=metric,
            n_points=args.n_points,
            is_vis=is_vis,
            device=device,
            pruning_method=pruning_method)
        results['jaccard'].append(current_jaccard)
        results['cka'].append(current_cka)
        results['fs'].append(current_fs)

    return results


def save_results(args, results: Dict[str, List[float]]):
    """
    Save the calculated results.

    Args:
        args: Command-line arguments.
        results: Dictionary containing the calculated similarities.
    """
    if not GlobalState.debug:
        os.makedirs(args.out_dir, exist_ok=True)
        save_path = os.path.join(args.out_dir,
                                 f"{args.task1}_{args.seed1}_{args.task2}_{args.seed2}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        Logger.info(f"Results saved to {save_path}.")
    else:
        Logger.info(f"Results are not saved in debug mode.")


def main(args):

    # Initialize logging and debug mode
    GlobalState.debug = args.debug
    Logger.initialise(args.debug)

    set_seed(args.run_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = load_data_loaders(args)
    model_info = load_models_and_masks(args, device, args.pruning_method)
    results = calculate_similarities(args, model_info, train_loader, val_loader, device, args.pruning_method)
    save_results(args, results)


def parse_args(cli_args=None):

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
                             "Default is 'results/retrained/'. \
                              For iterative magnitude pruning use the finetune directory 'results/finetune/' ")
    parser.add_argument("--mask_dir",
                        type=str,
                        default='results/masks/',
                        help="Directory which contains the masks. Default is 'results/masks'.")
    parser.add_argument("--out_dir",
                        type=str,
                        default='results/compare/',
                        help="Directory to save the output. Default is './output'.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")
    parser.add_argument("--pruning_method ",
                        type=str,
                        choices=['structured', MAGNITUDE_PRUNING_GLOBAL, MAGNITUDE_PRUNING_COMPONENT_WISE, RANDOM_PRUNING],
                        )
    # Parse the arguments
    args = parser.parse_args()
    args.task1 = 'cola'
    args.task2 = 'cola'
    args.seed1 = 0
    args.seed2 = 0
    args.debug = True
    args.pruning_method = MAGNITUDE_PRUNING_GLOBAL
    args.retrain_dir = 'results/finetune/'

    return args


if __name__ == "__main__":

    args = parse_args()

    # Execute the main function
    main(args)
