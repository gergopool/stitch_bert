import os
import argparse
import torch

# Import custom modules
from src.data import load_data_from_args
from src.models import build_pretrained_transformer
from src.utils import set_seed
from src.metrics import get_metric_for
from src.trainer import train
from src import Logger, GlobalState, TASKS
from src.magnitude_pruning import rewind, MAGNITUDE_PRUNING_COMPONENT_WISE, MAGNITUDE_PRUNING_GLOBAL, RANDOM_PRUNING
from src.trainer import iterative_magnitude_pruning
import pandas as pd
from train import load_datasets

def finetuned_result_from_csv(args):
    eval = pd.read_csv(args.evaluation_csv_dir)
    pd_task = eval[eval['task']==args.task]
    pd_seed = pd_task[pd_task['seed']==args.seed]
    finetuned_best_metric = float(pd_seed['orig'].iloc[0])
    return finetuned_best_metric

def main(args):

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
    
    # see https://github.com/VITA-Group/BERT-Tickets/blob/master/LT_glue.py#L788
    origin_model_dict  = rewind(model.state_dict()) # create new model to avoid the problem

    # get the finetuned_result_from_csv(args)
    finetuned_best_metric = finetuned_result_from_csv(args)

    magnitude_pruning_type = f'_{args.pruning_method}'

    model, mask_dict = iterative_magnitude_pruning(model,
        train_loader,
        val_loader,
        metric,
        is_vis = args.task in TASKS['vis'],
        orig = origin_model_dict,
        pruning_method = args.pruning_method,
        prune_every=args.prune_every,
        finetuned_best_metric = finetuned_best_metric)
    
    save_path = os.path.join('results','masks', f"{args.task}_{args.seed}{magnitude_pruning_type}.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(mask_dict, save_path)
    
    # Save the trained model
    if not GlobalState.debug:
        save_path = os.path.join(args.out_dir, f"{args.task}_{args.seed}{magnitude_pruning_type}.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        Logger.info(f"Model saved to {save_path}")
    else:
        Logger.info(f"Model not saved due to debug mode.")


def parse_args(cli_args=None):

    parser = argparse.ArgumentParser()

    # Define the argparse arguments
    parser.add_argument("--task",
                        type=str,
                        choices=TASKS['nlp'] + TASKS['vis'],
                        help="Name of the task (dataset).")
    parser.add_argument("--seed", type=int, help="The random seed of the run, for reproducibility.")
    parser.add_argument("--data_dir",
                        type=str,
                        default='./data/shared/data/',
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
                        help="Directory to save the output. Default is './results/finetune'. \
                              For iterative magnitude pruning set it to 'results/iterative_magnitude/'")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode. Default is False.")
    parser.add_argument("--pruning_method ",
                        type=str,
                        choices=[MAGNITUDE_PRUNING_GLOBAL, MAGNITUDE_PRUNING_COMPONENT_WISE, RANDOM_PRUNING],
                        )
    parser.add_argument("--prune_every",
                        type=int,
                        default=200,
                        help="How many steps does it take between two pruning steps")
    parser.add_argument("--evaluation_csv_dir",
                        type=str,
                        default='./evaluation.csv',
                        help="Directory where the finetuned results are written")
    # Parse the arguments
    args = parser.parse_args(cli_args)

    return args


if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Execute the main function
    main(args)