import os
import argparse
import torch
from copy import deepcopy
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from torch.utils.data import TensorDataset

# Import custom modules
from src.data import load_data_from_args
from src.models import build_pretrained_transformer
from src.utils import set_seed
from src.task_metrics import get_metric_for
from src.evaluator import evaluate
from src import Logger, GlobalState


def train(model, train_dataset, val_dataset, metric, n_iters, batch_size, task, tokenizer, head_mask=None):
    """
    Train a given model on a given train dataset for a specified number of iterations and batch size.
    Return with the best model based on the validation metric.

    Args:
        model: The model to be trained.
        dataset: The dataset to be used for training.
        metric: The metric to be used for evaluation.
        n_iters: The number of iterations for training.
        batch_size: The size of the batch for training.

    Returns:
        The trained model.
    """

    # Set number of iterations to 3 if in debug mode
    if GlobalState.debug:
        n_iters = 3
        batch_size = 4

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model to trainable
    was_trainable = model.training
    model.train().to(device)

    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer) if task == 'mlm' else None
    # Prepare data loaders

    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            persistent_workers=True,
                            num_workers=2,
                            collate_fn = collate_fn)
    test_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            persistent_workers=True,
                            num_workers=2,
                            collate_fn = collate_fn)

    # Initialize training related objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(n_iters * 0.1), n_iters)
    scaler = torch.cuda.amp.GradScaler()

    # Train the model
    finished = False
    best_metric = None
    best_model = model
    iters_done = 0
    no_improvement = 0

    

    while not finished:
        for batch in train_loader:

            if isinstance(train_loader.dataset, TensorDataset):
                # Prepare input
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "token_type_ids": batch[2],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
            else:
                inputs = {key: value.to(device) for key, value in batch.items()}

            if head_mask is not None:
                inputs["head_mask"] = head_mask

            # Calculate loss
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**inputs)
                loss = outputs[0]

            # Perform backward pass and update weights
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Save model if this is the best
            iters_done += 1
            if iters_done % min(200, n_iters) == 0:
                metric_value = evaluate(model, test_loader, metric)
                if best_metric is None or best_metric < metric_value:
                    Logger.info(
                        f"{iters_done:4d}/{n_iters}: Improved to {metric_value:.2f}! Saved.")
                    best_metric = metric_value
                    best_model = deepcopy(model)
                    no_improvement = 0
                else:
                    Logger.info(f"{iters_done:4d}/{n_iters}: ({metric_value:.2f})")
                    no_improvement += 1

            # Check if training should be finished
            finished = iters_done >= n_iters or no_improvement >= 10
            if finished:
                break

    Logger.info(f"Best {metric.name}: {best_metric:.2f}")

    # Restore the original training state of the model
    model = best_model
    model.train(was_trainable)

    return model


def load_datasets(args, tokenizer):
    train_dataset = load_data_from_args(args, tokenizer, dev=False)
    val_dataset = load_data_from_args(args, tokenizer, dev=True)
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
    model = train(model,
                  train_dataset,
                  val_dataset,
                  metric,
                  n_iters=args.n_iterations,
                  batch_size=args.batch_size,
                  task = args.task,
                  tokenizer = tokenizer)
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
        choices=['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli','mlm'],
        help="Name of the task (dataset).")
    parser.add_argument("--data_dir",
                        type=str,
                        default='/data/shared/data/glue_data',
                        help="Directory where the data is located.")
    parser.add_argument("--model_type",
                        type=str,
                        default='bert-base-uncased',
                        help="Type of the model. Default is 'bert-base-uncased'. Use xlm-roberta-base for MLM task")
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
