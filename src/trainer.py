import torch
from torch import nn
from copy import deepcopy
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

from src.evaluator import evaluate
from .static import GlobalState, Logger
from .glue_metrics import Metric


def train(model: nn.Module,
          train_dataset: Dataset,
          val_dataset: Dataset,
          metric: Metric,
          n_iters: int,
          batch_size: int,
          head_mask: torch.Tensor = None,
          force_eval_mode: bool = False,
          verbose: bool = True) -> nn.Module:
    """
    Train a given model on a given train dataset for a specified number of iterations and batch size.
    Return with the best model based on the validation metric.

    Args:
        model: The model to be trained.
        dataset: The dataset to be used for training.
        metric: The metric to be used for evaluation.
        n_iters: The number of iterations for training.
        batch_size: The size of the batch for training and evaluation.
        head_mask: The mask applied on heads.
        force_eval_mode: If True, the model will be set to eval mode during training.
        verbose: If True, the training progress will be printed.
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

    if not force_eval_mode:
        model.train()
    model.to(device)

    # Prepare data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              persistent_workers=True,
                              num_workers=2)
    test_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             num_workers=2)

    # Initialize training related objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(n_iters * 0.1), n_iters)

    if not GlobalState.debug:
        scaler = torch.cuda.amp.GradScaler()

    # Train the model
    finished = False
    best_metric = None
    best_model = model
    iters_done = 0
    no_improvement = 0

    while not finished:
        for batch in train_loader:
            # Prepare input
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[2],
                "attention_mask": batch[1],
                "labels": batch[3]
            }

            if head_mask is not None:
                inputs["head_mask"] = head_mask

            # Calculate loss
            loss = model(**inputs)[0]

            # Perform backward pass and update weights
            optimizer.zero_grad()
            if GlobalState.debug:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            # Save model if this is the best
            iters_done += 1
            if iters_done % min(200, n_iters) == 0:
                metric_value = evaluate(model, test_loader, metric)
                if best_metric is None or best_metric < metric_value:
                    if verbose:
                        Logger.info(
                            f"{iters_done:4d}/{n_iters}: Improved to {metric_value:.2f}! Saved.")
                    best_metric = metric_value
                    best_model = deepcopy(model)
                    no_improvement = 0
                else:
                    if verbose:
                        Logger.info(f"{iters_done:4d}/{n_iters}: ({metric_value:.2f})")
                    no_improvement += 1

            # Check if training should be finished
            finished = iters_done >= n_iters or no_improvement >= 10
            if finished:
                break

    if verbose:
        Logger.info(f"Best {metric.name}: {best_metric:.2f}")

    # Restore the original training state of the model
    model = best_model
    model.train(was_trainable)

    return model, best_metric