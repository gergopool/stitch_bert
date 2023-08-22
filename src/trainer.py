import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from src.evaluator import evaluate
from .static import GlobalState, Logger
from .glue_metrics import Metric


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          metric: Metric,
          head_mask: torch.Tensor = None,
          force_eval_mode: bool = False,
          is_vis: bool = False,
          verbose: bool = True) -> nn.Module:
    """
    Train a given model on a given train dataset for a specified number of iterations and batch size.
    Return with the best model based on the validation metric.

    Args:
        model: The model to be trained.
        train_loader: The data loader for the training dataset.
        val_loader: The data loader for the validation dataset.
        metric: The metric to be used for evaluation.
        head_mask: The mask applied on heads.
        force_eval_mode: If True, the model will be set to eval mode during training.
        is_vis: If True, we assume it is a vision task.
        verbose: If True, the training progress will be printed.
    Returns:
        The trained model.
    """
    n_iters = len(train_loader)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model to trainable
    was_trainable = model.training

    if not force_eval_mode:
        model.train()
    model.to(device)

    # Initialize training related objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(n_iters * 0.1), n_iters)
    if not GlobalState.debug:
        scaler = torch.cuda.amp.GradScaler()

    # Reset data_loader in case we want to use it again for exactly n_iters iterations
    train_loader.batch_sampler.reset()

    # Variables used for early stopping and saving the best model
    best_metric = None
    best_model = model
    no_improvement = 0

    # Start training
    data_iter = tqdm(train_loader, desc='Training') if verbose else train_loader

    for iter_i, batch in enumerate(data_iter):
        # Prepare input
        batch = tuple(t.to(device) for t in batch)
        if is_vis:
            inputs = {"pixel_values": batch[0], "labels": batch[1]}
        else:
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[2],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
        if head_mask is not None:
            inputs["head_mask"] = head_mask

        # Calculate loss
        if GlobalState.debug:
            loss = model(**inputs)[0]
        else:
            with torch.cuda.amp.autocast():
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
        if (iter_i + 1) % min(200, n_iters) == 0:
            metric_value = evaluate(model, val_loader, metric, is_vis)
            if best_metric is None or best_metric < metric_value:
                if verbose:
                    Logger.info(f"{iter_i:4d}/{n_iters}: Improved to {metric_value:.4f}! Saved.")
                best_metric = metric_value
                best_model = deepcopy(model)
                no_improvement = 0
            else:
                if verbose:
                    Logger.info(f"{iter_i:4d}/{n_iters}: ({metric_value:.2f})")
                no_improvement += 1

        # Check if any stopping criterion is met
        if no_improvement >= 5:
            break

    if verbose:
        Logger.info(f"Best {metric.name}: {best_metric:.2f}")

    # Restore the original training state of the model
    model = best_model
    model.train(was_trainable)

    return model, best_metric