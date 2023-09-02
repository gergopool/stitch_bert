import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from .magnitude_pruning import pruning_model, component_wise_pruning_model, see_weight_rate
from .magnitude_pruning import MAGNITUDE_PRUNING_COMPONENT_WISE, MAGNITUDE_PRUNING_GLOBAL, RANDOM_PRUNING
import os

from src.evaluator import evaluate
from .static import GlobalState, Logger
from .metrics import Metric


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          metric: Metric,
          head_mask: torch.Tensor = None,
          force_eval_mode: bool = False,
          is_vis: bool = False,
          verbose: bool = True,
          lr: float = 1e-5,
          weight_decay: float = 0.001,
          warmup_fraction: float = 0.1,
          early_stopping_patience: int = 5) -> nn.Module:
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
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        warmup_fraction: Fraction of iterations for learning rate warmup.
        early_stopping_patience: Number of iterations without improvement before stopping.
    Returns:
        The trained model.
    """
    n_iters = len(train_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preserve original training state
    was_model_trainable = model.training

    if not force_eval_mode:
        model.train()
    model.to(device)

    # Initialize training related objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(n_iters * warmup_fraction), n_iters)
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
            optimizer.step()
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
                    Logger.info(f"{iter_i+1:4d}/{n_iters}: Improved to {metric_value:.4f}! Saved.")
                best_metric = metric_value
                best_model = deepcopy(model)
                no_improvement = 0
            else:
                if verbose:
                    Logger.info(f"{iter_i+1:4d}/{n_iters}: ({metric_value:.2f})")
                no_improvement += 1

        # Check if any stopping criterion is met
        if no_improvement >= early_stopping_patience:
            break

    if verbose:
        Logger.info(f"Best {metric.name}: {best_metric:.2f}")

    # Restore the original training state of the model
    model = best_model
    model.train(was_model_trainable)

    return model, best_metric

def iterative_magnitude_pruning(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          metric: Metric,
          force_eval_mode: bool = False,
          is_vis: bool = False,
          verbose: bool = True,
          lr: float = 1e-5,
          weight_decay: float = 0.001,
          warmup_fraction: float = 0.1,
          early_stopping_patience: int = 5,
          orig: dict = {},
          pruning_method: str = MAGNITUDE_PRUNING_COMPONENT_WISE,
          prune_every: int= 200,
          finetuned_best_metric: float=0):
    """
    Iteratively train the model, save, prune and rewind

    Args:
        model: The model to be trained.
        train_loader: The data loader for the training dataset.
        val_loader: The data loader for the validation dataset.
        metric: The metric to be used for evaluation.
        force_eval_mode: If True, the model will be set to eval mode during training.
        is_vis: If True, we assume it is a vision task.
        verbose: If True, the training progress will be printed.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        warmup_fraction: Fraction of iterations for learning rate warmup.
        early_stopping_patience: Number of iterations without improvement before stopping.
        orig: The dictionary containing the parameters of the model.
        pruning_method: Whether we prune the p% of each module i.e. "component_wise" or globally
        prune_every: We save the model every prune_every and then prune and rewind
    Returns:
        model: the model with the highest sparsity that has metric_value >= 0.9 * finetuned_best_metric
        mask_dict: dict which contains the masked_modules
    """
    n_iters = len(train_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preserve original training state
    was_model_trainable = model.training

    if not force_eval_mode:
        model.train()
    model.to(device)

    # Initialize training related objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(n_iters * warmup_fraction), n_iters)
    if not GlobalState.debug:
        scaler = torch.cuda.amp.GradScaler()

    # Reset data_loader in case we want to use it again for exactly n_iters iterations
    train_loader.batch_sampler.reset()

    # Variables used for early stopping and saving the best model
    best_metric_current_sparsity = None
    best_model = model
    no_improvement = 0
    pruning_step = 0

    # Start training
    data_iter = tqdm(train_loader, desc='Training') if verbose else train_loader
    mask_dict = {} 
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
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Save model if this is the best
        if (iter_i + 1) % min(200, n_iters) == 0:
            metric_value = evaluate(model, val_loader, metric, is_vis)
            # (first time or best for this sparsity) and (at_least >= 0.9 finetuned accuracy or debug)
            if (best_metric_current_sparsity is None or best_metric_current_sparsity <= metric_value) \
                     and (metric_value >= 0.9 * finetuned_best_metric or GlobalState.debug):
                if verbose:
                    Logger.info(f"{iter_i+1:4d}/{n_iters}: Improved to {metric_value:.4f}! Saved.")
                best_metric_current_sparsity = metric_value
                best_model = deepcopy(model)
                no_improvement = 0
                mask_dict = {}
                for key in model_dict.keys():
                    if 'mask' in key:
                        mask_dict[key] = model_dict[key]   

            else:
                if verbose:
                    Logger.info(f"{iter_i+1:4d}/{n_iters}: ({metric_value:.2f})")
                no_improvement += 1

        # check https://github.com/VITA-Group/BERT-Tickets/blob/master/LT_glue.py#L353 
        # where the only difference is that they save every checkpoint while I save above the one with the highest sparsity and score
        if iter_i % prune_every == 0 and prune_every >0:
            best_metric_current_sparsity = 0 # init again because we will prune below
            Logger.info(f'starting pruning {1/(10-pruning_step)}')
            if pruning_method == MAGNITUDE_PRUNING_COMPONENT_WISE:
                component_wise_pruning_model(model, 1/(10-pruning_step))
            elif pruning_method == MAGNITUDE_PRUNING_GLOBAL:
                pruning_model(model, 1/(10-pruning_step))
            rate_weight_equal_zero = see_weight_rate(model)
            pruning_step += 1
            Logger.info(f'zero_rate = {rate_weight_equal_zero}')

            Logger.info('rewinding')
            # orig has weight_orig not weight_mask or weight while
            # model_dict has weight_mask, weight_orig

            # for key in orig.keys():
            #     assert(torch.sum(model.state_dict()[key]) == torch.sum(orig[key]) )

            # orig is different from the pretrained version

            # from transformers import AutoModelForSequenceClassification
            # pretrained =AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
            # module = 'bert.encoder.layer.11.intermediate.dense.weight'
            # assert (torch.sum(pretrained.state_dict()[module]) != \
            #         torch.sum(orig[f'{module}_orig']))

            model_dict = model.state_dict()
            model_dict.update(orig) #hoping to reset remaing values to weight_orig
            model.load_state_dict(model_dict) #load the modified state_dict to the model

            # after pruning rewind scheduler and lr to initial values
            Logger.info('optimizer rewinding')
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, int(n_iters * warmup_fraction), n_iters)
        

        # Check if any stopping criterion is met
        if no_improvement >= early_stopping_patience:
            break

    if verbose:
        Logger.info(f"Best {metric.name}: {best_metric_current_sparsity:.2f}")

    # Restore the original training state of the model
    model = best_model
    model.train(was_model_trainable)
    return model, mask_dict
