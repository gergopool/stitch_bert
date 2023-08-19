import torch
import numpy as np
from .glue_metrics import Metric
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification as Bert
from typing import Tuple

from .static import GlobalState


def predict(model: Bert,
            data_loader: DataLoader,
            mask: torch.tensor = None) -> Tuple[np.array, np.array]:
    """
    Run the model on the data loader and return the predictions and labels.

    Args:
        model: The Bert model.
        data_loader: The DataLoader object.
        mask: The mask tensor.

    Returns:
        A tuple of numpy arrays for predictions and labels.
    """

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model trainable
    trainable_was = model.training
    model.eval().to(device)

    # Prepare mask
    mask = None if mask is None else mask.clone().detach().to(device)

    preds, labels = [], []
    for iter_i, batch in enumerate(data_loader):

        # Prepare input
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[2],
            "attention_mask": batch[1],
            "labels": batch[3],
            "head_mask": mask,
        }

        # Calculate loss
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**inputs)

        # Get predictions
        preds.append(outputs[1].cpu().detach())
        labels.append(batch[3].cpu().detach())

        # Quick after three iterations in debug mode
        if GlobalState.debug and iter_i >= 2:
            break

    # Concatenate predictions and labels
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Set model back to trainable
    model.train(trainable_was)

    return preds, labels


def evaluate(model: Bert,
             data_loader: DataLoader,
             metric: Metric,
             mask: torch.tensor = None) -> float:
    """
    Evaluate the model on the data loader using the provided metric.

    Args:
        model: The Bert model.
        data_loader: The DataLoader object.
        metric: The Metric object to be used for evaluation.
        mask: The mask tensor.

    Returns:
        The evaluation score.
    """
    preds, labels = predict(model, data_loader, mask=mask)
    return metric(preds, labels)