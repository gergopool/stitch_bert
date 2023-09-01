import torch
from .metrics import Metric
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification as Bert

from copy import deepcopy

from .static import GlobalState


    


def evaluate(model: Bert,
             data_loader: DataLoader,
             metric: Metric,
             is_vis: bool = False,
             mask: torch.tensor = None) -> float:
    """
    Evaluate the model on the data loader using the provided metric.

    Args:
        model: The Bert model.
        data_loader: The DataLoader object.
        metric: The Metric object to be used for evaluation.
        is_vis: If True, we assume it is a vision task.
        mask: The mask tensor.

    Returns:
        The evaluation score.
    """
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model trainable
    was_model_trainable = model.training
    model.eval().to(device)

    # Reset metric running parameters
    metric.reset()

    # Prepare mask
    mask = None if mask is None else mask.clone().detach().to(device)

    preds, labels = [], []
    for iter_i, batch in enumerate(data_loader):

        if hasattr(batch, 'input_ids'):
            inputs = {key: value.to(device) for key, value in batch.items()}
        else:
            # Prepare input
            batch = tuple(t.to(device) for t in batch)
            if is_vis:
                inputs = {'pixel_values': batch[0], 'labels': batch[1]}
            else:
                inputs = {
                    "input_ids": batch[0],
                    "token_type_ids": batch[2],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }

        inputs['head_mask'] = mask

        # Calculate loss
        if GlobalState.debug:
            with torch.inference_mode():
                outputs = model(**inputs)
        else:
            with torch.cuda.amp.autocast():
                with torch.inference_mode():
                    outputs = model(**inputs)

        # Get predictions
        metric.accumulate(outputs[1].cpu().detach().numpy(),
                          inputs['labels'].cpu().detach().numpy())

        # Quit after three iterations in debug mode
        if GlobalState.debug and iter_i >= 2:
            break

    # Set model back to trainable
    model.train(was_model_trainable)

    return deepcopy(metric.value)