import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple


def get_activations(model: nn.Module,
                    mask: torch.Tensor,
                    layer_i: int,
                    **inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    
    Parameters:
    - model: The model to retrieve activations from.
    - mask: The mask to be applied during the forward pass.
    - layer_i: Index of the layer to retrieve activations from.
    - inputs: Additional inputs required for the forward pass.

    Returns:
    - hidden_states: Activations from the specified layer.
    """
    output = model(**inputs, head_mask=mask, output_hidden_states=True)
    hidden_states = output['hidden_states'][layer_i + 1]
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    if 'attention_mask' in inputs:
        mask = inputs['attention_mask'].view(-1).bool()
        hidden_states = hidden_states[mask]
    return hidden_states


def process_batch(batch: Tuple[torch.Tensor], is_vis: bool,
                  device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Process a batch of data for either visual or non-visual tasks.

    Parameters:
    - batch: A tuple containing the batch data.
    - is_vis: Flag indicating whether it's a visual task.
    - device: Device to move the batch data to.

    Returns:
    - inputs: Processed inputs for the forward pass.
    """
    batch = tuple(t.to(device) for t in batch)
    if is_vis:
        return {"pixel_values": batch[0]}
    else:
        return {"input_ids": batch[0], "token_type_ids": batch[2], "attention_mask": batch[1]}


def calculate_embeddings(model1: nn.Module,
                         model2: nn.Module,
                         mask1: torch.Tensor,
                         mask2: torch.Tensor,
                         layer_i: int,
                         n_points: int,
                         data_loader: DataLoader,
                         is_vis: bool,
                         device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Pre-calculate embeddings for two models using the given masks.
    It is then used for calculating the CKA or stitching matrix initialisation.

    Parameters:
    - model1: First model for which to calculate embeddings.
    - model2: Second model for which to calculate embeddings.
    - mask1: Mask for the first model.
    - mask2: Mask for the second model.
    - layer_i: Index of the layer to retrieve activations from.
    - n_points: Number of points to retrieve.
    - data_loader: DataLoader containing the data.
    - is_vis: Flag indicating whether it's a visual task.
    - device: Device to move the models and data to.

    Returns:
    - Dictionary containing the activations 'x1' and 'x2'.
    """

    # Move models and masks to the given device and set them to evaluation mode
    model1 = model1.eval().to(device)
    model2 = model2.eval().to(device)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)

    # Collect activations from forward passes
    act1, act2 = [], []
    count = 0
    for batch in data_loader:
        inputs = process_batch(batch, is_vis, device)
        with torch.no_grad():
            act1.append(get_activations(model1, mask1, layer_i, **inputs))
            act2.append(get_activations(model2, mask2, layer_i, **inputs))

        count += len(act1)

        if count >= n_points:
            break

    # Concatenate and truncate activations to the required number of points
    act1 = torch.cat(act1, dim=0)[:n_points]
    act2 = torch.cat(act2, dim=0)[:n_points]

    return {'x1': act1, 'x2': act2}
