import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict

from ..static import GlobalState


def calculate_embeddings(model1: nn.Module,
                         model2: nn.Module,
                         mask1: torch.Tensor,
                         mask2: torch.Tensor,
                         layer_i: int,
                         n_points: int,
                         data_loader: DataLoader,
                         is_vis: bool,
                         device: torch.device) -> Dict[str, torch.Tensor]:

    # Set models to eval mode
    model1 = model1.eval().to(device)
    model2 = model2.eval().to(device)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)

    # Method for retrieving activations
    def get_activations(model, mask, **inputs):
        output = model(**inputs, head_mask=mask, output_hidden_states=True)
        hidden_states = output['hidden_states'][layer_i + 1]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if 'attention_mask' in inputs:
            mask = inputs['attention_mask'].view(-1).bool()
            hidden_states = hidden_states[mask]
        return hidden_states

    # Run forward passes and save activations
    act1, act2 = [], []
    count = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        if is_vis:
            inputs = {"pixel_values": batch[0]}
        else:
            inputs = {"input_ids": batch[0], "token_type_ids": batch[2], "attention_mask": batch[1]}
        with torch.no_grad():
            act1.append(get_activations(model1, mask1, **inputs))
            act2.append(get_activations(model2, mask2, **inputs))

        if count >= n_points:
            break

    # Concatenate activations
    act1 = torch.cat(act1, dim=0)[:n_points]
    act2 = torch.cat(act2, dim=0)[:n_points]

    return {'x1': act1, 'x2': act2}
