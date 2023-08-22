import torch
from torch import nn


def _open_forward_override_hook(model: nn.Module, layer_i: int) -> None:
    """Open a forward hook to load activations."""
    model.cache = None

    def override_output(module, m_in, m_out):
        assert model.cache is not None
        activation = model.cache
        model.cache = None
        return (activation,)

    if hasattr(model, 'bert'):
        layer = model.bert.encoder.layer[layer_i]
    elif hasattr(model, 'vit'):
        layer = model.vit.encoder.layer[layer_i]
    model.hook = layer.register_forward_hook(override_output)


def _close_forward_override_hook(model: nn.Module) -> None:
    """Close the forward hook."""
    model.hook.remove()