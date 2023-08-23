from torch import nn


def _open_forward_override_hook(model: nn.Module, layer_i: int) -> None:
    """
    Open a forward hook to load activations.

    Parameters:
    - model: The model on which to register the forward hook.
    - layer_i: Index of the layer for the hook.
    """
    model.cache = None  # Initialize cache to None

    def override_output(module, m_in, m_out):
        assert model.cache is not None  # Ensure cache is not None before proceeding
        activation = model.cache
        model.cache = None
        return (activation,)

    # Determine the layer based on the model type (BERT or ViT)
    if hasattr(model, 'bert'):
        layer = model.bert.encoder.layer[layer_i]
    elif hasattr(model, 'vit'):
        layer = model.vit.encoder.layer[layer_i]
    else:
        raise ValueError("Model must have either 'bert' or 'vit' attribute.")

    model.hook = layer.register_forward_hook(override_output)


def _close_forward_override_hook(model: nn.Module) -> None:
    """
    Close the forward hook.

    Parameters:
    - model: The model on which to remove the forward hook.
    """
    model.hook.remove()  # Remove the registered forward hook
