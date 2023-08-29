import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from typing import Dict
from collections import OrderedDict

from .hooks import _open_forward_override_hook, _close_forward_override_hook
from ..trainer import train
from ..metrics import Metric
from ..static import Logger


def functional_similarity(model1: nn.Module,
                          model2: nn.Module,
                          mask1: torch.Tensor,
                          mask2: torch.Tensor,
                          layer_i: int,
                          embeddings: Dict[str, torch.Tensor],
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          model2_performance: float,
                          metric: Metric,
                          is_vis: bool,
                          device: torch.device,
                          trans_type: str) -> float:
    """
    Calculate the functional similarity between two models by stitching them together.

    Parameters:
    - model1, model2: The models to be stitched.
    - mask1, mask2: Masks to be applied to the models.
    - layer_i: Index of the layer for stitching.
    - embeddings: Embeddings for initializing the least squares method.
    - train_loader, val_loader: Data loaders for training and validation.
    - model2_performance: Original performance of model2 (must be non-zero).
    - metric: Metric for evaluation.
    - is_vis: Flag indicating whether it's a visual task.
    - device: Device to move the models and data to.
    - trans_type: Type of transformation to be applied to the activations. Choose from 'linear' and 'linearbn'.

    Returns:
    - The functional similarity as a float value.
    """

    assert model2_performance != 0, "Original performance cannot be zero."

    # Set up stitched model
    stitch_net = StitchNet(model1, model2, mask1, mask2, layer_i, device, trans_type)
    stitch_net.set_least_squares_init(**embeddings)

    # Train stitched model
    _, best_performance = train(stitch_net,
                           train_loader,
                           val_loader,
                           metric,
                           head_mask=None,
                           force_eval_mode=True,
                           is_vis=is_vis,
                           lr=3e-4,
                           weight_decay=0.,
                           verbose=True)

    # Close hooks on stitched model
    stitch_net.close()

    Logger.info(f"Best achieved metric: {best_performance:.4f}")

    return best_performance / model2_performance


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class StitchNet(nn.Module):
    """
    Stitched Network class to combine two models with specified masks and layer index.

    Methods:
    - set_least_squares_init: Initialize the transformation with least squares method.
    - forward: Forward pass through the stitched model.
    - close: Close hooks on the stitched model.
    """

    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 mask1: torch.Tensor,
                 mask2: torch.Tensor,
                 layer_idx: int,
                 device: torch.device,
                 trans:str = 'linear'):
        super().__init__()
        self.model1 = model1.eval().to(device)
        self.model2 = model2.eval().to(device)
        self.mask1 = mask1.to(device)
        self.mask2 = mask2.to(device)
        self.layer_idx = layer_idx
        self.hidden_size = self.model1.config.hidden_size
        self.n_heads = self.model1.config.num_attention_heads
        self.n_dim = self.hidden_size // self.n_heads
        self.trans = trans

        # self.transform = nn.Linear(self.n_heads, self.n_heads, bias=False)
        if trans == 'linear':
            self.transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif trans == 'linearbn':
            self.transform = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.hidden_size, self.hidden_size, bias=False)),
                ('perm1', Permute(0, 2, 1)),
                ('bn', nn.BatchNorm1d(self.hidden_size, affine=True)),
                ('perm2', Permute(0, 2, 1))
            ]))
            self.transform.bn.bias.requires_grad = False

        self._freeze_nets()
        self._setup_hooks()

    def set_least_squares_init(self, x1: torch.Tensor, x2: torch.Tensor) -> None:
        """Initialize the transformation weight with the pseudo inverse of the given tensors."""
        with torch.no_grad():
            w = self._pseudo_inverse(x1, x2)
            if self.trans == 'linear':
                self.transform.weight.data = w
            else:
                vec_sizes = torch.norm(w, dim=1, keepdim=True)
                self.transform.linear.weight.data = w / vec_sizes
                self.transform.bn.weight.data = vec_sizes
                w = (w.T / vec_sizes).T

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the stitched model."""

        # Calculate activations for model1
        with torch.no_grad():
            if 'input_ids' in kwargs:
                outputs = self.model1(kwargs['input_ids'],
                                      attention_mask=kwargs['attention_mask'],
                                      head_mask=self.mask1,
                                      output_hidden_states=True)
            else:
                outputs = self.model1(kwargs['pixel_values'],
                                      head_mask=self.mask1,
                                      output_hidden_states=True)

        # Load and transform activations
        act = outputs['hidden_states'][self.layer_idx + 1].detach()
        act = self.transform(act)

        # Cache activations in model2
        self.model2.cache = act

        # Run model2
        if 'input_ids' in kwargs:
            outputs = self.model2(kwargs['input_ids'],
                                  attention_mask=kwargs['attention_mask'],
                                  labels=kwargs['labels'],
                                  head_mask=self.mask2)
        else:
            outputs = self.model2(kwargs['pixel_values'],
                                  labels=kwargs['labels'],
                                  head_mask=self.mask2)

        return outputs

    def close(self):
        _close_forward_override_hook(self.model2)

    # =======================================================================
    # Private functions
    # =======================================================================

    def _freeze_nets(self):
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False

    def _pseudo_inverse(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Calculate the pseudo-inverse of the given tensors."""

        x1 = x1.reshape(-1, x1.shape[-1])
        x2 = x2.reshape(-1, x2.shape[-1])

        if not x1.shape[0] == x2.shape[0]:
            raise ValueError('Spatial size of compared neurons must match when ' \
                            'calculating psuedo inverse matrix.')

        # Calculate pseudo inverse
        w = (torch.linalg.pinv(x1) @ x2).T

        return w

    def _setup_hooks(self):
        _open_forward_override_hook(self.model2, self.layer_idx)
