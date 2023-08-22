import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict

from .hooks import _open_forward_override_hook, _close_forward_override_hook
from ..trainer import train
from ..glue_metrics import Metric
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
                          device: torch.device) -> float:

    assert model2_performance != 0, "Original performance cannot be zero."

    # Set up stitched model
    stitch_net = StitchNet(model1, model2, mask1, mask2, layer_i, device)
    stitch_net.set_least_squares_init(**embeddings)

    # Train stitched model
    _, best_performance = train(stitch_net,
                           train_loader,
                           val_loader,
                           metric,
                           head_mask=None,
                           force_eval_mode=True,
                           is_vis=is_vis,
                           verbose=True)

    # Close hooks on stitched model
    stitch_net.close()

    Logger.info(f"Best achieved metric: {best_performance:.4f}")

    return best_performance / model2_performance


class StitchNet(nn.Module):

    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 mask1: torch.Tensor,
                 mask2: torch.Tensor,
                 layer_idx: int,
                 device: torch.device):
        super().__init__()
        self.model1 = model1.eval().to(device)
        self.model2 = model2.eval().to(device)
        self.mask1 = mask1.to(device)
        self.mask2 = mask2.to(device)
        self.layer_idx = layer_idx
        self.hidden_size = self.model1.config.hidden_size
        self.n_heads = self.model1.config.num_attention_heads
        self.n_dim = self.hidden_size // self.n_heads

        # self.transform = nn.Linear(self.n_heads, self.n_heads, bias=False)
        self.transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self._freeze_nets()
        self._setup_hooks()

    def set_least_squares_init(self, x1: torch.Tensor, x2: torch.Tensor) -> None:
        with torch.no_grad():
            self.transform.weight.data = self._pseudo_inverse(x1, x2)

    def forward(self, **kwargs):

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
