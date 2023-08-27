import torch
from torch import Tensor


def cka(x1: Tensor, x2: Tensor) -> float:
    """Linear Centered Kernel Alignment (CKA) between two tensors.

    :param x1: The first tensor of shape (n_samples, *)
    :param x2: The second tensor of shape (n_samples, *)
    :return: The CKA between the two tensors, ranging from 0 to 1.
    """
    x1 = _gram_linear(rearrange_activations(x1))
    x2 = _gram_linear(rearrange_activations(x2))
    similarity = _cka(x1, x2)
    return similarity


def rearrange_activations(activations: Tensor) -> Tensor:
    """Rearrange activations to be of shape (n_samples, n_features)."""
    batch_size = activations.shape[0]
    flat_activations = activations.view(batch_size, -1)
    return flat_activations


def _cka(gram_x: Tensor, gram_y: Tensor, debiased: bool = False) -> float:
    """Compute the CKA similarity between two Gram matrices."""
    gram_x = _center_gram(gram_x, unbiased=debiased)
    gram_y = _center_gram(gram_y, unbiased=debiased)

    scaled_hsic = (gram_x.view(-1) * gram_y.view(-1)).sum()
    normalization_x = torch.linalg.norm(gram_x)
    normalization_y = torch.linalg.norm(gram_y)

    return scaled_hsic / (normalization_x * normalization_y)


def _gram_linear(x: Tensor) -> Tensor:
    """Compute the Gram matrix of the given tensor."""
    return torch.mm(x, x.T)


def _center_gram(gram: Tensor, unbiased: bool = False) -> Tensor:
    """Center the Gram matrix.

    :param gram: Input symmetric matrix.
    :param unbiased: Flag indicating whether to use unbiased centering.
    :return: Centered Gram matrix.
    """
    if not torch.allclose(gram, gram.T, rtol=1e-03, atol=1e-04):
        raise ValueError('Input must be a symmetric matrix.')

    if unbiased:
        pass
        # TODO
    else:
        means = torch.mean(gram, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram -= torch.unsqueeze(means, len(means.shape))
        gram -= torch.unsqueeze(means, 0)

    return gram
