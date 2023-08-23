import torch


def jaccard_similarity(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Computes the Jaccard similarity between two masks.

    :param mask1: The first head mask of shape (n_layers, n_heads)
    :param mask2: The second head mask of shape (n_layers, n_heads)
    :return: The Jaccard similarity between the two masks, ranging from 0 to 1.
    """

    # Check that the masks have the same shape
    assert mask1.shape == mask2.shape, "Masks must have the same shape."

    mask1 = mask1.bool()
    mask2 = mask2.bool()

    # Heads active in both masks
    intersection = (mask1 & mask2).sum()

    # Heads active in either mask
    union = (mask1 | mask2).sum()

    return (intersection / union).item()