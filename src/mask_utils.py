import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification as Bert
from typing import Tuple
from .static import Logger, GlobalState


def compute_heads_importance(
        model: Bert,
        data_loader: DataLoader,
        head_mask: torch.tensor = None) -> Tuple[torch.tensor, np.array, np.array]:
    """
    Compute the head importance scores according to http://arxiv.org/abs/1905.10650

    Args:
        model: The Bert model.
        data_loader: DataLoader for the dataset.
        head_mask: Binary mask to apply to the heads. If None, no mask is applied.

    Returns:
        Tuple containing the head importance tensor, predictions array, and labels array.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)

    # Prepare the head mask
    head_mask = torch.ones(n_layers, n_heads) if head_mask is None else head_mask.clone().detach()
    head_mask = head_mask.to(device)
    head_mask.requires_grad_(True)

    preds, labels = [], []
    tot_tokens = 0.0

    fake_opt = torch.optim.Adam([head_mask] + list(model.parameters()), lr=0.0)

    for iter_i, batch in enumerate(tqdm(data_loader, desc="Importance score computation")):
        input_ids, input_mask, segment_ids, label_ids = tuple(t.to(device) for t in batch)

        outputs = model(input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=label_ids,
                        head_mask=head_mask)
        loss, logits = outputs[:2]

        # Backward pass to compute gradients
        fake_opt.zero_grad()
        loss.backward()
        head_importance += head_mask.grad.abs().detach()

        preds.append(logits.detach().cpu())
        labels.append(label_ids.detach().cpu())
        tot_tokens += input_mask.float().detach().sum().data

        if GlobalState.debug and iter_i >= 2:
            Logger.info("Breaking the loop for debug purposes.")
            break

    # Normalize the importance over layers
    head_importance /= tot_tokens
    head_importance = torch.nn.functional.normalize(head_importance, p=2, dim=-1)

    return head_importance, torch.cat(preds, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def mask_heads(model: Bert,
               eval_dataloader: DataLoader,
               metric,
               threshold: float = 0.9,
               masking_amount: float = 0.1) -> torch.tensor:
    """
    Find masks based on importance scores as described in http://arxiv.org/abs/1905.10650

    Args:
        model: The Bert model.
        eval_dataloader: DataLoader for the evaluation dataset.
        metric: The metric to be used for evaluation.
        threshold: The threshold for the performance loss.
        masking_amount: The percentage of heads to be masked.

    Returns:
        The binary mask for the heads.
    """
    # Compute importance scores
    head_importance, preds, labels = compute_heads_importance(model, eval_dataloader)
    original_score = metric(preds, labels)
    stop_at = original_score * threshold
    Logger.info(f"Pruning: original score: {original_score:.2f}, threshold: {stop_at:.2f}")

    # Initialize mask and compute mask size
    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * masking_amount))

    # Ensure original score is non-zero in debug mode
    if GlobalState.debug and original_score == 0.0:
        original_score = 1.

    current_score = original_score
    while current_score >= stop_at:
        head_mask = new_head_mask.clone()  # save current head mask

        # Sort heads by importance
        head_importance[head_mask == 0.0] = float("Inf")
        head_importance, heads_to_mask = head_importance.view(-1).sort()
        heads_to_mask = heads_to_mask[head_importance != float("Inf")]

        if len(heads_to_mask) < num_to_mask:
            Logger.info("Nothing more to mask")
            break

        selected_heads_to_mask = heads_to_mask[:num_to_mask]

        # Mask heads
        for head in selected_heads_to_mask:
            layer_idx = head.item() // model.bert.config.num_attention_heads
            head_idx = head.item() % model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0

        head_importance, preds, labels = compute_heads_importance(model, eval_dataloader, head_mask=new_head_mask)
        current_score = metric(preds, labels)
        performance_meter = current_score / original_score * 100.
        orig_size = new_head_mask.numel()
        new_size = int(new_head_mask.sum())
        size_meter = new_size / orig_size * 100
        Logger.info(
            f"Model performance: {current_score:.2f}/{original_score:.2} ({performance_meter:.2f}%)"
        )
        Logger.info(f"Model size: {new_size}/{orig_size} ({size_meter:.2f}%) ")

        if GlobalState.debug:
            Logger.info("Breaking the loop for debug purposes.")
            break

    return head_mask.detach().cpu()