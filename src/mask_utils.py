import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from typing import Tuple
import copy
from src.analysis_tools import jaccard_for_magnitude_pruning
from .static import Logger, GlobalState
from .evaluator import evaluate
from .magnitude_pruning import get_params_to_prune
from transformers import AutoModelForSequenceClassification

def compute_heads_importance(
        model: nn.Module,
        data_loader: DataLoader,
        is_vis: bool = False,
        head_mask: torch.tensor = None) -> Tuple[torch.tensor, np.array, np.array]:
    """
    Compute the head importance scores according to http://arxiv.org/abs/1905.10650

    Args:
        model: The transformer model.
        data_loader: DataLoader for the dataset.
        is_vis: If True, we assume it is a vision task.
        head_mask: Binary mask to apply to the heads. If None, no mask is applied.

    Returns:
        Tuple containing the head importance tensor, predictions array, and labels array.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)

    # Prepare the head mask
    head_mask = torch.ones(n_layers, n_heads) if head_mask is None else head_mask.clone().detach()
    head_mask = head_mask.to(device)
    head_mask.requires_grad_(True)

    preds, labels = [], []

    fake_opt = torch.optim.Adam([head_mask] + list(model.parameters()), lr=0.0)

    for iter_i, batch in enumerate(tqdm(data_loader, desc="Importance score computation")):

        batch = tuple(t.to(device) for t in batch)
        if is_vis:
            inputs = {"pixel_values": batch[0], "labels": batch[1]}
        else:
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[2],
                "attention_mask": batch[1],
                "labels": batch[3]
            }

        outputs = model(**inputs, head_mask=head_mask)
        loss, logits = outputs[:2]

        # Backward pass to compute gradients
        fake_opt.zero_grad()
        loss.backward()
        head_importance += head_mask.grad.abs().detach()

        preds.append(logits.detach().cpu())
        labels.append(inputs['labels'].detach().cpu())

        if GlobalState.debug and iter_i >= 2:
            Logger.info("Breaking the loop for debug purposes.")
            break

    # Normalize the importance over layers
    head_importance = torch.nn.functional.normalize(head_importance, p=2, dim=-1)

    all_preds = torch.cat(preds, dim=0).numpy()
    all_labels = torch.cat(labels, dim=0).numpy()

    return head_importance, all_preds, all_labels


def mask_heads(model: nn.Module,
               eval_dataloader: DataLoader,
               metric,
               threshold: float = 0.9,
               masking_amount: float = 0.1,
               is_vis: bool = False) -> torch.tensor:
    """
    Find masks based on importance scores as described in http://arxiv.org/abs/1905.10650

    Args:
        model: The Bert model.
        eval_dataloader: DataLoader for the evaluation dataset.
        metric: The metric to be used for evaluation.
        threshold: The threshold for the performance loss.
        masking_amount: The percentage of heads to be masked.
        is_vis: If True, we assume it is a vision task.

    Returns:
        The binary mask for the heads.
    """
    # Compute importance scores
    head_importance, preds, labels = compute_heads_importance(model, eval_dataloader, is_vis)
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
            layer_idx = head.item() // model.config.num_attention_heads
            head_idx = head.item() % model.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0

        head_importance, preds, labels = compute_heads_importance(
            model, eval_dataloader, is_vis, head_mask=new_head_mask)
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

from src.magnitude_pruning import component_wise_pruning_model, pruning_model, see_weight_rate, see_updated_weight_rate

def count_params_per_head(model):

    # every transformer layer (block) has the same number of parameters so it's not important which one we choose
    # parameters for W_k, W_v and W_q. 
    # Huggingface has one W_k, one W_v and one W_q matrix for all attention_heads so we divide with the number of attention_heads
    # we multiply with 3 because we have query, keys and values
    q_k_v = dict(model.named_parameters())['bert.encoder.layer.1.attention.self.query.weight'].shape.numel() / model.config.num_attention_heads*3

    # we multiply with 3 because we have query, keys and values
    q_k_v_bias = dict(model.named_parameters())['bert.encoder.layer.1.attention.self.query.bias'].numel() / model.config.num_attention_heads*3

    # W_out is of dimension [h*d_value, d_model] = [12*64 = 768, 768] so we divide with number of heads
    output = dict(model.named_parameters())['bert.encoder.layer.1.attention.output.dense.weight'].shape.numel() / model.config.num_attention_heads
    w_o_bias = dict(model.named_parameters())['bert.encoder.layer.1.attention.output.dense.bias'].numel() / model.config.num_attention_heads
    return q_k_v + q_k_v_bias + output + w_o_bias

def magnitude_masks(model: nn.Module,
               eval_dataloader: DataLoader,
               metric,
               pruning_method:str,
               threshold: float = 0.9,
               masking_amount: float = 0.1,
               is_vis: bool = False) -> torch.tensor:
    """
    Find masks based on magnitude pruning

    Args:
        model: The Bert model.
        eval_dataloader: DataLoader for the evaluation dataset.
        metric: The metric to be used for evaluation.
        threshold: The threshold for the performance loss.
        masking_amount: The percentage to be masked.
        is_vis: If True, we assume it is a vision task.

    Returns:
        The binary mask for the heads.
    """
    original_score = 0.99
    # original_score = evaluate(model,
    #          eval_dataloader,
    #          metric,
    #          is_vis,
    #          mask=None)

    # how many parameters do we prune at every iteration
    head_params = count_params_per_head(model)
    num_heads = int(masking_amount*144)
    total_pruned_per_iter = int(head_params * num_heads)


    pruning_step = 0
    if pruning_method=='magnitude_uniform':
        component_wise_pruning_model(model, total_pruned_per_iter)
    elif pruning_method=='magnitude_all':
        pruning_model(model, total_pruned_per_iter)
    else:
        raise ValueError("pruning_method must be magnitude_uniform or magnitude_all")
    
    #0.1*144 = 14 heads are pruned at every iteration when using structured pruning.
    # One head has in total

    rate_weight_equal_zero = see_weight_rate(model)
    Logger.info(f'zero_rate = {rate_weight_equal_zero} %')

    pruning_step += 1

    stop_at = original_score * threshold
    Logger.info(f"Pruning: original score: {original_score:.2f}, threshold: {stop_at:.2f}")

    # Ensure original score is non-zero in debug mode
    if GlobalState.debug and original_score == 0.0:
        original_score = 1.

    current_score = original_score
    new_model_dict = model.state_dict()
    i = 0
    while current_score >= stop_at:
        i+=1
        if i>2:
            break
        # save model before pruning
        model_dict = copy.deepcopy(new_model_dict)
        model.load_state_dict(model_dict)
        if pruning_method=='magnitude_uniform':
            component_wise_pruning_model(model, total_pruned_per_iter)
        elif pruning_method=='magnitude_all':
            pruning_model(model, total_pruned_per_iter)
        else:
            raise ValueError("pruning_method must be magnitude_uniform or magnitude_all")

        # evaluation_score = evaluate(model, eval_dataloader, metric, is_vis, mask=None)
        params = get_params_to_prune(model)
        rate_weight_equal_zero = see_weight_rate(model)
        Logger.info(f'zero_rate = {rate_weight_equal_zero} %')

        # calculate new model after pruning
        new_model_dict = model.state_dict()
        if GlobalState.debug:
            Logger.info("Breaking the loop for debug purposes.")
            # break
    
    mask_dict = {}
    for module in model_dict.keys():
        if 'mask' in module:
            # last iteration more sparse than the second to last iteration
            assert(torch.sum(new_model_dict[module]) < torch.sum(model_dict[module]))
            assert(torch.sum(torch.logical_and(new_model_dict[module], model_dict[module])) == torch.sum(new_model_dict[module]) )
            mask_dict[module] = model_dict[module]

    # code not correct see https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
    # new_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
    #                                                           num_labels=2)
    # pruning_model(new_model, 0)
    
    # new_model.load_state_dict(model_dict)
    # rate_weight_equal_zero = see_weight_rate(new_model)
    # Logger.info(f'zero_rate = {rate_weight_equal_zero} %')

    # rate_weight_equal_zero = see_updated_weight_rate(new_model)
    # Logger.info(f'UPDATED zero_rate = {rate_weight_equal_zero} %')

    return mask_dict