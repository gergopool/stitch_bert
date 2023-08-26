"""
    This code is based on multiLMs-lang-neutral-subnets/helpers/pruning_utils.py
"""

import torch.nn.utils.prune as prune
import torch

def get_params_to_prune(model_to_prune):
    """ Returns the parameters we want to prune for mBERT model """
    parameters_to_prune = []
    if isinstance(model_to_prune, torch.nn.DataParallel):
        model = model_to_prune.module
    else:
        model = model_to_prune
   
    for layer_num in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            parameters_to_prune.append((eval(f"model.bert.encoder.layer[{layer_num}].{module_name}"), 'weight'))

    if model.bert.pooler is not None:
        parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
        
    parameters_to_prune = tuple(parameters_to_prune)
    return parameters_to_prune


def component_wise_pruning_model(model, px):
    """ Performs component-wise magnitude pruning for mBERT model 
    Args:
        model: mBERT model
        px (float): Pruning rate (a number between 0 and 1)
    """
    parameters_to_prune = get_params_to_prune(model)
    for param in parameters_to_prune:
        prune.l1_unstructured(param[0], name="weight", amount=px)


def pruning_model(model, px):
    """ Performs magnitude pruning for mBERT model
    Args:
        model: mBERT model
        px (float): Pruning rate (a number between 0 and 1)
    """
    parameters_to_prune = get_params_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def random_pruning_model(model, px):
    """ Performs random pruning for mBERT model
    Args:
        model: mBERT model
        px (float): Pruning rate (a number between 0 and 1)
    """
    parameters_to_prune = get_params_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

def pruning_model_custom(model, mask_dict):
    """ Performs custom pruning given a pruning mask for mBERT model """
    parameters_to_prune = []
    mask_list = []
    for layer_num in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            parameters_to_prune.append(eval(f"model.bert.encoder.layer[{layer_num}].{module_name}"))
            mask_list.append(mask_dict[f'bert.encoder.layer.{layer_num}.{module_name}.weight_mask'])

    if model.bert.pooler is not None and 'bert.pooler.dense.weight_mask' in mask_dict:
        parameters_to_prune.append(model.bert.pooler.dense)
        mask_list.append(mask_dict['bert.pooler.dense.weight_mask'])

    for idx in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[idx], 'weight', mask=mask_list[idx])

def see_weight_rate(model): 
    """ Computes the sparsity level of the given mBERT model """
    sum_list = 0
    zero_sum = 0
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for idx in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            sum_list = sum_list + float(eval(f"model.bert.encoder.layer[{idx}].{module_name}.weight.nelement()"))
            zero_sum = zero_sum +\
                float(torch.sum(eval(f"model.bert.encoder.layer[{idx}].{module_name}.weight") == 0))

    if model.bert.pooler is not None:
        sum_list = sum_list + float(model.bert.pooler.dense.weight.nelement())
        zero_sum = zero_sum + float(torch.sum(model.bert.pooler.dense.weight == 0))
 
    return 100.0 * zero_sum / sum_list
