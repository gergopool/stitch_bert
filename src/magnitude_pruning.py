"""
    This code is based on https://github.com/negar-foroutan/multiLMs-lang-neutral-subnets/blob/main/helpers/pruning_utils.py
"""

import torch.nn.utils.prune as prune
import torch

MAGNITUDE_PRUNING_COMPONENT_WISE = "magnitude_pruning_component_wise" # p% from every module
MAGNITUDE_PRUNING_GLOBAL = "magnitude_pruning_global" # p% overall from all all parameters based on the weight norm
RANDOM_PRUNING = 'random_pruning' # p% overall from all all parameters randomly

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
        px (float): Pruning rate (a number between 0 and 1) or int number of parameters to prune
    """
    parameters_to_prune = get_params_to_prune(model)
    for param in parameters_to_prune:
        prune.l1_unstructured(param[0], name="weight", amount=px)


def pruning_model(model, px):
    """ Performs magnitude pruning for mBERT model
    Args:
        model: mBERT model
        px (float): Pruning rate (a number between 0 and 1) or int number of parameters to prune
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
        px (float): Pruning rate (a number between 0 and 1) or int number of parameters to prune
    """
    parameters_to_prune = get_params_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

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

def rewind(pre_weight):

    recover_dict = {}
    name_list = []
    for ii in range(12):
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.self.query.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.self.key.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.self.value.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight')
        name_list.append('bert.encoder.layer.'+str(ii)+'.output.dense.weight')
    name_list.append('bert.pooler.dense.weight')

    for key in pre_weight.keys():

        if 'bert' in key:
            if key in name_list:
                new_key = key+'_orig'
            else:
                new_key = key

            recover_dict[new_key] = pre_weight[key]

    return recover_dict

def get_mbert_mask_ones_overlap(mask_list):
    """ Computes number of active neurons of two pruned masks. """
    total_similarity = 0
    total_size = 0
    all_layers_similarity = []
    for ii in range(12):
        layer_similarity = 0
        layer_size = 0

        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            data_list = []
            for mask_dict in mask_list:
                data_list.append(mask_dict[f'bert.encoder.layer.{ii}.{module_name}.weight_mask'])
            result = torch.stack(data_list, dim=0).sum(dim=0)
            layer_size += len(result[result != 0])
            layer_similarity += len(result[result == len(mask_list)])

        total_similarity += layer_similarity
        total_size += layer_size
        similarity = float(layer_similarity / layer_size)
        # print("Layer: {}, similarity rate: {:.4f}".format(ii, similarity))
        all_layers_similarity.append(similarity)

    if "bert.pooler.dense.weight_mask" in mask_list[0]:
        data_list = []
        for mask_dict in mask_list:
            data_list.append(mask_dict['bert.pooler.dense.weight_mask'])
        result = torch.stack(data_list, dim=0).sum(dim=0)
        layer_size += len(result[result != 0])
        layer_similarity += len(result[result == len(mask_list)])

        total_similarity += layer_similarity
        total_size += layer_size

    print("Total similarity rate: {:.4f}".format(float(total_similarity / total_size)))
    return all_layers_similarity


def see_mask_zero_rate(mask_dict):
    """ Compute the percenrage of the pruned neurons using the given mask (mBERT). """
    
    total_size = 0.0
    zero_size = 0.0
    for key in list(mask_dict.keys()):
        total_size += float(mask_dict[key].nelement())
        zero_size += float(torch.sum(mask_dict[key] == 0))

    zero_rate = (100 * zero_size) / total_size
    return zero_rate