"""
    This code is based on https://github.com/negar-foroutan/multiLMs-lang-neutral-subnets/blob/main/helpers/analysis_tools.py
"""

from asyncio.log import logger
import torch

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


def jaccard_for_magnitude_pruning(mask_dict1, mask_dict2, last_layer):
    """ Computes jaccard similarity till given last_layer """

    components = ["attention.self.query", "attention.self.key", "attention.self.value",
                  "attention.output.dense", "intermediate.dense", "output.dense"]

    all_and = 0
    all_union = 0
    for comp in components:
        for idx in range(last_layer):
            key = f'bert.encoder.layer.{idx}.' + comp + '.weight_mask'
            module_union = torch.logical_or(mask_dict1[key], mask_dict2[key]).sum().item()
            module_and = torch.logical_and(mask_dict1[key], mask_dict2[key]).sum().item()
            all_and += module_and
            all_union += module_union

    jaccard = all_and * 1.0 / all_union


    return jaccard