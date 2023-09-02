import torch


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