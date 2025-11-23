import torch
import torch.nn.utils.prune as prune


# Set Model Weights Using Mask
def set_weight_by_mask(model, mask):
    with torch.no_grad():
        model_dict = model.state_dict()
        new_dict = {}
        for key in model_dict.keys():
            mask_key = key + '_mask'
            new_dict[key] = model_dict[key] * mask[mask_key] if mask_key in mask else model_dict[key]
        model.load_state_dict(new_dict)


# Helper Function to Identify Layers to Prune
def get_prune_params(model, include_fc=True, include_bias=True, is_resnet=False):
    """Retrieve parameters for pruning from the model, including Conv2d and optionally Linear layers."""
    parameters_to_prune = []

    if is_resnet:
        # Handle ResNet specific pruning
        state_dict_keys = model.state_dict().keys()
        keys = [key for key in state_dict_keys if 'weight' in key]
        if include_bias:
            keys += [key for key in state_dict_keys if 'bias' in key]

        for key in keys:
            temp_key_split = key.split('.')
            temp_module = model
            for part in temp_key_split[:-1]:
                temp_module = getattr(temp_module, part)
            parameters_to_prune.append((temp_module, temp_key_split[-1]))

    else:
        # Handle general CNNs
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or (include_fc and isinstance(module, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
                if include_bias and hasattr(module, 'bias') and module.bias is not None:
                    parameters_to_prune.append((module, 'bias'))

    return parameters_to_prune


# Unified Pruning Function
def prune_model(model, sparsity, is_resnet=False, include_fc=True, include_bias=True):
    """Prune the model globally using L1 Unstructured pruning."""
    prune_params = get_prune_params(model, include_fc=include_fc, include_bias=include_bias, is_resnet=is_resnet)
    prune.global_unstructured(prune_params, pruning_method=prune.L1Unstructured, amount=sparsity)

    # Collect and return masks
    mask_dict = {key: model.state_dict()[key] for key in model.state_dict().keys() if key.endswith('_mask')}

    # Remove pruning reparameterization
    for module, name in prune_params:
        prune.remove(module, name)

    return mask_dict
