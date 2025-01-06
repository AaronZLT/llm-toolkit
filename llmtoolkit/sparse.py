import torch
import torch.nn as nn

from .utils import (
    print_rank_0,
)


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def prune_magnitude(model, sparsity_ratio: float = 0.5, prune_n=0, prune_m=0):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                print_rank_0(
                    f"Pruning layer {i} - {name}, sparsity ratio = {prune_n}:{prune_m}"
                )
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                print_rank_0(f"Pruning layer {i} - {name}, sparsity ratio = {sparsity_ratio}")
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            W[W_mask] = 0
