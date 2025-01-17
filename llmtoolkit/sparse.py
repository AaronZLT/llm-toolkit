import torch
import torch.nn as nn
from peft import PeftModel
from typing import Dict, List

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


def find_base_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers and "base_layer" in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


@torch.no_grad()
def prune_magnitude(model, sparsity_ratio: float = 0.5, prune_n=0, prune_m=0):
    def _prune_magnitude(
        named_modules: Dict,
        sparsity_ratio: float,
        prune_n: int,
        prune_m: int,
    ):
        for name in named_modules:
            W = named_modules[name].weight.data
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
                print_rank_0(
                    f"Pruning layer {i} - {name}, sparsity ratio = {sparsity_ratio}"
                )
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            W[W_mask] = 0

    if sparsity_ratio > 1 or sparsity_ratio < 0:
        raise ValueError("sparsity_ratio should be in (0,1).")
    if prune_n % 2 != 0 or prune_m % 2 != 0 or prune_n > prune_m:
        raise ValueError("prune_n, prune_m need to be even, and prune_n < prune_m.")

    if isinstance(model, PeftModel):
        model.merge_adapter()
        layers = model.base_model.model.model.layers
        for i in range(len(layers)):
            layer = layers[i]
            named_modules = find_base_layers(layer)
            _prune_magnitude(named_modules, sparsity_ratio, prune_n, prune_m)
        model.unmerge_adapter()
    else:
        layers = model.model.layers
        for i in range(len(layers)):
            layer = layers[i]
            named_modules = find_layers(layer)
            _prune_magnitude(named_modules, sparsity_ratio, prune_n, prune_m)
