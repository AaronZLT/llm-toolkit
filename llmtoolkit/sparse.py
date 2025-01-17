import torch
import torch.nn as nn
from peft import PeftModel
from typing import Dict, List

from .utils import (
    print_rank_0,
)


@torch.no_grad()
def prune_magnitude(model, sparsity_ratio: float = 0.5, prune_n=0, prune_m=0):
    def _get_mask_prune_magnitude(
        module,
        sparsity_ratio: float,
        prune_n: int,
        prune_m: int,
    ) -> torch.tensor:
        W = module.weight.data
        W_metric = torch.abs(W)
        if prune_n != 0:
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
            thresh = torch.sort(W_metric.flatten().cuda())[0][
                int(W.numel() * sparsity_ratio)
            ].cpu()
            W_mask = W_metric <= thresh
        return W_mask

    if sparsity_ratio > 1 or sparsity_ratio < 0:
        raise ValueError("sparsity_ratio should be in (0,1).")
    if prune_n % 2 != 0 or prune_m % 2 != 0 or prune_n > prune_m:
        raise ValueError("prune_n, prune_m need to be even, and prune_n < prune_m.")

    named_mask = {}
    # todo: check if lora is attached
    if isinstance(model, PeftModel):
        model.merge_adapter()
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and "base_layer" in n:
                named_mask.update(
                    {n: _get_mask_prune_magnitude(m, sparsity_ratio, prune_n, prune_m)}
                )
        model.unmerge_adapter()
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and "base_layer" in n:
                if n in named_mask:
                    print_rank_0(
                        f"Pruning layer - {n}, sparsity ratio = {sparsity_ratio}"
                    )
                    m.weight.data[named_mask[n]] = 0
                else:
                    raise ValueError(
                        f"No sparse mask for module {n}! This is unexpected."
                    )
    else:
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                named_mask.update(
                    {n: _get_mask_prune_magnitude(m, sparsity_ratio, prune_n, prune_m)}
                )
                print_rank_0(f"Pruning layer - {n}, sparsity ratio = {sparsity_ratio}")
                m.weight.data[named_mask[n]]
