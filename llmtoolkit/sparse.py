import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import peft
from peft import PeftModel
from typing import Dict, List
import bitsandbytes as bnb
from bitsandbytes.functional import quantize_4bit, dequantize_4bit

from .utils import (
    print_rank_0,
)


r"""
FYI

Typically, a lora layer contains following sub-layers:
lora.Linear(
  (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
  (lora_dropout): ModuleDict(
    (default): Identity()
  )
  (lora_A): ModuleDict(
    (default): Linear(in_features=4096, out_features=1, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=1, out_features=4096, bias=False)
  )
  (lora_embedding_A): ParameterDict()
  (lora_embedding_B): ParameterDict()
  (lora_magnitude_vector): ModuleDict()
)

If with bitsandbytes 4bit, a lora layer then contains following:
lora.Linear4bit(
  (base_layer): Linear4bit(in_features=4096, out_features=4096, bias=False)
  (lora_dropout): ModuleDict(
    (default): Identity()
  )
  (lora_A): ModuleDict(
    (default): Linear(in_features=4096, out_features=1, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=1, out_features=4096, bias=False)
  )
  (lora_embedding_A): ParameterDict()
  (lora_embedding_B): ParameterDict()
  (lora_magnitude_vector): ModuleDict()
)
"""


def find_module_name(model, target_module):
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return None

# todo: modify uint8 directly without dequantize and quantize
@torch.no_grad()
def apply_spare(model, named_mask: dict):
    for n, m in model.named_modules():
        if n in named_mask:
            print_rank_0(f"Applying sparse on layer - {n}")
            if isinstance(m, bnb.nn.Linear4bit):
                quant_state = copy.deepcopy(m.quant_state)
                _dequantize = dequantize_4bit(
                    A=m.weight.data,
                    quant_state=quant_state,
                    blocksize=quant_state.blocksize,
                    quant_type=quant_state.quant_type,
                )
                _dequantize[named_mask[n].cuda()] = 0
                m.weight.data, _ = quantize_4bit(
                    A=_dequantize,
                    absmax=quant_state.absmax,
                    blocksize=quant_state.blocksize,
                    quant_type=quant_state.quant_type,
                )
            elif isinstance(m, bnb.nn.Linear8bit):
                pass
            else:
                m.weight.data[named_mask[n].cuda()] = 0


@torch.no_grad()
def prune_magnitude(
    model, sparsity_ratio: float = 0.5, prune_n=0, prune_m=0, offload=True
) -> List:
    def _get_mask_prune_magnitude(
        W,
        sparsity_ratio: float,
        prune_n: int,
        prune_m: int,
    ) -> torch.tensor:
        # W = module.weight.data
        # W_cpu = W.detach().cpu().clone()
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
            thresh = torch.sort(W_metric.flatten())[0][
                int(W.numel() * sparsity_ratio)
            ].cpu()
            W_mask = W_metric <= thresh

        if offload:
            return W_mask.cpu()
        else:
            return W_mask

    if sparsity_ratio > 1 or sparsity_ratio < 0:
        raise ValueError("sparsity_ratio should be in (0,1).")
    if prune_n % 2 != 0 or prune_m % 2 != 0 or prune_n > prune_m:
        raise ValueError("prune_n, prune_m need to be even, and prune_n < prune_m.")

    named_mask = {}
    # todo: check if lora is attached

    if hasattr(model, "hf_quantizer"):
        print_rank_0(
            "The base_model is quantized. Proceed with dequantize and quantize."
        )
        quantization_config = model.hf_quantizer.quantization_config
        if quantization_config.load_in_4bit:
            for n, m in model.named_modules():
                if isinstance(m, peft.tuners.lora.bnb.Linear4bit):
                    print_rank_0(
                        f"Pruning layer - {n}, sparsity ratio = {sparsity_ratio}"
                    )
                    # with autocast(dtype=torch.bfloat16):
                    base_layer = m.base_layer
                    base_layer_name = find_module_name(model, base_layer)
                    quant_state = copy.deepcopy(base_layer.quant_state)
                    dequantize_base_layer_data = dequantize_4bit(
                        A=base_layer.weight.data,
                        quant_state=quant_state,
                        blocksize=quant_state.blocksize,
                        quant_type=quant_state.quant_type,
                    )
                    target_layer_data = dequantize_base_layer_data + (
                        (m.lora_B.default.weight.data @ m.lora_A.default.weight.data)
                        * model.peft_config["default"].lora_alpha
                        / model.peft_config["default"].r
                    )
                    named_mask.update(
                        {
                            base_layer_name: _get_mask_prune_magnitude(
                                target_layer_data, sparsity_ratio, prune_n, prune_m
                            )
                        }
                    )
                    dequantize_base_layer_data[named_mask[base_layer_name].cuda()] = 0
                    m.base_layer.weight.data, _ = quantize_4bit(
                        A=dequantize_base_layer_data,
                        absmax=quant_state.absmax,
                        blocksize=quant_state.blocksize,
                        quant_type=quant_state.quant_type,
                    )
        elif quantization_config.load_in_8bit:
            raise ValueError("Sparse on 8bit model is not supported for now.")
        else:
            raise ValueError(
                "Quantized model detected, however it is neither load_in_4bit or load_in_8bit."
            )
    else:
        if isinstance(model, PeftModel):
            model.merge_adapter()
            for n, m in model.named_modules():
                if isinstance(m, peft.tuners.lora.layer.Linear):
                    base_layer_name = find_module_name(model, m.base_layer)
                    named_mask.update(
                        {
                            base_layer_name: _get_mask_prune_magnitude(
                                m.base_layer.weight.data,
                                sparsity_ratio,
                                prune_n,
                                prune_m,
                            )
                        }
                    )
            model.unmerge_adapter()
            for n, m in model.named_modules():
                if n in named_mask:
                    print_rank_0(
                        f"Pruning layer - {n}, sparsity ratio = {sparsity_ratio}"
                    )
                    m.weight.data[named_mask[n].cuda()] = 0
        else:
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    named_mask.update(
                        {
                            n: _get_mask_prune_magnitude(
                                m.weight.data, sparsity_ratio, prune_n, prune_m
                            )
                        }
                    )
                    print_rank_0(
                        f"Pruning layer - {n}, sparsity ratio = {sparsity_ratio}"
                    )
                    m.weight.data[named_mask[n].cuda()] = 0

    return named_mask
