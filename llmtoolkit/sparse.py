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


def find_module_name(model, target_module):
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return None


# TODO: check if the model is quantized
@torch.no_grad()
def check_sparsity(model):
    pass


# TODO: modify uint8 directly without dequantize and quantize
# TODO: add sparse config to model config
@torch.no_grad()
def apply_sparse(model, named_mask: dict):
    pass


@torch.no_grad()
def mergeW2AB(W, A, B, lora_scaling):
    pass


@torch.no_grad()
def prune_magnitude(
    model,
    sparsity_ratio: float = 0.5,
    prune_n=0,
    prune_m=0,
    offload=True,
    sparse_preserve_accuracy=False,
    sparse_prune_largest=False,
) -> List:
    return []
