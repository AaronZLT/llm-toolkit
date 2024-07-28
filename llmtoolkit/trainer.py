import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset

import transformers
from transformers import (
    Seq2SeqTrainer,
)

from .utils import (
    get_rank,
    rank_0,
    print_rank_0,
    safe_dict2file,
    get_unique_key,
    is_ipex_available,
    hardware_info,
)


class Seq2SeqTrainer_llmtoolkit(Seq2SeqTrainer):
    """
    Features:
    1. save sequence length on every step
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_seq = []

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self.step_seq.append(inputs["input_ids"].numel())

        return super().training_step(model=model, inputs=inputs)

    def get_trained_seq(self):
        return self.step_seq
