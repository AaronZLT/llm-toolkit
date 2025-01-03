from typing import Any, Dict, Union

import torch
from torch import nn

from transformers import (
    Seq2SeqTrainer,
)


class Seq2SeqTrainer_llmtoolkit(Seq2SeqTrainer):
    """
    Features:
    1. save sequence length on every step
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_seq = []

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch,
    ) -> torch.Tensor:
        self.step_seq.append(inputs["input_ids"].numel())

        return super().training_step(
            model=model, inputs=inputs, num_items_in_batch=num_items_in_batch
        )

    def get_trained_seq(self):
        return self.step_seq
