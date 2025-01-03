from typing import Any, Dict, Union

import torch
from torch import nn

from transformers import (
    Seq2SeqTrainer,
)

from .optim import (
    Adam,
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


class Seq2SeqTrainer_llmtoolkit_lorafa(Seq2SeqTrainer):
    """
    Features:
    1. save sequence length on every step
    2. lorafa optimizer
    """

    def __init__(self, *args, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_seq = []
        self.scaling_factor = scaling_factor

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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer_params = {
            "param_groups": [
                {
                    "params": self.model.parameters(),
                    "lr": self.args.learning_rate,
                    "names": [name for name, _ in self.model.named_parameters()],
                    "scaling_factor": self.scaling_factor,
                    "betas": (0.9, 0.999),
                    "weight_decay": self.args.weight_decay,
                },
            ]
        }
        self.optimizer = Adam(optimizer_params["param_groups"])

        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )
        