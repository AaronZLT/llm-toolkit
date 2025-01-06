from typing import Any, Dict, Union

import torch
from torch import nn

from transformers import (
    Seq2SeqTrainer,
)

from .optim import (
    AdamW_lorafa,
)
from .utils import (
    print_rank_0,
)


class BaseSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Base class for Seq2Seq trainers with shared functionality.
    Features:
    1. Save sequence length at every step.
    """

    def __init__(self, *args, **kwargs):
        print_rank_0("Initializing BaseSeq2SeqTrainer.")
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

    def get_trained_seq(self) -> list:
        """
        Get the list of sequence lengths recorded during training.
        """
        return self.step_seq


class Seq2SeqTrainer_lorafa(BaseSeq2SeqTrainer):
    """
    Seq2Seq trainer with LoRA-specific functionality.
    Features:
    1. Save sequence length at every step.
    2. Support for AdamW_lorafa optimizer.
    """

    def __init__(self, *args, lora_scale=2.0, **kwargs):
        print_rank_0(
            f"Initializing Seq2SeqTrainer_lorafa with lora_scale {lora_scale}."
        )
        super().__init__(*args, **kwargs)
        self.scaling_factor = lora_scale

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create optimizer and learning rate scheduler with LoRA-specific scaling.
        """
        param_groups = [
            {
                "params": self.model.parameters(),
                "lr": self.args.learning_rate,
                "names": [name for name, _ in self.model.named_parameters()],
                "scaling_factor": self.scaling_factor,
                "betas": (0.9, 0.999),
                "weight_decay": self.args.weight_decay,
            }
        ]

        self.optimizer = AdamW_lorafa(param_groups)
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )
