from typing import Tuple, Union

import numpy as np
import torch

from typing import Callable, Iterable, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor
from torch._utils import is_compiling
from torch.cuda.amp import autocast

import math


class AdamW_lorafa(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    def is_same(self, name_list):
        return name_list[0].split(".")[:-3] == name_list[1].split(".")[:-3]

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            scaling_factor = group["scaling_factor"]
            param_list = []
            name_list = []
            for p, n in zip(group["params"], group["names"]):
                # for p in group["params"]:
                # Skip non-lora no-grad module, since we need lora_A which is no-grad.
                if "lora" not in n and p.grad is None:
                    continue
                grad = p.grad
                # if grad.is_sparse:
                #     raise RuntimeError(
                #         "Adam does not support sparse gradients, please consider SparseAdam instead"
                #     )

                if "lora" in n:
                    param_list.append(p)
                    name_list.append(n)
                    if len(param_list) == 2:
                        name = n[: n.find("lora")] + "lora"
                        # assert self.is_same(name_list)
                    elif len(param_list) == 1:
                        continue
                else:
                    name = n

                # param_list -> [A,B]

                state = self.state[name]
                # State initialization
                if len(state) == 0:
                    if len(param_list) == 2:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg_B"] = torch.zeros_like(param_list[1])
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq_B"] = torch.zeros_like(param_list[1])
                    else:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                if len(param_list) == 2:
                    A = param_list[0]
                    B = param_list[1]
                    # grad_A_orin = A.grad
                    grad_B_orin = B.grad

                    # projection
                    delta = 1e-8

                    # computing the inverse matrix
                    AA_T = A @ A.T
                    # B_TB = B.T @ B
                    AA_T_inv = torch.linalg.pinv(
                        AA_T + delta * torch.eye(A.shape[0]).to(A.device)
                    )
                    # B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device))

                    with autocast(dtype=torch.bfloat16):
                        grad_B = (1 / scaling_factor**2) * (grad_B_orin @ AA_T_inv)
                    if grad_B.dtype != B.grad.dtype:
                        grad_B = grad_B.to(B.grad.dtype)

                    exp_avg_B, exp_avg_sq_B = state["exp_avg_B"], state["exp_avg_sq_B"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    exp_avg_B.mul_(beta1).add_(grad_B, alpha=(1.0 - beta1))
                    exp_avg_sq_B.mul_(beta2).addcmul_(grad_B, grad_B, value=1.0 - beta2)

                    denom_B = exp_avg_sq_B.sqrt().add_(group["eps"])
                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )
                    B.addcdiv_(exp_avg_B, denom_B, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        B.add_(B, alpha=(-group["lr"] * group["weight_decay"]))
                    param_list = []
                    name_list = []
                else:
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )

                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    # Just adding the square of the weights to the loss function is *not*
                    # the correct way of using L2 regularization/weight decay with Adam,
                    # since that will interact with the m and v parameters in strange ways.
                    #
                    # Instead we want to decay the weights in a manner that doesn't interact
                    # with the m/v parameters. This is equivalent to adding the square
                    # of the weights to the loss with plain (non-momentum) SGD.
                    # Add weight decay at the end (fixed version)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
