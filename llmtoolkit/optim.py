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


def _dispatch_sqrt(
    x: float,
):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)


def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        return x.item() if isinstance(x, torch.Tensor) else x


def _get_scalar_dtype():
    return (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )


class AdamW_lorafa(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        # todo: assert all lora_A is untrainable, and all lora_B is trainable.
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.step_ = 0
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def is_same(self, name_list):
        return name_list[0].split(".")[:-3] == name_list[1].split(".")[:-3]

    def step_efficient(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            scaling_factor = group["scaling_factor"]

            param_list = []
            name_list = []
            for p, n in zip(group["params"], group["names"]):
                if "lora" not in n and p.grad is None:
                    continue

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

                state = self.state[name]

                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    if len(param_list) == 2:
                        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                        # Exponential moving average of gradient values
                        # state["exp_avg_A"] = torch.zeros(param_list[0].shape).to(p.device).to(p.dtype)
                        state["exp_avg_B"] = (
                            torch.zeros(param_list[1].shape).to(p.device).to(p.dtype)
                        )

                        # Exponential moving average of squared gradient values
                        # state["exp_avg_sq_A"] = torch.zeros(param_list[0].shape).to(p.device).to(p.dtype)
                        state["exp_avg_sq_B"] = (
                            torch.zeros(param_list[1].shape).to(p.device).to(p.dtype)
                        )

                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            # state["max_exp_avg_sq_A"] = torch.zeros(param_list[0].shape).to(p.device).to(p.dtype)
                            state["max_exp_avg_sq_B"] = (
                                torch.zeros(param_list[1].shape)
                                .to(p.device)
                                .to(p.dtype)
                            )
                    else:
                        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros(p.shape).to(p.device).to(p.dtype)

                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = (
                            torch.zeros(p.shape).to(p.device).to(p.dtype)
                        )

                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = (
                                torch.zeros(p.shape).to(p.device).to(p.dtype)
                            )

                # step
                if len(param_list) == 2:
                    # note: param_list = [A, B]
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
                    ).to(torch.bfloat16)
                    # B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device))

                    # grad_A = (1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin + X @ A
                    # grad_B = (1 / scaling_factor ** 2) * ((torch.eye(B.shape[0]).to(B.device) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv) - B @ X

                    with autocast(dtype=torch.float32):
                        grad_B = (1 / scaling_factor**2) * (grad_B_orin @ AA_T_inv)
                    grad_B = grad_B.to(B.grad.dtype)

                    # exp_avg_A = state["exp_avg_A"]
                    # exp_avg_sq_A = state["exp_avg_sq_A"]

                    exp_avg_B = state["exp_avg_B"]
                    exp_avg_sq_B = state["exp_avg_sq_B"]

                    step_t = state["step"]

                    step_t += 1

                    # exp_avg_A.lerp_(grad_A, 1 - beta1)
                    exp_avg_B.lerp_(grad_B, 1 - beta1)
                    # exp_avg_sq_A.mul_(beta2).addcmul_(grad_A, grad_A.conj(), value=1 - beta2)
                    exp_avg_sq_B.mul_(beta2).addcmul_(
                        grad_B, grad_B.conj(), value=1 - beta2
                    )

                    step = _get_value(step_t)

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    step_size = group["lr"]

                    bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

                    if group["amsgrad"]:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        # torch.maximum(state["max_exp_avg_sq_A"], exp_avg_sq_A, out=state["max_exp_avg_sq_A"])
                        torch.maximum(
                            state["max_exp_avg_sq_B"],
                            exp_avg_sq_B,
                            out=state["max_exp_avg_sq_B"],
                        )

                        # Use the max. for normalizing running avg. of gradient
                        # denom_A = (state["max_exp_avg_sq_A"].sqrt() / bias_correction2_sqrt).add_(group['eps'])
                        denom_B = (
                            state["max_exp_avg_sq_B"].sqrt() / bias_correction2_sqrt
                        ).add_(group["eps"])
                    else:
                        # denom_A = (exp_avg_sq_A.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                        denom_B = (exp_avg_sq_B.sqrt() / bias_correction2_sqrt).add_(
                            group["eps"]
                        )

                    if group["weight_decay"] != 0:
                        # A.mul_(1 - group["weight_decay"] * group["lr"])
                        B.mul_(1 - group["weight_decay"] * group["lr"])

                    # A.addcdiv_(exp_avg_A / bias_correction1, denom_A, value=-step_size)
                    B.addcdiv_(exp_avg_B / bias_correction1, denom_B, value=-step_size)
                    param_list = []
                    name_list = []
                else:
                    grad = p.grad
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step_t = state["step"]

                    step_t += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    step = _get_value(step_t)

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    step_size = group["lr"]

                    bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(
                        group["eps"]
                    )
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["weight_decay"] * group["lr"])

                    p.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_efficient()

        return loss
