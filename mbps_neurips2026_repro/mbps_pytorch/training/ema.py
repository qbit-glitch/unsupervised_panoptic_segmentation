"""Exponential Moving Average (EMA) for Teacher Model.

Maintains an EMA copy of model parameters for:
    1. PQ proxy loss computation (teacher predictions)
    2. Self-training pseudo-label generation
    3. Training stability

theta_ema <- mu * theta_ema + (1-mu) * theta   where mu = 0.999
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn


class EMAState:
    """EMA state container that shadows a PyTorch ``nn.Module``.

    The EMA parameters are stored as a ``state_dict`` (an ``OrderedDict``
    of tensors).  Updates are performed in-place under ``torch.no_grad()``
    to avoid tracking in the autograd graph.

    Args:
        model: The PyTorch model whose parameters should be shadowed.
        momentum: EMA momentum (0.999).
    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.999,
    ):
        self.momentum = momentum
        self.step = 0
        # Deep-copy the state dict so EMA params are independent
        self.ema_state_dict: Dict[str, torch.Tensor] = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters.

        theta_ema <- mu * theta_ema + (1-mu) * theta

        Args:
            model: The model with current (trained) parameters.
        """
        for key, param in model.state_dict().items():
            if key in self.ema_state_dict:
                self.ema_state_dict[key].mul_(self.momentum).add_(
                    param.detach(), alpha=1.0 - self.momentum
                )
        self.step += 1

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get current EMA parameters as a state dict.

        Returns:
            EMA parameter state dict.
        """
        return self.ema_state_dict

    def apply_to_model(self, model: nn.Module) -> None:
        """Load the EMA parameters into a model (in-place).

        This is useful for evaluation or pseudo-label generation using
        the EMA teacher.

        Args:
            model: Target model to receive EMA parameters.
        """
        model.load_state_dict(self.ema_state_dict, strict=False)

    @torch.no_grad()
    def apply_bias_correction(self) -> Dict[str, torch.Tensor]:
        """Apply bias correction for early steps.

        Corrects for initialization bias:
            theta_corrected = theta_ema / (1 - mu^step)

        Returns:
            Bias-corrected EMA state dict.
        """
        correction = 1.0 / (1.0 - self.momentum ** (self.step + 1))
        return {k: v * correction for k, v in self.ema_state_dict.items()}


def create_ema(
    model: nn.Module,
    momentum: float = 0.999,
) -> EMAState:
    """Create an EMA state from a PyTorch model.

    Args:
        model: Initial model whose parameters are copied.
        momentum: EMA decay rate.

    Returns:
        EMAState instance.
    """
    return EMAState(model, momentum)
