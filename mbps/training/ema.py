"""Exponential Moving Average (EMA) for Teacher Model.

Maintains an EMA copy of model parameters for:
    1. PQ proxy loss computation (teacher predictions)
    2. Self-training pseudo-label generation
    3. Training stability

θ_ema ← μ · θ_ema + (1-μ) · θ   where μ = 0.999
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class EMAState:
    """EMA state container.

    Args:
        params: Model parameters (pytree).
        momentum: EMA momentum (0.999).
        step: Current step count.
    """

    def __init__(
        self,
        params: Any,
        momentum: float = 0.999,
    ):
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)
        self.momentum = momentum
        self.step = 0

    def update(self, new_params: Any) -> None:
        """Update EMA parameters.

        θ_ema ← μ · θ_ema + (1-μ) · θ

        Args:
            new_params: Current model parameters.
        """
        self.ema_params = jax.tree.map(
            lambda ema, new: self.momentum * ema + (1 - self.momentum) * new,
            self.ema_params,
            new_params,
        )
        self.step += 1

    def get_params(self) -> Any:
        """Get current EMA parameters.

        Returns:
            EMA parameter pytree.
        """
        return self.ema_params

    def apply_bias_correction(self) -> Any:
        """Apply bias correction for early steps.

        Corrects for initialization bias: θ_corrected = θ_ema / (1 - μ^step)

        Returns:
            Bias-corrected EMA parameters.
        """
        correction = 1.0 / (1.0 - self.momentum ** (self.step + 1))
        return jax.tree.map(lambda x: x * correction, self.ema_params)


def create_ema(
    params: Any,
    momentum: float = 0.999,
) -> EMAState:
    """Create an EMA state from model parameters.

    Args:
        params: Initial model parameters.
        momentum: EMA decay rate.

    Returns:
        EMAState instance.
    """
    return EMAState(params, momentum)
