"""Gradient Balancing for Multi-Task Learning.

Implements PCGrad-style gradient projection to resolve conflicts
between semantic and instance gradients.

g_inst_proj = g_inst - (min(0, <g_inst, g_sem>) / ||g_sem||² ) · g_sem
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import optax


def project_conflicting_gradients(
    grad_primary: jnp.ndarray,
    grad_secondary: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Project secondary gradient to remove conflict with primary.

    If the secondary gradient conflicts with the primary (negative
    dot product), project it onto the normal plane of primary.

    Args:
        grad_primary: Primary gradient (e.g., semantic).
        grad_secondary: Secondary gradient (e.g., instance).
        eps: Numerical stability.

    Returns:
        Projected secondary gradient.
    """
    dot = jnp.sum(grad_primary * grad_secondary)
    conflict = jnp.minimum(dot, 0.0)
    primary_norm_sq = jnp.sum(grad_primary ** 2) + eps
    projected = grad_secondary - (conflict / primary_norm_sq) * grad_primary
    return projected


class GradientBalancer:
    """Multi-task gradient balancing with PCGrad-style projection.

    Resolves gradient conflicts between semantic and instance tasks
    by projecting the instance gradient to be non-conflicting.

    Args:
        beta_schedule: Weight schedule for instance gradient ramp-up.
    """

    def __init__(self, beta_start: float = 0.0, beta_end: float = 1.0):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get_beta(self, epoch: int, phase_b_start: int, phase_b_end: int) -> float:
        """Get instance loss weight based on curriculum.

        Args:
            epoch: Current epoch.
            phase_b_start: Phase B start epoch.
            phase_b_end: Phase B end epoch.

        Returns:
            Beta weight for instance gradient.
        """
        if epoch < phase_b_start:
            return 0.0
        elif epoch >= phase_b_end:
            return self.beta_end

        # Linear ramp-up during Phase B
        progress = (epoch - phase_b_start) / (phase_b_end - phase_b_start)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def balance_gradients(
        self,
        grad_semantic: dict,
        grad_instance: dict,
        beta: float,
    ) -> dict:
        """Balance semantic and instance gradients.

        Args:
            grad_semantic: Gradient pytree from semantic loss.
            grad_instance: Gradient pytree from instance loss.
            beta: Instance gradient weight.

        Returns:
            Combined gradient pytree.
        """

        def _combine(g_sem, g_inst):
            if g_sem is None or g_inst is None:
                return g_sem if g_sem is not None else g_inst

            # Flatten for projection
            flat_sem = jnp.ravel(g_sem)
            flat_inst = jnp.ravel(g_inst)

            # Project instance gradient
            flat_proj = project_conflicting_gradients(flat_sem, flat_inst)

            # Reshape
            projected = jnp.reshape(flat_proj, g_inst.shape)

            # Combine
            return g_sem + beta * projected

        return jax.tree.map(_combine, grad_semantic, grad_instance)
