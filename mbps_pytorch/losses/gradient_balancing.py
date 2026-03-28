"""Gradient Balancing for Multi-Task Learning.

Implements PCGrad-style gradient projection to resolve conflicts
between semantic and instance gradients.

g_inst_proj = g_inst - (min(0, <g_inst, g_sem>) / ||g_sem||^2 ) * g_sem
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def project_conflicting_gradients(
    grad_primary: torch.Tensor,
    grad_secondary: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
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
    dot = torch.sum(grad_primary * grad_secondary)
    conflict = torch.minimum(dot, torch.tensor(0.0, device=dot.device))
    primary_norm_sq = torch.sum(grad_primary ** 2) + eps
    projected = grad_secondary - (conflict / primary_norm_sq) * grad_primary
    return projected


class GradientBalancer:
    """Multi-task gradient balancing with PCGrad-style projection.

    Resolves gradient conflicts between semantic and instance tasks
    by projecting the instance gradient to be non-conflicting.

    In PyTorch, gradient manipulation works on parameter `.grad` attributes
    rather than JAX's functional gradient pytrees.

    Args:
        beta_start: Starting weight for instance gradient ramp-up.
        beta_end: Ending weight for instance gradient ramp-up.
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
        grad_semantic: Dict[str, torch.Tensor],
        grad_instance: Dict[str, torch.Tensor],
        beta: float,
    ) -> Dict[str, torch.Tensor]:
        """Balance semantic and instance gradients.

        Operates on dictionaries mapping parameter names to gradient tensors.
        This replaces JAX's pytree-based gradient manipulation.

        Args:
            grad_semantic: Dict mapping param names to semantic loss gradients.
            grad_instance: Dict mapping param names to instance loss gradients.
            beta: Instance gradient weight.

        Returns:
            Combined gradient dict.
        """
        combined = {}
        for name in set(grad_semantic.keys()) | set(grad_instance.keys()):
            g_sem = grad_semantic.get(name)
            g_inst = grad_instance.get(name)

            if g_sem is None or g_inst is None:
                combined[name] = g_sem if g_sem is not None else g_inst
                continue

            # Flatten for projection
            flat_sem = g_sem.reshape(-1)
            flat_inst = g_inst.reshape(-1)

            # Project instance gradient
            flat_proj = project_conflicting_gradients(flat_sem, flat_inst)

            # Reshape
            projected = flat_proj.reshape(g_inst.shape)

            # Combine
            combined[name] = g_sem + beta * projected

        return combined

    def balance_gradients_inplace(
        self,
        model: nn.Module,
        loss_semantic: torch.Tensor,
        loss_instance: torch.Tensor,
        beta: float,
    ) -> None:
        """Balance gradients in-place on model parameters.

        Convenience method that computes semantic and instance gradients
        via backward passes and applies PCGrad projection directly to
        the model's `.grad` attributes.

        Args:
            model: The model whose parameter gradients to balance.
            loss_semantic: Scalar semantic loss tensor.
            loss_instance: Scalar instance loss tensor.
            beta: Instance gradient weight.
        """
        # Compute semantic gradients
        model.zero_grad()
        loss_semantic.backward(retain_graph=True)
        grads_semantic = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        # Compute instance gradients
        model.zero_grad()
        loss_instance.backward(retain_graph=True)
        grads_instance = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        # Balance
        combined = self.balance_gradients(grads_semantic, grads_instance, beta)

        # Write combined gradients back to parameters
        model.zero_grad()
        for name, p in model.named_parameters():
            if name in combined:
                p.grad = combined[name]
