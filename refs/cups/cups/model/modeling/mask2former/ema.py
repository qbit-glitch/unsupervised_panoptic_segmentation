"""Exponential-moving-average wrapper for teacher models (G1, N4, Stage-3)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["EMAModel"]


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite model parameters with shadow copy (used at eval / Stage-3)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])
