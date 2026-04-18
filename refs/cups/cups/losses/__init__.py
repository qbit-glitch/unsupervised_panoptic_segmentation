"""Stage-2 auxiliary semantic-head losses (P1-P4).

Each loss is a free function with a uniform signature:

    loss(logits, targets, ctx: dict) -> Tensor

where ``ctx`` may contain optional tensors (``depth``, ``dino_features``,
``rgb``, ``class_weight``, ``ignore_index``, ``params``) used by specific
losses.  Losses silently accept missing optional keys; required keys raise
``KeyError``.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch

__all__ = ["build_aux_losses"]


def build_aux_losses() -> Dict[str, Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]]:
    """Return a dict {name: loss_fn} for all P1-P4 aux losses."""
    from .boundary import boundary_ce
    from .dense_affinity import gated_crf, neco
    from .depth_smoothness import depth_smoothness
    from .lovasz import lovasz_softmax
    from .stego_adapter import stego_corr

    return {
        "lovasz_softmax": lovasz_softmax,
        "boundary_ce": boundary_ce,
        "stego_corr": stego_corr,
        "depth_smoothness": depth_smoothness,
        "gated_crf": gated_crf,
        "neco": neco,
    }
