"""Boundary-weighted cross-entropy (P1 aux-loss).

The CE loss is computed per-pixel and up-weighted inside a small band
around label boundaries. The band is derived from the pseudo-labels
themselves by detecting pixels whose dilated neighbourhood contains
more than one class. ``boundary_ce_mult`` controls how aggressively the
boundary pixels are emphasised; values in 2x-4x are typical.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.nn import functional as F


def compute_boundary_mask(targets: torch.Tensor, dilate_px: int = 3) -> torch.Tensor:
    """Dilated label-change detector.

    A pixel is flagged as boundary if the max and min labels inside its
    ``(2*dilate_px+1)`` neighbourhood disagree — i.e. at least one
    neighbour carries a different class. Ignore pixels (value 255) are
    treated as legitimate labels so the caller should mask them out
    downstream if that is undesirable.

    Args:
        targets: (B, H, W) integer label map.
        dilate_px: half-width of the neighbourhood; ``3`` gives a 7-px
            band.

    Returns:
        (B, H, W) bool tensor (``True`` inside the boundary band).
    """
    t = targets.unsqueeze(1).float()
    k = 2 * dilate_px + 1
    max_pool = F.max_pool2d(t, kernel_size=k, stride=1, padding=dilate_px)
    # Min-pool via the max-of-negative trick.
    min_pool = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=dilate_px)
    mask = (max_pool != min_pool).squeeze(1)
    return mask


def boundary_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx: Dict[str, Any],
) -> torch.Tensor:
    """Cross-entropy with an extra multiplier on boundary pixels.

    The base CE uses ``ctx['class_weight']`` and ``ctx['ignore_index']``
    so it lines up with the main ``loss_sem_seg`` term. The multiplier
    is applied multiplicatively: pixels inside the boundary band receive
    weight ``boundary_ce_mult``, all other valid pixels receive weight
    ``1``.
    """
    params = ctx.get("params", {})
    ignore = int(ctx.get("ignore_index", 255))
    dilate = int(params.get("boundary_dilate_px", 3))
    mult = float(params.get("boundary_ce_mult", 2.0))
    class_weight = ctx.get("class_weight", None)
    if isinstance(class_weight, (list, tuple)):
        class_weight = torch.tensor(class_weight, device=logits.device, dtype=logits.dtype)

    ce_px = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        ignore_index=ignore,
        weight=class_weight,
    )  # (B, H, W)

    mask = compute_boundary_mask(targets, dilate_px=dilate).to(logits.dtype)
    pixel_weights = 1.0 + (mult - 1.0) * mask
    valid = (targets != ignore).to(logits.dtype)
    weighted = ce_px * pixel_weights * valid
    denom = valid.sum().clamp(min=1.0)
    return weighted.sum() / denom


__all__ = ["boundary_ce", "compute_boundary_mask"]
