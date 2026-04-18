"""Lovász-Softmax loss (Berman et al. 2018, CVPR).

Differentiable surrogate for the Jaccard (IoU) index, used in Pass 1 of
the Stage-2 loss-augmentation plan to directly optimise per-class IoU.
Ported from the reference implementation at
https://github.com/bermanmaxim/LovaszSoftmax.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.nn import functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension for the Jaccard index."""
    p = gt_sorted.sum()
    intersection = p - gt_sorted.cumsum(0)
    union = p + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-12)
    # Discrete-derivative form required by the Lovász extension.
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    """Lovász-Softmax on flattened (N, C) probabilities and (N,) labels."""
    if probas.numel() == 0:
        return probas.sum() * 0.0

    C = probas.size(1)
    if classes == "present":
        present = torch.unique(labels)
        class_iter = present.tolist()
    else:
        class_iter = list(range(C))

    losses = []
    for c in class_iter:
        fg = (labels == c).to(probas.dtype)
        if fg.sum() == 0:
            continue
        class_pred = probas[:, int(c)]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx: Dict[str, Any],
) -> torch.Tensor:
    """Compute Lovász-Softmax loss on multi-class logits.

    Args:
        logits: (B, C, H, W) unnormalised scores.
        targets: (B, H, W) integer class indices; ``ctx['ignore_index']``
            pixels are masked out before the loss is computed.
        ctx: dict accepting ``ignore_index`` (default 255) and optional
            ``classes`` (``"present"`` or ``"all"``; default ``"present"``
            matches the Berman reference behaviour).

    Returns:
        Scalar tensor with gradient flowing back to ``logits``.
    """
    ignore = int(ctx.get("ignore_index", 255))
    classes = str(ctx.get("classes", "present"))

    probas = F.softmax(logits, dim=1)
    _, C, _, _ = probas.shape
    probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = targets.reshape(-1)
    valid = labels_flat != ignore
    if not valid.any():
        return probas.sum() * 0.0
    return _lovasz_softmax_flat(probas_flat[valid], labels_flat[valid], classes=classes)


__all__ = ["lovasz_softmax"]
