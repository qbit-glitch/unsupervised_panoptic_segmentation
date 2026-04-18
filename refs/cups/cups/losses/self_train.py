"""N5 Self-training: confidence-thresholded pseudo-label filter."""
from __future__ import annotations

from typing import Dict

import torch

__all__ = ["filter_by_confidence"]


def filter_by_confidence(pseudo: Dict[str, torch.Tensor], threshold: float = 0.95) -> Dict[str, torch.Tensor]:
    scores = pseudo["scores"]
    keep = scores >= threshold
    H = pseudo["masks"].shape[-2]
    W = pseudo["masks"].shape[-1]
    return {
        "labels": pseudo["labels"][keep],
        "masks": pseudo["masks"][keep] if keep.any() else torch.zeros(0, H, W, dtype=torch.bool, device=pseudo["masks"].device),
        "scores": pseudo["scores"][keep],
    }
