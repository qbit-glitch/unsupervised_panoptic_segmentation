"""N5 Self-training: confidence-thresholded pseudo-label filter."""
from __future__ import annotations

from typing import Dict

import torch

__all__ = ["filter_by_confidence"]


def filter_by_confidence(pseudo: Dict[str, torch.Tensor], threshold: float = 0.95) -> Dict[str, torch.Tensor]:
    scores = pseudo["scores"]
    masks = pseudo["masks"]
    keep = scores >= threshold
    # Preserve input dtype + trailing shape in the empty-fallback so
    # downstream torch.cat across batches does not break when a caller
    # passes soft (float) masks or higher-dim mask tensors.
    empty_masks = torch.zeros(0, *masks.shape[1:], dtype=masks.dtype, device=masks.device)
    return {
        "labels": pseudo["labels"][keep],
        "masks": masks[keep] if keep.any() else empty_masks,
        "scores": pseudo["scores"][keep],
    }
