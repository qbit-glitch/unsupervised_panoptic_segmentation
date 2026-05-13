"""FRACAL post-hoc calibration for long-tail dead-class recovery.

Reference:
    Alexandridis et al., "FRACAL: Fractal Calibration for Long-tailed Object
    Detection", CVPR 2025. arXiv:2410.11774.

Idea: shift per-class logits by lambda * (mean_D - D_c), where D_c is the
fractal dimension of class c's spatial occupancy across the val set. Tail
classes (low D, compact regions) receive a positive shift; head classes
(high D, broadly distributed) receive a negative shift. Total shift is
zero-mean, so overall confidence stays calibrated.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def box_counting_dimension(mask: np.ndarray) -> float:
    """Compute the box-counting fractal dimension of a binary mask.

    Uses powers-of-two box sizes and fits log(N(s)) = -D * log(s) + c.

    Args:
        mask: 2-D boolean array (H, W).

    Returns:
        Estimated fractal dimension. Returns 0.0 for empty or near-empty masks.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if not mask.any():
        return 0.0

    n_pixels = int(mask.sum())
    if n_pixels < 2:
        # Single-pixel / degenerate masks have no fractal scaling
        return 0.0

    h, w = mask.shape
    max_size = min(h, w)

    # Use box sizes from 1 up to max_size // 4 in powers of two
    sizes: list[int] = []
    s = 1
    while s <= max(2, max_size // 4):
        sizes.append(s)
        s *= 2

    if len(sizes) < 2:
        return 0.0

    counts: list[int] = []
    for size in sizes:
        # Crop to integer multiple of `size`
        H = (h // size) * size
        W = (w // size) * size
        if H == 0 or W == 0:
            counts.append(0)
            continue
        cropped = mask[:H, :W]
        # Reshape into non-overlapping size x size blocks and OR-reduce
        blocks = cropped.reshape(H // size, size, W // size, size)
        block_occupancy = blocks.any(axis=(1, 3))
        counts.append(int(block_occupancy.sum()))

    # Drop any (size, count) pairs with zero count (log undefined)
    log_pairs = [
        (np.log(size), np.log(count))
        for size, count in zip(sizes, counts)
        if count > 0
    ]
    if len(log_pairs) < 2:
        return 0.0

    log_sizes = np.array([p[0] for p in log_pairs])
    log_counts = np.array([p[1] for p in log_pairs])

    # Linear regression: log_counts ~ slope * log_sizes + intercept
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return float(-slope)


def per_class_fractal_dim(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute fractal dimension per class from a stack of integer label maps.

    Args:
        labels: (N, H, W) integer tensor of class indices in [0, num_classes).
        num_classes: number of classes.

    Returns:
        (num_classes,) float tensor of estimated fractal dimensions. Classes
        absent from all images get 0.0.
    """
    if labels.dim() != 3:
        raise ValueError(f"labels must be (N, H, W), got {tuple(labels.shape)}")

    labels_np = labels.cpu().numpy()
    n_imgs, h, w = labels_np.shape

    out = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        # Aggregate all per-image binary masks for class c into a single mask
        # by OR-pooling across images. Per-image fractal dim averaging would
        # also be valid; OR-pooling is simpler and matches the FRACAL paper's
        # "global mask per class" formulation.
        union_mask = (labels_np == c).any(axis=0)  # (H, W)
        if not union_mask.any():
            out[c] = 0.0
            continue
        out[c] = box_counting_dimension(union_mask)

    return torch.from_numpy(out)


def fracal_calibrate(
    logits: torch.Tensor,
    per_class_d: torch.Tensor,
    lam: float = 1.0,
) -> torch.Tensor:
    """Apply FRACAL post-hoc logit shift.

    Per-class shift: shift_c = lam * (mean_D - D_c). Shifts sum to zero
    (when averaged uniformly across classes), so the calibration is
    confidence-preserving in expectation.

    Args:
        logits: (B, C, H, W) or (B, C) tensor of pre-softmax logits.
        per_class_d: (C,) fractal dimensions per class.
        lam: calibration strength (0.0 = no calibration, 1.0 = paper default).

    Returns:
        Same shape as `logits`, with per-channel shift applied.
    """
    if lam == 0.0:
        return logits

    if per_class_d.ndim != 1:
        raise ValueError(f"per_class_d must be 1-D, got shape {tuple(per_class_d.shape)}")

    num_classes = per_class_d.shape[0]
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"logits channel dim {logits.shape[1]} does not match "
            f"per_class_d size {num_classes}"
        )

    mean_d = per_class_d.mean()
    shifts = lam * (mean_d - per_class_d)  # (C,)

    # Broadcast per-channel shift across spatial dims
    shape = [1, num_classes] + [1] * (logits.dim() - 2)
    shifts = shifts.to(logits.device, dtype=logits.dtype).reshape(shape)
    return logits + shifts
