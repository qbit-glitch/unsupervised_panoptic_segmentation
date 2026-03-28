"""v2 Semantic Loss: Cross-entropy with label smoothing.

Replaces v1's STEGO + DepthG unsupervised loss with supervised
cross-entropy against pseudo-labels from K-means clustering.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def semantic_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy loss with label smoothing for pseudo-labels.

    Args:
        logits: (B, N, K) class logits from semantic head.
        labels: (B, N) int pseudo-labels from K-means.
        ignore_index: Label value to ignore (unlabeled pixels).
        label_smoothing: Label smoothing factor.

    Returns:
        Scalar loss.
    """
    b, n, k = logits.shape

    # Flatten for cross_entropy
    logits_flat = logits.reshape(-1, k)  # (B*N, K)
    labels_flat = labels.reshape(-1).long()  # (B*N,)

    # Mask invalid tokens
    valid_mask = labels_flat != ignore_index

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Cross-entropy with label smoothing
    loss = F.cross_entropy(
        logits_flat[valid_mask],
        labels_flat[valid_mask],
        label_smoothing=label_smoothing,
        reduction="mean",
    )

    return loss
