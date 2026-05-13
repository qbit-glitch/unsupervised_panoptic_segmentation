"""Confidence-based sample filtering inspired by LoRA-TTT (ICML 2025).

Ported from: https://github.com/ykojima4020/LoRA-TTT
Reference: "Low-Rank Test-Time Training for Vision-Language Models"
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def select_confident_samples(
    logits: torch.Tensor,
    selection_p: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-p fraction of samples with lowest entropy (highest confidence).

    Args:
        logits: (N, C) unnormalized logits.
        selection_p: Fraction of samples to keep (0 < p <= 1.0).

    Returns:
        selected_logits: (M, C) where M = int(N * selection_p)
        selected_idx: (M,) indices of selected samples in original batch.
    """
    if selection_p >= 1.0:
        return logits, torch.arange(logits.size(0), device=logits.device)

    # Per-sample entropy: H = -sum(p * log(p))
    probs = logits.softmax(dim=1)
    log_probs = logits.log_softmax(dim=1)
    batch_entropy = -(probs * log_probs).sum(dim=1)  # (N,)

    # Lower entropy = higher confidence
    k = max(1, int(batch_entropy.size(0) * selection_p))
    idx = torch.argsort(batch_entropy, descending=False)[:k]
    return logits[idx], idx


def avg_entropy(outputs: torch.Tensor) -> torch.Tensor:
    """Marginal entropy minimization: entropy of the averaged log-probability.

    Args:
        outputs: (M, C) logits from confident samples.

    Returns:
        Scalar entropy of the mean log-probability distribution.
    """
    # logits = outputs.log_softmax(dim=1) is numerically unstable; use logsumexp
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # (M, C)
    avg_logits = logits.logsumexp(dim=0) - math.log(logits.shape[0])  # (C,)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def weighted_entropy(
    logits: torch.Tensor,
    scaling_factor: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample entropy and confidence-based weighting coefficient.

    Args:
        logits: (M, C) logits.
        scaling_factor: Entropy offset for weight calculation.

    Returns:
        coefficient: (M,) weight per sample = 1 / exp(entropy - scaling_factor)
        entropy: (M,) per-sample entropy values.
    """
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    min_val = torch.finfo(log_probs.dtype).min
    log_probs = torch.clamp(log_probs, min=min_val)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=1)

    # Prevent overflow
    max_entropy = 88.0
    entropy = torch.clamp(entropy, max=max_entropy)
    coefficient = 1.0 / torch.exp(entropy.clone().detach() - scaling_factor)
    return coefficient, entropy


def confidence_weighted_loss(
    loss_per_sample: torch.Tensor,
    logits: torch.Tensor,
    selection_p: float = 0.1,
    scaling_factor: float = 0.4,
) -> torch.Tensor:
    """Apply confidence-based weighting to a per-sample loss.

    Args:
        loss_per_sample: (N,) unreduced loss per sample.
        logits: (N, C) corresponding logits for confidence estimation.
        selection_p: Fraction of samples to consider.
        scaling_factor: Entropy offset for weighting.

    Returns:
        Scalar weighted loss.
    """
    selected_logits, selected_idx = select_confident_samples(logits, selection_p)
    selected_loss = loss_per_sample[selected_idx]
    coefficient, _ = weighted_entropy(selected_logits, scaling_factor)
    return (selected_loss * coefficient).mean()
