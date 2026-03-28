"""Semantic loss for MBPS v2: cross-entropy against pseudo-labels.

Replaces the STEGO + DepthG correlation losses from v1 with simple
supervised cross-entropy, trained on pseudo-labels from DINOv3 K-means.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def semantic_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = 255,
    label_smoothing: float = 0.1,
) -> jnp.ndarray:
    """Per-token cross-entropy loss against pseudo semantic labels.

    Args:
        logits: (B, N, K) unnormalized class logits.
        labels: (B, N) integer pseudo-labels in [0, K-1] or ignore_index.
        ignore_index: Label value to ignore (typically 255).
        label_smoothing: Label smoothing factor (0 = no smoothing).

    Returns:
        Scalar loss (mean over valid tokens).
    """
    B, N, K = logits.shape

    # Create valid mask
    valid = labels != ignore_index  # (B, N)
    n_valid = jnp.maximum(valid.sum(), 1.0)

    # Clip labels to valid range for one_hot (invalid labels get masked out)
    safe_labels = jnp.where(valid, labels, 0)

    # One-hot encode
    targets = jax.nn.one_hot(safe_labels, K)  # (B, N, K)

    # Label smoothing
    if label_smoothing > 0:
        targets = (1.0 - label_smoothing) * targets + label_smoothing / K

    # Log-softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, N, K)

    # Cross-entropy
    loss_per_token = -jnp.sum(targets * log_probs, axis=-1)  # (B, N)

    # Mask invalid tokens
    loss_per_token = jnp.where(valid, loss_per_token, 0.0)

    return loss_per_token.sum() / n_valid
