"""STEGO Correspondence Loss.

Ported from mhamilton723/STEGO. Implements the self-supervised
contrastive loss using KNN positive pair mining in DINO feature space.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


def compute_cosine_similarity(
    x: jnp.ndarray,
    y: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute pairwise cosine similarity.

    Args:
        x: Features of shape (N, D).
        y: Features of shape (M, D).
        eps: Epsilon for numerical stability.

    Returns:
        Cosine similarity matrix of shape (N, M).
    """
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)
    return x_norm @ y_norm.T


def find_knn_pairs(
    features: jnp.ndarray,
    k: int = 7,
) -> jnp.ndarray:
    """Find K nearest neighbors in feature space.

    Args:
        features: Feature vectors of shape (N, D).
        k: Number of nearest neighbors.

    Returns:
        KNN indices of shape (N, k).
    """
    sim = compute_cosine_similarity(features, features)
    # Exclude self-similarity by setting diagonal to -inf
    sim = sim - jnp.eye(sim.shape[0]) * 1e9
    # Get top-k indices
    _, indices = jax.lax.top_k(sim, k)
    return indices


def stego_loss(
    semantic_codes: jnp.ndarray,
    dino_features: jnp.ndarray,
    temperature: float = 0.1,
    knn_k: int = 7,
    num_negatives: int = 64,
) -> jnp.ndarray:
    """Compute STEGO correspondence loss (InfoNCE).

    Uses KNN in DINO feature space to find positive pairs,
    with random negatives. Encourages semantic codes to be
    similar for DINO-similar patches.

    Args:
        semantic_codes: Semantic codes of shape (B, N, D_s).
        dino_features: DINO features of shape (B, N, D_f).
        temperature: Temperature for InfoNCE loss.
        knn_k: Number of nearest neighbors for positive pairs.
        num_negatives: Number of negative samples.

    Returns:
        Scalar STEGO loss.
    """
    b, n, d_s = semantic_codes.shape

    total_loss = 0.0

    for i in range(b):
        codes_i = semantic_codes[i]  # (N, D_s)
        feats_i = dino_features[i]  # (N, D_f)

        # L2 normalize semantic codes
        codes_norm = codes_i / (
            jnp.linalg.norm(codes_i, axis=-1, keepdims=True) + 1e-8
        )

        # Find KNN positive pairs in DINO space
        knn_idx = find_knn_pairs(feats_i, k=knn_k)  # (N, k)

        # Compute positive similarities
        # For each anchor i, its positive is a random KNN neighbor
        # Use first neighbor for simplicity (most similar)
        pos_idx = knn_idx[:, 0]  # (N,)
        pos_codes = codes_norm[pos_idx]  # (N, D_s)

        # Positive similarity
        pos_sim = jnp.sum(codes_norm * pos_codes, axis=-1) / temperature  # (N,)

        # Negative similarities: random subset
        # Use jax.random for reproducibility
        neg_idx = jnp.arange(n)
        neg_codes = codes_norm  # All codes as potential negatives
        all_sim = (codes_norm @ neg_codes.T) / temperature  # (N, N)

        # InfoNCE: log(exp(pos) / sum(exp(all)))
        log_sum_exp = jax.nn.logsumexp(all_sim, axis=-1)
        loss_i = -pos_sim + log_sum_exp

        total_loss = total_loss + jnp.mean(loss_i)

    return total_loss / b


def depth_guided_correlation_loss(
    semantic_codes: jnp.ndarray,
    depth: jnp.ndarray,
    sigma_d: float = 0.5,
    num_pairs: int = 1024,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """Compute DepthG depth-guided feature correlation loss.

    Weights feature similarity by depth proximity:
    L_DepthG = Σ w_ij * (1 - cos(s_i, s_j))²
    where w_ij = exp(-|D_i - D_j|² / 2σ²)

    Args:
        semantic_codes: Semantic codes of shape (B, N, D_s).
        depth: Flattened depth values of shape (B, N).
        sigma_d: Depth similarity bandwidth.
        num_pairs: Number of random pixel pairs to sample.
        key: PRNG key for random sampling.

    Returns:
        Scalar depth correlation loss.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    b, n, d = semantic_codes.shape

    total_loss = 0.0

    for batch_idx in range(b):
        codes = semantic_codes[batch_idx]  # (N, D)
        d_vals = depth[batch_idx]  # (N,)

        # Sample random pairs
        k1, k2 = jax.random.split(key)
        idx_i = jax.random.randint(k1, (num_pairs,), 0, n)
        idx_j = jax.random.randint(k2, (num_pairs,), 0, n)

        # Depth weights
        depth_diff = d_vals[idx_i] - d_vals[idx_j]
        w_ij = jnp.exp(-depth_diff**2 / (2 * sigma_d**2))

        # Cosine similarity
        codes_i = codes[idx_i]
        codes_j = codes[idx_j]
        cos_sim = jnp.sum(codes_i * codes_j, axis=-1) / (
            jnp.linalg.norm(codes_i, axis=-1)
            * jnp.linalg.norm(codes_j, axis=-1)
            + 1e-8
        )

        # Loss: w * (1 - cos)²
        loss = w_ij * (1.0 - cos_sim) ** 2
        total_loss = total_loss + jnp.mean(loss)

    return total_loss / b
