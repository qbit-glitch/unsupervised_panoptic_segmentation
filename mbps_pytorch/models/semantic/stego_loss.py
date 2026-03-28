"""STEGO Correspondence Loss.

Ported from mhamilton723/STEGO. Implements the self-supervised
contrastive loss using KNN positive pair mining in DINO feature space.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute pairwise cosine similarity.

    Args:
        x: Features of shape (N, D).
        y: Features of shape (M, D).
        eps: Epsilon for numerical stability.

    Returns:
        Cosine similarity matrix of shape (N, M).
    """
    x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + eps)
    y_norm = y / (torch.norm(y, dim=-1, keepdim=True) + eps)
    return x_norm @ y_norm.T


def find_knn_pairs(
    features: torch.Tensor,
    k: int = 7,
) -> torch.Tensor:
    """Find K nearest neighbors in feature space.

    Args:
        features: Feature vectors of shape (N, D).
        k: Number of nearest neighbors.

    Returns:
        KNN indices of shape (N, k).
    """
    sim = compute_cosine_similarity(features, features)
    # Exclude self-similarity by setting diagonal to -inf
    sim = sim - torch.eye(sim.shape[0], device=sim.device) * 1e9
    # Get top-k indices
    _, indices = torch.topk(sim, k, dim=-1)
    return indices


def stego_loss(
    semantic_codes: torch.Tensor,
    dino_features: torch.Tensor,
    temperature: float = 0.1,
    knn_k: int = 7,
    num_negatives: int = 64,
) -> torch.Tensor:
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

    total_loss = torch.tensor(0.0, device=semantic_codes.device)

    for i in range(b):
        codes_i = semantic_codes[i]  # (N, D_s)
        feats_i = dino_features[i]  # (N, D_f)

        # L2 normalize semantic codes
        codes_norm = codes_i / (
            torch.norm(codes_i, dim=-1, keepdim=True) + 1e-8
        )

        # Find KNN positive pairs in DINO space
        knn_idx = find_knn_pairs(feats_i, k=knn_k)  # (N, k)

        # Compute positive similarities
        # For each anchor i, its positive is a random KNN neighbor
        # Use first neighbor for simplicity (most similar)
        pos_idx = knn_idx[:, 0]  # (N,)
        pos_codes = codes_norm[pos_idx]  # (N, D_s)

        # Positive similarity
        pos_sim = torch.sum(codes_norm * pos_codes, dim=-1) / temperature  # (N,)

        # Negative similarities: all codes as potential negatives
        all_sim = (codes_norm @ codes_norm.T) / temperature  # (N, N)

        # InfoNCE: log(exp(pos) / sum(exp(all)))
        log_sum_exp = torch.logsumexp(all_sim, dim=-1)
        loss_i = -pos_sim + log_sum_exp

        total_loss = total_loss + torch.mean(loss_i)

    return total_loss / b


def depth_guided_correlation_loss(
    semantic_codes: torch.Tensor,
    depth: torch.Tensor,
    sigma_d: float = 0.5,
    num_pairs: int = 1024,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Compute DepthG depth-guided feature correlation loss.

    Weights feature similarity by depth proximity:
    L_DepthG = sum w_ij * (1 - cos(s_i, s_j))^2
    where w_ij = exp(-|D_i - D_j|^2 / 2*sigma^2)

    Args:
        semantic_codes: Semantic codes of shape (B, N, D_s).
        depth: Flattened depth values of shape (B, N).
        sigma_d: Depth similarity bandwidth.
        num_pairs: Number of random pixel pairs to sample.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Scalar depth correlation loss.
    """
    b, n, d = semantic_codes.shape
    device = semantic_codes.device

    total_loss = torch.tensor(0.0, device=device)

    for batch_idx in range(b):
        codes = semantic_codes[batch_idx]  # (N, D)
        d_vals = depth[batch_idx]  # (N,)

        # Sample random pairs
        idx_i = torch.randint(0, n, (num_pairs,), device=device, generator=generator)
        idx_j = torch.randint(0, n, (num_pairs,), device=device, generator=generator)

        # Depth weights
        depth_diff = d_vals[idx_i] - d_vals[idx_j]
        w_ij = torch.exp(-depth_diff ** 2 / (2 * sigma_d ** 2))

        # Cosine similarity
        codes_i = codes[idx_i]
        codes_j = codes[idx_j]
        cos_sim = torch.sum(codes_i * codes_j, dim=-1) / (
            torch.norm(codes_i, dim=-1)
            * torch.norm(codes_j, dim=-1)
            + 1e-8
        )

        # Loss: w * (1 - cos)^2
        loss = w_ij * (1.0 - cos_sim) ** 2
        total_loss = total_loss + torch.mean(loss)

    return total_loss / b
