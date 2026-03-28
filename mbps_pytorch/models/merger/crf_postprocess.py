"""CRF Post-Processing for Semantic Segmentation.

Implements a mean-field approximation CRF for refining semantic
predictions using spatial and bilateral potentials.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def pairwise_bilateral(
    image: torch.Tensor,
    positions: torch.Tensor,
    theta_alpha: float = 80.0,
    theta_beta: float = 13.0,
) -> torch.Tensor:
    """Compute bilateral pairwise potentials.

    k(i,j) = exp(-|p_i - p_j|^2/(2*theta_alpha^2) - |I_i - I_j|^2/(2*theta_beta^2))

    Args:
        image: RGB image of shape (N, 3), flattened.
        positions: Spatial positions of shape (N, 2).
        theta_alpha: Spatial bandwidth.
        theta_beta: Color bandwidth.

    Returns:
        Bilateral kernel of shape (N, N).
    """
    # Spatial distance
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
    pos_sq = torch.sum(pos_diff ** 2, dim=-1)  # (N, N)

    # Color distance
    color_diff = image.unsqueeze(1) - image.unsqueeze(0)  # (N, N, 3)
    color_sq = torch.sum(color_diff ** 2, dim=-1)  # (N, N)

    return torch.exp(
        -pos_sq / (2 * theta_alpha ** 2)
        - color_sq / (2 * theta_beta ** 2)
    )


def pairwise_smoothness(
    positions: torch.Tensor,
    theta_gamma: float = 3.0,
) -> torch.Tensor:
    """Compute smoothness pairwise potentials.

    k(i,j) = exp(-|p_i - p_j|^2/(2*theta_gamma^2))

    Args:
        positions: Spatial positions of shape (N, 2).
        theta_gamma: Spatial bandwidth.

    Returns:
        Smoothness kernel of shape (N, N).
    """
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
    pos_sq = torch.sum(pos_diff ** 2, dim=-1)  # (N, N)
    return torch.exp(-pos_sq / (2 * theta_gamma ** 2))


def crf_inference(
    unary_potentials: torch.Tensor,
    image: torch.Tensor,
    spatial_h: int,
    spatial_w: int,
    num_iterations: int = 10,
    theta_alpha: float = 80.0,
    theta_beta: float = 13.0,
    theta_gamma: float = 3.0,
    w_bilateral: float = 5.0,
    w_smoothness: float = 3.0,
) -> torch.Tensor:
    """Run CRF mean-field inference.

    Args:
        unary_potentials: Log-probabilities (N, K) from classifier.
        image: RGB image (N, 3), values in [0, 255].
        spatial_h: Image height.
        spatial_w: Image width.
        num_iterations: Number of mean-field iterations.
        theta_alpha, theta_beta, theta_gamma: Kernel bandwidths.
        w_bilateral, w_smoothness: Kernel weights.

    Returns:
        Refined probabilities of shape (N, K).
    """
    n, k = unary_potentials.shape
    device = unary_potentials.device

    # Create spatial positions grid
    y_pos = torch.arange(spatial_h, device=device).repeat_interleave(spatial_w)
    x_pos = torch.arange(spatial_w, device=device).repeat(spatial_h)
    positions = torch.stack([y_pos, x_pos], dim=-1).float()

    # For efficiency, subsample for kernel computation if N is large
    max_points = 2048
    if n > max_points:
        # Subsample for pairwise computation
        stride = n // max_points
        sub_idx = torch.arange(0, n, stride, device=device)[:max_points]

        # Compute kernels on subsample
        k_bilateral = pairwise_bilateral(
            image[sub_idx], positions[sub_idx], theta_alpha, theta_beta
        )
        k_smoothness = pairwise_smoothness(positions[sub_idx], theta_gamma)

        # Initialize Q from unary
        Q = F.softmax(unary_potentials, dim=-1)

        for _ in range(num_iterations):
            # Message passing on subsample
            Q_sub = Q[sub_idx]
            msg = w_bilateral * (k_bilateral @ Q_sub) + w_smoothness * (
                k_smoothness @ Q_sub
            )

            # Compatibility transform (Potts model)
            pairwise = msg - torch.mean(msg, dim=-1, keepdim=True)

            # Update only subsampled points as approximation
            Q_new = F.softmax(
                unary_potentials[sub_idx] - pairwise, dim=-1
            )

            Q = Q.clone()
            Q[sub_idx] = Q_new

        return Q
    else:
        # Full computation for small inputs
        k_bilateral = pairwise_bilateral(
            image, positions, theta_alpha, theta_beta
        )
        k_smoothness = pairwise_smoothness(positions, theta_gamma)

        Q = F.softmax(unary_potentials, dim=-1)

        for _ in range(num_iterations):
            msg = w_bilateral * (k_bilateral @ Q) + w_smoothness * (
                k_smoothness @ Q
            )
            pairwise = msg - torch.mean(msg, dim=-1, keepdim=True)
            Q = F.softmax(unary_potentials - pairwise, dim=-1)

        return Q
