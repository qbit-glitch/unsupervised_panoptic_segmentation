"""CRF Post-Processing for Semantic Segmentation.

Implements a mean-field approximation CRF for refining semantic
predictions using spatial and bilateral potentials.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def pairwise_bilateral(
    image: jnp.ndarray,
    positions: jnp.ndarray,
    theta_alpha: float = 80.0,
    theta_beta: float = 13.0,
) -> jnp.ndarray:
    """Compute bilateral pairwise potentials.

    k(i,j) = exp(-|p_i - p_j|²/(2θ_α²) - |I_i - I_j|²/(2θ_β²))

    Args:
        image: RGB image of shape (N, 3), flattened.
        positions: Spatial positions of shape (N, 2).
        theta_alpha: Spatial bandwidth.
        theta_beta: Color bandwidth.

    Returns:
        Bilateral kernel of shape (N, N).
    """
    # Spatial distance
    pos_diff = positions[:, None, :] - positions[None, :, :]
    pos_sq = jnp.sum(pos_diff ** 2, axis=-1)

    # Color distance
    color_diff = image[:, None, :] - image[None, :, :]
    color_sq = jnp.sum(color_diff ** 2, axis=-1)

    return jnp.exp(
        -pos_sq / (2 * theta_alpha ** 2)
        - color_sq / (2 * theta_beta ** 2)
    )


def pairwise_smoothness(
    positions: jnp.ndarray,
    theta_gamma: float = 3.0,
) -> jnp.ndarray:
    """Compute smoothness pairwise potentials.

    k(i,j) = exp(-|p_i - p_j|²/(2θ_γ²))

    Args:
        positions: Spatial positions of shape (N, 2).
        theta_gamma: Spatial bandwidth.

    Returns:
        Smoothness kernel of shape (N, N).
    """
    pos_diff = positions[:, None, :] - positions[None, :, :]
    pos_sq = jnp.sum(pos_diff ** 2, axis=-1)
    return jnp.exp(-pos_sq / (2 * theta_gamma ** 2))


def crf_inference(
    unary_potentials: jnp.ndarray,
    image: jnp.ndarray,
    spatial_h: int,
    spatial_w: int,
    num_iterations: int = 10,
    theta_alpha: float = 80.0,
    theta_beta: float = 13.0,
    theta_gamma: float = 3.0,
    w_bilateral: float = 5.0,
    w_smoothness: float = 3.0,
) -> jnp.ndarray:
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

    # Create spatial positions grid
    y_pos = jnp.repeat(jnp.arange(spatial_h), spatial_w)
    x_pos = jnp.tile(jnp.arange(spatial_w), spatial_h)
    positions = jnp.stack([y_pos, x_pos], axis=-1).astype(jnp.float32)

    # For efficiency, subsample for kernel computation if N is large
    max_points = 2048
    if n > max_points:
        # Subsample for pairwise computation
        stride = n // max_points
        sub_idx = jnp.arange(0, n, stride)[:max_points]

        # Compute kernels on subsample
        k_bilateral = pairwise_bilateral(
            image[sub_idx], positions[sub_idx], theta_alpha, theta_beta
        )
        k_smoothness = pairwise_smoothness(positions[sub_idx], theta_gamma)

        # Initialize Q from unary
        Q = jax.nn.softmax(unary_potentials, axis=-1)

        for _ in range(num_iterations):
            # Message passing on subsample
            Q_sub = Q[sub_idx]
            msg = w_bilateral * (k_bilateral @ Q_sub) + w_smoothness * (
                k_smoothness @ Q_sub
            )

            # Compatibility transform (Potts model)
            pairwise = msg - jnp.mean(msg, axis=-1, keepdims=True)

            # Update only subsampled points as approximation
            Q_new = jax.nn.softmax(
                unary_potentials[sub_idx] - pairwise, axis=-1
            )

            Q = Q.at[sub_idx].set(Q_new)

        return Q
    else:
        # Full computation for small inputs
        k_bilateral = pairwise_bilateral(
            image, positions, theta_alpha, theta_beta
        )
        k_smoothness = pairwise_smoothness(positions, theta_gamma)

        Q = jax.nn.softmax(unary_potentials, axis=-1)

        for _ in range(num_iterations):
            msg = w_bilateral * (k_bilateral @ Q) + w_smoothness * (
                k_smoothness @ Q
            )
            pairwise = msg - jnp.mean(msg, axis=-1, keepdims=True)
            Q = jax.nn.softmax(unary_potentials - pairwise, axis=-1)

        return Q
