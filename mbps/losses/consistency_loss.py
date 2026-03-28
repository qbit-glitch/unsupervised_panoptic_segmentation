"""Cross-Branch Consistency Losses.

L_consistency = λ_u · L_uniform + λ_b · L_boundary + λ_dbc · L_DBC

Components:
    - Uniformity: Instance-Semantic entropy alignment
    - Boundary: Semantic-Instance boundary co-occurrence
    - DBC: Depth-Boundary Coherence
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def uniformity_loss(
    instance_masks: jnp.ndarray,
    semantic_pred: jnp.ndarray,
    num_classes: int = 27,
) -> jnp.ndarray:
    """Instance-Semantic Uniformity Loss.

    Each instance mask should contain pixels of predominantly one
    semantic class. Minimizes entropy of semantic distribution within
    each instance.

    L_uniform = Σ_m H(P(class | mask_m))

    Args:
        instance_masks: Instance mask probs of shape (B, M, N).
        semantic_pred: Semantic class predictions of shape (B, N).
        num_classes: Number of semantic classes.

    Returns:
        Scalar uniformity loss.
    """
    b, m, n = instance_masks.shape

    # One-hot encode semantic predictions: (B, N, K)
    one_hot = jax.nn.one_hot(semantic_pred, num_classes)

    # Weighted class counts per instance: (B, M, N) @ (B, N, K) -> (B, M, K)
    class_counts = jnp.einsum("bmn,bnk->bmk", instance_masks, one_hot)

    # Normalize to probabilities
    mask_sums = jnp.sum(instance_masks, axis=-1, keepdims=True) + 1e-6  # (B, M, 1)
    class_prob = class_counts / mask_sums
    class_prob = jnp.clip(class_prob, 1e-7, 1.0)

    # Entropy per instance
    entropy = -jnp.sum(
        jnp.where(
            class_prob > 1e-6,
            class_prob * jnp.log(class_prob + 1e-10),
            0.0,
        ),
        axis=-1,
    )  # (B, M)

    return jnp.mean(entropy)


def boundary_alignment_loss(
    semantic_pred: jnp.ndarray,
    instance_masks: jnp.ndarray,
    spatial_h: int = 64,
    spatial_w: int = 64,
) -> jnp.ndarray:
    """Semantic-Instance Boundary Alignment Loss.

    Encourages semantic boundaries to align with instance boundaries.

    Args:
        semantic_pred: Semantic predictions of shape (B, N).
        instance_masks: Instance mask probs of shape (B, M, N).
        spatial_h: Spatial height for 2D reshaping.
        spatial_w: Spatial width for 2D reshaping.

    Returns:
        Scalar boundary alignment loss.
    """
    b, m, n = instance_masks.shape

    # Compute semantic boundary map
    # Use top-1 instance for simplicity
    top_mask = instance_masks[:, 0, :]  # (B, N)

    # Reshape to spatial
    sem_2d = jnp.reshape(semantic_pred, (b, spatial_h, spatial_w))
    mask_2d = jnp.reshape(top_mask, (b, spatial_h, spatial_w))

    # Compute boundaries via pixel differences
    sem_grad_x = jnp.abs(jnp.diff(sem_2d, axis=-1, prepend=sem_2d[:, :, :1]))
    sem_grad_y = jnp.abs(jnp.diff(sem_2d, axis=-2, prepend=sem_2d[:, :1, :]))
    sem_boundary = ((sem_grad_x > 0) | (sem_grad_y > 0)).astype(jnp.float32)

    mask_grad_x = jnp.abs(jnp.diff(mask_2d, axis=-1, prepend=mask_2d[:, :, :1]))
    mask_grad_y = jnp.abs(jnp.diff(mask_2d, axis=-2, prepend=mask_2d[:, :1, :]))
    inst_boundary = ((mask_grad_x > 0.3) | (mask_grad_y > 0.3)).astype(jnp.float32)

    # Alignment: semantic boundaries should colocate with instance boundaries
    intersection = jnp.sum(sem_boundary * inst_boundary, axis=(-2, -1))
    union = jnp.sum(
        jnp.maximum(sem_boundary, inst_boundary), axis=(-2, -1)
    ) + 1e-8

    # IoU of boundaries (maximize → minimize 1-IoU)
    boundary_iou = intersection / union
    return 1.0 - jnp.mean(boundary_iou)


def depth_boundary_coherence_loss(
    semantic_pred: jnp.ndarray,
    instance_masks: jnp.ndarray,
    depth: jnp.ndarray,
    spatial_h: int = 64,
    spatial_w: int = 64,
) -> jnp.ndarray:
    """Depth-Boundary Coherence (DBC) Loss.

    Ensures that prediction boundaries align with depth discontinuities.
    Boundaries in predictions should occur where depth changes sharply.

    Args:
        semantic_pred: Semantic predictions of shape (B, N).
        instance_masks: Instance mask probs of shape (B, M, N).
        depth: Depth values of shape (B, N).
        spatial_h: Spatial height.
        spatial_w: Spatial width.

    Returns:
        Scalar DBC loss.
    """
    b = depth.shape[0]

    # Reshape to spatial
    depth_2d = jnp.reshape(depth, (b, spatial_h, spatial_w))
    sem_2d = jnp.reshape(semantic_pred, (b, spatial_h, spatial_w))

    # Depth boundaries
    depth_grad_x = jnp.abs(jnp.diff(depth_2d, axis=-1, prepend=depth_2d[:, :, :1]))
    depth_grad_y = jnp.abs(jnp.diff(depth_2d, axis=-2, prepend=depth_2d[:, :1, :]))
    depth_boundary = jnp.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)

    # Semantic boundaries
    sem_grad_x = (jnp.diff(sem_2d, axis=-1, prepend=sem_2d[:, :, :1]) != 0).astype(
        jnp.float32
    )
    sem_grad_y = (jnp.diff(sem_2d, axis=-2, prepend=sem_2d[:, :1, :]) != 0).astype(
        jnp.float32
    )
    sem_boundary = jnp.maximum(sem_grad_x, sem_grad_y)

    # Loss: semantic boundaries should align with depth boundaries
    # High depth gradient → should have semantic boundary
    # Low depth gradient → should NOT have semantic boundary
    # Clamp depth_boundary to prevent exp explosion with large gradients
    depth_boundary_clamped = jnp.clip(depth_boundary, 0.0, 2.0)
    weight = 1.0 + depth_boundary_clamped * 3.0  # Linear weighting, no exp overflow
    loss = jnp.mean(weight * (1.0 - sem_boundary) * depth_boundary_clamped)

    return loss


class ConsistencyLoss:
    """Combined cross-branch consistency loss.

    Args:
        lambda_uniform: Weight for uniformity loss.
        lambda_boundary: Weight for boundary alignment.
        lambda_dbc: Weight for depth-boundary coherence.
        num_classes: Number of semantic classes.
    """

    def __init__(
        self,
        lambda_uniform: float = 0.3,
        lambda_boundary: float = 0.2,
        lambda_dbc: float = 0.2,
        num_classes: int = 27,
    ):
        self.lambda_uniform = lambda_uniform
        self.lambda_boundary = lambda_boundary
        self.lambda_dbc = lambda_dbc
        self.num_classes = num_classes

    def __call__(
        self,
        semantic_pred: jnp.ndarray,
        instance_masks: jnp.ndarray,
        depth: jnp.ndarray,
        spatial_h: int = 64,
        spatial_w: int = 64,
    ) -> Dict[str, jnp.ndarray]:
        """Compute combined consistency loss.

        Args:
            semantic_pred: Semantic predictions (B, N).
            instance_masks: Instance mask probs (B, M, N).
            depth: Depth values (B, N).
            spatial_h: Spatial height for 2D ops.
            spatial_w: Spatial width for 2D ops.

        Returns:
            Dict with loss components and total.
        """
        losses = {}

        l_uniform = uniformity_loss(
            instance_masks, semantic_pred, self.num_classes
        )
        losses["uniform"] = l_uniform

        l_boundary = boundary_alignment_loss(
            semantic_pred, instance_masks, spatial_h, spatial_w
        )
        losses["boundary"] = l_boundary

        l_dbc = depth_boundary_coherence_loss(
            semantic_pred, instance_masks, depth, spatial_h, spatial_w
        )
        losses["dbc"] = l_dbc

        losses["total"] = (
            self.lambda_uniform * l_uniform
            + self.lambda_boundary * l_boundary
            + self.lambda_dbc * l_dbc
        )

        return losses
