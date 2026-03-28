"""Cross-Branch Consistency Losses.

L_consistency = lambda_u * L_uniform + lambda_b * L_boundary + lambda_dbc * L_DBC

Components:
    - Uniformity: Instance-Semantic entropy alignment
    - Boundary: Semantic-Instance boundary co-occurrence
    - DBC: Depth-Boundary Coherence
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def uniformity_loss(
    instance_masks: torch.Tensor,
    semantic_pred: torch.Tensor,
    num_classes: int = 27,
) -> torch.Tensor:
    """Instance-Semantic Uniformity Loss.

    Each instance mask should contain pixels of predominantly one
    semantic class. Minimizes entropy of semantic distribution within
    each instance.

    L_uniform = sum_m H(P(class | mask_m))

    Args:
        instance_masks: Instance mask probs of shape (B, M, N).
        semantic_pred: Semantic class predictions of shape (B, N).
        num_classes: Number of semantic classes.

    Returns:
        Scalar uniformity loss.
    """
    b, m, n = instance_masks.shape
    device = instance_masks.device

    total_loss = torch.tensor(0.0, device=device)

    for batch in range(b):
        for inst in range(m):
            mask = instance_masks[batch, inst]  # (N,)
            mask_sum = torch.sum(mask) + 1e-6

            # Distribution of semantic classes within this mask
            class_hist = torch.zeros(num_classes, device=device)
            for c in range(num_classes):
                class_count = torch.sum(
                    mask * (semantic_pred[batch] == c).float()
                )
                class_hist[c] = class_count

            # Normalize to probability
            class_prob = class_hist / mask_sum
            class_prob = torch.clamp(class_prob, 1e-7, 1.0)

            # Entropy (with stable log)
            entropy = -torch.sum(torch.where(
                class_prob > 1e-6,
                class_prob * torch.log(class_prob + 1e-10),
                torch.zeros_like(class_prob),
            ))
            total_loss = total_loss + entropy

    return total_loss / (b * m + 1e-8)


def boundary_alignment_loss(
    semantic_pred: torch.Tensor,
    instance_masks: torch.Tensor,
    spatial_h: int = 64,
    spatial_w: int = 64,
) -> torch.Tensor:
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
    sem_2d = semantic_pred.reshape(b, spatial_h, spatial_w).float()
    mask_2d = top_mask.reshape(b, spatial_h, spatial_w)

    # Compute boundaries via pixel differences
    # Prepend first column/row to match JAX diff with prepend behavior
    sem_grad_x = torch.abs(
        sem_2d - torch.cat([sem_2d[:, :, :1], sem_2d[:, :, :-1]], dim=-1)
    )
    sem_grad_y = torch.abs(
        sem_2d - torch.cat([sem_2d[:, :1, :], sem_2d[:, :-1, :]], dim=-2)
    )
    sem_boundary = ((sem_grad_x > 0) | (sem_grad_y > 0)).float()

    mask_grad_x = torch.abs(
        mask_2d - torch.cat([mask_2d[:, :, :1], mask_2d[:, :, :-1]], dim=-1)
    )
    mask_grad_y = torch.abs(
        mask_2d - torch.cat([mask_2d[:, :1, :], mask_2d[:, :-1, :]], dim=-2)
    )
    inst_boundary = ((mask_grad_x > 0.3) | (mask_grad_y > 0.3)).float()

    # Alignment: semantic boundaries should colocate with instance boundaries
    intersection = torch.sum(sem_boundary * inst_boundary, dim=(-2, -1))
    union = torch.sum(
        torch.maximum(sem_boundary, inst_boundary), dim=(-2, -1)
    ) + 1e-8

    # IoU of boundaries (maximize -> minimize 1-IoU)
    boundary_iou = intersection / union
    return 1.0 - torch.mean(boundary_iou)


def depth_boundary_coherence_loss(
    semantic_pred: torch.Tensor,
    instance_masks: torch.Tensor,
    depth: torch.Tensor,
    spatial_h: int = 64,
    spatial_w: int = 64,
) -> torch.Tensor:
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
    depth_2d = depth.reshape(b, spatial_h, spatial_w)
    sem_2d = semantic_pred.reshape(b, spatial_h, spatial_w).float()

    # Depth boundaries
    depth_grad_x = torch.abs(
        depth_2d - torch.cat([depth_2d[:, :, :1], depth_2d[:, :, :-1]], dim=-1)
    )
    depth_grad_y = torch.abs(
        depth_2d - torch.cat([depth_2d[:, :1, :], depth_2d[:, :-1, :]], dim=-2)
    )
    depth_boundary = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)

    # Semantic boundaries
    sem_grad_x = (
        sem_2d - torch.cat([sem_2d[:, :, :1], sem_2d[:, :, :-1]], dim=-1)
    ) != 0
    sem_grad_y = (
        sem_2d - torch.cat([sem_2d[:, :1, :], sem_2d[:, :-1, :]], dim=-2)
    ) != 0
    sem_boundary = torch.maximum(
        sem_grad_x.float(), sem_grad_y.float()
    )

    # Loss: semantic boundaries should align with depth boundaries
    # High depth gradient -> should have semantic boundary
    # Low depth gradient -> should NOT have semantic boundary
    # Clamp depth_boundary to prevent exp explosion with large gradients
    depth_boundary_clamped = torch.clamp(depth_boundary, 0.0, 2.0)
    weight = 1.0 + depth_boundary_clamped * 3.0  # Linear weighting, no exp overflow
    loss = torch.mean(weight * (1.0 - sem_boundary) * depth_boundary_clamped)

    return loss


class ConsistencyLoss(nn.Module):
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
        super().__init__()
        self.lambda_uniform = lambda_uniform
        self.lambda_boundary = lambda_boundary
        self.lambda_dbc = lambda_dbc
        self.num_classes = num_classes

    def forward(
        self,
        semantic_pred: torch.Tensor,
        instance_masks: torch.Tensor,
        depth: torch.Tensor,
        spatial_h: int = 64,
        spatial_w: int = 64,
    ) -> Dict[str, torch.Tensor]:
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
