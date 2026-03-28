"""Instance Segmentation Loss.

Combines mask loss (Dice + BCE) with score supervision and box losses.

L_instance = L_dice + lambda_drop * L_bce + lambda_box * L_box
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute Dice loss for mask prediction.

    Dice = 2|P intersection G| / (|P| + |G|)

    Args:
        pred_masks: Predicted mask logits of shape (B, M, N).
        target_masks: Target binary masks of shape (B, M, N).
        smooth: Smoothing factor.

    Returns:
        Scalar Dice loss.
    """
    pred_prob = torch.sigmoid(pred_masks)

    intersection = torch.sum(pred_prob * target_masks, dim=-1)  # (B, M)
    union = torch.sum(pred_prob, dim=-1) + torch.sum(target_masks, dim=-1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - torch.mean(dice)


def mask_bce_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss for mask prediction.

    Args:
        pred_masks: Predicted mask logits of shape (B, M, N).
        target_masks: Target binary masks of shape (B, M, N).

    Returns:
        Scalar BCE loss.
    """
    bce = F.binary_cross_entropy_with_logits(
        pred_masks,
        target_masks,
        reduction="mean",
    )
    return bce


def unsupervised_mask_loss(
    pred_masks: torch.Tensor,
    pred_scores: torch.Tensor,
    features: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Unsupervised mask loss based on feature coherence.

    Encourages masks to group features with high cosine similarity
    and separate features with low similarity, using a margin-based
    contrastive formulation bounded in [0, 2 + margin].

    L = sum_m score_m * max(0, inter_cos_m - intra_cos_m + margin)

    Args:
        pred_masks: Predicted mask logits of shape (B, M, N).
        pred_scores: Instance scores of shape (B, M).
        features: Feature vectors of shape (B, N, D).
        margin: Contrastive margin.

    Returns:
        Scalar unsupervised mask loss (non-negative).
    """
    b, m, n = pred_masks.shape
    mask_probs = torch.sigmoid(pred_masks)  # (B, M, N)

    # L2-normalize features for cosine similarity (bounded to [-1, 1])
    features_norm = F.normalize(features, dim=-1)  # (B, N, D)

    total_loss = torch.tensor(0.0, device=pred_masks.device)
    count = 0

    for batch in range(b):
        for inst in range(m):
            mask = mask_probs[batch, inst]  # (N,)
            score = pred_scores[batch, inst]

            if score.item() < 0.1:
                continue

            mask_sum = torch.sum(mask) + 1e-8
            anti_mask_sum = torch.sum(1.0 - mask) + 1e-8

            # Intra-mask cosine similarity (want high)
            weighted_feats = features_norm[batch] * mask[:, None]  # (N, D)
            centroid = F.normalize(
                torch.sum(weighted_feats, dim=0, keepdim=True), dim=-1
            )  # (1, D)
            intra_cos = torch.sum(
                mask * (features_norm[batch] @ centroid.T).squeeze(-1)
            ) / mask_sum

            # Inter-mask cosine similarity (want low)
            inter_cos = torch.sum(
                (1.0 - mask) * (features_norm[batch] @ centroid.T).squeeze(-1)
            ) / anti_mask_sum

            # Margin-based contrastive: penalize when inter >= intra - margin
            gap_loss = torch.clamp(inter_cos - intra_cos + margin, min=0.0)
            total_loss = total_loss + score * gap_loss
            count += 1

    return total_loss / max(count, 1)


class InstanceLoss(nn.Module):
    """Combined instance segmentation loss.

    Args:
        lambda_dice: Weight for Dice loss.
        lambda_bce: Weight for BCE loss.
        lambda_unsup: Weight for unsupervised coherence loss.
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_bce: float = 0.5,
        lambda_unsup: float = 0.3,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_unsup = lambda_unsup

    def forward(
        self,
        pred_masks: torch.Tensor,
        pred_scores: torch.Tensor,
        features: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined instance loss.

        In unsupervised setting, uses feature coherence.
        If target masks provided, uses supervised Dice + BCE.

        Args:
            pred_masks: Predicted mask logits (B, M, N).
            pred_scores: Instance scores (B, M).
            features: Feature vectors (B, N, D).
            target_masks: Optional target masks (B, M, N).

        Returns:
            Dict with loss components and total.
        """
        losses = {}

        if target_masks is not None:
            # Supervised losses
            l_dice = dice_loss(pred_masks, target_masks)
            l_bce = mask_bce_loss(pred_masks, target_masks)
            losses["dice"] = l_dice
            losses["bce"] = l_bce
            losses["total"] = self.lambda_dice * l_dice + self.lambda_bce * l_bce
        else:
            # Unsupervised loss
            l_unsup = unsupervised_mask_loss(pred_masks, pred_scores, features)
            losses["unsup_mask"] = l_unsup
            losses["total"] = self.lambda_unsup * l_unsup

        return losses
