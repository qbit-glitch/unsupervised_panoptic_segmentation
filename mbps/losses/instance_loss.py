"""Instance Segmentation Loss.

Combines mask loss (Dice + BCE) with score supervision and box losses.

L_instance = L_dice + λ_drop · L_bce + λ_box · L_box
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import optax


def dice_loss(
    pred_masks: jnp.ndarray,
    target_masks: jnp.ndarray,
    smooth: float = 1.0,
) -> jnp.ndarray:
    """Compute Dice loss for mask prediction.

    Dice = 2|P ∩ G| / (|P| + |G|)

    Args:
        pred_masks: Predicted mask logits of shape (B, M, N).
        target_masks: Target binary masks of shape (B, M, N).
        smooth: Smoothing factor.

    Returns:
        Scalar Dice loss.
    """
    pred_prob = jax.nn.sigmoid(pred_masks)

    intersection = jnp.sum(pred_prob * target_masks, axis=-1)  # (B, M)
    union = jnp.sum(pred_prob, axis=-1) + jnp.sum(target_masks, axis=-1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - jnp.mean(dice)


def mask_bce_loss(
    pred_masks: jnp.ndarray,
    target_masks: jnp.ndarray,
) -> jnp.ndarray:
    """Binary cross-entropy loss for mask prediction.

    Args:
        pred_masks: Predicted mask logits of shape (B, M, N).
        target_masks: Target binary masks of shape (B, M, N).

    Returns:
        Scalar BCE loss.
    """
    bce = optax.sigmoid_binary_cross_entropy(
        logits=pred_masks,
        labels=target_masks,
    )
    return jnp.mean(bce)


def unsupervised_mask_loss(
    pred_masks: jnp.ndarray,
    pred_scores: jnp.ndarray,
    features: jnp.ndarray,
) -> jnp.ndarray:
    """Unsupervised mask loss based on feature coherence.

    Encourages masks to group features with high mutual similarity
    and separate features with low similarity.

    L = -Σ_m score_m · (intra_sim_m - inter_sim_m)

    Args:
        pred_masks: Predicted mask logits of shape (B, M, N).
        pred_scores: Instance scores of shape (B, M).
        features: Feature vectors of shape (B, N, D).

    Returns:
        Scalar unsupervised mask loss.
    """
    b, m, n = pred_masks.shape
    mask_probs = jax.nn.sigmoid(pred_masks)  # (B, M, N)

    total_loss = jnp.array(0.0)

    for batch in range(b):
        for inst in range(m):
            mask = mask_probs[batch, inst]  # (N,)
            score = pred_scores[batch, inst]

            if score < 0.1:
                continue

            mask_sum = jnp.sum(mask) + 1e-8
            anti_mask_sum = jnp.sum(1.0 - mask) + 1e-8

            # Intra-mask feature similarity
            weighted_feats = features[batch] * mask[:, None]  # (N, D)
            centroid = jnp.sum(weighted_feats, axis=0) / mask_sum  # (D,)
            intra_sim = jnp.sum(
                mask * jnp.sum(features[batch] * centroid[None, :], axis=-1)
            ) / mask_sum

            # Inter-mask feature similarity (should be low)
            anti_feats = features[batch] * (1.0 - mask)[:, None]
            anti_centroid = jnp.sum(anti_feats, axis=0) / anti_mask_sum
            inter_sim = jnp.sum(
                (1.0 - mask)
                * jnp.sum(features[batch] * centroid[None, :], axis=-1)
            ) / anti_mask_sum

            # Maximize gap between intra and inter similarity
            total_loss = total_loss - score * (intra_sim - inter_sim)

    return total_loss / (b * m)


class InstanceLoss:
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
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_unsup = lambda_unsup

    def __call__(
        self,
        pred_masks: jnp.ndarray,
        pred_scores: jnp.ndarray,
        features: jnp.ndarray,
        target_masks: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
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
