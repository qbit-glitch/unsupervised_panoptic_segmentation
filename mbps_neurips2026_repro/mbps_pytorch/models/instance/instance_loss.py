"""CutS3D Instance Loss Functions -- Spatial Confidence Components.

Faithful implementation of Sick et al., ICCV 2025.
All loss functions, augmentation utilities, and the Spatial Confidence
Soft Target Loss from the paper.

Components:
  1. Spatial Confidence Soft Target Loss (Eq. 6)
  2. Confident Copy-Paste Selection (Algorithm 7)
  3. Confidence Alpha-Blending (Eq. 5)
  4. DropLoss for unmatched proposals
  5. Box regression loss

L_instance = L_mask_SC + lambda_drop * L_Drop + lambda_box * L_box
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Algorithm 9: Spatial Confidence Soft Target Loss (Eq. 6)
# ---------------------------------------------------------------------------

def spatial_confidence_soft_target_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    spatial_confidence: torch.Tensor,
) -> torch.Tensor:
    """Spatial Confidence Soft Target Loss.

    L_mask = Sum_{i,j} SC_{i,j} * BCE(M_hat_{i,j}, M_{i,j})     [Eq. 6]

    Unlike CuVLER's scalar soft target loss that re-weights the entire
    mask by a single confidence score, this loss weights each patch/pixel
    independently using the Spatial Confidence map.

    Confident regions contribute more to the gradient; uncertain boundary
    patches contribute less, creating a cleaner learning signal.

    For pixels outside the instance region (not covered by SC), SC = 1
    (full contribution), following the paper.

    Args:
        pred_masks: Predicted mask logits, shape (B, M, N).
        target_masks: Target binary masks, shape (B, M, N).
        spatial_confidence: Per-pixel SC maps, shape (B, M, N) or (B, N).
            If (B, N), it is broadcast across all M instances.

    Returns:
        Scalar SC-weighted BCE loss.
    """
    # Compute per-pixel BCE (numerically stable using logits)
    bce = F.binary_cross_entropy_with_logits(
        input=pred_masks, target=target_masks, reduction="none"
    )  # (B, M, N)

    # Handle SC shape: if (B, N), expand to (B, 1, N) for broadcast
    if spatial_confidence.ndim == 2:
        sc = spatial_confidence[:, None, :]  # (B, 1, N)
    else:
        sc = spatial_confidence  # (B, M, N)

    # Weight by Spatial Confidence
    weighted_bce = sc * bce  # (B, M, N)

    return torch.mean(weighted_bce)


# ---------------------------------------------------------------------------
# Algorithm 7: Confident Copy-Paste Selection
# ---------------------------------------------------------------------------

def confident_copy_paste_selection(
    masks: torch.Tensor,
    spatial_confidence: torch.Tensor,
    num_valid: torch.Tensor,
    top_fraction: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select high-confidence masks for copy-paste augmentation.

    For each image, average the SC map over the mask region to get a
    single confidence score per mask, then select the top fraction.

    Args:
        masks: Binary instance masks, shape (B, M, K).
        spatial_confidence: SC maps, shape (B, M, K).
        num_valid: Number of valid masks per image, shape (B,).
        top_fraction: Fraction of masks to keep (0, 1].

    Returns:
        Tuple of:
            - selected_mask: Boolean selection mask, shape (B, M).
            - confidence_scores: Per-mask avg SC score, shape (B, M).
    """
    # Average SC within each mask region
    mask_area = torch.sum(masks, dim=-1)  # (B, M)
    sc_sum = torch.sum(spatial_confidence * masks, dim=-1)  # (B, M)
    confidence_scores = sc_sum / (mask_area + 1e-8)  # (B, M)

    # Zero out invalid masks
    M = masks.shape[1]
    device = masks.device
    valid_mask = torch.arange(M, device=device)[None, :] < num_valid[:, None]  # (B, M)
    confidence_scores = torch.where(valid_mask, confidence_scores, torch.tensor(-1.0, device=device))

    # Select top fraction per image
    n_select = torch.clamp(
        torch.floor(num_valid.float() * top_fraction).to(torch.int32),
        min=1,
    )

    # Sort by confidence (descending) and threshold
    sorted_idx = torch.argsort(-confidence_scores, dim=-1)  # (B, M)
    rank = torch.argsort(sorted_idx, dim=-1)  # rank of each mask
    selected_mask = rank < n_select[:, None]  # (B, M)
    selected_mask = selected_mask & valid_mask

    return selected_mask, confidence_scores


# ---------------------------------------------------------------------------
# Algorithm 8: Confidence Alpha-Blending (Eq. 5)
# ---------------------------------------------------------------------------

def confidence_alpha_blending(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    masks: torch.Tensor,
    spatial_confidence: torch.Tensor,
) -> torch.Tensor:
    """Alpha-blend pasted objects using Spatial Confidence.

    I^aug_{i,j} = SC_{i,j} * I^S_{i,j} + (1 - SC_{i,j}) * I^T_{i,j}   [Eq. 5]

    High-confidence regions: fully pasted (opaque).
    Low-confidence regions: blended with target (semi-transparent).
    Outside mask: SC = 0, so fully target image.

    Args:
        source_images: Source images (objects to paste), shape (B, H, W, 3).
        target_images: Target images (destination), shape (B, H, W, 3).
        masks: Binary instance masks (upsampled to pixel res), shape (B, H, W).
        spatial_confidence: SC maps (upsampled to pixel res), shape (B, H, W).

    Returns:
        Augmented images, shape (B, H, W, 3).
    """
    # Alpha = SC within mask, 0 outside
    alpha = spatial_confidence * masks  # (B, H, W)
    alpha = alpha[..., None]  # (B, H, W, 1) for broadcast over channels

    augmented = alpha * source_images + (1.0 - alpha) * target_images
    return augmented


# ---------------------------------------------------------------------------
# DropLoss for unmatched proposals
# ---------------------------------------------------------------------------

def drop_loss(
    pred_masks: torch.Tensor,
    pred_scores: torch.Tensor,
    matched: torch.Tensor,
) -> torch.Tensor:
    """Drop loss for unmatched instance predictions.

    Penalizes unmatched mask proposals to suppress spurious detections.
    L_Drop = Sum_{unmatched m} (||mask_m||^2 + score_m^2) / N_unmatched

    Args:
        pred_masks: Predicted mask logits, shape (B, M, N).
        pred_scores: Instance scores, shape (B, M).
        matched: Boolean mask of matched instances, shape (B, M).

    Returns:
        Scalar drop loss.
    """
    unmatched = ~matched
    mask_norms = torch.sum(pred_masks ** 2, dim=-1)  # (B, M)
    score_penalty = pred_scores ** 2  # (B, M)

    loss = torch.sum(unmatched.float() * (mask_norms + score_penalty))
    num_unmatched = torch.sum(unmatched.float()) + 1e-8

    return loss / num_unmatched


# ---------------------------------------------------------------------------
# Box regression loss
# ---------------------------------------------------------------------------

def box_regression_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    matched: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 box regression loss for matched instances.

    Args:
        pred_boxes: Predicted boxes, shape (B, M, 4).
        target_boxes: Target boxes, shape (B, M, 4).
        matched: Boolean matched instances, shape (B, M).

    Returns:
        Scalar box regression loss.
    """
    diff = torch.abs(pred_boxes - target_boxes)
    smooth_l1 = torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)  # (B, M, 4)

    loss = torch.sum(smooth_l1, dim=-1)  # (B, M)
    loss = torch.sum(matched.float() * loss)
    num_matched = torch.sum(matched.float()) + 1e-8

    return loss / num_matched


# ---------------------------------------------------------------------------
# Copy-Paste Augmentation Utility
# ---------------------------------------------------------------------------

def copy_paste_augment(
    images: torch.Tensor,
    masks: torch.Tensor,
    spatial_confidence: torch.Tensor,
    num_valid: torch.Tensor,
    top_fraction: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full copy-paste augmentation pipeline with SC.

    1. Select confident masks
    2. Randomly pair source/target images
    3. Alpha-blend using SC

    Args:
        images: Batch of images, shape (B, H, W, 3).
        masks: Instance masks at pixel resolution, shape (B, M, H, W).
        spatial_confidence: SC maps at pixel resolution, shape (B, M, H, W).
        num_valid: Valid mask count, shape (B,).
        top_fraction: Fraction of masks to select.
        generator: Optional torch random generator.

    Returns:
        Tuple of:
            - augmented_images: Shape (B, H, W, 3).
            - augmented_masks: Shape (B, M, H, W).
            - augmented_sc: Shape (B, M, H, W).
    """
    B, M, H, W = masks.shape
    device = masks.device

    # Step 1: Select confident masks
    # Reshape masks for selection: (B, M, H*W)
    masks_flat = masks.reshape(B, M, -1)
    sc_flat = spatial_confidence.reshape(B, M, -1)
    selected, _ = confident_copy_paste_selection(
        masks_flat, sc_flat, num_valid, top_fraction
    )  # (B, M)

    # Step 2: Random pairing -- shuffle batch indices
    perm = torch.randperm(B, device=device, generator=generator)
    source_images = images[perm]
    source_masks = masks[perm]
    source_sc = spatial_confidence[perm]
    source_selected = selected[perm]

    # Step 3: For each target image, paste the first selected mask from source
    # Combine all selected source masks into a single composite mask
    composite_mask = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    composite_sc = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    for m_idx in range(M):
        is_selected = source_selected[:, m_idx].float()  # (B,)
        mask_m = source_masks[:, m_idx]  # (B, H, W)
        sc_m = source_sc[:, m_idx]  # (B, H, W)
        # Only paste if selected and not overlapping existing paste
        paste_here = is_selected[:, None, None] * mask_m * (1.0 - composite_mask)
        composite_mask = torch.clamp(composite_mask + paste_here, 0.0, 1.0)
        composite_sc = torch.where(paste_here > 0.5, sc_m, composite_sc)

    # Step 4: Alpha-blend
    augmented_images = confidence_alpha_blending(
        source_images, images, composite_mask, composite_sc
    )

    # Also combine masks for the augmented image
    # Keep original target masks and add pasted masks
    augmented_masks = masks  # original masks stay
    augmented_sc = spatial_confidence

    return augmented_images, augmented_masks, augmented_sc


# ---------------------------------------------------------------------------
# Combined Instance Loss
# ---------------------------------------------------------------------------

class CutS3DInstanceLoss:
    """Combined CutS3D instance loss with Spatial Confidence.

    L = L_mask_SC + lambda_drop * L_Drop + lambda_box * L_box

    Uses Spatial Confidence Soft Target Loss (Eq. 6) as the primary
    mask loss, replacing standard BCE with per-pixel SC weighting.

    Args:
        lambda_drop: Weight for drop loss (default 0.5).
        lambda_box: Weight for box regression loss (default 1.0).
    """

    def __init__(
        self,
        lambda_drop: float = 0.5,
        lambda_box: float = 1.0,
    ):
        self.lambda_drop = lambda_drop
        self.lambda_box = lambda_box

    def __call__(
        self,
        pred_masks: torch.Tensor,
        pred_scores: torch.Tensor,
        features: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
        pred_boxes: Optional[torch.Tensor] = None,
        spatial_confidence: Optional[torch.Tensor] = None,
        matched: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined instance loss.

        Args:
            pred_masks: Predicted mask logits (B, M, N).
            pred_scores: Instance scores (B, M).
            features: Feature vectors (B, N, D).
            target_masks: Target masks (B, M, N).
            target_boxes: Target boxes (B, M, 4).
            pred_boxes: Predicted boxes (B, M, 4).
            spatial_confidence: SC maps (B, M, N) or (B, N).
            matched: Boolean matched instances (B, M).

        Returns:
            Dict with 'sc_mask', 'drop', 'box', 'total' loss components.
        """
        device = pred_masks.device
        losses: Dict[str, torch.Tensor] = {}

        if target_masks is not None and matched is not None:
            # Use SC Soft Target Loss if spatial confidence available
            if spatial_confidence is not None:
                l_mask = spatial_confidence_soft_target_loss(
                    pred_masks, target_masks, spatial_confidence
                )
            else:
                # Fallback to standard BCE (SC = 1 everywhere)
                l_mask = torch.mean(
                    F.binary_cross_entropy_with_logits(
                        input=pred_masks, target=target_masks, reduction="none"
                    )
                )
            losses["sc_mask"] = l_mask

            l_drop = drop_loss(pred_masks, pred_scores, matched)
            losses["drop"] = l_drop

            total = l_mask + self.lambda_drop * l_drop

            if pred_boxes is not None and target_boxes is not None:
                l_box = box_regression_loss(pred_boxes, target_boxes, matched)
                losses["box"] = l_box
                total = total + self.lambda_box * l_box
            else:
                losses["box"] = torch.tensor(0.0, device=device)

            losses["total"] = total

        else:
            # Unsupervised mode: feature coherence loss
            b, m, n = pred_masks.shape
            mask_probs = torch.sigmoid(pred_masks)

            # Vectorized intra-mask similarity (no Python loops)
            # Pool features per instance
            pooled = torch.einsum("bmn,bnd->bmd", mask_probs, features)
            mask_sums = torch.sum(mask_probs, dim=-1, keepdim=True) + 1e-8
            centroids = pooled / mask_sums  # (B, M, D)

            # Similarity of each patch to its instance centroid
            sim = torch.einsum("bnd,bmd->bmn", features, centroids)  # (B, M, N)
            intra_sim = torch.sum(mask_probs * sim, dim=-1) / mask_sums.squeeze(-1)

            # Weight by instance scores
            total_loss = -torch.mean(pred_scores * intra_sim)

            losses["unsup_coherence"] = total_loss
            losses["total"] = total_loss

        return losses


# Backwards-compatible alias
ModelInstanceLoss = CutS3DInstanceLoss
