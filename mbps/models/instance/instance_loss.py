"""CutS3D Instance Loss Functions — Spatial Confidence Components.

Faithful implementation of Sick et al., ICCV 2025.
All loss functions, augmentation utilities, and the Spatial Confidence
Soft Target Loss from the paper.

Components:
  1. Spatial Confidence Soft Target Loss (Eq. 6)
  2. Confident Copy-Paste Selection (Algorithm 7)
  3. Confidence Alpha-Blending (Eq. 5)
  4. DropLoss for unmatched proposals
  5. Box regression loss
  6. Greedy IoU matching (proposal-GT assignment)

L_instance = L_mask_SC + λ_drop · L_Drop + λ_box · L_box
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Algorithm 9: Spatial Confidence Soft Target Loss (Eq. 6)
# ---------------------------------------------------------------------------

def spatial_confidence_soft_target_loss(
    pred_masks: jnp.ndarray,
    target_masks: jnp.ndarray,
    spatial_confidence: jnp.ndarray,
) -> jnp.ndarray:
    """Spatial Confidence Soft Target Loss.

    L_mask = Σ_{i,j} SC_{i,j} · BCE(M̂_{i,j}, M_{i,j})     [Eq. 6]

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
    bce = optax.sigmoid_binary_cross_entropy(
        logits=pred_masks, labels=target_masks
    )  # (B, M, N)

    # Handle SC shape: if (B, N), expand to (B, 1, N) for broadcast
    if spatial_confidence.ndim == 2:
        sc = spatial_confidence[:, None, :]  # (B, 1, N)
    else:
        sc = spatial_confidence  # (B, M, N)

    # Weight by Spatial Confidence
    weighted_bce = sc * bce  # (B, M, N)

    return jnp.mean(weighted_bce)


# ---------------------------------------------------------------------------
# Algorithm 7: Confident Copy-Paste Selection
# ---------------------------------------------------------------------------

def confident_copy_paste_selection(
    masks: jnp.ndarray,
    spatial_confidence: jnp.ndarray,
    num_valid: jnp.ndarray,
    top_fraction: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    mask_area = jnp.sum(masks, axis=-1)  # (B, M)
    sc_sum = jnp.sum(spatial_confidence * masks, axis=-1)  # (B, M)
    confidence_scores = sc_sum / (mask_area + 1e-8)  # (B, M)

    # Zero out invalid masks
    M = masks.shape[1]
    valid_mask = jnp.arange(M)[None, :] < num_valid[:, None]  # (B, M)
    confidence_scores = jnp.where(valid_mask, confidence_scores, -1.0)

    # Select top fraction per image
    n_select = jnp.maximum(1, jnp.floor(num_valid * top_fraction).astype(jnp.int32))

    # Sort by confidence (descending) and threshold
    sorted_idx = jnp.argsort(-confidence_scores, axis=-1)  # (B, M)
    rank = jnp.argsort(sorted_idx, axis=-1)  # rank of each mask
    selected_mask = rank < n_select[:, None]  # (B, M)
    selected_mask = selected_mask & valid_mask

    return selected_mask, confidence_scores


# ---------------------------------------------------------------------------
# Algorithm 8: Confidence Alpha-Blending (Eq. 5)
# ---------------------------------------------------------------------------

def confidence_alpha_blending(
    source_images: jnp.ndarray,
    target_images: jnp.ndarray,
    masks: jnp.ndarray,
    spatial_confidence: jnp.ndarray,
) -> jnp.ndarray:
    """Alpha-blend pasted objects using Spatial Confidence.

    I^aug_{i,j} = SC_{i,j} · I^S_{i,j} + (1 - SC_{i,j}) · I^T_{i,j}   [Eq. 5]

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
# Greedy IoU Matching (Proposal-GT Assignment)
# ---------------------------------------------------------------------------

def _soft_iou(pred_masks: jnp.ndarray, target_masks: jnp.ndarray) -> jnp.ndarray:
    """Pairwise soft IoU between two mask sets.

    Args:
        pred_masks: (M, N) predicted mask probabilities.
        target_masks: (T, N) target mask probabilities.

    Returns:
        IoU matrix of shape (M, T).
    """
    intersection = jnp.einsum("mn,tn->mt", pred_masks, target_masks)
    pred_sum = jnp.sum(pred_masks, axis=-1)[:, None]
    target_sum = jnp.sum(target_masks, axis=-1)[None, :]
    union = pred_sum + target_sum - intersection + 1e-8
    return intersection / union


def _greedy_match_single(
    iou_matrix: jnp.ndarray,
    num_valid: jnp.ndarray,
    iou_threshold: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Greedy matching for a single image (called via vmap).

    For each valid GT (largest area first), assigns the best unassigned
    prediction above iou_threshold.

    Args:
        iou_matrix: (M, T) IoU between predictions and targets.
        num_valid: scalar int, number of valid targets.
        iou_threshold: minimum IoU to accept a match.

    Returns:
        pred_to_gt: (M,) int — GT index assigned to each pred (-1 if unmatched).
        matched: (M,) bool — which predictions are matched.
    """
    M, T = iou_matrix.shape

    init_state = (
        jnp.zeros(M, dtype=jnp.bool_),        # pred_assigned
        jnp.full(M, -1, dtype=jnp.int32),      # pred_to_gt
    )

    def body_fn(t, state):
        pred_assigned, pred_to_gt = state

        # IoU of all preds with GT t
        ious = iou_matrix[:, t]  # (M,)

        # Mask out already-assigned preds and invalid GTs
        ious = jnp.where(pred_assigned, -1.0, ious)
        valid_gt = t < num_valid
        ious = jnp.where(valid_gt, ious, -1.0)

        best_pred = jnp.argmax(ious)
        best_iou = ious[best_pred]
        do_match = best_iou > iou_threshold

        new_assigned = pred_assigned.at[best_pred].set(True)
        new_pred_to_gt = pred_to_gt.at[best_pred].set(t)

        pred_assigned = jnp.where(do_match, new_assigned, pred_assigned)
        pred_to_gt = jnp.where(do_match, new_pred_to_gt, pred_to_gt)

        return (pred_assigned, pred_to_gt)

    matched, pred_to_gt = jax.lax.fori_loop(0, T, body_fn, init_state)
    return pred_to_gt, matched


def greedy_iou_matching(
    pred_masks: jnp.ndarray,
    target_masks: jnp.ndarray,
    num_valid: jnp.ndarray,
    iou_threshold: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Match predicted masks to GT masks using greedy IoU assignment.

    For each image in the batch, computes pairwise IoU between all
    predictions and all valid GT masks, then greedily assigns each GT
    to its best-matching unassigned prediction.

    Args:
        pred_masks: (B, M, N) predicted mask probabilities (after sigmoid).
        target_masks: (B, T, N) target masks (padded, first num_valid are real).
        num_valid: (B,) number of valid targets per image.
        iou_threshold: minimum IoU for a match (default 0.1).

    Returns:
        matched: (B, M) bool — which predictions are matched.
        aligned_targets: (B, M, N) — targets reordered to align with preds.
        pred_to_gt: (B, M) int — GT index for each pred (-1 if unmatched).
        match_ious: (B, M) float — IoU of each match (0 for unmatched).
    """
    B, M, N = pred_masks.shape
    T = target_masks.shape[1]

    # Compute pairwise IoU per batch element
    iou_matrices = jax.vmap(_soft_iou)(pred_masks, target_masks)  # (B, M, T)

    # Run greedy matching per batch element
    match_fn = partial(_greedy_match_single, iou_threshold=iou_threshold)
    pred_to_gt, matched = jax.vmap(match_fn)(iou_matrices, num_valid)  # (B, M), (B, M)

    # Gather aligned targets: for each matched pred, pick its assigned GT
    safe_idx = jnp.maximum(pred_to_gt, 0)  # clamp -1 to 0 for gather
    batch_idx = jnp.arange(B)[:, None]  # (B, 1)
    aligned_targets = target_masks[batch_idx, safe_idx]  # (B, M, N)
    # Zero out unmatched entries
    aligned_targets = jnp.where(matched[:, :, None], aligned_targets, 0.0)

    # Extract match IoUs
    match_ious = iou_matrices[batch_idx, jnp.arange(M)[None, :], safe_idx]  # (B, M)
    match_ious = jnp.where(matched, match_ious, 0.0)

    return matched, aligned_targets, pred_to_gt, match_ious


# ---------------------------------------------------------------------------
# DropLoss for unmatched proposals
# ---------------------------------------------------------------------------

def drop_loss(
    pred_masks: jnp.ndarray,
    pred_scores: jnp.ndarray,
    matched: jnp.ndarray,
) -> jnp.ndarray:
    """Drop loss for unmatched instance predictions.

    Penalizes unmatched mask proposals to suppress spurious detections.
    L_Drop = Σ_{unmatched m} (‖mask_m‖² + score_m²) / N_unmatched

    Args:
        pred_masks: Predicted mask logits, shape (B, M, N).
        pred_scores: Instance scores, shape (B, M).
        matched: Boolean mask of matched instances, shape (B, M).

    Returns:
        Scalar drop loss.
    """
    unmatched = ~matched
    mask_norms = jnp.mean(pred_masks ** 2, axis=-1)  # (B, M)
    score_penalty = pred_scores ** 2  # (B, M)

    loss = jnp.sum(unmatched * (mask_norms + score_penalty))
    num_unmatched = jnp.sum(unmatched) + 1e-8

    return loss / num_unmatched


# ---------------------------------------------------------------------------
# Box regression loss
# ---------------------------------------------------------------------------

def box_regression_loss(
    pred_boxes: jnp.ndarray,
    target_boxes: jnp.ndarray,
    matched: jnp.ndarray,
) -> jnp.ndarray:
    """Smooth L1 box regression loss for matched instances.

    Args:
        pred_boxes: Predicted boxes, shape (B, M, 4).
        target_boxes: Target boxes, shape (B, M, 4).
        matched: Boolean matched instances, shape (B, M).

    Returns:
        Scalar box regression loss.
    """
    diff = jnp.abs(pred_boxes - target_boxes)
    smooth_l1 = jnp.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)  # (B, M, 4)

    loss = jnp.sum(smooth_l1, axis=-1)  # (B, M)
    loss = jnp.sum(matched * loss)
    num_matched = jnp.sum(matched) + 1e-8

    return loss / num_matched


# ---------------------------------------------------------------------------
# Copy-Paste Augmentation Utility
# ---------------------------------------------------------------------------

def copy_paste_augment(
    images: jnp.ndarray,
    masks: jnp.ndarray,
    spatial_confidence: jnp.ndarray,
    num_valid: jnp.ndarray,
    rng: jnp.ndarray,
    top_fraction: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full copy-paste augmentation pipeline with SC.

    1. Select confident masks
    2. Randomly pair source/target images
    3. Alpha-blend using SC

    Args:
        images: Batch of images, shape (B, H, W, 3).
        masks: Instance masks at pixel resolution, shape (B, M, H, W).
        spatial_confidence: SC maps at pixel resolution, shape (B, M, H, W).
        num_valid: Valid mask count, shape (B,).
        rng: JAX random key.
        top_fraction: Fraction of masks to select.

    Returns:
        Tuple of:
            - augmented_images: Shape (B, H, W, 3).
            - augmented_masks: Shape (B, M, H, W).
            - augmented_sc: Shape (B, M, H, W).
    """
    B, M, H, W = masks.shape

    # Step 1: Select confident masks
    # Reshape masks for selection: (B, M, H*W)
    masks_flat = masks.reshape(B, M, -1)
    sc_flat = spatial_confidence.reshape(B, M, -1)
    selected, _ = confident_copy_paste_selection(
        masks_flat, sc_flat, num_valid, top_fraction
    )  # (B, M)

    # Step 2: Random pairing — shuffle batch indices
    rng_perm, rng_inst = jax.random.split(rng)
    perm = jax.random.permutation(rng_perm, B)
    source_images = images[perm]
    source_masks = masks[perm]
    source_sc = spatial_confidence[perm]
    source_selected = selected[perm]

    # Step 3: For each target image, paste the first selected mask from source
    # Combine all selected source masks into a single composite mask
    composite_mask = jnp.zeros((B, H, W), dtype=jnp.float32)
    composite_sc = jnp.zeros((B, H, W), dtype=jnp.float32)

    for m_idx in range(M):
        is_selected = source_selected[:, m_idx]  # (B,)
        mask_m = source_masks[:, m_idx]  # (B, H, W)
        sc_m = source_sc[:, m_idx]  # (B, H, W)
        # Only paste if selected and not overlapping existing paste
        paste_here = is_selected[:, None, None] * mask_m * (1.0 - composite_mask)
        composite_mask = jnp.clip(composite_mask + paste_here, 0.0, 1.0)
        composite_sc = jnp.where(paste_here > 0.5, sc_m, composite_sc)

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

    L = L_mask_SC + λ_drop · L_Drop + λ_box · L_box

    Uses Spatial Confidence Soft Target Loss (Eq. 6) as the primary
    mask loss, replacing standard BCE with per-pixel SC weighting.

    When num_valid is provided, performs IoU-based greedy matching between
    predictions and GT masks before computing losses.

    Args:
        lambda_drop: Weight for drop loss (default 0.1).
        lambda_box: Weight for box regression loss (default 1.0).
        iou_threshold: Minimum IoU for matching (default 0.1).
    """

    def __init__(
        self,
        lambda_drop: float = 0.5,
        lambda_box: float = 1.0,
        iou_threshold: float = 0.1,
    ):
        self.lambda_drop = lambda_drop
        self.lambda_box = lambda_box
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        pred_masks: jnp.ndarray,
        pred_scores: jnp.ndarray,
        features: jnp.ndarray,
        target_masks: jnp.ndarray | None = None,
        target_boxes: jnp.ndarray | None = None,
        pred_boxes: jnp.ndarray | None = None,
        spatial_confidence: jnp.ndarray | None = None,
        matched: jnp.ndarray | None = None,
        num_valid: jnp.ndarray | None = None,
    ) -> Dict[str, jnp.ndarray]:
        """Compute combined instance loss.

        Args:
            pred_masks: Predicted mask logits (B, M, N).
            pred_scores: Instance scores (B, M).
            features: Feature vectors (B, N, D).
            target_masks: Target masks (B, T, N) — padded GT masks.
            target_boxes: Target boxes (B, M, 4).
            pred_boxes: Predicted boxes (B, M, 4).
            spatial_confidence: SC maps (B, T, N) — per-GT SC maps.
            matched: Boolean matched instances (B, M) — legacy, ignored if num_valid given.
            num_valid: (B,) int — number of valid GT masks per image.
                When provided, IoU matching is used instead of pre-computed matched.

        Returns:
            Dict with 'sc_mask', 'drop', 'box', 'total' loss components.
        """
        losses = {}

        if target_masks is not None and (num_valid is not None or matched is not None):
            # --- IoU-based matching ---
            if num_valid is not None:
                pred_probs = jax.nn.sigmoid(pred_masks)  # (B, M, N)
                matched, aligned_targets, pred_to_gt, match_ious = \
                    greedy_iou_matching(
                        pred_probs, target_masks, num_valid,
                        iou_threshold=self.iou_threshold,
                    )

                # Align SC maps to matched predictions
                if spatial_confidence is not None:
                    B = pred_masks.shape[0]
                    safe_idx = jnp.maximum(pred_to_gt, 0)
                    batch_idx = jnp.arange(B)[:, None]
                    aligned_sc = spatial_confidence[batch_idx, safe_idx]  # (B, M, N)
                    aligned_sc = jnp.where(matched[:, :, None], aligned_sc, 1.0)
                else:
                    aligned_sc = None

                losses["mean_match_iou"] = jnp.sum(match_ious) / (jnp.sum(matched) + 1e-8)
                losses["n_matched"] = jnp.sum(matched).astype(jnp.float32)
            else:
                # Legacy path: pre-computed matched array
                aligned_targets = target_masks
                aligned_sc = spatial_confidence

            # --- SC Soft Target Loss (only on matched predictions) ---
            bce = optax.sigmoid_binary_cross_entropy(
                logits=pred_masks, labels=aligned_targets,
            )  # (B, M, N)

            # Weight by SC if available
            if aligned_sc is not None:
                if aligned_sc.ndim == 2:
                    sc = aligned_sc[:, None, :]
                else:
                    sc = aligned_sc
                weighted_bce = sc * bce
            else:
                weighted_bce = bce

            # Only count matched predictions in mask loss
            matched_3d = matched[:, :, None].astype(jnp.float32)  # (B, M, 1)
            l_mask = jnp.sum(weighted_bce * matched_3d) / (jnp.sum(matched_3d) * pred_masks.shape[2] + 1e-8)
            losses["sc_mask"] = l_mask

            # --- DropLoss on unmatched ---
            l_drop = drop_loss(pred_masks, pred_scores, matched)
            losses["drop"] = l_drop

            total = l_mask + self.lambda_drop * l_drop

            if pred_boxes is not None and target_boxes is not None:
                l_box = box_regression_loss(pred_boxes, target_boxes, matched)
                losses["box"] = l_box
                total = total + self.lambda_box * l_box
            else:
                losses["box"] = jnp.array(0.0)

            losses["total"] = total

        else:
            # Unsupervised mode: feature coherence loss
            b, m, n = pred_masks.shape
            mask_probs = jax.nn.sigmoid(pred_masks)

            pooled = jnp.einsum("bmn,bnd->bmd", mask_probs, features)
            mask_sums = jnp.sum(mask_probs, axis=-1, keepdims=True) + 1e-8
            centroids = pooled / mask_sums

            sim = jnp.einsum("bnd,bmd->bmn", features, centroids)
            intra_sim = jnp.sum(mask_probs * sim, axis=-1) / mask_sums.squeeze(-1)

            total_loss = -jnp.mean(pred_scores * intra_sim)

            losses["unsup_coherence"] = total_loss
            losses["total"] = total_loss

        return losses


# Backwards-compatible alias
ModelInstanceLoss = CutS3DInstanceLoss
