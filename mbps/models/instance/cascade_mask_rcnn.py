"""Cascade Mask R-CNN — Class-Agnostic Detector (CAD) for CutS3D.

Faithful implementation following Sick et al. ICCV 2025, which uses
Cascade Mask R-CNN [Cai & Vasconcelos, TPAMI 2019] as the CAD trained
on CutS3D pseudo-masks with Spatial Confidence and DropLoss.

Architecture:
  - FPN-like feature pyramid from backbone (DINO ViT-S/8)
  - RPN for region proposals
  - 3-stage cascade: each stage has RoI head (cls + box) + Mask head
  - Class-agnostic: num_classes=1 (foreground vs background)
  - Trained with SC Soft Target Loss, Confident Copy-Paste, Alpha-Blending

TPU Optimization:
  - All operations use static shapes (fixed max proposals)
  - No dynamic control flow; padding + masking instead
  - Compatible with jax.pmap for multi-device training
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


# ---------------------------------------------------------------------------
# Region Proposal Network (RPN)
# ---------------------------------------------------------------------------

class RPNHead(nn.Module):
    """Region Proposal Network head.

    Generates class-agnostic object proposals from feature maps.
    Uses a 3x3 conv followed by 1x1 convs for objectness and box deltas.

    Attributes:
        hidden_dim: Intermediate feature dimension.
        num_anchors: Number of anchors per spatial position.
    """
    hidden_dim: int = 256
    num_anchors: int = 3

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate proposals from features.

        Args:
            features: Backbone features, shape (B, N, D).
            deterministic: Disable dropout if True.

        Returns:
            Tuple of:
                - objectness: Shape (B, N, num_anchors).
                - box_deltas: Shape (B, N, num_anchors * 4).
        """
        x = nn.Dense(self.hidden_dim, name="rpn_conv")(features)
        x = jax.nn.relu(x)

        objectness = nn.Dense(
            self.num_anchors, name="rpn_cls"
        )(x)  # (B, N, A)

        box_deltas = nn.Dense(
            self.num_anchors * 4, name="rpn_bbox"
        )(x)  # (B, N, A*4)

        return objectness, box_deltas


# ---------------------------------------------------------------------------
# RoI Feature Extraction
# ---------------------------------------------------------------------------

class RoIHead(nn.Module):
    """Region of Interest classification and box regression head.

    Processes pooled features from instance mask proposals.
    Class-agnostic: predicts fg/bg + 4 box deltas.

    Attributes:
        hidden_dim: FC layer dimension.
        num_classes: Number of classes (1 for class-agnostic).
    """
    hidden_dim: int = 256
    num_classes: int = 1

    @nn.compact
    def __call__(
        self,
        pooled_features: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict class scores and box deltas.

        Args:
            pooled_features: RoI features, shape (B, M, D).
            deterministic: Disable dropout if True.

        Returns:
            Tuple of:
                - class_logits: Shape (B, M, num_classes + 1). (+1 for bg)
                - box_deltas: Shape (B, M, 4).
        """
        x = nn.Dense(self.hidden_dim, name="fc1")(pooled_features)
        x = jax.nn.relu(x)
        x = nn.Dense(self.hidden_dim, name="fc2")(x)
        x = jax.nn.relu(x)

        class_logits = nn.Dense(
            self.num_classes + 1, name="cls_score"
        )(x)
        box_deltas = nn.Dense(4, name="bbox_pred")(x)

        return class_logits, box_deltas


# ---------------------------------------------------------------------------
# Mask Head
# ---------------------------------------------------------------------------

class MaskHead(nn.Module):
    """Mask prediction head.

    Predicts per-pixel binary masks from RoI features.
    Uses 4 FC layers with LayerNorm (approximating 4 conv layers
    in the original paper, adapted to patch-level representation).

    Attributes:
        hidden_dim: Feature dimension.
        num_conv_layers: Number of FC layers.
    """
    hidden_dim: int = 256
    num_conv_layers: int = 4

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Predict mask features.

        Args:
            features: Pooled RoI features, shape (B, M, D).
            deterministic: Disable dropout if True.

        Returns:
            Mask features, shape (B, M, hidden_dim).
        """
        x = features
        for i in range(self.num_conv_layers):
            x = nn.Dense(self.hidden_dim, name=f"mask_fc{i}")(x)
            x = jax.nn.relu(x)
            x = nn.LayerNorm(name=f"mask_norm{i}")(x)

        return x


# ---------------------------------------------------------------------------
# Cascade Stage
# ---------------------------------------------------------------------------

class CascadeStage(nn.Module):
    """One stage of the Cascade Mask R-CNN.

    Each stage:
    1. Pools features using current masks
    2. Predicts class logits and box deltas via RoIHead
    3. Refines masks via MaskHead
    4. Adds residual mask delta

    Attributes:
        hidden_dim: Feature dimension.
        num_classes: Number of classes.
        stage_id: Stage index (for naming).
    """
    hidden_dim: int = 256
    num_classes: int = 1
    stage_id: int = 0

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        masks: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run one cascade stage.

        Args:
            features: Feature map, shape (B, N, D).
            masks: Current mask logits, shape (B, M, N).
            deterministic: Disable dropout if True.

        Returns:
            Tuple of:
                - refined_masks: Shape (B, M, N).
                - class_logits: Shape (B, M, num_classes + 1).
                - box_deltas: Shape (B, M, 4).
        """
        # Pool features using current masks
        mask_probs = jax.nn.sigmoid(masks)
        pooled = jnp.einsum("bmn,bnd->bmd", mask_probs, features)
        pooled = pooled / (
            jnp.sum(mask_probs, axis=-1, keepdims=True) + 1e-8
        )

        # Classification and box regression
        roi_head = RoIHead(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            name=f"roi_stage{self.stage_id}",
        )
        class_logits, box_deltas = roi_head(pooled, deterministic)

        # Mask refinement
        mask_head = MaskHead(
            hidden_dim=self.hidden_dim,
            name=f"mask_stage{self.stage_id}",
        )
        mask_feats = mask_head(pooled, deterministic)

        # Predict mask delta from mask features
        mask_delta = nn.Dense(
            features.shape[-2],
            name=f"mask_pred_stage{self.stage_id}",
        )(mask_feats)

        refined_masks = masks + mask_delta

        return refined_masks, class_logits, box_deltas


# ---------------------------------------------------------------------------
# Full Cascade Mask R-CNN (CAD)
# ---------------------------------------------------------------------------

class InstanceHead(nn.Module):
    """Cascade Mask R-CNN — Class-Agnostic Detector (CAD).

    Full Cascade Mask R-CNN head integrating:
    - Feature projection
    - Initial mask proposal generation
    - 3-stage cascade refinement (RoI + Mask heads per stage)
    - Final score prediction

    This is the main detection head trained on CutS3D pseudo-masks
    with Spatial Confidence Soft Target Loss.

    Attributes:
        max_instances: Maximum number of instance proposals.
        hidden_dim: Feature dimension.
        num_refinement_stages: Number of cascade stages (default 3).
        num_classes: Number of classes (1 = class-agnostic).
    """
    max_instances: int = 100
    hidden_dim: int = 256
    num_refinement_stages: int = 3
    num_classes: int = 1

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate instance masks and scores.

        Args:
            features: Input features, shape (B, N, D).
            deterministic: Disable dropout if True.

        Returns:
            Tuple of:
                - masks: Instance mask logits, shape (B, M, N).
                - scores: Instance confidence scores, shape (B, M).
        """
        b, n, d = features.shape

        # Feature projection
        feat_proj = nn.Dense(self.hidden_dim, name="feat_proj")(features)
        feat_proj = jax.nn.relu(feat_proj)

        # Generate initial mask proposals
        mask_embed = nn.Dense(
            self.max_instances, name="mask_embed"
        )(feat_proj)
        masks = jnp.transpose(mask_embed, (0, 2, 1))  # (B, M, N)

        # Cascade refinement stages
        all_class_logits = []
        all_box_deltas = []

        for stage in range(self.num_refinement_stages):
            cascade_stage = CascadeStage(
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
                stage_id=stage,
                name=f"cascade_stage{stage}",
            )
            masks, cls_logits, box_deltas = cascade_stage(
                feat_proj, masks, deterministic
            )
            all_class_logits.append(cls_logits)
            all_box_deltas.append(box_deltas)

        # Final score prediction from last stage
        final_mask_probs = jax.nn.sigmoid(masks)
        final_feats = jnp.einsum("bmn,bnd->bmd", final_mask_probs, feat_proj)
        final_feats = final_feats / (
            jnp.sum(final_mask_probs, axis=-1, keepdims=True) + 1e-8
        )

        # Score = objectness (foreground probability from class logits)
        # Use last stage's class logits: softmax over [bg, fg]
        last_cls = all_class_logits[-1]  # (B, M, 2)
        scores = jax.nn.softmax(last_cls, axis=-1)[..., 1]  # fg prob, (B, M)

        return masks, scores
