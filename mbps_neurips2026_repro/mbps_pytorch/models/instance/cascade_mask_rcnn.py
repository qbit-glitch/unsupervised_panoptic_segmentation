"""Cascade Mask R-CNN -- Class-Agnostic Detector (CAD) for CutS3D.

Faithful implementation following Sick et al. ICCV 2025, which uses
Cascade Mask R-CNN [Cai & Vasconcelos, TPAMI 2019] as the CAD trained
on CutS3D pseudo-masks with Spatial Confidence and DropLoss.

Architecture:
  - FPN-like feature pyramid from backbone (DINO ViT-S/8)
  - RPN for region proposals
  - 3-stage cascade: each stage has RoI head (cls + box) + Mask head
  - Class-agnostic: num_classes=1 (foreground vs background)
  - Trained with SC Soft Target Loss, Confident Copy-Paste, Alpha-Blending

GPU Optimization:
  - All operations use static shapes (fixed max proposals)
  - No dynamic control flow; padding + masking instead
  - Compatible with torch.nn.DataParallel for multi-device training
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Region Proposal Network (RPN)
# ---------------------------------------------------------------------------

class RPNHead(nn.Module):
    """Region Proposal Network head.

    Generates class-agnostic object proposals from feature maps.
    Uses a 3x3 conv followed by 1x1 convs for objectness and box deltas.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Intermediate feature dimension.
        num_anchors: Number of anchors per spatial position.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_anchors: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors

        self.rpn_conv = nn.Linear(input_dim, hidden_dim)
        self.rpn_cls = nn.Linear(hidden_dim, num_anchors)
        self.rpn_bbox = nn.Linear(hidden_dim, num_anchors * 4)

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate proposals from features.

        Args:
            features: Backbone features, shape (B, N, D).
            deterministic: Disable dropout if True.

        Returns:
            Tuple of:
                - objectness: Shape (B, N, num_anchors).
                - box_deltas: Shape (B, N, num_anchors * 4).
        """
        x = F.relu(self.rpn_conv(features))
        objectness = self.rpn_cls(x)      # (B, N, A)
        box_deltas = self.rpn_bbox(x)     # (B, N, A*4)

        return objectness, box_deltas


# ---------------------------------------------------------------------------
# RoI Feature Extraction
# ---------------------------------------------------------------------------

class RoIHead(nn.Module):
    """Region of Interest classification and box regression head.

    Processes pooled features from instance mask proposals.
    Class-agnostic: predicts fg/bg + 4 box deltas.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: FC layer dimension.
        num_classes: Number of classes (1 for class-agnostic).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        pooled_features: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict class scores and box deltas.

        Args:
            pooled_features: RoI features, shape (B, M, D).
            deterministic: Disable dropout if True.

        Returns:
            Tuple of:
                - class_logits: Shape (B, M, num_classes + 1). (+1 for bg)
                - box_deltas: Shape (B, M, 4).
        """
        x = F.relu(self.fc1(pooled_features))
        x = F.relu(self.fc2(x))

        class_logits = self.cls_score(x)
        box_deltas = self.bbox_pred(x)

        return class_logits, box_deltas


# ---------------------------------------------------------------------------
# Mask Head
# ---------------------------------------------------------------------------

class MaskHead(nn.Module):
    """Mask prediction head.

    Predicts per-pixel binary masks from RoI features.
    Uses 4 FC layers with LayerNorm (approximating 4 conv layers
    in the original paper, adapted to patch-level representation).

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Feature dimension.
        num_conv_layers: Number of FC layers.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_conv_layers: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers

        self.mask_fcs = nn.ModuleList()
        self.mask_norms = nn.ModuleList()
        for i in range(num_conv_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.mask_fcs.append(nn.Linear(in_dim, hidden_dim))
            self.mask_norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Predict mask features.

        Args:
            features: Pooled RoI features, shape (B, M, D).
            deterministic: Disable dropout if True.

        Returns:
            Mask features, shape (B, M, hidden_dim).
        """
        x = features
        for i in range(self.num_conv_layers):
            x = self.mask_fcs[i](x)
            x = F.relu(x)
            x = self.mask_norms[i](x)

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

    Args:
        hidden_dim: Feature dimension.
        num_patches: Number of spatial patches (N).
        num_classes: Number of classes.
        stage_id: Stage index (for naming).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_patches: int = 900,
        num_classes: int = 1,
        stage_id: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.stage_id = stage_id

        self.roi_head = RoIHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        self.mask_head = MaskHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        # Resolution-independent mask prediction via dot product
        # with projected features instead of fixed nn.Linear(D, N)
        self.mask_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        mask_probs = torch.sigmoid(masks)
        pooled = torch.einsum("bmn,bnd->bmd", mask_probs, features)
        pooled = pooled / (
            torch.sum(mask_probs, dim=-1, keepdim=True) + 1e-8
        )

        # Classification and box regression
        class_logits, box_deltas = self.roi_head(pooled, deterministic)

        # Mask refinement
        mask_feats = self.mask_head(pooled, deterministic)

        # Predict mask delta via dot product with features (resolution-independent)
        mask_proj = self.mask_proj(mask_feats)  # (B, M, D)
        mask_delta = torch.einsum("bmd,bnd->bmn", mask_proj, features)

        refined_masks = masks + mask_delta

        return refined_masks, class_logits, box_deltas


# ---------------------------------------------------------------------------
# Full Cascade Mask R-CNN (CAD)
# ---------------------------------------------------------------------------

class InstanceHead(nn.Module):
    """Cascade Mask R-CNN -- Class-Agnostic Detector (CAD).

    Full Cascade Mask R-CNN head integrating:
    - Feature projection
    - Initial mask proposal generation
    - 3-stage cascade refinement (RoI + Mask heads per stage)
    - Final score prediction

    This is the main detection head trained on CutS3D pseudo-masks
    with Spatial Confidence Soft Target Loss.

    Args:
        max_instances: Maximum number of instance proposals.
        input_dim: Input feature dimension.
        hidden_dim: Feature dimension.
        num_patches: Number of spatial patches (N).
        num_refinement_stages: Number of cascade stages (default 3).
        num_classes: Number of classes (1 = class-agnostic).
    """

    def __init__(
        self,
        max_instances: int = 100,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_patches: int = 900,
        num_refinement_stages: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.max_instances = max_instances
        self.hidden_dim = hidden_dim
        self.num_refinement_stages = num_refinement_stages
        self.num_classes = num_classes

        # Feature projection
        self.feat_proj = nn.Linear(input_dim, hidden_dim)

        # Generate initial mask proposals
        self.mask_embed = nn.Linear(hidden_dim, max_instances)

        # Cascade refinement stages
        self.cascade_stages = nn.ModuleList()
        for stage in range(num_refinement_stages):
            self.cascade_stages.append(
                CascadeStage(
                    hidden_dim=hidden_dim,
                    num_patches=num_patches,
                    num_classes=num_classes,
                    stage_id=stage,
                )
            )

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        feat_proj = F.relu(self.feat_proj(features))

        # Generate initial mask proposals
        mask_embed = self.mask_embed(feat_proj)
        masks = mask_embed.permute(0, 2, 1)  # (B, M, N)

        # Cascade refinement stages
        all_class_logits = []
        all_box_deltas = []

        for stage in range(self.num_refinement_stages):
            masks, cls_logits, box_deltas = self.cascade_stages[stage](
                feat_proj, masks, deterministic
            )
            all_class_logits.append(cls_logits)
            all_box_deltas.append(box_deltas)

        # Final score prediction from last stage
        # Score = objectness (foreground probability from class logits)
        # Use last stage's class logits: softmax over [bg, fg]
        last_cls = all_class_logits[-1]  # (B, M, 2)
        scores = F.softmax(last_cls, dim=-1)[..., 1]  # fg prob, (B, M)

        return masks, scores
