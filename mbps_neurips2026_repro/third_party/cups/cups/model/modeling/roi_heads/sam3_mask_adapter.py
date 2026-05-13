"""SAM3 Mask-Adapter: cross-attention between ROI box features and SAM3 mask embeddings.

For each image, pools DINOv3 features under each SAM3 binary mask → 256-dim
per-mask embedding.  ROI box features then cross-attend to the set of mask
embeddings (1 head, residual), improving semantic disambiguation for classes
that differ in shape but share depth profiles (motorcycle vs bicycle vs rider).

Architecture inspired by:
  Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation
  Shi et al., CVPR 2025.

Disabled by default (MODEL.ROI_BOX_HEAD.SAM3_MASK_ADAPTER = False).
Enable for Ablation 3 evaluation.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from detectron2.structures import Instances
from torch import nn


class SAM3MaskAdapter(nn.Module):
    """Cross-attend ROI features to SAM3 mask embeddings.

    Per-image, each SAM3 mask is embedded by mean-pooling DINOv3 patch tokens
    under the mask.  The set of embeddings is used as key/value in a single-head
    cross-attention where the ROI features are the queries.

    Args:
        vit_dim: DINOv3 patch feature dimension (768 for ViT-B/16).
        roi_dim: ROI box head output dimension (1024 for FastRCNN default).
        adapter_dim: Internal cross-attention dimension.
        n_max_masks: Maximum SAM3 masks retained per image (padded to this count).
    """

    def __init__(
        self,
        vit_dim: int = 768,
        roi_dim: int = 1024,
        adapter_dim: int = 256,
        n_max_masks: int = 20,
    ) -> None:
        super().__init__()
        self.n_max_masks = n_max_masks
        self.adapter_dim = adapter_dim

        # Project mask embeddings from vit_dim → adapter_dim
        self.mask_emb_proj = nn.Sequential(
            nn.Linear(vit_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
        )
        # Project ROI features from roi_dim → adapter_dim (query)
        self.roi_q_proj = nn.Linear(roi_dim, adapter_dim)
        # Single-head attention (QKV manually — avoids full MHA overhead)
        self.k_proj = nn.Linear(adapter_dim, adapter_dim)
        self.v_proj = nn.Linear(adapter_dim, adapter_dim)
        # Project adapter output back to roi_dim
        self.out_proj = nn.Linear(adapter_dim, roi_dim)
        self.norm = nn.LayerNorm(roi_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _pool_mask_embeddings(
        self,
        vit_feats: torch.Tensor,
        sam3_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Pool DINOv3 features per SAM3 mask.

        Args:
            vit_feats: (B, HW, C) — DINOv3 patch tokens (already flattened).
            sam3_masks: (B, N_max, HW) — flattened binary masks (float), padded with 0.

        Returns:
            mask_embs: (B, N_max, C) — per-mask pooled embeddings.
        """
        # Normalise mask weights (safe mean = sum / area, handle empty masks)
        area = sam3_masks.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, N_max, 1)
        weights = sam3_masks / area  # (B, N_max, HW)
        # (B, N_max, HW) x (B, HW, C) → (B, N_max, C)
        mask_embs = torch.bmm(weights, vit_feats)
        return mask_embs

    def forward(
        self,
        box_features: torch.Tensor,
        vit_feats_flat: torch.Tensor,
        sam3_masks_flat: torch.Tensor,
        sam3_key_padding: torch.Tensor,
        proposals: List[Instances],
    ) -> torch.Tensor:
        """Apply SAM3-guided cross-attention to ROI features.

        Args:
            box_features: (N_rois, roi_dim) — ROI box head outputs.
            vit_feats_flat: (B, HW, vit_dim) — DINOv3 patch tokens.
            sam3_masks_flat: (B, N_max, HW) — binary SAM3 masks (float), padded.
            sam3_key_padding: (B, N_max) bool — True for padded (empty) slots.
            proposals: list of Instances (one per image) to map ROIs → image index.

        Returns:
            Updated box_features of shape (N_rois, roi_dim).
        """
        B = vit_feats_flat.shape[0]
        device = box_features.device

        # 1. Compute mask embeddings for each image: (B, N_max, vit_dim)
        raw_embs = self._pool_mask_embeddings(vit_feats_flat, sam3_masks_flat)
        # Project to adapter_dim: (B, N_max, adapter_dim)
        mask_embs = self.mask_emb_proj(raw_embs)

        # 2. Build ROI → image index mapping
        rois_per_image = [len(p) for p in proposals]
        img_indices = torch.repeat_interleave(
            torch.arange(B, device=device), torch.tensor(rois_per_image, device=device)
        )  # (N_rois,)

        # 3. Project ROI features to query space
        q = self.roi_q_proj(box_features)  # (N_rois, adapter_dim)

        # 4. Cross-attention — process per-image to respect variable proposal counts
        updated_rois = torch.zeros_like(q)
        scale = self.adapter_dim ** -0.5

        for img_idx in range(B):
            roi_sel = img_indices == img_idx
            if not roi_sel.any():
                continue

            q_i = q[roi_sel]  # (n_i, adapter_dim)
            kv = mask_embs[img_idx]  # (N_max, adapter_dim)
            k = self.k_proj(kv)
            v = self.v_proj(kv)

            # Mask out padded slots
            pad_mask = sam3_key_padding[img_idx]  # (N_max,) bool — True=padded
            attn = torch.mm(q_i, k.T) * scale  # (n_i, N_max)
            if pad_mask.any():
                attn = attn.masked_fill(pad_mask.unsqueeze(0), float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            updated_rois[roi_sel] = torch.mm(attn, v)  # (n_i, adapter_dim)

        # 5. Project back to roi_dim and residual-add
        out = self.out_proj(updated_rois)  # (N_rois, roi_dim)
        return self.norm(box_features + out)
