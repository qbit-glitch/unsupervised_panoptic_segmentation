#!/usr/bin/env python3
"""Cross-Attention Fusion module for multi-modal NCut (Round 8).

Replaces the uniform weighted concatenation in mmgd_cut.py with per-token
learned fusion weights. A lightweight cross-attention block learns WHERE each
modality is most informative — DINOv3 for semantics, SSD-1B for
boundaries/texture.

Architecture:
    Q: DINOv3 tokens (N, D_dino) — primary semantic modality
    K, V: SSD-1B tokens (N, D_ssd) — secondary structural modality
    Output: per-token scalar weight α ∈ [0, 1]
    Fused: f = (1 - α) * dinov3_proj + α * ssd1b_proj

Self-supervised training:
    Loss: maximise intra-cluster cosine similarity − inter-cluster similarity
    under pseudo-labels from the best current MMGD-Cut model.
    This is a consistency loss: the fused representation should produce
    compact, separable clusters matching the NCut segmentation.

Usage:
    # Train the fusion module (requires cached NCut pseudo-segments)
    python train_cross_attention_fusion.py \
        --coco_root /path/to/coco \
        --seg_dir /path/to/coco/mmgd_segments/<config_key>/val2017 \
        --output_dir checkpoints/crossattn_fusion \
        --device mps

    # After training, integrate into mmgd_cut.py by adding:
    #   --fusion_ckpt checkpoints/crossattn_fusion/best.pth
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Per-token learned fusion weight for DINOv3 + SSD-1B.

    Uses a single cross-attention head to compute how much each DINOv3
    token should defer to the corresponding SSD-1B token.

    The fusion weight α ∈ [0, 1] is per-token and spatially adaptive:
    - Regions where DINOv3 and SSD-1B agree → α near 0.5 (balanced)
    - Regions where SSD-1B provides sharp boundaries → α near 1.0
    - Semantic regions with weak diffusion structure → α near 0.0

    Args:
        d_dino: DINOv3 feature dimension (default 1024 for ViT-L).
        d_ssd:  SSD-1B feature dimension (default 1280).
        d_proj: Shared projection dimension for cross-attention.
        n_heads: Number of cross-attention heads.
        d_out:  Output (fused) feature dimension.
    """

    def __init__(
        self,
        d_dino: int = 1024,
        d_ssd: int = 1280,
        d_proj: int = 256,
        n_heads: int = 4,
        d_out: int = 512,
    ) -> None:
        super().__init__()
        self.d_proj = d_proj
        self.d_out = d_out

        # Project each modality to common d_proj space
        self.dino_proj = nn.Linear(d_dino, d_proj, bias=False)
        self.ssd_proj = nn.Linear(d_ssd, d_proj, bias=False)

        # Cross-attention: Q=dinov3, K=V=ssd1b
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_proj,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Alpha gate: scalar per-token weight ∈ [0, 1]
        self.alpha_gate = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_proj // 2, 1),
            nn.Sigmoid(),
        )

        # Output projection: fused d_proj → d_out
        self.out_proj = nn.Linear(d_proj, d_out, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialise alpha_gate bias to 0 → α starts at 0.5
        if hasattr(self.alpha_gate[-2], "bias") and self.alpha_gate[-2].bias is not None:
            nn.init.zeros_(self.alpha_gate[-2].bias)

    def forward(
        self,
        dino_feats: torch.Tensor,
        ssd_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fused features and per-token alpha weights.

        Args:
            dino_feats: (B, N, d_dino) L2-normalised DINOv3 tokens.
            ssd_feats:  (B, N, d_ssd) L2-normalised SSD-1B tokens
                        (already aligned to same spatial grid as dino).

        Returns:
            fused:  (B, N, d_out) L2-normalised fused features.
            alpha:  (B, N, 1) per-token fusion weight ∈ [0, 1].
        """
        B, N, _ = dino_feats.shape

        # Project to common space
        q = self.dino_proj(dino_feats)   # (B, N, d_proj)
        kv = self.ssd_proj(ssd_feats)   # (B, N, d_proj)

        # Cross-attention: each DINOv3 token attends to all SSD-1B tokens
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, N, d_proj)

        # Alpha: how much to blend SSD-1B signal into each DINOv3 token
        alpha = self.alpha_gate(attn_out)  # (B, N, 1)

        # Fuse: weighted combination in projection space
        fused_proj = (1.0 - alpha) * q + alpha * attn_out  # (B, N, d_proj)

        # Output projection + L2-normalise (compatible with mmgd_cut.py fuse_features)
        fused = F.normalize(self.out_proj(fused_proj), p=2, dim=-1)

        return fused, alpha


class FusionWrapper(nn.Module):
    """Convenience wrapper that handles feature loading and applies CrossAttentionFusion.

    Designed to be a drop-in replacement for the uniform fuse_features() call
    in mmgd_cut.py. Used during evaluation after training.

    Args:
        fusion_module: Trained CrossAttentionFusion.
        target_res:    Spatial grid resolution (default 32 for 32×32).
    """

    def __init__(
        self,
        fusion_module: CrossAttentionFusion,
        target_res: int = 32,
    ) -> None:
        super().__init__()
        self.fusion = fusion_module
        self.target_res = target_res

    @torch.no_grad()
    def fuse(
        self,
        dino_feats: torch.Tensor,
        ssd_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse DINOv3 and SSD-1B features via learned cross-attention.

        Args:
            dino_feats: (N_dino, D_dino) — may be different resolution than SSD.
            ssd_feats:  (N_ssd, D_ssd).

        Returns:
            (target_res², d_out) L2-normalised fused features.
        """
        import torch.nn.functional as F_
        import math

        def _align(feats: torch.Tensor, res: int) -> torch.Tensor:
            n, d = feats.shape
            g = int(math.sqrt(n))
            if g == res:
                return F.normalize(feats, p=2, dim=1)
            feat_2d = feats.reshape(g, g, d).permute(2, 0, 1).unsqueeze(0)
            out = F_.interpolate(feat_2d, size=(res, res), mode="bilinear", align_corners=False)
            return F.normalize(out[0].permute(1, 2, 0).reshape(res * res, d), p=2, dim=1)

        d = _align(dino_feats, self.target_res).unsqueeze(0)  # (1, N, D_dino)
        s = _align(ssd_feats, self.target_res).unsqueeze(0)   # (1, N, D_ssd)

        fused, _ = self.fusion(d, s)
        return fused[0]  # (N, d_out)
