"""Mamba2 Panoptic Refiner (M2PR).

Takes precomputed DINOv2 features, CAUSE semantic logits, instance
descriptors, and depth maps as input. Outputs refined semantic logits
and instance embeddings via 4-directional cross-modal Mamba2 scanning
with depth FiLM conditioning.

Architecture:
    1. Input projections → two streams (semantic, instance)
    2. Depth FiLM conditioning
    3. Geometric feature injection
    4. 4-directional cross-modal Mamba2 scan (H→, ←H, V↓, ↑V)
    5. Output heads (semantic 27, instance 64, boundary 1)
    6. Gated residual connection
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbps_pytorch.models.bridge.depth_conditioning import UnifiedDepthConditioning
from mbps_pytorch.models.refiner.four_dir_scan import FourDirectionalCrossModalScan


class Mamba2PanopticRefiner(nn.Module):
    """Mamba2 Panoptic Refiner for unsupervised pseudo-label enhancement.

    Args:
        dino_dim: DINOv2 feature dimension (768).
        sem_dim: Semantic logit dimension (27 for CAUSE).
        geo_dim: Geometric feature dimension (18 = 13 depth_enc + 2 grad + 3 normal).
        inst_dim: Instance descriptor dimension (8).
        bridge_dim: Internal bridge dimension (256).
        inst_embed_dim: Output instance embedding dimension (64).
        mamba_layers: Mamba2 layers per scan direction (2).
        state_dim: SSM state dimension (64).
        chunk_size: GPU chunk size (128).
        spatial_h: Spatial grid height (32).
        spatial_w: Spatial grid width (64).
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        dino_dim: int = 768,
        sem_dim: int = 27,
        geo_dim: int = 18,
        inst_dim: int = 8,
        bridge_dim: int = 256,
        inst_embed_dim: int = 64,
        mamba_layers: int = 2,
        state_dim: int = 64,
        chunk_size: int = 128,
        spatial_h: int = 32,
        spatial_w: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.dino_dim = dino_dim
        self.sem_dim = sem_dim
        self.bridge_dim = bridge_dim
        self.inst_embed_dim = inst_embed_dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # --- 1. Input Projections ---
        self.proj_dino = nn.Sequential(
            nn.Linear(dino_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
        )
        self.proj_sem = nn.Sequential(
            nn.Linear(sem_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
        )
        self.proj_geo = nn.Sequential(
            nn.Linear(geo_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
        )
        self.proj_inst = nn.Sequential(
            nn.Linear(inst_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
        )

        # --- 2. Depth FiLM Conditioning ---
        self.depth_cond = UnifiedDepthConditioning(bridge_dim=bridge_dim)

        # --- 3. Geometric Injection (learned scales) ---
        self.geo_scale_sem = nn.Parameter(torch.tensor(0.1))
        self.geo_scale_inst = nn.Parameter(torch.tensor(0.1))

        # --- 4. Four-Directional Cross-Modal Mamba2 Scan ---
        self.four_dir_scan = FourDirectionalCrossModalScan(
            dim=bridge_dim,
            num_layers=mamba_layers,
            state_dim=state_dim,
            chunk_size=chunk_size,
            dropout_rate=dropout_rate,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )

        # --- 5. Output Heads ---
        self.sem_head = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim),
            nn.GELU(),
            nn.Linear(bridge_dim, sem_dim),
        )
        self.inst_head = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim),
            nn.GELU(),
            nn.Linear(bridge_dim, inst_embed_dim),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(bridge_dim, 1),
        )

        # --- 6. Gated Residual ---
        # sigmoid(-4) ≈ 0.018 — minimal bridge contribution at start
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        dino_features: torch.Tensor,
        cause_logits: torch.Tensor,
        geo_features: torch.Tensor,
        inst_descriptor: torch.Tensor,
        depth: torch.Tensor,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through M2PR.

        Args:
            dino_features: DINOv2 features (B, N, 768). Frozen, no grad.
            cause_logits: CAUSE semantic soft logits (B, N, 27).
            geo_features: Geometric features (B, N, 18).
            inst_descriptor: Instance descriptors (B, N, 8).
            depth: Depth values (B, N) in [0, 1].
            deterministic: Disable dropout if True.

        Returns:
            Dict with:
                refined_sem: (B, N, 27) refined semantic logits
                inst_embeddings: (B, N, 64) instance embeddings
                boundary: (B, N, 1) boundary predictions
                gate: scalar gate value
                cause_logits: (B, N, 27) original input (for KL loss)
        """
        outputs: Dict[str, torch.Tensor] = {}
        B, N, _ = dino_features.shape

        # --- 1. Input Projections ---
        h_dino = self.proj_dino(dino_features)     # (B, N, D)
        h_sem = self.proj_sem(cause_logits)        # (B, N, D)
        h_geo = self.proj_geo(geo_features)        # (B, N, D)
        h_inst = self.proj_inst(inst_descriptor)   # (B, N, D)

        # Build two streams:
        # Semantic stream = DINOv2 context + semantic signal
        stream_sem = h_dino + h_sem   # (B, N, D)
        # Instance stream = DINOv2 context + instance signal
        stream_inst = h_dino + h_inst  # (B, N, D)

        # --- 2. Depth FiLM Conditioning ---
        stream_sem, stream_inst, depth_loss = self.depth_cond(
            depth, stream_sem, stream_inst
        )
        outputs["depth_cond_loss"] = depth_loss

        # --- 3. Geometric Injection ---
        stream_sem = stream_sem + self.geo_scale_sem * h_geo
        stream_inst = stream_inst + self.geo_scale_inst * h_geo

        # --- 4. Four-Directional Cross-Modal Mamba2 Scan ---
        fused_sem, fused_inst = self.four_dir_scan(
            stream_sem, stream_inst, deterministic=deterministic
        )

        # --- 5. Output Heads ---
        sem_correction = self.sem_head(fused_sem)        # (B, N, 27)
        inst_embeddings = self.inst_head(fused_inst)     # (B, N, 64)
        boundary = torch.sigmoid(
            self.boundary_head(fused_sem + fused_inst)   # (B, N, 1)
        )

        # --- 6. Gated Residual for Semantics ---
        gate = torch.sigmoid(self.gate_logit)
        refined_sem = cause_logits + gate * sem_correction
        refined_sem = torch.clamp(refined_sem, -50.0, 50.0)

        outputs["refined_sem"] = refined_sem
        outputs["refined_sem_probs"] = F.softmax(refined_sem, dim=-1)
        outputs["inst_embeddings"] = inst_embeddings
        outputs["boundary"] = boundary
        outputs["gate"] = gate.detach()
        outputs["cause_logits"] = cause_logits
        outputs["fused_sem"] = fused_sem
        outputs["fused_inst"] = fused_inst

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}
