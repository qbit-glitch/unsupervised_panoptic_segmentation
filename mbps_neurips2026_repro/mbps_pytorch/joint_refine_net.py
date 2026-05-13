"""JointRefineNet: Unified Semantic + Instance Pseudo-Label Refinement.

Jointly refines CAUSE-TR 27-class semantic pseudo-labels AND produces
instance boundary/embedding predictions using a single shared Conv2d
backbone with three output heads.

Architecture:
    DINOv2 (768) → SemanticProjection → sem (bridge_dim)
    DINOv2 (768) + depth → DepthFeatureProjection → dfeat (bridge_dim)
    [sem, dfeat] → N × CoupledConvBlock → [sem_out, dfeat_out]
    sem_out → semantic_head → (27) class logits
    dfeat_out → boundary_head → (1) split probability
    [sem_out ∥ dfeat_out] → embedding_head → (embed_dim) L2-normalized

References:
    - CSCMRefineNet (this project): Proven Conv2d > Mamba at 32×64
    - de Brabandere et al. (2017): Discriminative instance embeddings
    - DepthG (CVPR 2024): Depth-boundary alignment
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .refine_net import (
    SemanticProjection,
    DepthFeatureProjection,
    CoupledConvBlock,
)


class JointRefineNet(nn.Module):
    """Joint semantic + instance refinement network.

    Takes DINOv2 features + depth (NO CAUSE logits as input — prevents
    identity shortcut) and outputs refined semantics, boundary probability,
    and instance embeddings.

    Args:
        num_classes: number of semantic classes (27 for CAUSE-TR)
        feature_dim: DINOv2 feature dimension (768 for ViT-B/14)
        bridge_dim: internal bridge dimension
        embed_dim: instance embedding dimension
        num_blocks: number of coupled conv blocks
        coupling_strength: initial alpha/beta for cross-chain coupling
        gradient_checkpointing: enable gradient checkpointing for memory
    """

    def __init__(
        self,
        num_classes: int = 27,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        embed_dim: int = 32,
        num_blocks: int = 4,
        coupling_strength: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.gradient_checkpointing = gradient_checkpointing

        # Shared projections (reused from CSCMRefineNet)
        self.sem_proj = SemanticProjection(feature_dim, bridge_dim)
        self.depth_feat_proj = DepthFeatureProjection(feature_dim, bridge_dim)

        # Shared backbone
        self.blocks = nn.ModuleList([
            CoupledConvBlock(
                d_model=bridge_dim,
                coupling_strength=coupling_strength,
            )
            for _ in range(num_blocks)
        ])

        # --- Head 1: Semantic (reads from semantic stream) ---
        self.semantic_head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )
        # Small random init (avoids identity-minimum trap)
        nn.init.normal_(self.semantic_head[-1].weight, std=0.01)
        nn.init.zeros_(self.semantic_head[-1].bias)

        # --- Head 2: Boundary (reads from depth-feature stream) ---
        self.boundary_head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

        # --- Head 3: Instance embedding (reads concat of both streams) ---
        self.embedding_head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim * 2),
            nn.Conv2d(bridge_dim * 2, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, 1),
        )

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            dinov2_features: (B, 768, H, W) — DINOv2 patch features
            depth: (B, 1, H, W) — normalized depth [0, 1]
            depth_grads: (B, 2, H, W) — Sobel gradients of depth
        Returns:
            dict with:
                semantic_logits: (B, 27, H, W)
                boundary_logits: (B, 1, H, W) — raw logits (apply sigmoid for prob)
                embeddings: (B, embed_dim, H, W) — L2-normalized
        """
        sem = self.sem_proj(dinov2_features)
        depth_feat = self.depth_feat_proj(dinov2_features, depth, depth_grads)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    block, sem, depth_feat, use_reentrant=False,
                )
            else:
                sem, depth_feat = block(sem, depth_feat)

        # Three output heads
        semantic_logits = self.semantic_head(sem)
        boundary_logits = self.boundary_head(depth_feat)

        combined = torch.cat([sem, depth_feat], dim=1)
        embeddings = self.embedding_head(combined)
        embeddings = F.normalize(embeddings, dim=1)  # L2 normalize per-pixel

        return {
            "semantic_logits": semantic_logits,
            "boundary_logits": boundary_logits,
            "embeddings": embeddings,
        }
