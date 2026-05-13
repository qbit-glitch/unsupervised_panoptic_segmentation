"""MBPS v2 Model: DINOv3 + Mamba Bridge Panoptic Segmentation (PyTorch).

Simplified architecture compared to v1:
  - DINOv3 ViT-B/16 backbone (768-dim) instead of DINO ViT-S/8 (384-dim)
  - Simple MLP semantic head with cross-entropy (no STEGO/DepthG)
  - Per-pixel instance embeddings with discriminative loss (no Cascade Mask R-CNN)
  - Same Mamba2 bridge for cross-modal fusion (our contribution)
  - Trained on pseudo-labels (not end-to-end unsupervised)

Architecture:
    Image -> DINOv3 ViT-B (frozen) -> features (B, N, 768)
        |-- Semantic head: MLP(768->K) -> class logits
        |-- Instance head: MLP(768->D_inst) -> instance embeddings
        |-- Mamba Bridge:
        |     |-- Project (K + D_inst -> bridge_dim)
        |     |-- Depth FiLM conditioning
        |     |-- Mamba2 BiCMS (bidirectional cross-modal scan)
        |     |-- Inverse project -> refined logits + embeddings
        |-- Panoptic merge
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbps_pytorch.models.backbone.dinov3_vitb import DINOv3ViTB
from mbps_pytorch.models.bridge.bicms import BidirectionalCrossModalScan
from mbps_pytorch.models.bridge.depth_conditioning import UnifiedDepthConditioning
from mbps_pytorch.models.bridge.projection import (
    AdaptiveProjectionBridge,
    InverseProjection,
)


class SemanticHeadV2(nn.Module):
    """Simple MLP semantic head for v2.

    Predicts per-token class logits from backbone features.

    Args:
        backbone_dim: Input feature dimension from backbone.
        num_classes: Number of semantic classes.
    """

    def __init__(self, backbone_dim: int = 768, num_classes: int = 19) -> None:
        super().__init__()
        self.fc1 = nn.Linear(backbone_dim, backbone_dim)
        self.ln1 = nn.LayerNorm(backbone_dim)
        self.fc2 = nn.Linear(backbone_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, N, backbone_dim) -> (B, N, num_classes) logits."""
        x = self.fc1(features)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class InstanceEmbeddingHead(nn.Module):
    """Per-pixel instance embedding head for v2.

    Args:
        backbone_dim: Input feature dimension from backbone.
        embed_dim: Instance embedding dimension.
    """

    def __init__(self, backbone_dim: int = 768, embed_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(backbone_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, N, backbone_dim) -> (B, N, embed_dim)."""
        x = self.fc1(features)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class MBPSv2Model(nn.Module):
    """MBPS v2 Panoptic Segmentation Model (PyTorch).

    Args:
        num_classes: Number of semantic classes.
        backbone_dim: DINOv3 feature dimension (768).
        instance_embed_dim: Instance embedding dimension.
        bridge_dim: Bridge projection dimension.
        mamba_layers: Number of Mamba2 layers per direction.
        mamba_state_dim: SSM state dimension.
        chunk_size: GPU-aligned chunk size for Mamba2.
        use_depth_conditioning: Whether to use depth FiLM conditioning.
        use_mamba_bridge: Whether to use Mamba2 bridge (vs MLP fallback).
        use_bidirectional: Whether to use bidirectional scan.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 19,
        backbone_dim: int = 768,
        instance_embed_dim: int = 64,
        bridge_dim: int = 384,
        mamba_layers: int = 4,
        mamba_state_dim: int = 64,
        chunk_size: int = 128,
        use_depth_conditioning: bool = True,
        use_mamba_bridge: bool = True,
        use_bidirectional: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_dim = backbone_dim
        self.instance_embed_dim = instance_embed_dim
        self.bridge_dim = bridge_dim
        self.use_depth_conditioning = use_depth_conditioning
        self.use_mamba_bridge = use_mamba_bridge
        self.use_bidirectional = use_bidirectional

        # Backbone (frozen DINOv3 ViT-B/16)
        self.backbone = DINOv3ViTB(
            patch_size=16,
            embed_dim=backbone_dim,
            freeze=True,
        )

        # Semantic head
        self.semantic_head = SemanticHeadV2(
            backbone_dim=backbone_dim,
            num_classes=num_classes,
        )

        # Instance embedding head
        self.instance_head = InstanceEmbeddingHead(
            backbone_dim=backbone_dim,
            embed_dim=instance_embed_dim,
        )

        # Bridge components
        self.projection = AdaptiveProjectionBridge(
            semantic_dim=num_classes,
            feature_dim=instance_embed_dim,
            bridge_dim=bridge_dim,
        )

        if use_depth_conditioning:
            self.depth_cond = UnifiedDepthConditioning(
                bridge_dim=bridge_dim,
            )

        if use_mamba_bridge:
            self.bicms = BidirectionalCrossModalScan(
                dim=bridge_dim,
                num_layers=mamba_layers,
                state_dim=mamba_state_dim,
                chunk_size=chunk_size,
                dropout_rate=dropout_rate,
            )
        else:
            self.fusion_mlp = nn.Linear(bridge_dim * 2, bridge_dim)

        # Inverse projections
        self.inv_proj_semantic = InverseProjection(
            input_dim=bridge_dim,
            output_dim=num_classes,
        )
        self.inv_proj_instance = InverseProjection(
            input_dim=bridge_dim,
            output_dim=instance_embed_dim,
        )

        # Bridge gate — starts near 0, ramps up during training
        # sigmoid(-4) ≈ 0.018 — prevents untrained bridge from corrupting
        self.bridge_gate_logit = nn.Parameter(torch.full((1,), -4.0))

    def forward(
        self,
        image: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        use_bridge: bool = True,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through MBPS v2.

        Args:
            image: Input images (B, C, H, W), ImageNet-normalized (NCHW).
            depth: Optional depth maps (B, H, W) or (B, 1, H, W).
            use_bridge: Whether to enable bridge fusion.
            deterministic: If True, disable dropout.

        Returns:
            Dict with model outputs.
        """
        outputs: Dict[str, torch.Tensor] = {}

        # 1. BACKBONE: Frozen DINOv3 ViT-B/16
        with torch.no_grad():
            features = self.backbone(image)  # (B, N, 768)
        features = features.detach()
        outputs["backbone_features"] = features

        n = features.shape[1]

        # 2. SEMANTIC HEAD
        semantic_logits = self.semantic_head(features)  # (B, N, K)
        outputs["semantic_logits"] = semantic_logits
        outputs["semantic_probs"] = F.softmax(semantic_logits, dim=-1)

        # 3. INSTANCE HEAD
        instance_embeds = self.instance_head(features)  # (B, N, D_inst)
        outputs["instance_embeddings"] = instance_embeds

        # 4. BRIDGE (optional)
        if use_bridge and (self.use_mamba_bridge or hasattr(self, "fusion_mlp")):
            bridge_gate = torch.sigmoid(self.bridge_gate_logit)
            outputs["bridge_gate"] = bridge_gate[0]

            # Project to bridge dimension
            sem_proj, inst_proj, align_loss = self.projection(
                semantic_logits, instance_embeds
            )
            outputs["align_loss"] = align_loss

            # Depth conditioning
            if self.use_depth_conditioning and depth is not None:
                depth_flat = self._flatten_depth(depth, n)
                sem_proj, inst_proj, depth_loss = self.depth_cond(
                    depth_flat, sem_proj, inst_proj
                )
                outputs["depth_loss"] = depth_loss
            else:
                outputs["depth_loss"] = torch.tensor(0.0, device=image.device)

            # Cross-modal fusion
            if self.use_mamba_bridge:
                fused_sem, fused_inst = self.bicms(
                    sem_proj, inst_proj, deterministic=deterministic
                )
            else:
                concat = torch.cat([sem_proj, inst_proj], dim=-1)
                fused = self.fusion_mlp(concat)
                fused = F.gelu(fused)
                fused_sem = fused
                fused_inst = fused

            outputs["fused_semantic"] = fused_sem
            outputs["fused_instance"] = fused_inst

            # Inverse project back to original dims
            recon_sem = self.inv_proj_semantic(fused_sem)
            recon_inst = self.inv_proj_instance(fused_inst)
            outputs["reconstructed_semantic"] = recon_sem
            outputs["reconstructed_instance"] = recon_inst

            # Gated residual: refine with bridge outputs
            semantic_logits = semantic_logits + bridge_gate * recon_sem
            instance_embeds = instance_embeds + bridge_gate * recon_inst

            # Clip for numerical safety
            semantic_logits = torch.clamp(semantic_logits, -50.0, 50.0)

            # Update outputs with refined predictions
            outputs["semantic_logits"] = semantic_logits
            outputs["semantic_probs"] = F.softmax(semantic_logits, dim=-1)
            outputs["instance_embeddings"] = instance_embeds
        else:
            outputs["bridge_gate"] = torch.tensor(0.0, device=image.device)

        return outputs

    def _flatten_depth(
        self, depth: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Flatten spatial depth map to token-level depth.

        Args:
            depth: Depth of shape (B, H, W) or (B, 1, H, W).
            num_tokens: Number of spatial tokens N.

        Returns:
            Flattened depth of shape (B, N).
        """
        if depth.ndim == 4:
            depth = depth.squeeze(1)

        b, h, w = depth.shape
        patch_size = 16
        h_tokens = h // patch_size
        w_tokens = w // patch_size

        depth_4d = depth.unsqueeze(1)
        depth_resized = F.interpolate(
            depth_4d, size=(h_tokens, w_tokens), mode="bilinear", align_corners=False,
        )
        return depth_resized.squeeze(1).reshape(b, -1)
