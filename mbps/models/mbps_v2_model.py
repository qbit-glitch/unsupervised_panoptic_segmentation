"""MBPS v2 Model: DINOv3 + Mamba Bridge Panoptic Segmentation.

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

import jax
import jax.numpy as jnp
from flax import linen as nn

from mbps.models.backbone.dinov3_vitb import DINOv3ViTB
from mbps.models.bridge.bicms import BidirectionalCrossModalScan
from mbps.models.bridge.depth_conditioning import UnifiedDepthConditioning
from mbps.models.bridge.projection import (
    AdaptiveProjectionBridge,
    InverseProjection,
)


class SemanticHeadV2(nn.Module):
    """Simple MLP semantic head for v2.

    Predicts per-token class logits from backbone features.
    Trained with cross-entropy against pseudo-labels.

    Attributes:
        backbone_dim: Input feature dimension from backbone.
        num_classes: Number of semantic classes.
    """

    backbone_dim: int = 768
    num_classes: int = 19

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Predict per-token semantic logits.

        Args:
            features: (B, N, backbone_dim) backbone features.

        Returns:
            (B, N, num_classes) class logits (unnormalized).
        """
        x = nn.Dense(self.backbone_dim, name="fc1")(features)
        x = nn.LayerNorm(name="ln1")(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.num_classes, name="fc2")(x)
        return x


class InstanceEmbeddingHead(nn.Module):
    """Per-pixel instance embedding head for v2.

    Predicts per-token embeddings that are pulled together for same-instance
    tokens and pushed apart for different-instance tokens (discriminative loss).

    Attributes:
        backbone_dim: Input feature dimension from backbone.
        embed_dim: Instance embedding dimension.
    """

    backbone_dim: int = 768
    embed_dim: int = 64

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Predict per-token instance embeddings.

        Args:
            features: (B, N, backbone_dim) backbone features.

        Returns:
            (B, N, embed_dim) instance embeddings.
        """
        x = nn.Dense(256, name="fc1")(features)
        x = nn.LayerNorm(name="ln1")(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.embed_dim, name="fc2")(x)
        return x


class MBPSv2Model(nn.Module):
    """MBPS v2 Panoptic Segmentation Model.

    Attributes:
        num_classes: Number of semantic classes.
        backbone_dim: DINOv3 feature dimension (768).
        instance_embed_dim: Instance embedding dimension.
        bridge_dim: Bridge projection dimension.
        mamba_layers: Number of Mamba2 layers per direction.
        mamba_state_dim: SSM state dimension.
        chunk_size: TPU-aligned chunk size for Mamba2.
        use_depth_conditioning: Whether to use depth FiLM conditioning.
        use_mamba_bridge: Whether to use Mamba2 bridge (vs MLP fallback).
        use_bidirectional: Whether to use bidirectional scan.
        dropout_rate: Dropout rate.
    """

    num_classes: int = 19
    backbone_dim: int = 768
    instance_embed_dim: int = 64
    bridge_dim: int = 384
    mamba_layers: int = 4
    mamba_state_dim: int = 64
    chunk_size: int = 128
    use_depth_conditioning: bool = True
    use_mamba_bridge: bool = True
    use_bidirectional: bool = True
    dropout_rate: float = 0.1

    def setup(self):
        """Initialize all sub-modules."""
        # Backbone (frozen DINOv3 ViT-B/16)
        self.backbone = DINOv3ViTB(name="backbone")

        # Semantic head
        self.semantic_head = SemanticHeadV2(
            backbone_dim=self.backbone_dim,
            num_classes=self.num_classes,
            name="semantic_head",
        )

        # Instance embedding head
        self.instance_head = InstanceEmbeddingHead(
            backbone_dim=self.backbone_dim,
            embed_dim=self.instance_embed_dim,
            name="instance_head",
        )

        # Bridge components
        self.projection = AdaptiveProjectionBridge(
            semantic_dim=self.num_classes,
            feature_dim=self.instance_embed_dim,
            bridge_dim=self.bridge_dim,
            name="projection",
        )

        if self.use_depth_conditioning:
            self.depth_cond = UnifiedDepthConditioning(
                bridge_dim=self.bridge_dim,
                name="depth_cond",
            )

        if self.use_mamba_bridge:
            self.bicms = BidirectionalCrossModalScan(
                dim=self.bridge_dim,
                num_layers=self.mamba_layers,
                state_dim=self.mamba_state_dim,
                chunk_size=self.chunk_size,
                dropout_rate=self.dropout_rate,
                name="bicms",
            )
        else:
            # Fallback: simple concatenation + MLP fusion
            self.fusion_mlp = nn.Dense(
                self.bridge_dim, name="fusion_mlp"
            )

        # Inverse projections
        self.inv_proj_semantic = InverseProjection(
            output_dim=self.num_classes,
            name="inv_proj_semantic",
        )
        self.inv_proj_instance = InverseProjection(
            output_dim=self.instance_embed_dim,
            name="inv_proj_instance",
        )

        # Bridge gate — starts near 0, ramps up during training
        # sigmoid(-4) ≈ 0.018 — prevents untrained bridge from corrupting
        self.bridge_gate_logit = self.param(
            "bridge_gate_logit",
            lambda rng, shape: jnp.full(shape, -4.0),
            (1,),
        )

    def __call__(
        self,
        image: jnp.ndarray,
        depth: Optional[jnp.ndarray] = None,
        use_bridge: bool = True,
        deterministic: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass through MBPS v2.

        Args:
            image: Input images (B, H, W, 3), ImageNet-normalized.
            depth: Optional depth maps (B, H, W), normalized [0, 1].
            use_bridge: Whether to enable bridge fusion.
            deterministic: If True, disable dropout.

        Returns:
            Dict with:
                - backbone_features: (B, N, 768)
                - semantic_logits: (B, N, K) class logits
                - semantic_probs: (B, N, K) softmax probabilities
                - instance_embeddings: (B, N, D_inst)
                - fused_semantic: (B, N, bridge_dim) [if bridge]
                - fused_instance: (B, N, bridge_dim) [if bridge]
                - reconstructed_semantic: (B, N, K) [if bridge]
                - reconstructed_instance: (B, N, D_inst) [if bridge]
                - align_loss: scalar [if bridge]
                - bridge_gate: scalar
        """
        outputs = {}

        # 1. BACKBONE: Frozen DINOv3 ViT-B/16
        features = self.backbone(image, deterministic=True)  # (B, N, 768)
        outputs["backbone_features"] = features

        # 2. SEMANTIC HEAD
        semantic_logits = self.semantic_head(features)  # (B, N, K)
        outputs["semantic_logits"] = semantic_logits
        outputs["semantic_probs"] = jax.nn.softmax(semantic_logits, axis=-1)

        # 3. INSTANCE HEAD
        instance_embeds = self.instance_head(features)  # (B, N, D_inst)
        outputs["instance_embeddings"] = instance_embeds

        # 4. BRIDGE (optional)
        if use_bridge and (self.use_mamba_bridge or hasattr(self, "fusion_mlp")):
            bridge_gate = jax.nn.sigmoid(self.bridge_gate_logit)
            outputs["bridge_gate"] = bridge_gate[0]

            # Project to bridge dimension
            sem_proj, inst_proj, align_loss = self.projection(
                semantic_logits, instance_embeds
            )
            outputs["align_loss"] = align_loss

            # Depth conditioning
            if self.use_depth_conditioning and depth is not None:
                sem_proj, inst_proj = self.depth_cond(
                    depth, sem_proj, inst_proj
                )

            # Cross-modal fusion
            if self.use_mamba_bridge:
                fused_sem, fused_inst = self.bicms(
                    sem_proj, inst_proj,
                    deterministic=deterministic,
                )
            else:
                # MLP fallback
                concat = jnp.concatenate([sem_proj, inst_proj], axis=-1)
                fused = self.fusion_mlp(concat)
                fused_sem = fused
                fused_inst = fused

            outputs["fused_semantic"] = fused_sem
            outputs["fused_instance"] = fused_inst

            # Inverse project back to original dims
            recon_sem = self.inv_proj_semantic(fused_sem)  # (B, N, K)
            recon_inst = self.inv_proj_instance(fused_inst)  # (B, N, D_inst)
            outputs["reconstructed_semantic"] = recon_sem
            outputs["reconstructed_instance"] = recon_inst

            # Gated residual: refine with bridge outputs
            semantic_logits = semantic_logits + bridge_gate * recon_sem
            instance_embeds = instance_embeds + bridge_gate * recon_inst

            # Clip for numerical safety
            semantic_logits = jnp.clip(semantic_logits, -50.0, 50.0)

            # Update outputs with refined predictions
            outputs["semantic_logits"] = semantic_logits
            outputs["semantic_probs"] = jax.nn.softmax(semantic_logits, axis=-1)
            outputs["instance_embeddings"] = instance_embeds
        else:
            outputs["bridge_gate"] = jnp.array(0.0)

        return outputs
