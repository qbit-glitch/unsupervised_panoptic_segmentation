"""Unified MBPS Model.

End-to-end Unsupervised Mamba-Bridge Panoptic Segmentation model
combining all components:
    1. Frozen DINO ViT-S/8 backbone
    2. DepthG semantic head
    3. CutS3D + Cascade instance head
    4. Adaptive Projection Bridge
    5. Unified Depth Conditioning
    6. Bidirectional Cross-Modal Scan (Mamba2)
    7. Stuff-Things Classifier
    8. Panoptic Merger
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from mbps.models.backbone.dino_vits8 import DINOViTS8
from mbps.models.bridge.bicms import BidirectionalCrossModalScan
from mbps.models.bridge.depth_conditioning import UnifiedDepthConditioning
from mbps.models.bridge.projection import (
    AdaptiveProjectionBridge,
    InverseProjection,
)
from mbps.models.classifier.stuff_things_mlp import StuffThingsClassifier
from mbps.models.instance.cascade_mask_rcnn import InstanceHead
from mbps.models.semantic.depthg_head import DepthGHead


class MBPSModel(nn.Module):
    """Unified MBPS Panoptic Segmentation Model.

    Architecture flow:
        Image → DINO backbone (frozen) → features (B, N, 384)
            ├─ Semantic branch: DepthG head → codes (B, N, 90)
            └─ Instance branch: InstanceHead → masks (B, M, N), scores (B, M)
        
        If bridge enabled:
            Projection: codes(90) + features(384) → bridge_dim(192)
            Depth conditioning: modulate projections with depth
            BiCMS: Mamba2 bidirectional scan on interleaved tokens
            Inverse projection: bridge_dim(192) → original dims
        
        Stuff-Things: classify clusters using DBD, FCC, IDF cues

    Attributes:
        num_classes: Number of semantic classes.
        semantic_dim: Semantic code dimension (90).
        feature_dim: DINO feature dimension (384).
        bridge_dim: Bridge dimension (192).
        max_instances: Maximum instances per image.
        mamba_layers: Number of Mamba2 layers per direction.
        mamba_state_dim: SSM state dimension.
        chunk_size: TPU-aligned chunk size.
        use_depth_conditioning: Whether to use depth conditioning.
        use_mamba_bridge: Whether to use Mamba2 bridge.
        use_bidirectional: Whether to use bidirectional scan.
        dropout_rate: Dropout rate.
    """

    num_classes: int = 27
    semantic_dim: int = 90
    feature_dim: int = 384
    bridge_dim: int = 192
    max_instances: int = 100
    mamba_layers: int = 4
    mamba_state_dim: int = 16
    chunk_size: int = 64
    use_depth_conditioning: bool = True
    use_mamba_bridge: bool = True
    use_bidirectional: bool = True
    dropout_rate: float = 0.1

    def setup(self):
        """Initialize all sub-modules."""
        # Backbone (frozen)
        self.backbone = DINOViTS8(name="backbone")

        # Semantic branch
        self.semantic_head = DepthGHead(
            hidden_dim=self.feature_dim,
            code_dim=self.semantic_dim,
            name="semantic_head",
        )

        # Instance branch
        self.instance_head = InstanceHead(
            max_instances=self.max_instances,
            hidden_dim=256,
            num_refinement_stages=3,
            name="instance_head",
        )

        # Bridge components
        self.projection = AdaptiveProjectionBridge(
            semantic_dim=self.semantic_dim,
            feature_dim=self.feature_dim,
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
            output_dim=self.semantic_dim,
            name="inv_proj_semantic",
        )
        self.inv_proj_features = InverseProjection(
            output_dim=self.feature_dim,
            name="inv_proj_features",
        )

        # Stuff-Things classifier
        self.stuff_things = StuffThingsClassifier(
            hidden_dims=(16, 8),
            name="stuff_things",
        )

    @nn.compact
    def __call__(
        self,
        image: jnp.ndarray,
        depth: Optional[jnp.ndarray] = None,
        use_bridge: bool = True,
        deterministic: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass through full MBPS model.

        Args:
            image: Input images of shape (B, H, W, 3).
            depth: Optional depth maps of shape (B, H, W).
            use_bridge: Whether to enable bridge fusion.
            deterministic: If True, disable dropout.

        Returns:
            Dict with all model outputs:
                - dino_features: (B, N, 384)
                - semantic_codes: (B, N, 90)
                - semantic_probs: (B, N, K) softmax over codes
                - instance_masks: (B, M, N)
                - instance_scores: (B, M)
                - fused_semantic: (B, N, D_b) (if bridge)
                - fused_features: (B, N, D_b) (if bridge)
                - reconstructed_semantic: (B, N, 90) (if bridge)
                - reconstructed_features: (B, N, 384) (if bridge)
                - align_loss: scalar (if bridge)
                - stuff_things_scores: (B, K)
        """
        outputs = {}
        b = image.shape[0]

        # ============================================
        # 1. BACKBONE: Frozen DINO ViT-S/8
        # ============================================
        # get patch tokens: (B, N, 384) where N = H/8 * W/8
        spatial_features = self.backbone(image, deterministic=True)
        outputs["dino_features"] = spatial_features

        n = spatial_features.shape[1]

        # ============================================
        # 2. SEMANTIC BRANCH: DepthG head
        # ============================================
        semantic_codes = self.semantic_head(spatial_features)  # (B, N, 90)
        outputs["semantic_codes"] = semantic_codes

        # Cluster assignments (argmax over semantic codes)
        # The 90-dim codes are clustered via linear probe to K classes
        semantic_proj = nn.Dense(
            self.num_classes, name="semantic_cluster_proj"
        )(semantic_codes)
        semantic_probs = jax.nn.softmax(semantic_proj, axis=-1)
        outputs["semantic_probs"] = semantic_probs
        outputs["semantic_pred"] = jnp.argmax(semantic_probs, axis=-1)

        # ============================================
        # 3. INSTANCE BRANCH: Cascade mask head
        # ============================================
        instance_masks, instance_scores = self.instance_head(
            spatial_features, deterministic=deterministic
        )  # (B, M, N), (B, M)
        outputs["instance_masks"] = instance_masks
        outputs["instance_scores"] = instance_scores

        # ============================================
        # 4. BRIDGE FUSION (if enabled)
        # ============================================
        if use_bridge:
            # 4a. Adaptive Projection
            sem_proj, feat_proj, align_loss = self.projection(
                semantic_codes, spatial_features
            )
            outputs["align_loss"] = align_loss

            # 4b. Depth Conditioning (optional)
            if self.use_depth_conditioning and depth is not None:
                # Flatten depth to (B, N) to match token dim
                depth_flat = self._flatten_depth(depth, n)
                sem_proj, feat_proj, depth_loss = self.depth_cond(
                    depth_flat, sem_proj, feat_proj
                )
                outputs["depth_loss"] = depth_loss
            else:
                outputs["depth_loss"] = jnp.array(0.0)

            # 4c. Cross-Modal Fusion
            if self.use_mamba_bridge:
                fused_sem, fused_feat = self.bicms(
                    sem_proj, feat_proj, deterministic=deterministic
                )
            else:
                # Simple concat + MLP fallback
                concat = jnp.concatenate([sem_proj, feat_proj], axis=-1)
                fused = self.fusion_mlp(concat)
                fused = jax.nn.gelu(fused)
                fused_sem = fused
                fused_feat = fused

            # Guard + clip fused outputs (clip alone doesn't help: clip(NaN)=NaN)
            fused_sem = jnp.where(jnp.isfinite(fused_sem), fused_sem, 0.0)
            fused_sem = jnp.clip(fused_sem, -50.0, 50.0)
            fused_feat = jnp.where(jnp.isfinite(fused_feat), fused_feat, 0.0)
            fused_feat = jnp.clip(fused_feat, -50.0, 50.0)
            outputs["fused_semantic"] = fused_sem
            outputs["fused_features"] = fused_feat

            # 4d. Inverse Projection (for reconstruction loss)
            recon_semantic = self.inv_proj_semantic(fused_sem)
            recon_features = self.inv_proj_features(fused_feat)
            # Guard against NaN from untrained inverse projections
            recon_semantic = jnp.where(
                jnp.isfinite(recon_semantic), recon_semantic, 0.0
            )
            recon_features = jnp.where(
                jnp.isfinite(recon_features), recon_features, 0.0
            )
            outputs["reconstructed_semantic"] = recon_semantic
            outputs["reconstructed_features"] = recon_features

            # Learnable bridge gate initialized near zero (sigmoid(-4) ≈ 0.018).
            # Prevents untrained bridge from corrupting trained semantic codes
            # at Phase C start. The gate learns to open as bridge trains.
            bridge_gate_logit = self.param(
                "bridge_gate",
                lambda key, shape: jnp.full(shape, -4.0),
                (1,),
            )
            bridge_scale = jax.nn.sigmoid(bridge_gate_logit)

            # Guard against NaN from untrained bridge pathway
            recon_safe = jnp.where(
                jnp.isfinite(recon_semantic), recon_semantic, 0.0
            )
            fused_semantic_codes = semantic_codes + bridge_scale * recon_safe
            outputs["semantic_codes_fused"] = fused_semantic_codes
            outputs["bridge_gate"] = bridge_scale

            # Re-cluster with fused codes
            fused_proj = nn.Dense(
                self.num_classes, name="fused_cluster_proj"
            )(fused_semantic_codes)
            fused_probs = jax.nn.softmax(fused_proj, axis=-1)
            outputs["semantic_probs"] = fused_probs
            outputs["semantic_pred"] = jnp.argmax(fused_probs, axis=-1)

        # ============================================
        # 5. STUFF-THINGS CLASSIFICATION
        # ============================================
        # Use simplified cues from output. Full cue computation
        # happens outside the model (in trainer) for efficiency.
        # Here we provide a placeholder based on semantic probs.
        k = self.num_classes
        cluster_features = jnp.zeros((b, k, 3))

        # Simple proxy cues from semantic predictions
        sem_pred = outputs["semantic_pred"]
        for c in range(k):
            mask = (sem_pred == c).astype(jnp.float32)
            cluster_size = jnp.sum(mask, axis=-1, keepdims=True) + 1e-8
            # Proxy DBD: variance of instance scores within cluster
            # Proxy FCC: semantic probability concentration
            # Proxy IDF: number of instances overlapping
            cluster_features = cluster_features.at[:, c, 0].set(
                jnp.mean(mask * jnp.max(jax.nn.sigmoid(instance_masks), axis=1), axis=-1)
            )
            cluster_features = cluster_features.at[:, c, 1].set(
                jnp.mean(jnp.max(semantic_probs * mask[:, :, None], axis=1), axis=-1)
            )
            cluster_features = cluster_features.at[:, c, 2].set(
                jnp.sum(
                    (jnp.sum(jax.nn.sigmoid(instance_masks) * mask[:, None, :], axis=-1) > 0.1).astype(jnp.float32),
                    axis=-1,
                ) / cluster_size.squeeze(-1)
            )

        stuff_things_scores = self.stuff_things(
            cluster_features, deterministic=deterministic
        )
        outputs["stuff_things_scores"] = stuff_things_scores

        return outputs

    def _flatten_depth(
        self, depth: jnp.ndarray, num_tokens: int
    ) -> jnp.ndarray:
        """Flatten spatial depth map to token-level depth.

        Args:
            depth: Depth of shape (B, H, W) or (B, H, W, 1).
            num_tokens: Number of spatial tokens N = (H/8)*(W/8).

        Returns:
            Flattened depth of shape (B, N).
        """
        if depth.ndim == 4:
            depth = depth.squeeze(-1)

        b, h, w = depth.shape

        # Compute token grid from actual image dimensions
        patch_size = 8  # DINO ViT-S/8 patch size
        h_tokens = h // patch_size
        w_tokens = w // patch_size

        # Average pool to token resolution
        depth_resized = jax.image.resize(
            depth[:, :, :, None],
            (b, h_tokens, w_tokens, 1),
            method="bilinear",
        )

        depth_resized = depth_resized.squeeze(-1)

        return depth_resized.reshape(b, -1)
