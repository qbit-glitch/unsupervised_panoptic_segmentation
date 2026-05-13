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

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbps_pytorch.models.backbone.dino_vits8 import DINOViTS8
from mbps_pytorch.models.bridge.bicms import BidirectionalCrossModalScan
from mbps_pytorch.models.bridge.depth_conditioning import UnifiedDepthConditioning
from mbps_pytorch.models.bridge.projection import (
    AdaptiveProjectionBridge,
    InverseProjection,
)
from mbps_pytorch.models.classifier.stuff_things_mlp import StuffThingsClassifier
from mbps_pytorch.models.instance.cascade_mask_rcnn import InstanceHead
from mbps_pytorch.models.semantic.depthg_head import DepthGHead


class MBPSModel(nn.Module):
    """Unified MBPS Panoptic Segmentation Model.

    Architecture flow:
        Image -> DINO backbone (frozen) -> features (B, N, 384)
            +-- Semantic branch: DepthG head -> codes (B, N, 90)
            +-- Instance branch: InstanceHead -> masks (B, M, N), scores (B, M)

        If bridge enabled:
            Projection: codes(90) + features(384) -> bridge_dim(192)
            Depth conditioning: modulate projections with depth
            BiCMS: Mamba2 bidirectional scan on interleaved tokens
            Inverse projection: bridge_dim(192) -> original dims

        Stuff-Things: classify clusters using DBD, FCC, IDF cues

    Attributes:
        num_classes: Number of semantic classes.
        semantic_dim: Semantic code dimension (90).
        feature_dim: DINO feature dimension (384).
        bridge_dim: Bridge dimension (192).
        max_instances: Maximum instances per image.
        mamba_layers: Number of Mamba2 layers per direction.
        mamba_state_dim: SSM state dimension.
        chunk_size: GPU-aligned chunk size.
        use_depth_conditioning: Whether to use depth conditioning.
        use_mamba_bridge: Whether to use Mamba2 bridge.
        use_bidirectional: Whether to use bidirectional scan.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 27,
        semantic_dim: int = 90,
        feature_dim: int = 384,
        bridge_dim: int = 192,
        max_instances: int = 100,
        mamba_layers: int = 4,
        mamba_state_dim: int = 64,
        chunk_size: int = 128,
        use_depth_conditioning: bool = True,
        use_mamba_bridge: bool = True,
        use_bidirectional: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize all sub-modules.

        Args:
            num_classes: Number of semantic classes.
            semantic_dim: Semantic code dimension (90).
            feature_dim: DINO feature dimension (384).
            bridge_dim: Bridge dimension (192).
            max_instances: Maximum instances per image.
            mamba_layers: Number of Mamba2 layers per direction.
            mamba_state_dim: SSM state dimension.
            chunk_size: GPU-aligned chunk size.
            use_depth_conditioning: Whether to use depth conditioning.
            use_mamba_bridge: Whether to use Mamba2 bridge.
            use_bidirectional: Whether to use bidirectional scan.
            dropout_rate: Dropout rate.
        """
        super().__init__()
        self.num_classes = num_classes
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim
        self.bridge_dim = bridge_dim
        self.max_instances = max_instances
        self.mamba_layers = mamba_layers
        self.mamba_state_dim = mamba_state_dim
        self.chunk_size = chunk_size
        self.use_depth_conditioning = use_depth_conditioning
        self.use_mamba_bridge = use_mamba_bridge
        self.use_bidirectional = use_bidirectional
        self.dropout_rate = dropout_rate

        # Backbone (frozen)
        self.backbone = DINOViTS8()

        # Semantic branch
        self.semantic_head = DepthGHead(
            hidden_dim=self.feature_dim,
            code_dim=self.semantic_dim,
        )

        # Instance branch
        self.instance_head = InstanceHead(
            max_instances=self.max_instances,
            hidden_dim=256,
            num_refinement_stages=3,
        )

        # Semantic cluster projection: semantic_dim -> num_classes
        self.semantic_cluster_proj = nn.Linear(self.semantic_dim, self.num_classes)

        # Bridge components
        self.projection = AdaptiveProjectionBridge(
            semantic_dim=self.semantic_dim,
            feature_dim=self.feature_dim,
            bridge_dim=self.bridge_dim,
        )

        if self.use_depth_conditioning:
            self.depth_cond = UnifiedDepthConditioning(
                bridge_dim=self.bridge_dim,
            )

        if self.use_mamba_bridge:
            self.bicms = BidirectionalCrossModalScan(
                dim=self.bridge_dim,
                num_layers=self.mamba_layers,
                state_dim=self.mamba_state_dim,
                chunk_size=self.chunk_size,
                dropout_rate=self.dropout_rate,
            )
        else:
            # Fallback: simple concatenation + MLP fusion
            self.fusion_mlp = nn.Linear(
                self.bridge_dim * 2, self.bridge_dim
            )

        # Inverse projections
        self.inv_proj_semantic = InverseProjection(
            output_dim=self.semantic_dim,
        )
        self.inv_proj_features = InverseProjection(
            output_dim=self.feature_dim,
        )

        # Fused cluster projection (used when bridge is enabled)
        self.fused_cluster_proj = nn.Linear(self.semantic_dim, self.num_classes)

        # Stuff-Things classifier
        self.stuff_things = StuffThingsClassifier(
            hidden_dims=(16, 8),
        )

    def forward(
        self,
        image: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        use_bridge: bool = True,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through full MBPS model.

        Args:
            image: Input images of shape (B, C, H, W) in PyTorch NCHW format.
            depth: Optional depth maps of shape (B, 1, H, W) or (B, H, W).
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
        outputs: Dict[str, torch.Tensor] = {}
        b = image.shape[0]

        # ============================================
        # 1. BACKBONE: Frozen DINO ViT-S/8
        # ============================================
        # get patch tokens: (B, N, 384) where N = H/8 * W/8
        with torch.no_grad():
            spatial_features = self.backbone(image)
        # Detach to ensure no gradients flow through backbone
        spatial_features = spatial_features.detach()
        outputs["dino_features"] = spatial_features

        n = spatial_features.shape[1]

        # ============================================
        # 2. SEMANTIC BRANCH: DepthG head
        # ============================================
        semantic_codes = self.semantic_head(spatial_features)  # (B, N, 90)
        outputs["semantic_codes"] = semantic_codes

        # Cluster assignments (argmax over semantic codes)
        # The 90-dim codes are clustered via linear probe to K classes
        semantic_proj = self.semantic_cluster_proj(semantic_codes)
        semantic_probs = F.softmax(semantic_proj, dim=-1)
        outputs["semantic_probs"] = semantic_probs
        outputs["semantic_pred"] = torch.argmax(semantic_probs, dim=-1)

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
                outputs["depth_loss"] = torch.tensor(
                    0.0, device=image.device
                )

            # 4c. Cross-Modal Fusion
            if self.use_mamba_bridge:
                fused_sem, fused_feat = self.bicms(
                    sem_proj, feat_proj, deterministic=deterministic
                )
            else:
                # Simple concat + MLP fallback
                concat = torch.cat([sem_proj, feat_proj], dim=-1)
                fused = self.fusion_mlp(concat)
                fused = F.gelu(fused)
                fused_sem = fused
                fused_feat = fused

            outputs["fused_semantic"] = fused_sem
            outputs["fused_features"] = fused_feat

            # 4d. Inverse Projection (for reconstruction loss)
            recon_semantic = self.inv_proj_semantic(fused_sem)
            recon_features = self.inv_proj_features(fused_feat)
            outputs["reconstructed_semantic"] = recon_semantic
            outputs["reconstructed_features"] = recon_features

            # Update semantic and instance with fused representations
            fused_semantic_codes = semantic_codes + recon_semantic
            outputs["semantic_codes_fused"] = fused_semantic_codes

            # Re-cluster with fused codes
            fused_proj = self.fused_cluster_proj(fused_semantic_codes)
            fused_probs = F.softmax(fused_proj, dim=-1)
            outputs["semantic_probs"] = fused_probs
            outputs["semantic_pred"] = torch.argmax(fused_probs, dim=-1)

        # ============================================
        # 5. STUFF-THINGS CLASSIFICATION
        # ============================================
        # Use simplified cues from output. Full cue computation
        # happens outside the model (in trainer) for efficiency.
        # Here we provide a placeholder based on semantic probs.
        k = self.num_classes
        cluster_features = torch.zeros(
            b, k, 3, device=image.device, dtype=image.dtype
        )

        # Simple proxy cues from semantic predictions
        sem_pred = outputs["semantic_pred"]
        instance_mask_probs = torch.sigmoid(instance_masks)

        for c in range(k):
            mask = (sem_pred == c).float()  # (B, N)
            cluster_size = torch.sum(mask, dim=-1, keepdim=True) + 1e-8  # (B, 1)

            # Proxy DBD: variance of instance scores within cluster
            max_inst_probs, _ = instance_mask_probs.max(dim=1)  # (B, N)
            cluster_features[:, c, 0] = torch.mean(
                mask * max_inst_probs, dim=-1
            )

            # Proxy FCC: semantic probability concentration
            max_sem_probs, _ = (
                semantic_probs * mask.unsqueeze(-1)
            ).max(dim=1)  # (B, K)
            cluster_features[:, c, 1] = torch.mean(max_sem_probs, dim=-1)

            # Proxy IDF: number of instances overlapping
            overlap_per_inst = torch.sum(
                instance_mask_probs * mask.unsqueeze(1), dim=-1
            )  # (B, M)
            num_overlapping = torch.sum(
                (overlap_per_inst > 0.1).float(), dim=-1
            )  # (B,)
            cluster_features[:, c, 2] = num_overlapping / cluster_size.squeeze(-1)

        stuff_things_scores = self.stuff_things(
            cluster_features, deterministic=deterministic
        )
        outputs["stuff_things_scores"] = stuff_things_scores

        return outputs

    def _flatten_depth(
        self, depth: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Flatten spatial depth map to token-level depth.

        Args:
            depth: Depth of shape (B, H, W), (B, 1, H, W).
            num_tokens: Number of spatial tokens N = (H/8)*(W/8).

        Returns:
            Flattened depth of shape (B, N).
        """
        if depth.ndim == 4:
            # (B, 1, H, W) -> (B, H, W)
            depth = depth.squeeze(1)

        b, h, w = depth.shape

        # Compute token grid from actual image dimensions
        patch_size = 8  # DINO ViT-S/8 patch size
        h_tokens = h // patch_size
        w_tokens = w // patch_size

        # Average pool to token resolution using adaptive avg pool
        # Need (B, 1, H, W) for F.interpolate
        depth_4d = depth.unsqueeze(1)
        depth_resized = F.interpolate(
            depth_4d,
            size=(h_tokens, w_tokens),
            mode="bilinear",
            align_corners=False,
        )
        # (B, 1, h_tokens, w_tokens) -> (B, h_tokens * w_tokens)
        return depth_resized.squeeze(1).reshape(b, -1)
