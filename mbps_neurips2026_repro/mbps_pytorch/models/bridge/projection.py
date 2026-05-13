"""Adaptive Projection Bridge (APB).

Projects semantic codes (90-dim) and DINO features (384-dim) into a
shared bridge dimension (192-dim) for Mamba2 fusion. Includes learned
inverse projections back to original dimensions.

Mathematical specification:
    S' = W_s * S + b_s  where W_s in R^{D_b x D_s}
    F' = W_f * F + b_f  where W_f in R^{D_b x D_f}
    L_align = ||S'||_F + ||F'||_F (Frobenius norm reg)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class AdaptiveProjectionBridge(nn.Module):
    """Adaptive Projection Bridge (APB).

    Projects heterogeneous feature streams into a common dimension
    for cross-modal fusion.

    Args:
        semantic_dim: Input semantic code dimension (90).
        feature_dim: Input DINO feature dimension (384).
        bridge_dim: Shared bridge dimension (192).
    """

    def __init__(
        self,
        semantic_dim: int = 90,
        feature_dim: int = 384,
        bridge_dim: int = 192,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim
        self.bridge_dim = bridge_dim

        # Semantic projection: 90 -> 192
        self.semantic_proj = nn.Linear(semantic_dim, bridge_dim)
        self.sem_norm = nn.LayerNorm(bridge_dim)

        # Feature projection: 384 -> 192
        self.feature_proj = nn.Linear(feature_dim, bridge_dim)
        self.feat_norm = nn.LayerNorm(bridge_dim)

    def forward(
        self,
        semantic_codes: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project features to bridge dimension.

        Args:
            semantic_codes: Shape (B, N, 90).
            dino_features: Shape (B, N, 384).

        Returns:
            Tuple of:
                - semantic_proj: Shape (B, N, 192).
                - feature_proj: Shape (B, N, 192).
                - align_loss: Scalar alignment regularization.
        """
        # Semantic projection: 90 -> 192
        sem_proj = self.semantic_proj(semantic_codes)
        sem_proj = self.sem_norm(sem_proj)

        # Feature projection: 384 -> 192
        feat_proj = self.feature_proj(dino_features)
        feat_proj = self.feat_norm(feat_proj)

        # Alignment loss: encourage projected features to have similar norms
        sem_norm_val = torch.mean(torch.sum(sem_proj ** 2, dim=-1))
        feat_norm_val = torch.mean(torch.sum(feat_proj ** 2, dim=-1))
        align_loss = torch.abs(sem_norm_val - feat_norm_val)

        return sem_proj, feat_proj, align_loss


class InverseProjection(nn.Module):
    """Inverse projection from bridge dimension back to original.

    Projects fused features from D_b back to original dimensions.

    Args:
        input_dim: Input bridge dimension.
        output_dim: Target output dimension.
    """

    def __init__(
        self,
        input_dim: int = 192,
        output_dim: int = 384,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.inv_proj = nn.Linear(input_dim, output_dim)
        self.inv_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse project from bridge dimension.

        Args:
            x: Fused features of shape (B, N, D_b).

        Returns:
            Projected features of shape (B, N, output_dim).
        """
        x = self.inv_proj(x)
        x = self.inv_norm(x)
        return x
