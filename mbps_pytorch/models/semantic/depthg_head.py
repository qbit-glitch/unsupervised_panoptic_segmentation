"""DepthG Semantic Segmentation Head.

Ported from visinf/depthg. Implements a 3-layer MLP projector on top
of frozen DINO features to produce a 90-dimensional semantic code space.

Architecture: Linear(384->384) -> ReLU -> Linear(384->384) -> ReLU -> Linear(384->90)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthGHead(nn.Module):
    """DepthG semantic segmentation head.

    Maps DINO ViT-S/8 features (384-dim) to a 90-dimensional semantic
    code space via a 3-layer MLP.

    Attributes:
        input_dim: Input feature dimension (384 for DINO ViT-S/8).
        hidden_dim: Hidden layer dimension.
        code_dim: Output code space dimension (90 per DepthG paper).
        use_batch_norm: Whether to use BatchNorm (False = use LayerNorm).
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 384,
        code_dim: int = 90,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.use_batch_norm = use_batch_norm

        # Layer 1: 384 -> 384
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Layer 2: 384 -> 384
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Layer 3: 384 -> 90 (code space)
        self.linear3 = nn.Linear(hidden_dim, code_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute semantic codes from DINO features.

        Args:
            features: DINO patch features of shape (B, N, 384).

        Returns:
            Semantic codes of shape (B, N, 90).
        """
        x = features

        # Layer 1: 384 -> 384
        x = self.linear1(x)
        x = self.norm1(x)
        x = F.relu(x)

        # Layer 2: 384 -> 384
        x = self.linear2(x)
        x = self.norm2(x)
        x = F.relu(x)

        # Layer 3: 384 -> 90 (code space)
        x = self.linear3(x)

        return x

    def get_cluster_assignments(
        self,
        codes: torch.Tensor,
        num_clusters: int = 27,
    ) -> torch.Tensor:
        """Get hard cluster assignments from soft codes via argmax.

        For actual clustering, use k-means on the code space.

        Args:
            codes: Semantic codes of shape (B, N, 90).
            num_clusters: Number of target clusters.

        Returns:
            Cluster assignments of shape (B, N).
        """
        # Simple approach: use first `num_clusters` dims as logits
        # In practice, k-means is used on the full 90-dim codes
        logits = codes[:, :, :num_clusters]
        return torch.argmax(logits, dim=-1)


class DepthGHeadSpatial(nn.Module):
    """DepthG head that operates on spatial (2D) feature maps.

    Same MLP but accepts and returns spatial format features.

    Attributes:
        hidden_dim: Hidden layer dimension.
        code_dim: Output code space dimension.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 384,
        code_dim: int = 90,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim

        # Layer 1: 384 -> 384
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Layer 2: 384 -> 384
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Layer 3: 384 -> 90 (code space)
        self.linear3 = nn.Linear(hidden_dim, code_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute semantic codes from spatial features.

        Args:
            features: Spatial features of shape (B, C, H, W) in PyTorch
                NCHW format.

        Returns:
            Semantic codes of shape (B, code_dim, H, W) in NCHW format.
        """
        b, c, h, w = features.shape

        # Reshape to token format: (B, C, H, W) -> (B, H*W, C)
        x = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Apply MLP
        x = self.linear1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.linear3(x)

        # Reshape back to spatial: (B, H*W, code_dim) -> (B, code_dim, H, W)
        x = x.reshape(b, h, w, self.code_dim).permute(0, 3, 1, 2)
        return x
