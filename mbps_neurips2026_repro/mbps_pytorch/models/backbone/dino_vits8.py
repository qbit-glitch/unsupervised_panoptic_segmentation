"""DINO ViT-S/8 backbone in PyTorch.

Ported from facebookresearch/dino. This module implements the
Vision Transformer (ViT-Small, patch size 8) architecture used as
the frozen feature extractor for both DepthG and CutS3D.

All parameters are frozen via requires_grad_(False).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings.

    Attributes:
        patch_size: Size of each image patch.
        embed_dim: Embedding dimension.
    """

    def __init__(self, patch_size: int = 8, embed_dim: int = 384) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Conv2D with kernel_size=patch_size, stride=patch_size
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings.

        Args:
            x: Input image of shape (B, 3, H, W) in PyTorch NCHW format.

        Returns:
            Patch embeddings of shape (B, N, embed_dim).
        """
        # Conv2D: (B, 3, H, W) -> (B, embed_dim, H//ps, W//ps)
        x = self.proj(x)
        b, c, h, w = x.shape
        # Reshape to (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer.

    Attributes:
        num_heads: Number of attention heads.
        dim: Total embedding dimension.
        qkv_bias: Whether to include bias in QKV projection.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input of shape (B, N, D).

        Returns:
            Output of shape (B, N, D).
        """
        b, n, d = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, d)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network in Transformer block.

    Attributes:
        dim: Input/output dimension.
        mlp_ratio: Hidden dimension multiplier.
        drop: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 384,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP.

        Args:
            x: Input of shape (B, N, D).

        Returns:
            Output of shape (B, N, D).
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block.

    Attributes:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        drop: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Transformer block (pre-norm).

        Args:
            x: Input of shape (B, N, D).

        Returns:
            Output of shape (B, N, D).
        """
        # Self-attention with pre-norm
        x = x + self.attn(self.norm1(x))

        # FFN with pre-norm
        x = x + self.mlp(self.norm2(x))

        return x


class DINOViTS8(nn.Module):
    """DINO ViT-S/8: Vision Transformer Small with patch size 8.

    Architecture: 12 transformer blocks, 384-dim embeddings, 6 heads.
    All parameters are frozen -- outputs are detached from the graph.

    Attributes:
        patch_size: Image patch size (8 for ViT-S/8).
        embed_dim: Embedding dimension (384 for ViT-Small).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        freeze: Whether to freeze backbone (detach gradients).
    """

    def __init__(
        self,
        patch_size: int = 8,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.freeze = freeze

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding -- initialized to zeros, loaded from checkpoint.
        # The size depends on the input image resolution; for standard DINO
        # ViT-S/8 on 224x224 images: num_patches = (224/8)^2 = 784, so
        # pos_embed has shape (1, 785, 384). We initialize with a placeholder
        # and resize in the weights loader if needed.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 784, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Freeze all parameters if requested
        if freeze:
            self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """Extract DINO features from input image.

        Args:
            x: Input image of shape (B, 3, H, W), normalized (PyTorch NCHW).
            return_all_tokens: If True, return CLS + patch tokens.

        Returns:
            Patch token features of shape (B, N, 384) where
            N = (H/8) * (W/8), or (B, N+1, 384) if return_all_tokens.
        """
        b = x.shape[0]

        # Patch embedding
        patch_embed = self.patch_embed(x)
        num_patches = patch_embed.shape[1]

        # Compute spatial dims of patch grid (for non-square inputs)
        patch_h = x.shape[2] // self.patch_size
        patch_w = x.shape[3] // self.patch_size

        # CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)

        # Position embedding -- interpolate if needed
        pos_embed = self._interpolate_pos_embed(num_patches, patch_h, patch_w)

        # Combine
        x = torch.cat([cls_tokens, patch_embed], dim=1)
        x = x + pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        if return_all_tokens:
            return x  # (B, N+1, D) -- CLS + patch tokens
        else:
            return x[:, 1:]  # (B, N, D) -- patch tokens only

    def _interpolate_pos_embed(
        self,
        num_patches: int,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
    ) -> torch.Tensor:
        """Interpolate position embeddings if input size differs from training.

        Args:
            num_patches: Number of patches in the current input.
            patch_h: Patch grid height (for non-square inputs).
            patch_w: Patch grid width (for non-square inputs).

        Returns:
            Position embeddings of shape (1, num_patches + 1, embed_dim).
        """
        stored_num_patches = self.pos_embed.shape[1] - 1

        if num_patches == stored_num_patches:
            return self.pos_embed

        # Separate CLS and patch position embeddings
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]

        # Reshape stored embeddings to spatial grid (assumed square)
        stored_h = stored_w = int(stored_num_patches ** 0.5)

        # Target spatial dims: use explicit dims if provided, else assume square
        if patch_h is not None and patch_w is not None:
            new_h, new_w = patch_h, patch_w
        else:
            new_h = new_w = int(num_patches ** 0.5)

        patch_pos = patch_pos.reshape(1, stored_h, stored_w, self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)  # (1, D, H, W)
        patch_pos = F.interpolate(
            patch_pos,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def get_spatial_features(
        self,
        x: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Extract features in spatial format.

        Args:
            x: Input image of shape (B, 3, H, W) in PyTorch NCHW format.
            image_size: Original image size (H, W) for reshaping.

        Returns:
            Features of shape (B, 384, H//8, W//8) in PyTorch NCHW format.
        """
        features = self(x)  # (B, N, D)
        h = image_size[0] // self.patch_size
        w = image_size[1] // self.patch_size
        b = features.shape[0]
        # Reshape to spatial: (B, N, D) -> (B, H, W, D) -> (B, D, H, W)
        features = features.reshape(b, h, w, self.embed_dim)
        features = features.permute(0, 3, 1, 2)  # NCHW
        return features
