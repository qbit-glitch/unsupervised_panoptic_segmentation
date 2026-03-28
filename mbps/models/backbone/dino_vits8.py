"""DINO ViT-S/8 backbone in JAX/Flax.

Ported from facebookresearch/dino. This module implements the
Vision Transformer (ViT-Small, patch size 8) architecture used as
the frozen feature extractor for both DepthG and CutS3D.

All parameters are frozen via jax.lax.stop_gradient.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings.

    Attributes:
        patch_size: Size of each image patch.
        embed_dim: Embedding dimension.
    """

    patch_size: int = 8
    embed_dim: int = 384

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract patch embeddings.

        Args:
            x: Input image of shape (B, H, W, 3).

        Returns:
            Patch embeddings of shape (B, N, embed_dim).
        """
        # Conv2D with kernel_size=patch_size, stride=patch_size
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="proj",
        )(x)
        b, h, w, c = x.shape
        x = jnp.reshape(x, (b, h * w, c))
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

    num_heads: int = 6
    dim: int = 384
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply multi-head self-attention.

        Args:
            x: Input of shape (B, N, D).
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (B, N, D).
        """
        b, n, d = x.shape
        head_dim = d // self.num_heads
        scale = head_dim**-0.5

        qkv = nn.Dense(3 * d, use_bias=self.qkv_bias, name="qkv")(x)
        qkv = jnp.reshape(qkv, (b, n, 3, self.num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        x = nn.Dense(d, name="proj")(x)
        x = nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)
        return x


class MLP(nn.Module):
    """Feed-forward network in Transformer block.

    Attributes:
        dim: Input/output dimension.
        mlp_ratio: Hidden dimension multiplier.
        drop: Dropout rate.
    """

    dim: int = 384
    mlp_ratio: float = 4.0
    drop: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply MLP.

        Args:
            x: Input of shape (B, N, D).
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (B, N, D).
        """
        hidden = int(self.dim * self.mlp_ratio)
        x = nn.Dense(hidden, name="fc1")(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        x = nn.Dense(self.dim, name="fc2")(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block.

    Attributes:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        drop: Dropout rate.
    """

    dim: int = 384
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply Transformer block (pre-norm).

        Args:
            x: Input of shape (B, N, D).
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (B, N, D).
        """
        # Self-attention with pre-norm
        residual = x
        x = nn.LayerNorm(name="norm1")(x)
        x = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            dim=self.dim,
            attn_drop=self.drop,
            proj_drop=self.drop,
            name="attn",
        )(x, deterministic=deterministic)
        x = residual + x

        # FFN with pre-norm
        residual = x
        x = nn.LayerNorm(name="norm2")(x)
        x = MLP(
            dim=self.dim,
            mlp_ratio=self.mlp_ratio,
            drop=self.drop,
            name="mlp",
        )(x, deterministic=deterministic)
        x = residual + x

        return x


class DINOViTS8(nn.Module):
    """DINO ViT-S/8: Vision Transformer Small with patch size 8.

    Architecture: 12 transformer blocks, 384-dim embeddings, 6 heads.
    All parameters are frozen — outputs are passed through stop_gradient.

    Attributes:
        patch_size: Image patch size (8 for ViT-S/8).
        embed_dim: Embedding dimension (384 for ViT-Small).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        freeze: Whether to freeze backbone (stop_gradient).
    """

    patch_size: int = 8
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    freeze: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        return_all_tokens: bool = False,
    ) -> jnp.ndarray:
        """Extract DINO features from input image.

        Args:
            x: Input image of shape (B, H, W, 3), normalized.
            deterministic: If True, disable dropout.
            return_all_tokens: If True, return CLS + patch tokens.

        Returns:
            Patch token features of shape (B, N, 384) where
            N = (H/8) * (W/8), or (B, N+1, 384) if return_all_tokens.
        """
        b, h, w, c = x.shape

        # Patch embedding
        patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name="patch_embed",
        )(x)
        num_patches = patch_embed.shape[1]

        # CLS token
        cls_token = self.param(
            "cls_token",
            nn.initializers.zeros,
            (1, 1, self.embed_dim),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (b, 1, self.embed_dim))

        # Position embedding
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, num_patches + 1, self.embed_dim),
        )

        # Combine
        x = jnp.concatenate([cls_tokens, patch_embed], axis=1)
        x = x + pos_embed

        # Transformer blocks
        for i in range(self.depth):
            x = TransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f"blocks_{i}",
            )(x, deterministic=deterministic)

        # Final layer norm
        x = nn.LayerNorm(name="norm")(x)

        # Freeze backbone
        if self.freeze:
            x = jax.lax.stop_gradient(x)

        if return_all_tokens:
            return x  # (B, N+1, D) — CLS + patch tokens
        else:
            return x[:, 1:]  # (B, N, D) — patch tokens only

    def get_spatial_features(
        self,
        x: jnp.ndarray,
        image_size: Tuple[int, int],
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Extract features in spatial format.

        Args:
            x: Input image of shape (B, H, W, 3).
            image_size: Original image size (H, W) for reshaping.
            deterministic: If True, disable dropout.

        Returns:
            Features of shape (B, H//8, W//8, 384).
        """
        features = self(x, deterministic=deterministic)
        h = image_size[0] // self.patch_size
        w = image_size[1] // self.patch_size
        return jnp.reshape(features, (features.shape[0], h, w, self.embed_dim))
