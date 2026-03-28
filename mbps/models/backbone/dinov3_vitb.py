"""DINOv3 ViT-B/16 backbone in JAX/Flax.

Ported from facebook/dinov3-vitb16-pretrain-lvd1689m.
Architecture: 12 transformer blocks, 768-dim embeddings, 12 heads, patch 16.
Includes 4 register tokens (DINOv3 feature) between CLS and patch tokens.

All parameters are frozen via jax.lax.stop_gradient.

Key differences from DINO ViT-S/8 (dino_vits8.py):
  - embed_dim: 384 -> 768
  - num_heads: 6 -> 12
  - patch_size: 8 -> 16
  - Has 4 register tokens (new)
  - Position embedding covers CLS + patches (no registers)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings.

    Attributes:
        patch_size: Size of each image patch.
        embed_dim: Embedding dimension.
    """

    patch_size: int = 16
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract patch embeddings.

        Args:
            x: Input image of shape (B, H, W, 3).

        Returns:
            Patch embeddings of shape (B, N, embed_dim).
        """
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

    num_heads: int = 12
    dim: int = 768
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
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
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

    dim: int = 768
    mlp_ratio: float = 4.0
    drop: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        hidden = int(self.dim * self.mlp_ratio)
        x = nn.Dense(hidden, name="fc1")(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        x = nn.Dense(self.dim, name="fc2")(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block (pre-norm).

    Attributes:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        drop: Dropout rate.
    """

    dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
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


class DINOv3ViTB(nn.Module):
    """DINOv3 ViT-B/16: Vision Transformer Base with patch size 16.

    Architecture: 12 transformer blocks, 768-dim embeddings, 12 heads.
    Includes 4 register tokens.
    All parameters are frozen — outputs are passed through stop_gradient.

    Token layout: [CLS, reg_0, reg_1, reg_2, reg_3, patch_0, ..., patch_N]

    Attributes:
        patch_size: Image patch size (16).
        embed_dim: Embedding dimension (768).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        num_register_tokens: Number of register tokens (DINOv3: 4).
        freeze: Whether to freeze backbone (stop_gradient).
    """

    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    freeze: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        return_all_tokens: bool = False,
    ) -> jnp.ndarray:
        """Extract DINOv3 features from input image.

        Args:
            x: Input image of shape (B, H, W, 3), normalized.
            deterministic: If True, disable dropout.
            return_all_tokens: If True, return all tokens including CLS and registers.

        Returns:
            Patch token features of shape (B, N, 768) where
            N = (H/16) * (W/16), or (B, N+1+num_register, 768) if return_all_tokens.
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

        # Position embedding (for CLS + patches, NOT registers)
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, num_patches + 1, self.embed_dim),
        )

        # Register tokens (no position embedding)
        register_tokens = self.param(
            "register_tokens",
            nn.initializers.zeros,
            (1, self.num_register_tokens, self.embed_dim),
        )
        reg_tokens = jnp.broadcast_to(
            register_tokens, (b, self.num_register_tokens, self.embed_dim)
        )

        # Combine: [CLS + patches] with pos_embed, then insert registers
        x = jnp.concatenate([cls_tokens, patch_embed], axis=1)  # (B, 1+N, D)
        x = x + pos_embed  # Add position embeddings

        # Insert registers after CLS token
        # Final layout: [CLS, reg_0, ..., reg_3, patch_0, ..., patch_N]
        x = jnp.concatenate([x[:, :1], reg_tokens, x[:, 1:]], axis=1)

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
            return x  # (B, 1 + num_register + N, D)
        else:
            # Return only patch tokens (skip CLS + registers)
            skip = 1 + self.num_register_tokens
            return x[:, skip:]  # (B, N, D)

    def get_spatial_features(
        self,
        x: jnp.ndarray,
        image_size: Tuple[int, int],
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Extract features in spatial (H, W) format.

        Args:
            x: Input image of shape (B, H, W, 3).
            image_size: Original image size (H, W) for reshaping.
            deterministic: If True, disable dropout.

        Returns:
            Features of shape (B, H//16, W//16, 768).
        """
        features = self(x, deterministic=deterministic)
        h = image_size[0] // self.patch_size
        w = image_size[1] // self.patch_size
        return jnp.reshape(features, (features.shape[0], h, w, self.embed_dim))

    def interpolate_pos_embed(
        self,
        pos_embed: jnp.ndarray,
        target_h: int,
        target_w: int,
    ) -> jnp.ndarray:
        """Interpolate position embeddings for different resolutions.

        Args:
            pos_embed: Original position embedding (1, 1+N_orig, D).
            target_h: Target number of patches in height.
            target_w: Target number of patches in width.

        Returns:
            Interpolated position embedding (1, 1+target_h*target_w, D).
        """
        # Separate CLS and patch position embeddings
        cls_pos = pos_embed[:, :1, :]  # (1, 1, D)
        patch_pos = pos_embed[:, 1:, :]  # (1, N_orig, D)

        N_orig = patch_pos.shape[1]
        orig_size = int(N_orig**0.5)

        if orig_size * orig_size != N_orig:
            # Non-square — assume original was from the training resolution
            # DINOv3 trains at 224x224 → 14x14 patches
            orig_h, orig_w = 14, 14
        else:
            orig_h, orig_w = orig_size, orig_size

        if orig_h == target_h and orig_w == target_w:
            return pos_embed

        # Reshape to spatial grid
        patch_pos = jnp.reshape(patch_pos, (1, orig_h, orig_w, -1))

        # Bilinear interpolation
        patch_pos = jax.image.resize(
            patch_pos,
            shape=(1, target_h, target_w, patch_pos.shape[-1]),
            method="bilinear",
        )

        # Flatten back
        patch_pos = jnp.reshape(patch_pos, (1, target_h * target_w, -1))

        return jnp.concatenate([cls_pos, patch_pos], axis=1)
