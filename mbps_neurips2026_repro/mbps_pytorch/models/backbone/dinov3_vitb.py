"""DINOv3 ViT-B/16 backbone in PyTorch.

Loads from HuggingFace: facebook/dinov3-vitb16-pretrain-lvd1689m
768-dim embeddings, 12 heads, patch size 16, 4 register tokens.

All parameters are frozen via requires_grad_(False).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""

    def __init__(self, patch_size: int = 16, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, embed_dim)."""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
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
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, d)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network in Transformer block."""

    def __init__(self, dim: int = 768, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block (pre-norm)."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv3ViTB(nn.Module):
    """DINOv3 ViT-B/16: Vision Transformer Base with patch size 16.

    Architecture: 12 transformer blocks, 768-dim, 12 heads, 4 register tokens.
    All parameters frozen — outputs are detached from graph.

    Attributes:
        patch_size: Image patch size (16).
        embed_dim: Embedding dimension (768).
        depth: Number of transformer blocks (12).
        num_heads: Number of attention heads (12).
        num_register_tokens: Number of register tokens (4).
        freeze: Whether to freeze backbone.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.freeze = freeze

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Register tokens (DINOv3 adds 4)
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim)
        )

        # Position embedding (CLS + patches; registers have no pos embed)
        # Default for 224x224: (224/16)^2 = 196 patches + 1 CLS = 197
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 196, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        if freeze:
            self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """Extract DINOv3 features.

        Args:
            x: Input image (B, 3, H, W), ImageNet-normalized (NCHW).
            return_all_tokens: If True, return CLS + reg + patch tokens.

        Returns:
            Patch tokens (B, N, 768) where N = (H/16)*(W/16),
            or all tokens if return_all_tokens.
        """
        b = x.shape[0]

        # Patch embedding
        patch_embed = self.patch_embed(x)
        num_patches = patch_embed.shape[1]

        patch_h = x.shape[2] // self.patch_size
        patch_w = x.shape[3] // self.patch_size

        # CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)

        # Register tokens
        reg_tokens = self.register_tokens.expand(b, -1, -1)

        # Position embedding (interpolate if needed) — applies to CLS + patches
        pos_embed = self._interpolate_pos_embed(num_patches, patch_h, patch_w)

        # DINOv3 token order: [CLS, reg_0..reg_3, patch_0..patch_N]
        # Position embeddings apply to CLS + patches only
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]

        cls_tokens = cls_tokens + cls_pos
        patch_embed = patch_embed + patch_pos

        # Concatenate: CLS + registers + patches
        x = torch.cat([cls_tokens, reg_tokens, patch_embed], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_all_tokens:
            return x

        # Return only patch tokens: skip CLS (1) + registers (num_register_tokens)
        offset = 1 + self.num_register_tokens
        return x[:, offset:]

    def _interpolate_pos_embed(
        self,
        num_patches: int,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
    ) -> torch.Tensor:
        """Interpolate position embeddings for different input sizes."""
        stored_num_patches = self.pos_embed.shape[1] - 1

        if num_patches == stored_num_patches:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]

        stored_h = stored_w = int(stored_num_patches ** 0.5)

        if patch_h is not None and patch_w is not None:
            new_h, new_w = patch_h, patch_w
        else:
            new_h = new_w = int(num_patches ** 0.5)

        patch_pos = patch_pos.reshape(1, stored_h, stored_w, self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(new_h, new_w), mode="bicubic", align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m") -> "DINOv3ViTB":
        """Load pretrained DINOv3 weights from HuggingFace.

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            DINOv3ViTB with pretrained weights loaded.
        """
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Install transformers>=4.56.0: pip install transformers>=4.56.0")

        hf_model = AutoModel.from_pretrained(model_name)

        model = cls(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_register_tokens=4,
            freeze=True,
        )

        # Map HuggingFace state dict to our module
        hf_sd = hf_model.state_dict()
        new_sd = {}

        # Patch embedding
        new_sd["patch_embed.proj.weight"] = hf_sd["dinov3.embeddings.patch_embeddings.projection.weight"]
        new_sd["patch_embed.proj.bias"] = hf_sd["dinov3.embeddings.patch_embeddings.projection.bias"]

        # CLS token
        new_sd["cls_token"] = hf_sd["dinov3.embeddings.cls_token"]

        # Register tokens
        new_sd["register_tokens"] = hf_sd["dinov3.embeddings.register_tokens"]

        # Position embedding
        new_sd["pos_embed"] = hf_sd["dinov3.embeddings.position_embeddings"]

        # Transformer blocks
        for i in range(12):
            hf_prefix = f"dinov3.encoder.layer.{i}"

            # Attention QKV
            q_w = hf_sd[f"{hf_prefix}.attention.attention.query.weight"]
            k_w = hf_sd[f"{hf_prefix}.attention.attention.key.weight"]
            v_w = hf_sd[f"{hf_prefix}.attention.attention.value.weight"]
            new_sd[f"blocks.{i}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_sd[f"{hf_prefix}.attention.attention.query.bias"]
            k_b = hf_sd[f"{hf_prefix}.attention.attention.key.bias"]
            v_b = hf_sd[f"{hf_prefix}.attention.attention.value.bias"]
            new_sd[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

            # Attention output
            new_sd[f"blocks.{i}.attn.proj.weight"] = hf_sd[f"{hf_prefix}.attention.output.dense.weight"]
            new_sd[f"blocks.{i}.attn.proj.bias"] = hf_sd[f"{hf_prefix}.attention.output.dense.bias"]

            # LayerNorm 1
            new_sd[f"blocks.{i}.norm1.weight"] = hf_sd[f"{hf_prefix}.norm1.weight"]
            new_sd[f"blocks.{i}.norm1.bias"] = hf_sd[f"{hf_prefix}.norm1.bias"]

            # MLP
            new_sd[f"blocks.{i}.mlp.fc1.weight"] = hf_sd[f"{hf_prefix}.mlp.fc1.weight"]
            new_sd[f"blocks.{i}.mlp.fc1.bias"] = hf_sd[f"{hf_prefix}.mlp.fc1.bias"]
            new_sd[f"blocks.{i}.mlp.fc2.weight"] = hf_sd[f"{hf_prefix}.mlp.fc2.weight"]
            new_sd[f"blocks.{i}.mlp.fc2.bias"] = hf_sd[f"{hf_prefix}.mlp.fc2.bias"]

            # LayerNorm 2
            new_sd[f"blocks.{i}.norm2.weight"] = hf_sd[f"{hf_prefix}.norm2.weight"]
            new_sd[f"blocks.{i}.norm2.bias"] = hf_sd[f"{hf_prefix}.norm2.bias"]

        # Final norm
        new_sd["norm.weight"] = hf_sd["dinov3.layernorm.weight"]
        new_sd["norm.bias"] = hf_sd["dinov3.layernorm.bias"]

        model.load_state_dict(new_sd)
        model.requires_grad_(False)
        return model
