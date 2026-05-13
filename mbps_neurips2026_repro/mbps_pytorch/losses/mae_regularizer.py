"""Lightweight MAE reconstruction regularizer inspired by LoRA-TTT (ICML 2025).

Ported from: https://github.com/ykojima4020/LoRA-TTT
Reference: "Low-Rank Test-Time Training for Vision-Language Models"

Provides a CLS-token MAE loss without requiring a heavy decoder.
Useful as an auxiliary regularizer during adapter training.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchShuffle(nn.Module):
    """Shuffle patches for MAE-style masking."""

    def __init__(self, mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor, patch_size: int = 14) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random patch shuffling / masking.

        Args:
            x: (B, C, H, W) image tensor.
            patch_size: Patch size for tokenization.

        Returns:
            shuffled_x: (B, C, H, W) with masked patches zeroed.
            mask: (B, num_patches) binary mask (1 = masked, 0 = kept).
        """
        B, C, H, W = x.shape
        assert H % patch_size == 0 and W % patch_size == 0
        h_p, w_p = H // patch_size, W // patch_size
        num_patches = h_p * w_p

        # Reshape to patches
        x_patches = x.reshape(B, C, h_p, patch_size, w_p, patch_size)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, -1)

        # Random mask
        noise = torch.rand(B, num_patches, device=x.device)
        num_keep = int(num_patches * (1 - self.mask_ratio))
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        # Create mask
        mask = torch.ones(B, num_patches, device=x.device)
        mask.scatter_(1, ids_keep, 0.0)  # 0 = keep, 1 = mask

        # Zero out masked patches
        x_patches_masked = x_patches.clone()
        x_patches_masked[mask.bool()] = 0.0

        # Reshape back
        x_patches_masked = x_patches_masked.reshape(B, h_p, w_p, C, patch_size, patch_size)
        x_patches_masked = x_patches_masked.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return x_patches_masked, mask


class CLSTokenMAELoss(nn.Module):
    """Lightweight MAE loss using CLS token reconstruction.

    Instead of reconstructing all pixels (expensive), this loss masks random
    patches and ensures the CLS token of the masked input matches the CLS token
    of the original input. No decoder needed.
    """

    def __init__(self, mask_ratio: float = 0.5, patch_size: int = 14):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffler = PatchShuffle(mask_ratio=mask_ratio)

    def forward(
        self,
        encoder: callable,
        images: torch.Tensor,
        confidence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CLS-token MAE loss.

        Args:
            encoder: Callable that takes (B, C, H, W) and returns features.
                     Expected to return a dict or tensor where CLS token is accessible.
            images: (B, 3, H, W) input images.
            confidence_mask: Optional (B,) bool mask for confident samples only.

        Returns:
            Scalar MAE loss.
        """
        B = images.shape[0]

        # Filter to confident samples if mask provided
        if confidence_mask is not None:
            images = images[confidence_mask]
            if images.shape[0] == 0:
                return torch.tensor(0.0, device=images.device)

        # Full image CLS token
        with torch.no_grad():
            feat_full = encoder(images)
            cls_full = self._extract_cls(feat_full)  # (B, D)

        # Masked image CLS token
        images_masked, mask = self.shuffler(images, patch_size=self.patch_size)
        feat_masked = encoder(images_masked)
        cls_masked = self._extract_cls(feat_masked)  # (B, D)

        # MSE between CLS tokens
        loss = F.mse_loss(cls_masked, cls_full)

        # Normalize by mask ratio
        loss = loss / self.mask_ratio

        return loss

    def _extract_cls(self, features) -> torch.Tensor:
        """Extract CLS token from encoder output.

        Supports:
            - (B, N, D) tensor: returns first token [CLS]
            - dict with 'last_hidden_state': returns first token
        """
        if isinstance(features, dict):
            features = features["last_hidden_state"]
        if features.dim() == 3:
            return features[:, 0, :]  # (B, D)
        return features


def mae_regularizer_loss(
    backbone: nn.Module,
    images: torch.Tensor,
    mask_ratio: float = 0.5,
    patch_size: int = 14,
) -> torch.Tensor:
    """Standalone function for CLS-token MAE regularization.

    Args:
        backbone: Frozen or adapted backbone that returns (B, N, D) features.
        images: (B, 3, H, W) images.
        mask_ratio: Fraction of patches to mask.
        patch_size: Patch size.

    Returns:
        Scalar MAE loss.
    """
    loss_fn = CLSTokenMAELoss(mask_ratio=mask_ratio, patch_size=patch_size)
    return loss_fn(backbone, images)
