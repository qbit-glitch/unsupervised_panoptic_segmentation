"""Depth FiLM (Feature-wise Linear Modulation) conditioning module.

Encodes depth geometry (depth + Sobel gradients + surface normals) into
per-patch affine parameters (gamma, beta) that modulate DINOv3 features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import sobel as scipy_sobel
import numpy as np


GRID_H, GRID_W = 32, 64
N_PATCHES = GRID_H * GRID_W


class DepthFiLM(nn.Module):
    """FiLM conditioning from depth geometry.

    Args:
        feat_dim: Feature dimension to modulate (1024 for DINOv3 ViT-L/16).
        n_freq: Number of sinusoidal frequency bands for depth encoding.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(self, feat_dim: int = 1024, n_freq: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_freq = n_freq

        # Depth encoding dimension: raw depth (1) + sin/cos (2*n_freq) + grads (2) = 2*n_freq + 3
        depth_enc_dim = 2 * n_freq + 3

        # MLP: depth_encoding → (gamma, beta) per patch
        self.mlp = nn.Sequential(
            nn.Linear(depth_enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim * 2),  # gamma + beta
        )

        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Frequency bands for sinusoidal encoding
        freqs = torch.linspace(0, np.log2(N_PATCHES), n_freq)
        self.register_buffer("freqs", 2.0 ** freqs * np.pi)

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode depth map into per-patch geometric features.

        Args:
            depth: (B, H, W) depth map at full resolution (512×1024).

        Returns:
            (B, N_PATCHES, depth_enc_dim) per-patch depth encoding.
        """
        B = depth.shape[0]

        # Downsample to patch grid
        depth_patches = F.adaptive_avg_pool2d(
            depth.unsqueeze(1), (GRID_H, GRID_W)
        ).squeeze(1)  # (B, 32, 64)

        # Compute Sobel gradients on patch-level depth
        depth_np = depth_patches.detach().cpu().numpy()
        grad_x = np.zeros_like(depth_np)
        grad_y = np.zeros_like(depth_np)
        for b in range(B):
            grad_x[b] = scipy_sobel(depth_np[b], axis=1)
            grad_y[b] = scipy_sobel(depth_np[b], axis=0)

        grad_x = torch.from_numpy(grad_x).to(depth.device, depth.dtype)
        grad_y = torch.from_numpy(grad_y).to(depth.device, depth.dtype)

        # Flatten to patch sequence
        depth_flat = depth_patches.reshape(B, N_PATCHES, 1)  # (B, N, 1)
        grad_x_flat = grad_x.reshape(B, N_PATCHES, 1)
        grad_y_flat = grad_y.reshape(B, N_PATCHES, 1)

        # Sinusoidal encoding of depth
        depth_scaled = depth_flat * self.freqs.unsqueeze(0).unsqueeze(0)  # (B, N, n_freq)
        sin_enc = torch.sin(depth_scaled)  # (B, N, n_freq)
        cos_enc = torch.cos(depth_scaled)  # (B, N, n_freq)

        # Concatenate: [depth, sin, cos, grad_x, grad_y]
        encoding = torch.cat([depth_flat, sin_enc, cos_enc, grad_x_flat, grad_y_flat], dim=-1)
        return encoding  # (B, N, 2*n_freq + 3)

    def forward(self, features: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation to features.

        Args:
            features: (B, N, D) DINOv3 patch features.
            depth: (B, H, W) depth map at original resolution.

        Returns:
            (B, N, D) depth-modulated features.
        """
        depth_enc = self.encode_depth(depth)  # (B, N, depth_enc_dim)
        film_params = self.mlp(depth_enc)  # (B, N, 2*D)

        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, N, D)
        gamma = 1.0 + gamma  # center around 1

        return gamma * features + beta
