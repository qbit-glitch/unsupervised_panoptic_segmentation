"""Unified Depth Conditioning Module (UDCM).

Encodes monocular depth via sinusoidal positional encoding and
modulates semantic/feature branch representations through an
FiLM-style gating mechanism.

Mathematical specification:
    gamma(D) = [sin(2^k pi D), cos(2^k pi D)]_{k=0}^{K-1}  (sinusoidal encoding)
    X_conditioned = X * (1 + W_gamma * gamma(D)) + W_beta * gamma(D)  (FiLM modulation)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_depth_encoding(
    depth: torch.Tensor,
    freq_bands: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> torch.Tensor:
    """Compute sinusoidal positional encoding of depth values.

    gamma(D) = [sin(2^k pi D), cos(2^k pi D)] for k in freq_bands.

    Args:
        depth: Depth values of shape (B, N) or (B, H, W), in [0, 1].
        freq_bands: Frequency bands for encoding.

    Returns:
        Depth encoding of shape (B, ..., 2*len(freq_bands)).
    """
    encodings = []
    pi = torch.pi
    for freq in freq_bands:
        encodings.append(torch.sin(freq * pi * depth))
        encodings.append(torch.cos(freq * pi * depth))
    return torch.stack(encodings, dim=-1)


class UnifiedDepthConditioning(nn.Module):
    """Unified Depth Conditioning Module (UDCM).

    Conditions both semantic and feature branches on monocular
    depth information using FiLM (Feature-wise Linear Modulation).

    Args:
        bridge_dim: Bridge feature dimension (192).
        freq_bands: Sinusoidal encoding frequencies.
        depth_mlp_dims: Hidden dims for depth processing MLP.
    """

    def __init__(
        self,
        bridge_dim: int = 192,
        freq_bands: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        depth_mlp_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.bridge_dim = bridge_dim
        self.freq_bands = freq_bands
        # Auto-scale MLP dims with bridge_dim if not explicitly set
        self.depth_mlp_dims = depth_mlp_dims or (64, bridge_dim)

        # Depth encoding dimension
        depth_enc_dim = 2 * len(freq_bands)

        # Depth MLP layers
        self.depth_mlps = nn.ModuleList()
        in_dim = depth_enc_dim
        for dim in self.depth_mlp_dims:
            self.depth_mlps.append(nn.Linear(in_dim, dim))
            in_dim = dim

        # Final projection to bridge_dim
        self.depth_proj = nn.Linear(in_dim, bridge_dim)

        # FiLM modulation parameters
        # Semantic branch: gamma_s, beta_s
        self.gamma_s = nn.Linear(bridge_dim, bridge_dim)
        self.beta_s = nn.Linear(bridge_dim, bridge_dim)

        # Feature branch: gamma_f, beta_f
        self.gamma_f = nn.Linear(bridge_dim, bridge_dim)
        self.beta_f = nn.Linear(bridge_dim, bridge_dim)

    def forward(
        self,
        depth: torch.Tensor,
        semantic_proj: torch.Tensor,
        feature_proj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Condition features on depth.

        Args:
            depth: Depth values of shape (B, N), in [0, 1].
            semantic_proj: Projected semantic of shape (B, N, D_b).
            feature_proj: Projected features of shape (B, N, D_b).

        Returns:
            Tuple of:
                - sem_conditioned: Depth-conditioned semantic (B, N, D_b).
                - feat_conditioned: Depth-conditioned features (B, N, D_b).
                - depth_loss: Depth consistency regularization loss.
        """
        # Sinusoidal depth encoding
        depth_enc = sinusoidal_depth_encoding(depth, self.freq_bands)
        # (B, N, 2*num_freqs)

        # Process depth encoding through MLP
        d = depth_enc
        for mlp in self.depth_mlps:
            d = F.relu(mlp(d))

        # Ensure output dim matches bridge_dim
        d = self.depth_proj(d)

        # FiLM modulation parameters
        # Semantic branch: gamma_s, beta_s
        gamma_s = self.gamma_s(d) + 1.0
        beta_s = self.beta_s(d)

        # Feature branch: gamma_f, beta_f
        gamma_f = self.gamma_f(d) + 1.0
        beta_f = self.beta_f(d)

        # Apply FiLM: x_cond = x * gamma + beta
        sem_conditioned = semantic_proj * gamma_s + beta_s
        feat_conditioned = feature_proj * gamma_f + beta_f

        # Depth consistency loss: nearby depth -> similar modulation
        # Encourage smooth depth conditioning
        depth_grad_x = torch.diff(depth, dim=-1)
        mod_grad_x_s = torch.diff(
            torch.mean(gamma_s, dim=-1), dim=-1
        )
        depth_loss = torch.mean(
            torch.abs(mod_grad_x_s)
            * torch.exp(-torch.abs(depth_grad_x) * 5.0)
        )

        return sem_conditioned, feat_conditioned, depth_loss
