"""Full Depth-Conditioned Slot Attention Decoder model.

Combines DepthFiLM + SlotAttention + SpatialBroadcastDecoder into a single
module for unsupervised instance discovery from DINOv3 features + depth.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_film import DepthFiLM, GRID_H, GRID_W, N_PATCHES
from .slot_attention import SlotAttention
from .decoder import SpatialBroadcastDecoder


@dataclass(frozen=True)
class DepthSlotDecoderConfig:
    """Configuration for DepthSlotDecoder."""

    feat_dim: int = 1024           # DINOv3 ViT-L/16 feature dimension
    slot_dim: int = 256            # Slot representation dimension
    num_slots: int = 20            # Max instances per image
    slot_iters: int = 5            # Slot attention iterations
    decoder_hidden: int = 2048     # Decoder MLP hidden dim
    decoder_layers: int = 3        # Decoder MLP depth
    film_n_freq: int = 16          # Sinusoidal frequency bands for depth
    film_hidden: int = 256         # FiLM MLP hidden dim
    slot_hidden: int = 512         # Slot attention MLP hidden dim


class DepthSlotDecoder(nn.Module):
    """Depth-conditioned slot attention decoder for instance segmentation.

    Pipeline:
        1. DINOv3 features (1024-D) → project to slot_dim
        2. DepthFiLM modulates projected features with depth geometry
        3. SlotAttention discovers K object slots via iterative competition
        4. SpatialBroadcastDecoder reconstructs features from slots
        5. Attention maps serve as instance masks

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: DepthSlotDecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Project DINOv3 features to slot dimension
        self.input_proj = nn.Sequential(
            nn.LayerNorm(cfg.feat_dim),
            nn.Linear(cfg.feat_dim, cfg.slot_dim),
            nn.GELU(),
            nn.Linear(cfg.slot_dim, cfg.slot_dim),
        )

        # Depth FiLM conditioning (operates on slot_dim features)
        self.depth_film = DepthFiLM(
            feat_dim=cfg.slot_dim,
            n_freq=cfg.film_n_freq,
            hidden_dim=cfg.film_hidden,
        )

        # Learnable 2D positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, cfg.slot_dim) * 0.02)

        # Slot attention
        self.slot_attention = SlotAttention(
            num_slots=cfg.num_slots,
            dim=cfg.slot_dim,
            iters=cfg.slot_iters,
            hidden_dim=cfg.slot_hidden,
        )

        # Spatial broadcast decoder
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=cfg.slot_dim,
            target_dim=cfg.feat_dim,
            hidden_dim=cfg.decoder_hidden,
            n_layers=cfg.decoder_layers,
        )

    def encode(self, features: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Encode DINOv3 features with depth conditioning.

        Args:
            features: (B, N, feat_dim) DINOv3 patch features.
            depth: (B, H, W) depth map at original resolution.

        Returns:
            (B, N, slot_dim) depth-conditioned encoded features.
        """
        x = self.input_proj(features)  # (B, N, slot_dim)
        x = self.depth_film(x, depth)  # FiLM modulation
        x = x + self.pos_embed  # positional encoding
        return x

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            features: (B, N, feat_dim) DINOv3 features (N=2048, D=1024).
            depth: (B, H, W) depth map (512×1024).

        Returns:
            Dict with keys:
                - recon: (B, N, feat_dim) reconstructed features
                - masks: (B, K, N) soft slot masks
                - slots: (B, K, slot_dim) slot representations
                - attn: (B, K, N) slot attention weights
        """
        # Encode with depth conditioning
        encoded = self.encode(features, depth)  # (B, N, slot_dim)

        # Slot attention: discover objects
        slots, attn = self.slot_attention(encoded)  # (B, K, D), (B, K, N)

        # Decode: reconstruct features from slots
        recon, masks = self.decoder(slots)  # (B, N, feat_dim), (B, K, N)

        return {
            "recon": recon,
            "masks": masks,
            "slots": slots,
            "attn": attn,
        }

    def get_instance_masks(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        min_area: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract hard instance masks for evaluation.

        Args:
            features: (B, N, feat_dim) DINOv3 features.
            depth: (B, H, W) depth map.
            min_area: Minimum number of patches for a valid instance.

        Returns:
            instance_map: (B, H, W) integer instance IDs (0=background).
            slot_scores: (B, K) confidence score per slot (fraction of image covered).
        """
        with torch.no_grad():
            out = self.forward(features, depth)
            masks = out["masks"]  # (B, K, N)

            B, K, N = masks.shape

            # Hard assignment: argmax over slots
            instance_map_flat = masks.argmax(dim=1)  # (B, N) values in [0, K)

            # Filter small instances
            slot_scores = torch.zeros(B, K, device=masks.device)
            for k in range(K):
                area = (instance_map_flat == k).sum(dim=-1).float()  # (B,)
                slot_scores[:, k] = area / N
                # Zero out slots below min_area
                too_small = area < min_area
                instance_map_flat[too_small & (instance_map_flat == k)] = 0

            # Reshape to spatial grid and upsample
            instance_map = instance_map_flat.reshape(B, GRID_H, GRID_W)
            instance_map = F.interpolate(
                instance_map.unsqueeze(1).float(),
                size=(512, 1024),
                mode="nearest",
            ).squeeze(1).long()

            return instance_map, slot_scores

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
