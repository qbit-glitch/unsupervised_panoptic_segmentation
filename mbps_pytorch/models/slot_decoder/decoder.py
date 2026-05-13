"""Spatial Broadcast Decoder for feature reconstruction from slots.

Based on DINOSAUR (Seitzer et al., ICLR 2023). Each slot is broadcast to all
spatial positions, combined with positional encoding, and decoded via MLP to
predict per-position features + mask logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


GRID_H, GRID_W = 32, 64
N_PATCHES = GRID_H * GRID_W


class SpatialBroadcastDecoder(nn.Module):
    """Decode slot representations back into spatial feature maps.

    For each slot:
      1. Broadcast slot vector to all N positions
      2. Add learned positional encoding
      3. MLP → (target_dim + 1) per position
      4. Split into predicted features and mask logit

    Final reconstruction combines all slots via softmax mask competition.

    Args:
        slot_dim: Dimension of slot representations.
        target_dim: Target feature dimension for reconstruction (1024 for DINOv3 ViT-L/16).
        hidden_dim: MLP hidden dimension.
        n_layers: Number of MLP layers.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        target_dim: int = 1024,
        hidden_dim: int = 2048,
        n_layers: int = 3,
    ):
        super().__init__()
        self.target_dim = target_dim

        # Learned positional encoding for decoder
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, slot_dim) * 0.02)

        # MLP: slot_dim → hidden → ... → (target_dim + 1)
        layers = []
        in_dim = slot_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, target_dim + 1))  # +1 for mask logit
        self.mlp = nn.Sequential(*layers)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode slots to feature reconstruction + mask logits.

        Args:
            slots: (B, K, slot_dim) slot representations.

        Returns:
            recon: (B, N, target_dim) reconstructed features.
            masks: (B, K, N) soft mask assignments (softmax over slots).
        """
        B, K, D = slots.shape

        # Broadcast each slot to all positions + add positional encoding
        # (B, K, D) → (B, K, N, D)
        slots_broadcast = slots.unsqueeze(2).expand(-1, -1, N_PATCHES, -1)
        pos = self.pos_embed.unsqueeze(1).expand(B, K, -1, -1)  # (B, K, N, D)
        decoder_input = slots_broadcast + pos  # (B, K, N, D)

        # MLP decode
        output = self.mlp(decoder_input)  # (B, K, N, target_dim + 1)

        # Split into features and mask logits
        features = output[..., :-1]  # (B, K, N, target_dim)
        mask_logits = output[..., -1]  # (B, K, N)

        # Softmax over slots → mask competition
        masks = F.softmax(mask_logits, dim=1)  # (B, K, N)

        # Reconstruct: weighted sum of per-slot features
        recon = (masks.unsqueeze(-1) * features).sum(dim=1)  # (B, N, target_dim)

        return recon, masks
