"""Slot Attention module with depth-conditioned feature input.

Based on Locatello et al. (2020) "Object-Centric Learning with Slot Attention"
and Seitzer et al. (2023) "Bridging the Gap to Real-World Object-Centric Learning" (DINOSAUR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Iterative slot attention for object discovery.

    Patches compete for slots via softmax over slot dimension. Each iteration
    refines slot representations through attention-weighted aggregation + GRU update.

    Args:
        num_slots: Number of object slots (max instances per image).
        dim: Slot feature dimension.
        iters: Number of attention iterations.
        hidden_dim: MLP hidden dimension for slot update.
        eps: Numerical stability for attention normalization.
    """

    def __init__(
        self,
        num_slots: int = 20,
        dim: int = 256,
        iters: int = 5,
        hidden_dim: int = 512,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        self.eps = eps

        # Learnable slot initialization (sample from learned Gaussian)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim) * (dim ** -0.5))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))

        # Attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Slot update (GRU + MLP residual)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run iterative slot attention.

        Args:
            inputs: (B, N, D) encoded patch features (after FiLM modulation + projection).

        Returns:
            slots: (B, K, D) final slot representations.
            attn: (B, K, N) attention weights (softmax over slots for each patch).
        """
        B, N, D = inputs.shape
        device = inputs.device

        # Initialize slots from learned Gaussian
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_logsigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # (B, N, D)
        v = self.to_v(inputs)  # (B, N, D)
        scale = D ** -0.5

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)  # (B, K, D)

            # Attention logits: (B, K, N)
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * scale

            # Softmax over slots → each patch distributed across slots
            attn = F.softmax(attn_logits, dim=1)  # (B, K, N)

            # Weighted mean of values (normalize per slot)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)  # (B, K, D)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D),
            ).reshape(B, -1, D)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Final attention pass for mask extraction
        slots_final = self.norm_slots(slots)
        q = self.to_q(slots_final)
        attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * scale
        attn = F.softmax(attn_logits, dim=1)  # (B, K, N)

        return slots, attn
