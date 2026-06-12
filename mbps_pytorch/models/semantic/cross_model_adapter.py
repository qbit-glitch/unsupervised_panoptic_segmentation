"""Cross-model fusion adapter: condition one frozen model's codes on another's.

Wraps the proven DCFA `DepthAdapter` (concat-conditioning, zero-init residual)
with a learned projection of the cross-model code into the 16-d conditioning
slot that DCFA used for sinusoidal depth. Also provides the stratified pair
sampler and the teacher-guided correlation loss (DCFA's depth kernel replaced
by clamped teacher-code cosine similarity — spec section 4.3 of
docs/superpowers/specs/2026-06-12-depthg-cause-fusion-adapter-design.md).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbps_pytorch.models.semantic.depth_adapter import DepthAdapter


class CrossModelAdapter(nn.Module):
    """codes' = codes + r([codes; W_proj @ cond]) with zero-init residual head.

    Side A: code_dim=90 (CAUSE), cond_dim=100 (DepthG).
    Side B: code_dim=100 (DepthG), cond_dim=90 (CAUSE).
    """

    def __init__(
        self,
        code_dim: int,
        cond_dim: int,
        proj_width: int = 16,
        hidden_dim: int = 384,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.cond_dim = cond_dim
        self.proj_width = proj_width
        self.proj = nn.Linear(cond_dim, proj_width)
        self.core = DepthAdapter(
            code_dim=code_dim, depth_dim=proj_width,
            hidden_dim=hidden_dim, num_layers=num_layers,
        )

    def forward(self, codes: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """codes: (B, N, code_dim); cond: (B, N, cond_dim), grid-aligned."""
        return self.core(codes, self.proj(cond))


def sample_stratified_pairs(
    h: int,
    w: int,
    n_short: int = 512,
    n_long: int = 512,
    r_short: int = 4,
    r_long: int = 8,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Sample flat-index pixel pairs on an (h, w) grid.

    Short pool: anchor + offset with L_inf in [1, r_short] (clamped to grid).
    Long pool: uniform pairs rejection-filtered to L_inf >= r_long.
    Returns ((idx_i_short, idx_j_short), (idx_i_long, idx_j_long)).
    """
    ai = torch.randint(0, h, (n_short,), device=device, generator=generator)
    aj = torch.randint(0, w, (n_short,), device=device, generator=generator)
    dr = torch.randint(-r_short, r_short + 1, (n_short,), device=device, generator=generator)
    dc = torch.randint(-r_short, r_short + 1, (n_short,), device=device, generator=generator)
    zero = (dr == 0) & (dc == 0)
    dr[zero] = 1  # nudge zero offsets to a valid neighbour
    bi = (ai + dr).clamp(0, h - 1)
    bj = (aj + dc).clamp(0, w - 1)
    # clamping can re-create zero offsets at borders; nudge the row index inward
    still_zero = (bi == ai) & (bj == aj)
    if still_zero.any():
        step = torch.where(ai[still_zero] < h - 1,
                           torch.ones_like(ai[still_zero]),
                           -torch.ones_like(ai[still_zero]))
        bi[still_zero] = (ai[still_zero] + step).clamp(0, h - 1)
    short = (ai * w + aj, bi * w + bj)

    n = h * w
    oversample = n_long * 4
    pi = torch.randint(0, n, (oversample,), device=device, generator=generator)
    pj = torch.randint(0, n, (oversample,), device=device, generator=generator)
    linf = torch.max((pi // w - pj // w).abs(), (pi % w - pj % w).abs())
    keep = (linf >= r_long).nonzero(as_tuple=True)[0]
    if keep.numel() < n_long:  # tiny grids: pad by repeating accepted pairs
        reps = (n_long + max(keep.numel(), 1) - 1) // max(keep.numel(), 1)
        keep = keep.repeat(reps)
    keep = keep[:n_long]
    return short, (pi[keep], pj[keep])


def teacher_guided_correlation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
) -> torch.Tensor:
    """w_ij * (1 - cos(student_i, student_j))^2 with w_ij = max(cos(teacher_i, teacher_j), 0).

    Same functional form as stego_loss.depth_guided_correlation_loss, with the
    depth kernel replaced by clamped teacher cosine similarity (spec 4.3).
    student: (N, Ds); teacher: (N, Dt); idx_*: (P,) flat indices.
    """
    with torch.no_grad():
        w = F.cosine_similarity(teacher[idx_i], teacher[idx_j], dim=-1).clamp_min(0.0)
    cos = F.cosine_similarity(student[idx_i], student[idx_j], dim=-1)
    return (w * (1.0 - cos) ** 2).mean()
