"""2D ↔ 1D scan/unscan functions for Vision Mamba2.

Provides different spatial scanning strategies to convert 2D image features
into 1D sequences for Mamba2 processing.

Scan strategies:
    - Raster: row-major flatten (left→right, top→bottom)
    - Cross-scan (VMamba 4-way): 4 directions batched into B dimension
      Dir 0: row-major forward
      Dir 1: row-major backward
      Dir 2: column-major forward
      Dir 3: column-major backward

All functions are pure tensor ops — no learnable parameters.
"""

import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Raster scan (simple row-major flatten)
# ---------------------------------------------------------------------------

def raster_scan(x: torch.Tensor) -> torch.Tensor:
    """Flatten 2D spatial features to 1D sequence in raster (row-major) order.

    Args:
        x: (B, C, H, W)
    Returns:
        (B, H*W, C)
    """
    assert x.dim() == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def raster_unscan(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Unflatten 1D sequence back to 2D spatial features.

    Args:
        x: (B, H*W, C)
        H: spatial height
        W: spatial width
    Returns:
        (B, C, H, W)
    """
    B, L, C = x.shape
    assert L == H * W
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


# ---------------------------------------------------------------------------
# Cross-scan (VMamba-style 4-way directional scan)
# ---------------------------------------------------------------------------

def cross_scan(x: torch.Tensor) -> torch.Tensor:
    """Scan 2D features in 4 directions, batched into B dimension.

    The 4 directions are:
        Dir 0: row-major forward  (left→right, top→bottom)
        Dir 1: row-major backward (right→left, bottom→top)
        Dir 2: col-major forward  (top→bottom, left→right)
        Dir 3: col-major backward (bottom→top, right→left)

    Args:
        x: (B, C, H, W)
    Returns:
        (4*B, H*W, C) — 4 scan directions stacked along batch dim
    """
    B, C, H, W = x.shape

    # Dir 0: row-major forward — standard raster
    dir0 = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

    # Dir 1: row-major backward — reverse of dir0
    dir1 = dir0.flip(1)  # (B, H*W, C)

    # Dir 2: col-major forward — transpose spatial dims, then raster
    x_t = x.permute(0, 1, 3, 2)  # (B, C, W, H) — swap H,W
    dir2 = x_t.flatten(2).transpose(1, 2)  # (B, H*W, C) but in col-major order

    # Dir 3: col-major backward — reverse of dir2
    dir3 = dir2.flip(1)  # (B, H*W, C)

    # Stack all 4 directions along batch dim
    return torch.cat([dir0, dir1, dir2, dir3], dim=0).contiguous()  # (4*B, H*W, C)


def cross_unscan(x: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    """Reverse 4-way cross-scan back to per-direction 2D features.

    Args:
        x: (4*B, H*W, C) — output from Mamba2 after cross_scan
        B: original batch size
        H: spatial height
        W: spatial width
    Returns:
        (B, 4, H*W, C) — 4 directions un-scanned and aligned to original spatial order
    """
    L = H * W
    C = x.shape[-1]

    # Split back into 4 directions
    dir0, dir1, dir2, dir3 = x.chunk(4, dim=0)  # each (B, H*W, C)

    # Dir 0: already in raster order — no change
    # dir0 = dir0

    # Dir 1: reverse back to raster order
    dir1 = dir1.flip(1)

    # Dir 2: col-major → raster order
    # Reshape to (B, C, W, H) spatial in col-major, then transpose back
    dir2_spatial = dir2.transpose(1, 2).reshape(B, C, W, H)  # (B, C, W, H)
    dir2_spatial = dir2_spatial.permute(0, 1, 3, 2)  # (B, C, H, W)
    dir2 = dir2_spatial.flatten(2).transpose(1, 2)  # (B, H*W, C) in raster order

    # Dir 3: reverse col-major → raster order
    dir3 = dir3.flip(1)
    dir3_spatial = dir3.transpose(1, 2).reshape(B, C, W, H)
    dir3_spatial = dir3_spatial.permute(0, 1, 3, 2)
    dir3 = dir3_spatial.flatten(2).transpose(1, 2)

    # Stack directions: (B, 4, H*W, C)
    return torch.stack([dir0, dir1, dir2, dir3], dim=1)


# ---------------------------------------------------------------------------
# Cross-modal interleaving
# ---------------------------------------------------------------------------

def interleave_tokens(
    semantic: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Interleave two token sequences: [s_1, f_1, s_2, f_2, ...].

    Args:
        semantic: (B, N, D)
        features: (B, N, D)
    Returns:
        (B, 2*N, D)
    """
    b, n, d = semantic.shape
    interleaved = torch.stack([semantic, features], dim=2)  # (B, N, 2, D)
    return interleaved.reshape(b, 2 * n, d)


def deinterleave_tokens(
    interleaved: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Separate interleaved tokens back into two streams.

    Args:
        interleaved: (B, 2*N, D)
    Returns:
        (semantic, features) each of shape (B, N, D)
    """
    b, l, d = interleaved.shape
    n = l // 2
    reshaped = interleaved.reshape(b, n, 2, d)
    return reshaped[:, :, 0, :], reshaped[:, :, 1, :]
