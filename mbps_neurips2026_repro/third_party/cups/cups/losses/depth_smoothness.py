"""Depth-guided logit regularizer (P3 aux-loss).

Penalises logit gradients where depth is locally smooth: regions of flat
depth (roads, sky, uniform walls) should also have spatially flat logits,
whereas depth discontinuities (object boundaries) are free to carry sharp
semantic transitions. The weighting follows the standard edge-aware
smoothness formulation :math:`w = \\exp(-\\alpha \\cdot \\lVert \\nabla
D \\rVert)` popular in self-supervised depth/SLAM literature.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn import functional as F


def _sobel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (Gx, Gy) per-channel Sobel responses for a (B, C, H, W) tensor."""
    device, dtype = x.device, x.dtype
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    c = x.shape[1]
    kx = kx.expand(c, 1, 3, 3)
    ky = ky.expand(c, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def depth_smoothness(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    ctx: Dict[str, Any],
) -> torch.Tensor:
    """Edge-aware smoothness penalty on logits, gated by depth gradients.

    Loss = mean(  exp(-alpha * (|dD/dx| + |dD/dy|))
                  * mean_c(|dL_c/dx| + |dL_c/dy|)  )

    Args:
        logits: (B, C, H, W) semantic logits at eval resolution.
        targets: Unused; kept for signature uniformity across aux losses.
        ctx: Must contain ``depth`` ((B, 1, H, W), (B, H, W) or another
            resolution -- will be bilinearly resized to logit resolution)
            and ``params["depth_smooth_alpha"]`` (float, defaults to 10.0).

    Raises:
        KeyError: if ``ctx`` is missing ``depth``.
    """
    if "depth" not in ctx:
        raise KeyError(
            "depth_smoothness requires ctx['depth']; the PanopticFPN wrapper "
            "must forward batch['depth'] into the semantic-head ctx."
        )
    params = ctx.get("params", {})
    alpha = float(params.get("depth_smooth_alpha", 10.0))

    depth = ctx["depth"]
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    elif depth.dim() != 4:
        raise ValueError(
            f"depth_smoothness expected depth with 3 or 4 dims, got shape {tuple(depth.shape)}"
        )
    depth = depth.float()

    _, _, H, W = logits.shape
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

    gx_l, gy_l = _sobel(logits)
    gx_d, gy_d = _sobel(depth)

    # Depth-edge gating: 1 inside smooth regions, ~0 across depth discontinuities.
    gate = torch.exp(-alpha * (gx_d.abs() + gy_d.abs()))

    logit_mag = (gx_l.abs() + gy_l.abs()).mean(dim=1, keepdim=True)
    return (gate * logit_mag).mean()


__all__ = ["depth_smoothness"]
