"""Dense-affinity aux losses (P4): Gated-CRF and NeCo.

* ``gated_crf`` is a simplified local-window Gated-CRF (Obukhov 2019):
  for each pixel and each of its K*K neighbours, the pair contributes a
  bilateral-weighted (1 - <p_i, p_j>) term. The center is excluded.
* ``neco`` is a neighbourhood-consistency term inspired by NeCo
  (Pariza et al., ICCV 2023): each patch pulls its top-k DINOv3
  neighbours closer in logit cosine space.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F


def _to_patches(dino: torch.Tensor) -> torch.Tensor:
    """Return ``(B, N, D)`` patch tensor from either layout."""
    if dino.dim() == 4:
        B, D, Hs, Ws = dino.shape
        return dino.permute(0, 2, 3, 1).reshape(B, Hs * Ws, D)
    if dino.dim() == 3:
        return dino
    raise ValueError(
        f"dino_features must have 3 or 4 dims, got shape {tuple(dino.shape)}"
    )


def gated_crf(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    ctx: Dict[str, Any],
) -> torch.Tensor:
    """Local-window Gated-CRF pairwise loss.

    Loss = mean over centres c and neighbours n of
              w(c, n) * (1 - <p_c, p_n>)

    where ``w(c, n) = exp(-|rgb_c - rgb_n|^2 / (2 * srgb^2))`` and
    ``<p, q>`` is the softmax dot-product.
    """
    if "rgb" not in ctx:
        raise KeyError(
            "gated_crf requires ctx['rgb']; the PanopticFPN wrapper must "
            "forward images.tensor into the semantic-head ctx."
        )
    params = ctx.get("params", {})
    K = int(params.get("gated_crf_kernel", 5))
    if K % 2 == 0:
        K += 1  # enforce odd kernel so padding stays symmetric
    srgb = float(params.get("gated_crf_rgb_sigma", 0.1))
    pad = K // 2
    mid = (K * K) // 2

    B, C, H, W = logits.shape
    rgb = ctx["rgb"].float()
    if rgb.shape[-2:] != (H, W):
        rgb = F.interpolate(rgb, size=(H, W), mode="bilinear", align_corners=False)

    probs = F.softmax(logits, dim=1)

    # (B, C*K*K, H*W) -> (B, C, K*K, H*W)
    p_u = F.unfold(probs, kernel_size=K, padding=pad).view(B, C, K * K, H * W)
    r_u = F.unfold(rgb, kernel_size=K, padding=pad).view(B, 3, K * K, H * W)

    p_center = probs.view(B, C, 1, H * W)
    r_center = rgb.view(B, 3, 1, H * W)

    rdiff = (r_u - r_center).pow(2).sum(dim=1)  # (B, K*K, H*W)
    w = torch.exp(-rdiff / (2.0 * srgb * srgb))

    dot = (p_u * p_center).sum(dim=1)  # (B, K*K, H*W)

    mask = torch.ones(K * K, device=logits.device, dtype=logits.dtype)
    mask[mid] = 0.0
    mask = mask.view(1, K * K, 1)

    return ((w * mask) * (1.0 - dot)).mean()


def neco(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    ctx: Dict[str, Any],
) -> torch.Tensor:
    """Neighbourhood-consistency on DINOv3 patch features.

    For each patch, find the top-k nearest DINO neighbours and pull the
    corresponding semantic logits closer in cosine space. Similar to
    STEGO but averaged over a neighbourhood instead of anchor-positive
    pairs.
    """
    if "dino_features" not in ctx:
        raise KeyError(
            "neco requires ctx['dino_features']; ensure the PanopticFPN "
            "wrapper populates the semantic-head ctx with DINOv3 features."
        )
    params = ctx.get("params", {})
    k = int(params.get("neco_k", 5))

    dino = _to_patches(ctx["dino_features"])  # (B, N, D)
    B, N, _ = dino.shape

    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise ValueError(
            f"neco expects a square DINO patch grid (N must be a perfect square); got N={N}"
        )
    Hs = Ws = side

    codes = F.adaptive_avg_pool2d(logits, output_size=(Hs, Ws))
    codes = codes.permute(0, 2, 3, 1).reshape(B, Hs * Ws, -1)  # (B, N, C)
    codes = F.normalize(codes, dim=-1)
    dn = F.normalize(dino, dim=-1)

    eff_k = min(k, N - 1)
    if eff_k <= 0:
        return codes.sum() * 0.0

    losses = []
    for b in range(B):
        sim = dn[b] @ dn[b].transpose(0, 1)
        sim.fill_diagonal_(float("-inf"))
        _, idx = sim.topk(eff_k, dim=-1)  # (N, eff_k)
        neigh = codes[b][idx]             # (N, eff_k, C)
        anchor = codes[b].unsqueeze(1)    # (N, 1, C)
        losses.append((1.0 - (anchor * neigh).sum(dim=-1)).mean())
    return torch.stack(losses).mean()


__all__ = ["gated_crf", "neco"]
