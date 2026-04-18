"""STEGO correspondence loss shim (P2 aux-loss).

Wraps :func:`mbps_pytorch.models.semantic.stego_loss.stego_loss` so it
can be invoked from the CUPS semantic head. The wrapper is responsible
for pooling the predicted logits down to the DINOv3 patch resolution
and for accepting both ``(B, D, Hs, Ws)`` and ``(B, Ns, D)`` feature
layouts.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F


_STEGO_IMPORTED: Optional[Any] = None


def _import_stego_loss():
    global _STEGO_IMPORTED
    if _STEGO_IMPORTED is not None:
        return _STEGO_IMPORTED
    # The CUPS fork ships under ``refs/cups``; the ``mbps_pytorch``
    # helpers live at the project root. Add that to sys.path on first
    # use so pytest / training runs both resolve the import.
    here = Path(__file__).resolve()
    # refs/cups/cups/losses/stego_adapter.py -> repo root is 4 levels up.
    repo_root = here.parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mbps_pytorch.models.semantic.stego_loss import stego_loss as _fn

    _STEGO_IMPORTED = _fn
    return _fn


def _to_patches(dino: torch.Tensor) -> torch.Tensor:
    """Coerce ``dino`` to ``(B, N, D)`` with ``N = Hs*Ws`` patches."""
    if dino.dim() == 4:
        B, D, Hs, Ws = dino.shape
        return dino.permute(0, 2, 3, 1).reshape(B, Hs * Ws, D)
    if dino.dim() == 3:
        return dino
    raise ValueError(
        f"stego_corr expected dino_features with 3 or 4 dims, got shape {tuple(dino.shape)}"
    )


def _resolve_patch_grid(num_patches: int, params: Dict[str, Any]) -> tuple[int, int]:
    explicit = params.get("stego_patch_grid")
    if explicit is not None:
        Hs, Ws = int(explicit[0]), int(explicit[1])
        if Hs * Ws != num_patches:
            raise ValueError(
                f"stego_patch_grid={explicit} does not match num_patches={num_patches}"
            )
        return Hs, Ws
    # Default: square grid if the patch count is a perfect square.
    side = int(round(math.sqrt(num_patches)))
    if side * side == num_patches:
        return side, side
    raise ValueError(
        f"Cannot infer DINO patch grid from {num_patches} patches — "
        "pass ctx['params']['stego_patch_grid'] = (Hs, Ws)."
    )


def stego_corr(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    ctx: Dict[str, Any],
) -> torch.Tensor:
    """Pool logits to the DINO patch resolution and apply STEGO InfoNCE.

    Raises:
        KeyError: if ``ctx`` is missing ``dino_features``.
    """
    if "dino_features" not in ctx:
        raise KeyError(
            "stego_corr requires ctx['dino_features']; ensure the semantic "
            "head ctx is populated by the PanopticFPN wrapper."
        )
    params = ctx.get("params", {})
    dino = _to_patches(ctx["dino_features"])  # (B, Ns, D)
    B, Ns, _ = dino.shape
    Hs, Ws = _resolve_patch_grid(Ns, params)

    codes = F.adaptive_avg_pool2d(logits, output_size=(Hs, Ws))  # (B, C, Hs, Ws)
    codes = codes.permute(0, 2, 3, 1).reshape(B, Hs * Ws, -1)   # (B, Ns, C)

    temperature = float(params.get("stego_temperature", 0.1))
    knn_k = int(params.get("stego_knn_k", 7))
    fn = _import_stego_loss()
    return fn(
        semantic_codes=codes,
        dino_features=dino,
        temperature=temperature,
        knn_k=knn_k,
    )


__all__ = ["stego_corr"]
