"""Freeze helpers for CAUSE-TR Mode B retraining.

Loads the official CAUSE-TR codebook (2048,768) and cluster_probe (27,90) from
the extracted .npz, installs them as frozen `nn.Parameter`s into a `Cluster`
instance, and provides verification utilities to ensure they stay immutable
through training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_frozen_codebook_and_probe(
    npz_path: str | Path,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load `codebook (2048, 768)` and `cluster_probe (27, 90)` from extracted npz.

    Tensors are returned as float32 with `requires_grad=False`. They retain
    bitwise-identical values to the originals from `cluster_tr.pth`.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Frozen centroids file not found: {npz_path}")

    z = np.load(npz_path, allow_pickle=False)
    if "codebook_2048x768" not in z or "cluster_probe_27x90" not in z:
        raise KeyError(
            f"npz missing expected keys. Found: {list(z.keys())}. "
            "Expected: codebook_2048x768 and cluster_probe_27x90."
        )

    codebook = torch.from_numpy(z["codebook_2048x768"]).to(device=device, dtype=torch.float32)
    probe = torch.from_numpy(z["cluster_probe_27x90"]).to(device=device, dtype=torch.float32)

    if codebook.shape != (2048, 768):
        raise ValueError(f"codebook shape mismatch: {tuple(codebook.shape)}, expected (2048, 768)")
    if probe.shape != (27, 90):
        raise ValueError(f"cluster_probe shape mismatch: {tuple(probe.shape)}, expected (27, 90)")

    codebook = codebook.contiguous()
    probe = probe.contiguous()
    codebook.requires_grad_(False)
    probe.requires_grad_(False)

    logger.info(
        "Loaded frozen artifacts from %s (codebook=%s, cluster_probe=%s)",
        npz_path, tuple(codebook.shape), tuple(probe.shape),
    )
    return codebook, probe


def install_frozen_into_cluster(
    cluster: nn.Module,
    codebook: torch.Tensor,
    cluster_probe: torch.Tensor,
) -> None:
    """Replace `cluster.codebook` and `cluster.cluster_probe` with frozen `nn.Parameter`s.

    Both wrapped Parameters have `requires_grad=False`. The values are clones of the
    inputs so the caller's tensors remain independent.
    """
    cluster.codebook = nn.Parameter(codebook.detach().clone(), requires_grad=False)
    cluster.cluster_probe = nn.Parameter(cluster_probe.detach().clone(), requires_grad=False)
    logger.info(
        "Installed frozen codebook (%s) and cluster_probe (%s) into Cluster",
        tuple(cluster.codebook.shape), tuple(cluster.cluster_probe.shape),
    )


def wire_codebook_into_segment(segment: nn.Module, cluster: nn.Module) -> None:
    """Point `segment.head.codebook` and `segment.head_ema.codebook` to the
    frozen `cluster.codebook`.

    `Decoder.__init__` defaults to `codebook=None`, but `Decoder.forward` does
    `vqt(feat, self.codebook)` which would fail. This wiring step is required.
    """
    segment.head.codebook = cluster.codebook
    segment.head_ema.codebook = cluster.codebook
    logger.info("Wired frozen codebook into Segment_TR.head and head_ema")


def verify_freeze(
    cluster: nn.Module,
    backbone: nn.Module | None = None,
    expected_codebook: torch.Tensor | None = None,
    expected_probe: torch.Tensor | None = None,
) -> None:
    """Assertion-based check that frozen artifacts are immutable and intact.

    Raises:
        AssertionError if any frozen tensor has `requires_grad=True` or has drifted
        from the expected values.
    """
    assert cluster.codebook.requires_grad is False, "cluster.codebook must be frozen"
    assert cluster.cluster_probe.requires_grad is False, "cluster.cluster_probe must be frozen"

    # Tolerance = float32 ULP (~3e-8). The codebook is referenced (not optimized),
    # but indexing operations like `vqt`'s `c[idx]` can produce 1-ULP drift across
    # PyTorch op chains. atol=1e-7 catches any real training drift while ignoring
    # this fp32 noise.
    if expected_codebook is not None:
        actual_cb = cluster.codebook.detach().to(expected_codebook.device)
        max_d = (actual_cb - expected_codebook).abs().max().item()
        assert torch.allclose(actual_cb, expected_codebook, atol=1e-7, rtol=0), (
            f"cluster.codebook drifted from expected values (max diff={max_d:.3e})"
        )
    if expected_probe is not None:
        actual_cp = cluster.cluster_probe.detach().to(expected_probe.device)
        max_d = (actual_cp - expected_probe).abs().max().item()
        assert torch.allclose(actual_cp, expected_probe, atol=1e-7, rtol=0), (
            f"cluster.cluster_probe drifted from expected values (max diff={max_d:.3e})"
        )

    if backbone is not None:
        bad = [n for n, p in backbone.named_parameters() if p.requires_grad]
        assert not bad, f"Backbone has trainable params: {bad[:5]} ..."

    logger.info("verify_freeze passed: codebook + cluster_probe + backbone are all frozen")


def collect_trainable_params(modules: list[nn.Module]) -> list[nn.Parameter]:
    """Collect all parameters with `requires_grad=True` across the listed modules.

    Useful for building optimizer groups while skipping frozen Parameters
    (codebook, cluster_probe, EMA twins).
    """
    params: list[nn.Parameter] = []
    for m in modules:
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    return params


__all__ = [
    "load_frozen_codebook_and_probe",
    "install_frozen_into_cluster",
    "wire_codebook_into_segment",
    "verify_freeze",
    "collect_trainable_params",
]
