from __future__ import annotations

import torch

from cups.model.modeling.mask2former.spm import SpatialPriorModule


def test_spm_emits_three_scales(tiny_image_batch: torch.Tensor) -> None:
    spm = SpatialPriorModule(in_channels=3, embed_dim=768).eval()
    with torch.no_grad():
        c2, c3, c4 = spm(tiny_image_batch)
    # Strides (4, 8, 16) relative to input 64x128.
    assert c2.shape == (2, 768, 16, 32)
    assert c3.shape == (2, 768, 8, 16)
    assert c4.shape == (2, 768, 4, 8)


def test_spm_gradients_flow(tiny_image_batch: torch.Tensor) -> None:
    spm = SpatialPriorModule(in_channels=3, embed_dim=768).train()
    out = spm(tiny_image_batch)
    loss = sum(t.sum() for t in out)
    loss.backward()
    for p in spm.parameters():
        assert p.grad is not None, f"no grad on {p.shape}"
