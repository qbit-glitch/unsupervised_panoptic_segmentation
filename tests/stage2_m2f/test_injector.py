from __future__ import annotations

import torch

from cups.model.modeling.mask2former.injector import Injector
from cups.model.modeling.mask2former.spm import SpatialPriorModule


def test_injector_preserves_vit_shape(tiny_image_batch: torch.Tensor, tiny_vit_patch: torch.Tensor) -> None:
    spm = SpatialPriorModule(embed_dim=768).eval()
    injector = Injector(embed_dim=768, num_heads=8).eval()
    with torch.no_grad():
        c2, c3, c4 = spm(tiny_image_batch)
        out = injector(vit_feat=tiny_vit_patch, c2=c2, c3=c3, c4=c4)
    assert out.shape == tiny_vit_patch.shape


def test_injector_gradients_flow(tiny_image_batch: torch.Tensor, tiny_vit_patch: torch.Tensor) -> None:
    spm = SpatialPriorModule(embed_dim=768).train()
    injector = Injector(embed_dim=768, num_heads=8).train()
    c2, c3, c4 = spm(tiny_image_batch)
    out = injector(vit_feat=tiny_vit_patch, c2=c2, c3=c3, c4=c4)
    out.sum().backward()
    for p in injector.parameters():
        assert p.grad is not None
