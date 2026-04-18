from __future__ import annotations

import torch

from cups.model.modeling.mask2former.extractor import Extractor
from cups.model.modeling.mask2former.spm import SpatialPriorModule


def test_extractor_preserves_spm_shapes(tiny_image_batch: torch.Tensor, tiny_vit_patch: torch.Tensor) -> None:
    spm = SpatialPriorModule(embed_dim=768).eval()
    extractor = Extractor(embed_dim=768, num_heads=8).eval()
    with torch.no_grad():
        c2, c3, c4 = spm(tiny_image_batch)
        c2_out, c3_out, c4_out = extractor(c2=c2, c3=c3, c4=c4, vit_feat=tiny_vit_patch)
    assert c2_out.shape == c2.shape
    assert c3_out.shape == c3.shape
    assert c4_out.shape == c4.shape
