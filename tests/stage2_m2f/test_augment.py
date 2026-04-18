from __future__ import annotations

import torch

from cups.model.modeling.mask2former.augment import ColorJitterModule, LargeScaleJitter


def test_lsj_scales_output_within_range() -> None:
    torch.manual_seed(0)
    lsj = LargeScaleJitter(min_scale=0.1, max_scale=2.0, target_size=(64, 128))
    img = torch.randn(3, 64, 128)
    lbl = torch.randint(0, 10, (1, 64, 128))
    out_img, out_lbl = lsj(img, lbl)
    assert out_img.shape == (3, 64, 128)
    assert out_lbl.shape == (1, 64, 128)
    assert out_img.abs().sum() > 0.0, "LSJ returned all-zeros image (forgot to copy content)"


def test_color_jitter_is_identity_when_all_zero() -> None:
    torch.manual_seed(0)
    cj = ColorJitterModule(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0)
    img = torch.rand(3, 64, 128)
    out = cj(img)
    assert torch.allclose(out, img)
