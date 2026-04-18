from __future__ import annotations

import torch

from cups.model.modeling.mask2former.msdeform_pixel_decoder import MSDeformAttnPixelDecoder


def test_pixel_decoder_shapes() -> None:
    torch.manual_seed(0)
    B, C = 2, 256
    feats = {
        "p2": torch.randn(B, C, 16, 32),
        "p3": torch.randn(B, C, 8, 16),
        "p4": torch.randn(B, C, 4, 8),
        "p5": torch.randn(B, C, 2, 4),
    }
    decoder = MSDeformAttnPixelDecoder(in_channels=C, hidden_dim=C, mask_dim=C, num_layers=3).eval()
    with torch.no_grad():
        mask_feat, multi_scale = decoder(feats)
    # mask_feat at stride 4 (same as p2)
    assert mask_feat.shape == (B, C, 16, 32)
    # multi_scale has 3 entries (p3, p4, p5 after transformer)
    assert len(multi_scale) == 3
    assert multi_scale[0].shape == (B, C, 8, 16)
    assert multi_scale[1].shape == (B, C, 4, 8)
    assert multi_scale[2].shape == (B, C, 2, 4)


def test_pixel_decoder_gradients() -> None:
    torch.manual_seed(1)
    B, C = 1, 128
    feats = {
        "p2": torch.randn(B, C, 16, 32, requires_grad=True),
        "p3": torch.randn(B, C, 8, 16, requires_grad=True),
        "p4": torch.randn(B, C, 4, 8, requires_grad=True),
        "p5": torch.randn(B, C, 2, 4, requires_grad=True),
    }
    decoder = MSDeformAttnPixelDecoder(in_channels=C, hidden_dim=C, mask_dim=C, num_layers=2).train()
    mask_feat, _ = decoder(feats)
    mask_feat.sum().backward()
    assert any(p.grad is not None for p in decoder.parameters())
