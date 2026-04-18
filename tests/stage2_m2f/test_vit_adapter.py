from __future__ import annotations

import torch

from cups.model.modeling.mask2former.vit_adapter import ViTAdapter


class _DummyDino(torch.nn.Module):
    """Replace DINOv3 with a frozen conv that emits (B, 768, H/16, W/16)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_vit_adapter_emits_p2_p5(tiny_image_batch: torch.Tensor) -> None:
    adapter = ViTAdapter(backbone=_DummyDino(), embed_dim=768, num_blocks=2).eval()
    with torch.no_grad():
        feats = adapter(tiny_image_batch)
    assert set(feats.keys()) == {"p2", "p3", "p4", "p5"}
    # Input 64x128 -> strides 4/8/16/32
    assert feats["p2"].shape == (2, 256, 16, 32)
    assert feats["p3"].shape == (2, 256, 8, 16)
    assert feats["p4"].shape == (2, 256, 4, 8)
    assert feats["p5"].shape == (2, 256, 2, 4)


def test_vit_adapter_backbone_frozen(tiny_image_batch: torch.Tensor) -> None:
    adapter = ViTAdapter(backbone=_DummyDino(), embed_dim=768, num_blocks=2)
    feats = adapter(tiny_image_batch)
    loss = sum(t.sum() for t in feats.values())
    loss.backward()
    # Backbone conv must have no grad (frozen); adapter pieces must have grads.
    assert adapter.backbone.conv.weight.grad is None
    assert any(p.grad is not None for p in adapter.spm.parameters())
