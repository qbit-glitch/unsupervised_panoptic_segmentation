from __future__ import annotations

import torch

from cups.model.modeling.meta_arch.mask2former_panoptic import Mask2FormerPanoptic
from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.masked_attention_decoder import MaskedAttentionDecoder
from cups.model.modeling.mask2former.msdeform_pixel_decoder import MSDeformAttnPixelDecoder
from cups.model.modeling.mask2former.query_pool import build_query_pool
from cups.model.modeling.mask2former.set_criterion import SetCriterion
from cups.model.modeling.mask2former.vit_adapter import ViTAdapter


class _DummyDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _tiny_meta_arch() -> Mask2FormerPanoptic:
    adapter = ViTAdapter(backbone=_DummyDino(), embed_dim=768, num_blocks=1, pyramid_channels=128)
    pixel = MSDeformAttnPixelDecoder(in_channels=128, hidden_dim=128, mask_dim=128, num_layers=2, num_heads=4)
    pool = build_query_pool(kind="standard", num_queries=10, embed_dim=128)
    dec = MaskedAttentionDecoder(hidden_dim=128, num_queries=10, num_classes=20, num_layers=2, num_heads=4, query_pool=pool)
    matcher = HungarianMatcher(num_points=64)
    criterion = SetCriterion(num_classes=20, matcher=matcher, weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}, num_points=64)
    return Mask2FormerPanoptic(
        backbone=adapter, pixel_decoder=pixel, transformer_decoder=dec, criterion=criterion,
        num_stuff_classes=12, num_thing_classes=8,
    )


def test_meta_arch_train_returns_loss_dict(tiny_gt_panoptic: list[dict]) -> None:
    torch.manual_seed(0)
    model = _tiny_meta_arch().train()
    batch = []
    for gt in tiny_gt_panoptic:
        batch.append({
            "image": torch.randn(3, 64, 128),
            "_m2f_targets": {"labels": gt["labels"], "masks": gt["masks"]},
        })
    loss_dict = model(batch)
    assert isinstance(loss_dict, dict)
    assert "loss_ce" in loss_dict
    assert all(torch.isfinite(v) for v in loss_dict.values())


def test_meta_arch_eval_returns_panoptic(tiny_gt_panoptic: list[dict]) -> None:
    torch.manual_seed(1)
    model = _tiny_meta_arch().eval()
    batch = [{"image": torch.randn(3, 64, 128)} for _ in range(2)]
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, list)
    assert len(out) == 2
    for sample in out:
        assert "sem_seg" in sample
        assert "panoptic_seg" in sample
        sem = sample["sem_seg"]
        assert sem.dim() == 3  # (K, H, W)
