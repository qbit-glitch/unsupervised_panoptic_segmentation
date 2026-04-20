"""Guards the segments_info schema emitted by Mask2FormerPanoptic.

The CUPS training loop feeds each step's eval output through
``prediction_to_label_format``, which requires every thing entry in
``segments_info`` to expose a ``"score"`` field. A previous run crashed
at step 872 with ``KeyError: 'score'`` because the meta-arch omitted it.
These tests lock the Cascade-compatible schema in place.
"""
from __future__ import annotations

import torch

from cups.model.model import prediction_to_label_format
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
    criterion = SetCriterion(
        num_classes=20, matcher=matcher,
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}, num_points=64,
    )
    return Mask2FormerPanoptic(
        backbone=adapter, pixel_decoder=pixel, transformer_decoder=dec, criterion=criterion,
        num_stuff_classes=12, num_thing_classes=8,
        object_mask_threshold=0.0,  # keep everything so segments_info is non-empty
        overlap_threshold=0.0,
    )


def _run_eval(num_samples: int = 2) -> tuple[list[dict], list[torch.Tensor]]:
    torch.manual_seed(42)
    model = _tiny_meta_arch().eval()
    images = [torch.randn(3, 64, 128) for _ in range(num_samples)]
    batch = [{"image": img} for img in images]
    with torch.no_grad():
        out = model(batch)
    return out, images


def test_segments_info_has_score_field() -> None:
    out, _ = _run_eval()
    # With object_mask_threshold=0.0 we keep all queries; at least one sample
    # should emit a non-empty segments_info list given the tiny random weights.
    total_segments = sum(len(s["panoptic_seg"][1]) for s in out)
    assert total_segments > 0, "tiny model produced no segments; thresholds may need tuning"
    for sample in out:
        for seg in sample["panoptic_seg"][1]:
            assert "score" in seg, f"missing 'score' in segments_info entry: {seg}"
            assert isinstance(seg["score"], float)
            assert 0.0 <= seg["score"] <= 1.0


def test_segments_info_required_keys() -> None:
    """Every segment must carry the full Cascade-compatible schema."""
    out, _ = _run_eval()
    required = {"id", "category_id", "isthing", "score"}
    for sample in out:
        for seg in sample["panoptic_seg"][1]:
            assert required.issubset(seg.keys()), f"expected {required}, got {set(seg.keys())}"
            assert isinstance(seg["id"], int)
            assert isinstance(seg["category_id"], int)
            assert isinstance(seg["isthing"], bool)
            assert isinstance(seg["score"], float)


def test_prediction_to_label_format_accepts_output() -> None:
    """Integration guard: the schema matches what the CUPS training loop expects.

    This is the exact call that crashed in production
    (pl_model_pseudo.py:176 -> prediction_to_label_format -> KeyError 'score').
    """
    out, images = _run_eval()
    # Must not raise KeyError on 'score'.
    labels = prediction_to_label_format(out, images, confidence_threshold=-1.0)
    assert isinstance(labels, list)
    assert len(labels) == len(out)
    for entry in labels:
        assert "image" in entry
        assert "sem_seg" in entry
        assert "instances" in entry


def test_empty_segments_info_is_handled() -> None:
    """When no queries pass thresholds, segments_info is an empty list (still valid)."""
    torch.manual_seed(7)
    model = _tiny_meta_arch().eval()
    # With an impossibly high threshold, nothing survives.
    model.object_mask_threshold = 1.1
    batch = [{"image": torch.randn(3, 64, 128)}]
    with torch.no_grad():
        out = model(batch)
    assert out[0]["panoptic_seg"][1] == []
    # prediction_to_label_format should still work on the empty case.
    labels = prediction_to_label_format(out, [batch[0]["image"]], confidence_threshold=-1.0)
    assert len(labels) == 1
