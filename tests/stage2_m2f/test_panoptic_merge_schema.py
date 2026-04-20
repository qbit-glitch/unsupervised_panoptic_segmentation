"""Guards the segments_info schema emitted by Mask2FormerPanoptic.

Two historical crashes motivated this file:
1. Step 872 crash: ``KeyError: 'score'`` in ``prediction_to_label_format``
   (model.py:345). Fixed by adding "score" to segments_info.
2. Step 3800 val crash: ``IndexError: tuple index out of range`` in
   ``prediction_to_standard_format`` (model.py:196) where
   ``thing_classes[pred["category_id"]]`` was indexed with a combined-space
   class id (e.g. 13 for an 8-thing dataset). Fixed by remapping
   category_id into Cascade-era space (stuff 1-indexed [1,S]; thing 0-indexed
   [0,T)) for external consumers while keeping the internal criterion on
   the combined space.

These tests lock both fixes + the Cascade-compatible schema in place.
"""
from __future__ import annotations

import torch

from cups.model.model import prediction_to_label_format, prediction_to_standard_format
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


def test_category_id_in_cascade_ranges() -> None:
    """Category ids must be in the Cascade-era space.

    Stuff: 1-indexed [1, S]. Thing: 0-indexed [0, T).
    Downstream code indexes ``stuff_classes[cid - 1]`` and ``thing_classes[cid]``,
    so emitting combined-space ids (e.g. 13 for a thing) corrupts semantics
    and crashes ``prediction_to_standard_format``.
    """
    num_stuff, num_thing = 12, 8  # matches _tiny_meta_arch()
    out, _ = _run_eval()
    for sample in out:
        for seg in sample["panoptic_seg"][1]:
            cid = seg["category_id"]
            if seg["isthing"]:
                assert 0 <= cid < num_thing, f"thing cid={cid} not in [0, {num_thing})"
            else:
                assert 1 <= cid <= num_stuff, f"stuff cid={cid} not in [1, {num_stuff}]"


def test_prediction_to_label_format_accepts_output() -> None:
    """Integration guard: output matches what the CUPS training loop expects.

    Exact call that crashed at step 872
    (pl_model_pseudo.py:176 -> prediction_to_label_format -> KeyError 'score').
    """
    out, images = _run_eval()
    labels = prediction_to_label_format(out, images, confidence_threshold=-1.0)
    assert isinstance(labels, list)
    assert len(labels) == len(out)
    for entry in labels:
        assert "image" in entry
        assert "sem_seg" in entry
        assert "instances" in entry


def test_prediction_to_standard_format_accepts_output() -> None:
    """Integration guard: output is indexable by stuff_classes / thing_classes.

    Exact call that crashed at step 3800
    (pl_model_pseudo.py:303 -> prediction_to_standard_format -> IndexError).
    Uses Cityscapes-like class tuples sized to _tiny_meta_arch()'s 12 stuff
    and 8 things.
    """
    out, _ = _run_eval()
    # Stuff classes: 12 dataset ids; thing classes: 8 dataset ids.
    stuff_classes = tuple(range(12))
    thing_classes = tuple(range(24, 32))
    for sample in out:
        panoptic = prediction_to_standard_format(
            sample["panoptic_seg"],
            stuff_classes=stuff_classes,
            thing_classes=thing_classes,
        )
        # Shape [H, W, 2] (semantic, instance).
        assert panoptic.dim() == 3 and panoptic.shape[-1] == 2


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
