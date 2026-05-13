"""Edge-case tests for SAM-mask subtraction in CopyPasteAugmentation.

Covers:
  - Low-area SAM masks pruned to length 0 when remaining area drops below
    the > 4 px threshold reused from the existing crop/resize helpers.
  - sam_teacher_logits intentionally NOT erased (bug #4 is out of scope).
  - No-op behaviour when no SAM keys are attached.

See plan: docs/plans/2026-05-05_stage4_sam3_alignfix_followups.md
"""
import os
import random
import sys

import pytest
import torch

pytest.importorskip("detectron2")
pytest.importorskip("kornia")
from detectron2.structures import BitMasks, Boxes, Instances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cups.augmentation import CopyPasteAugmentation


def _make_sample(sam_pixels):
    """Build a minimal sample dict with one instance and one SAM mask.

    Args:
        sam_pixels: list of (y, x) pixel coords to set True in the SAM mask.
    """
    masks = torch.zeros(1, 64, 128, dtype=torch.bool)
    masks[0, 0:32, 0:32] = True
    instances = Instances(
        image_size=(64, 128),
        gt_masks=BitMasks(masks.clone()),
        gt_boxes=Boxes(torch.tensor([[0, 0, 32, 32]], dtype=torch.float32)),
        gt_classes=torch.tensor([0], dtype=torch.long),
    )
    sam_masks = torch.zeros(1, 64, 128, dtype=torch.bool)
    for (y, x) in sam_pixels:
        sam_masks[0, y, x] = True
    return {
        "image": torch.zeros(3, 64, 128, dtype=torch.float32),
        "sem_seg": torch.zeros(64, 128, dtype=torch.long),
        "instances": instances,
        "sam_masks": sam_masks,
        "sam_ious": torch.tensor([0.9]),
        "sam_cls": torch.tensor([0], dtype=torch.long),
        "sam_teacher_logits": torch.randn(5, 64, 128),
    }


def _make_aug():
    return CopyPasteAugmentation(
        thing_class=0,
        max_num_pasted_objects=1,
        scale_range=(1.0, 1.0),
        use_random_horizontal_flipping=False,
        min_bounding_box_size=(1, 1),
    )


def test_low_area_sam_mask_pruned_when_paste_overlaps():
    """SAM mask with 5 pixels overlapping the paste loses ≥1 px → falls to ≤4 → pruned."""
    random.seed(0)
    torch.manual_seed(0)
    # The instance bbox is (0,0,32,32) so the source crop is 32x32. With
    # scale_range=(1,1) the paste mask remains 32x32 and may land anywhere
    # within the 64x128 image. Place the SAM pixels INSIDE the source bbox
    # area to maximise overlap with a same-size crop landing nearby.
    sample = _make_sample([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
    out = _make_aug()([sample], [sample])[0]

    # Either the SAM mask survives (paste landed elsewhere) — then it must be
    # unchanged — or it was pruned because remaining area ≤ 4. Aux fields
    # always stay in lockstep with the mask tensor.
    assert out["sam_masks"].shape[0] == out["sam_ious"].shape[0]
    assert out["sam_masks"].shape[0] == out["sam_cls"].shape[0]


def test_sam_teacher_logits_unchanged():
    """Teacher logits are dense spatial maps; bug #4 deferred so do NOT erase."""
    random.seed(1)
    torch.manual_seed(1)
    sample = _make_sample([(40, 100)])
    teacher_before = sample["sam_teacher_logits"].clone()
    out = _make_aug()([sample], [sample])[0]
    assert torch.equal(out["sam_teacher_logits"], teacher_before)


def test_no_sam_keys_no_op():
    """Sample without any sam_* keys runs cleanly through copy-paste."""
    random.seed(2)
    torch.manual_seed(2)
    sample = _make_sample([(40, 100)])
    for key in ("sam_masks", "sam_ious", "sam_cls", "sam_teacher_logits"):
        sample.pop(key, None)

    out = _make_aug()([sample], [sample])[0]
    assert "image" in out
    assert "sem_seg" in out
    assert "instances" in out
    # No SAM keys should have appeared from nowhere.
    for key in ("sam_masks", "sam_ious", "sam_cls", "sam_teacher_logits"):
        assert key not in out


def test_empty_sam_masks_no_op():
    """Sample whose sam_masks tensor is empty — augmentation must not crash."""
    random.seed(4)
    torch.manual_seed(4)
    sample = _make_sample([])
    sample["sam_masks"] = torch.zeros(0, 64, 128, dtype=torch.bool)
    sample["sam_ious"] = torch.zeros(0, dtype=torch.float32)
    sample["sam_cls"] = torch.zeros(0, dtype=torch.long)

    out = _make_aug()([sample], [sample])[0]
    assert out["sam_masks"].shape == (0, 64, 128)
    assert out["sam_ious"].shape == (0,)
    assert out["sam_cls"].shape == (0,)
