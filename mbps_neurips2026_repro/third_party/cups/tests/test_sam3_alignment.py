import os
import random
import sys

import pytest
import torch

detectron2 = pytest.importorskip("detectron2")
pytest.importorskip("kornia")
from detectron2.structures import BitMasks, Boxes, Instances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cups.augmentation import CopyPasteAugmentation, RandomCrop, ResolutionJitter


def _boxes_from_masks(masks):
    boxes = []
    for mask in masks:
        ys, xs = torch.where(mask)
        boxes.append(torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32))
    return torch.stack(boxes, dim=0)


def _sample_with_sam_fields():
    masks = torch.zeros(2, 80, 160, dtype=torch.bool)
    masks[0, 8:72, 16:144] = True
    masks[1, 30:58, 60:118] = True
    instances = Instances(
        image_size=(80, 160),
        gt_masks=BitMasks(masks.clone()),
        gt_boxes=Boxes(_boxes_from_masks(masks)),
        gt_classes=torch.tensor([3, 7], dtype=torch.long),
    )
    return {
        "image": torch.arange(3 * 80 * 160, dtype=torch.float32).reshape(3, 80, 160),
        "sem_seg": torch.zeros(80, 160, dtype=torch.long),
        "instances": instances,
        "sam_masks": masks.clone(),
        "sam_ious": torch.tensor([0.9, 0.7], dtype=torch.float32),
        "sam_cls": torch.tensor([0, 1], dtype=torch.long),
        "sam_teacher_logits": torch.randn(5, 80, 160),
        "file_name": "/tmp/aachen/aachen_000000_000019_leftImg8bit.png",
    }


def test_sam_masks_follow_crop_and_resolution_jitter():
    random.seed(13)
    torch.manual_seed(13)
    sample = _sample_with_sam_fields()

    cropped = RandomCrop(resolution_min=32, resolution_max=32, long_side_scale=2.0)([sample])[0]
    resized = ResolutionJitter(scales=None, resolutions=((40, 80),))([cropped])[0]

    assert resized["image"].shape[-2:] == (40, 80)
    assert resized["sam_teacher_logits"].shape[-2:] == resized["image"].shape[-2:]
    assert resized["sam_masks"].shape[-2:] == resized["image"].shape[-2:]
    assert resized["sam_masks"].shape[0] == resized["instances"].gt_masks.tensor.shape[0]
    assert torch.equal(resized["sam_masks"], resized["instances"].gt_masks.tensor)
    assert resized["sam_ious"].shape[0] == resized["sam_masks"].shape[0]
    assert resized["sam_cls"].shape[0] == resized["sam_masks"].shape[0]


def test_copy_paste_subtracts_pasted_region_from_sam_masks():
    """SAM masks must not bound regions overwritten by a copy-paste.

    Pre-fix this assertion was inverted (`torch.equal(out_sam, in_sam)`),
    which codified the bug. After the bug #3 fix in CopyPasteAugmentation,
    pasted regions are subtracted from sam_masks; sam_ious and sam_cls stay
    in lockstep via _filter_sam_aux_fields. sam_teacher_logits is NOT
    erased here (bug #4 deferred).
    """
    random.seed(3)
    torch.manual_seed(3)
    sample = _sample_with_sam_fields()
    aug = CopyPasteAugmentation(
        thing_class=0,
        max_num_pasted_objects=1,
        scale_range=(1.0, 1.0),
        use_random_horizontal_flipping=False,
        min_bounding_box_size=(1, 1),
    )

    out = aug([sample], [sample])[0]

    assert "sam_masks" in out
    assert "sam_ious" in out
    assert "sam_cls" in out
    assert "sam_teacher_logits" in out  # bug #4 deferred

    # The freshly pasted instance is the last row of gt_masks. SAM masks
    # must not overlap it.
    pasted = out["instances"].gt_masks.tensor[-1]
    if out["sam_masks"].shape[0] > 0:
        overlap = (out["sam_masks"] & pasted[None]).any()
        assert not overlap, "SAM masks must not overlap freshly pasted regions"

    # Aux fields stay in lockstep with the (possibly pruned) mask tensor.
    assert out["sam_ious"].shape[0] == out["sam_masks"].shape[0]
    assert out["sam_cls"].shape[0] == out["sam_masks"].shape[0]
