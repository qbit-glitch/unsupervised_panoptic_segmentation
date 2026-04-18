"""Tests for Mask2FormerPanoptic._collect_targets ID-space remapping.

Dataset convention (pseudo_label_dataset.py:430-491):
  - sem_seg value 0      -> thing region marker (when IGNORE_UNKNOWN_THING_REGIONS=False)
  - sem_seg value [1..S] -> stuff class (1-indexed, maps to [0..S) in combined space)
  - sem_seg value 255    -> void / ignore
  - instances.gt_classes -> thing class in [0..T) (0-indexed, maps to [S..S+T) in combined)

Combined class space expected by SetCriterion(num_classes=S+T):
  - [0, S)           -> stuff classes
  - [S, S+T)         -> thing classes
  - S+T              -> "no-object" phi (handled internally, not in targets)

_panoptic_merge line 133 confirms this: `isthing = cls >= num_stuff_classes`.
"""
from __future__ import annotations

import torch
from detectron2.structures import BitMasks, Instances

from cups.model.modeling.meta_arch.mask2former_panoptic import Mask2FormerPanoptic


class _StubMetaArch(Mask2FormerPanoptic):
    """Bypass __init__ so we can exercise _collect_targets without a full model."""

    def __init__(self, num_stuff: int, num_thing: int) -> None:  # noqa: D401
        torch.nn.Module.__init__(self)
        self.num_stuff_classes = num_stuff
        self.num_thing_classes = num_thing
        self.num_classes = num_stuff + num_thing
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device


def _make_instances(gt_classes: list[int], masks: torch.Tensor, H: int, W: int) -> Instances:
    inst = Instances((H, W))
    inst.gt_classes = torch.tensor(gt_classes, dtype=torch.long)
    inst.gt_masks = BitMasks(masks)
    return inst


def test_thing_labels_offset_by_num_stuff() -> None:
    """Bug A: gt_classes=[0,2] must become [num_stuff+0, num_stuff+2]."""
    S, T = 12, 8
    model = _StubMetaArch(num_stuff=S, num_thing=T)

    H, W = 32, 64
    masks = torch.zeros(2, H, W, dtype=torch.bool)
    masks[0, :16, :32] = True
    masks[1, 16:, 32:] = True
    inst = _make_instances([0, 2], masks, H, W)

    batch = [{"image": torch.zeros(3, H, W), "instances": inst}]
    targets = model._collect_targets(batch)
    labels = targets[0]["labels"].tolist()
    assert S + 0 in labels, f"thing gt_class 0 must map to {S}, got labels={labels}"
    assert S + 2 in labels, f"thing gt_class 2 must map to {S + 2}, got labels={labels}"
    assert 0 not in labels, f"raw gt_class 0 leaked (collides with stuff class 0), labels={labels}"


def test_stuff_labels_decremented() -> None:
    """Bug B: sem_seg value 1..S must become 0..S-1 (1-indexed -> 0-indexed)."""
    S, T = 12, 8
    model = _StubMetaArch(num_stuff=S, num_thing=T)

    H, W = 32, 64
    sem = torch.full((H, W), 255, dtype=torch.long)
    sem[:16, :32] = 1    # stuff class 0 (1-indexed as 1)
    sem[16:, 32:] = 5    # stuff class 4 (1-indexed as 5)

    batch = [{"image": torch.zeros(3, H, W), "sem_seg": sem}]
    targets = model._collect_targets(batch)
    labels = targets[0]["labels"].tolist()
    assert 0 in labels, f"sem_seg=1 must map to stuff class 0, got labels={labels}"
    assert 4 in labels, f"sem_seg=5 must map to stuff class 4, got labels={labels}"
    assert 1 not in labels, f"sem_seg=1 leaked as label 1, got labels={labels}"
    assert 5 not in labels, f"sem_seg=5 leaked as label 5, got labels={labels}"


def test_sem_seg_zero_skipped() -> None:
    """Bug C: sem_seg==0 is the thing-region marker (when IGNORE_UNKNOWN_THING_REGIONS=False).

    It must NOT produce a target with label=0 (which would collide with stuff class 0
    AND double-count thing regions that already live in 'instances').
    """
    S, T = 12, 8
    model = _StubMetaArch(num_stuff=S, num_thing=T)

    H, W = 32, 64
    sem = torch.zeros((H, W), dtype=torch.long)   # all thing-region
    sem[:8, :8] = 3   # one stuff patch (class 2 in 0-indexed)

    batch = [{"image": torch.zeros(3, H, W), "sem_seg": sem}]
    targets = model._collect_targets(batch)
    labels = targets[0]["labels"].tolist()
    assert 2 in labels, f"sem_seg=3 should remap to stuff class 2, got labels={labels}"
    # The only entry should be the stuff class 2; no stuff-class-0 from sem_seg==0.
    assert len(labels) == 1, f"sem_seg==0 should be skipped, got labels={labels}"


def test_combined_instances_and_sem_seg() -> None:
    """Full batch sample: both `instances` (things) and `sem_seg` (stuff) present.

    Verifies the combined-space target layout end-to-end:
      - thing gt_classes [0, 3] -> [S, S+3]
      - stuff sem_seg values {1, 7} -> {0, 6}
      - sem_seg==0 thing regions skipped (no label-0 collision)
    """
    S, T = 12, 8
    model = _StubMetaArch(num_stuff=S, num_thing=T)

    H, W = 32, 64

    thing_masks = torch.zeros(2, H, W, dtype=torch.bool)
    thing_masks[0, :8, :8] = True
    thing_masks[1, 8:16, 8:16] = True
    inst = _make_instances([0, 3], thing_masks, H, W)

    sem = torch.zeros((H, W), dtype=torch.long)   # default: thing region
    sem[:8, :8] = 0        # thing region (already in instances)
    sem[16:20, :] = 1      # stuff class 0
    sem[:, 32:36] = 7      # stuff class 6
    sem[24:, 48:] = 255    # void

    batch = [{"image": torch.zeros(3, H, W), "instances": inst, "sem_seg": sem}]
    targets = model._collect_targets(batch)
    labels = sorted(targets[0]["labels"].tolist())
    expected = sorted([S + 0, S + 3, 0, 6])
    assert labels == expected, f"expected {expected}, got {labels}"

    # All labels must be in the valid range [0, num_stuff + num_thing).
    assert all(0 <= lab < S + T for lab in labels), f"label out of range: {labels}"


def test_void_labels_skipped() -> None:
    """255 and -1 remain ignored after the remap."""
    S, T = 12, 8
    model = _StubMetaArch(num_stuff=S, num_thing=T)

    H, W = 32, 64
    sem = torch.full((H, W), 255, dtype=torch.long)
    sem[:4, :4] = -1

    batch = [{"image": torch.zeros(3, H, W), "sem_seg": sem}]
    targets = model._collect_targets(batch)
    assert targets[0]["labels"].numel() == 0, \
        f"only void sem_seg should yield empty targets, got {targets[0]['labels'].tolist()}"
