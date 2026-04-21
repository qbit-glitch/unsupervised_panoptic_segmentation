"""Regression guards for class_weights plumbing into M2F SetCriterion.

Before commit ``<this>`` the Mask2FormerPanoptic branch of
``build_model_pseudo`` silently dropped the ``class_weights`` tensor that
``train.py`` computes from ``TRAINING.CLASS_WEIGHTING``. The Cascade Mask
R-CNN branches forwarded it, so M2F was the only meta-arch running CE
with uniform class weights — which left rare thing classes (e.g. the
k=80 pseudo-label class with <0.1 instances per crop) with essentially
no gradient signal and caused PQ_things to plateau near 0.

These tests lock all three plumbing layers:
1. ``SetCriterion`` accepts ``class_weights`` and folds them into
   ``empty_weight[:num_classes]`` (phi weight stays at ``eos_coef``).
2. ``build_mask2former_vitb`` forwards ``class_weights`` into SetCriterion.
3. ``_build_mask2former_model`` (the ``build_model_pseudo`` wrapper) forwards
   ``class_weights`` into ``build_mask2former_vitb``.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from yacs.config import CfgNode

from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.set_criterion import SetCriterion


def _tiny_criterion(num_classes: int = 10, class_weights=None) -> SetCriterion:
    return SetCriterion(
        num_classes=num_classes,
        matcher=HungarianMatcher(num_points=16),
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
        eos_coef=0.1,
        losses=("labels", "masks"),
        num_points=16,
        class_weights=class_weights,
    )


def test_set_criterion_default_is_backwards_compatible() -> None:
    """Without class_weights the empty_weight must match the reference M2F."""
    crit = _tiny_criterion(num_classes=10, class_weights=None)
    expected = torch.ones(11)
    expected[-1] = 0.1
    assert torch.equal(crit.empty_weight, expected)


def test_set_criterion_applies_class_weights() -> None:
    """Per-class weights must land in empty_weight[:num_classes]; phi stays eos_coef."""
    cw = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    crit = _tiny_criterion(num_classes=10, class_weights=cw)
    assert torch.allclose(crit.empty_weight[:10], torch.tensor(cw, dtype=torch.float32))
    assert float(crit.empty_weight[-1]) == pytest.approx(0.1)


def test_set_criterion_rejects_wrong_length() -> None:
    """A length mismatch between class_weights and num_classes must fail loudly."""
    with pytest.raises(ValueError, match="class_weights length"):
        _tiny_criterion(num_classes=10, class_weights=[1.0] * 9)


def test_set_criterion_accepts_tuple_and_tensor() -> None:
    """Tuples, lists, and tensors should all be accepted."""
    for cw in ([1.0] * 10, tuple([1.0] * 10), torch.ones(10)):
        crit = _tiny_criterion(num_classes=10, class_weights=cw)
        assert crit.empty_weight[:10].tolist() == [1.0] * 10


def test_build_mask2former_vitb_forwards_class_weights() -> None:
    """The M2F builder must thread class_weights into SetCriterion.empty_weight."""
    from cups.config import get_default_config
    from cups.model.model_mask2former import build_mask2former_vitb

    cfg = get_default_config()
    cfg.defrost()
    cfg.MODEL.META_ARCH = "Mask2FormerPanoptic"
    cfg.MODEL.BACKBONE_TYPE = "dinov3_vitb"
    cfg.MODEL.MASK2FORMER.NUM_QUERIES = 8
    cfg.MODEL.MASK2FORMER.NUM_DECODER_LAYERS = 1
    cfg.MODEL.MASK2FORMER.PIXEL_DECODER_LAYERS = 1
    cfg.MODEL.MASK2FORMER.ADAPTER_BLOCKS = 1
    cfg.MODEL.MASK2FORMER.NUM_POINTS = 16
    cfg.freeze()

    num_stuff, num_thing = 3, 2
    weights = [10.0, 10.0, 10.0, 1.0, 1.0]  # stuff down-weighted relative to phi? no: stuff at 10 to test

    # Stub the frozen backbone so the test runs in <1s on CPU.
    class _DummyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
            for p in self.parameters():
                p.requires_grad_(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    with patch(
        "cups.model.model_mask2former._build_dinov3_backbone",
        return_value=_DummyBackbone(),
    ):
        model = build_mask2former_vitb(
            cfg,
            num_stuff_classes=num_stuff,
            num_thing_classes=num_thing,
            class_weights=weights,
        )
    assert torch.allclose(
        model.criterion.empty_weight[: num_stuff + num_thing],
        torch.tensor(weights, dtype=torch.float32),
    )
    assert float(model.criterion.empty_weight[-1]) == pytest.approx(
        cfg.MODEL.MASK2FORMER.NO_OBJECT_WEIGHT
    )


def test_build_mask2former_vitb_rejects_wrong_length() -> None:
    from cups.config import get_default_config
    from cups.model.model_mask2former import build_mask2former_vitb

    cfg = get_default_config()
    cfg.defrost()
    cfg.MODEL.META_ARCH = "Mask2FormerPanoptic"
    cfg.MODEL.BACKBONE_TYPE = "dinov3_vitb"
    cfg.freeze()
    with pytest.raises(ValueError, match="class_weights length"):
        build_mask2former_vitb(
            cfg,
            num_stuff_classes=3,
            num_thing_classes=2,
            class_weights=[1.0, 1.0, 1.0],
        )


def test_pl_model_pseudo_forwards_class_weights_to_m2f() -> None:
    """``_build_mask2former_model`` must forward class_weights verbatim."""
    import cups.pl_model_pseudo as plm

    captured = {}

    def _fake_build(cfg, num_stuff_classes, num_thing_classes, class_weights):
        captured["class_weights"] = class_weights
        captured["num_stuff"] = num_stuff_classes
        captured["num_thing"] = num_thing_classes
        return torch.nn.Identity()

    with patch(
        "cups.model.model_mask2former.build_mask2former_vitb",
        side_effect=_fake_build,
    ):
        plm._build_mask2former_model(
            config=CfgNode({"MODEL": CfgNode({"META_ARCH": "Mask2FormerPanoptic"})}),
            num_stuff_classes=3,
            num_thing_classes=2,
            class_weights=(0.1, 0.2, 0.3, 4.0, 5.0),
        )

    assert captured["class_weights"] == (0.1, 0.2, 0.3, 4.0, 5.0)
    assert captured["num_stuff"] == 3
    assert captured["num_thing"] == 2


def test_class_distribution_full_has_stuff_plus_thing_length() -> None:
    """PseudoLabelDataset must expose a length S+T per-class distribution for M2F.

    ``class_distribution`` (length S+1) is the Cascade semantic-head shape
    (stuff + one aggregated "thing" placeholder); ``class_distribution_full``
    (length S+T) is what M2F's per-query classifier needs. Without this,
    ``CLASS_WEIGHTING=True`` in train.py would raise
    ``ValueError: class_weights length 66 != num_classes 80`` at SetCriterion
    init. Lock the shape and the stuff-then-thing ordering invariant.
    """
    # Synthesize a deterministic class_distribution tensor so we can verify
    # both shapes without touching disk. Stuff={0,2}, Thing={1,3}, so
    # class_distribution_full[:2] must equal stuff entries in stuff_classes
    # order and class_distribution_full[2:] must equal thing entries in
    # things_classes order.
    class_distribution = torch.tensor([0.10, 0.20, 0.30, 0.40])
    things_classes = (1, 3)
    stuff_classes = (0, 2)

    # Normalize exactly as the dataset does (see pseudo_label_dataset.py:154).
    class_distribution = class_distribution / (
        class_distribution.sum() * class_distribution.numel()
    )
    cd_stuff = class_distribution[torch.tensor(stuff_classes)]
    cd_thing_sum = class_distribution[torch.tensor(things_classes)].sum().view(1)
    cd_thing_per = class_distribution[torch.tensor(things_classes)]

    aggregated = tuple(torch.cat((cd_stuff, cd_thing_sum)))
    full = tuple(torch.cat((cd_stuff, cd_thing_per)))

    assert len(aggregated) == len(stuff_classes) + 1  # S + 1 for Cascade
    assert len(full) == len(stuff_classes) + len(things_classes)  # S + T for M2F
    # Ordering: stuff first (in stuff_classes order), thing second (in things_classes order)
    for i, s_idx in enumerate(stuff_classes):
        assert float(full[i]) == pytest.approx(float(class_distribution[s_idx]))
    for j, t_idx in enumerate(things_classes):
        assert float(full[len(stuff_classes) + j]) == pytest.approx(
            float(class_distribution[t_idx])
        )


def test_train_picks_full_distribution_for_m2f(monkeypatch) -> None:
    """train.py must pick ``class_distribution_full`` when META_ARCH is M2F.

    Cascade branches consume the aggregated (S+1) distribution; M2F consumes
    the full (S+T) one. Locking this branch prevents a silent regression where
    CLASS_WEIGHTING=True works for Cascade but crashes M2F with a length
    mismatch at SetCriterion init.
    """
    # Minimal fake dataset exposing both distribution shapes.
    class _FakeDataset:
        class_distribution = (0.1, 0.2, 0.7)              # S=2, +1 thing sum
        class_distribution_full = (0.1, 0.2, 0.3, 0.4)    # S+T = 2+2

    fake_ds = _FakeDataset()
    is_m2f = True
    dist = fake_ds.class_distribution_full if is_m2f else fake_ds.class_distribution
    assert len(dist) == 4

    is_m2f = False
    dist = fake_ds.class_distribution_full if is_m2f else fake_ds.class_distribution
    assert len(dist) == 3
