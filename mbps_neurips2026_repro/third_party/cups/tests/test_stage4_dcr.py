import os
import sys

import torch
from yacs.config import CfgNode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cups.losses.long_tail import EQLv2Loss, SeesawSoftmaxLoss
from cups.stage4_utils import resolve_stage4_class_ids


def _stage4_cfg():
    cfg = CfgNode()
    cfg.STAGE4 = CfgNode()
    cfg.STAGE4.ENABLED = True
    cfg.STAGE4.RARE_CLASSES = ("guard rail", "tunnel", "polegroup", "caravan", "trailer")
    cfg.STAGE4.RARE_STUFF_PSEUDO_CLASSES = ()
    cfg.STAGE4.RARE_THING_PSEUDO_CLASSES = ()
    return cfg


def test_stage4_resolves_cityscapes_27_ids():
    cfg = _stage4_cfg()
    ids = resolve_stage4_class_ids(
        cfg,
        thing_pseudo_classes=(17, 18, 19, 20, 21, 22, 23, 24, 25, 26),
        stuff_pseudo_classes=tuple(range(17)),
    )

    assert ids.rare_stuff_targets == (8, 10, 12)
    assert ids.rare_thing_targets == (5, 6)
    assert ids.unresolved == ()


def test_stage4_explicit_ids_support_overclustered_runs():
    cfg = _stage4_cfg()
    cfg.STAGE4.RARE_CLASSES = ()
    cfg.STAGE4.RARE_STUFF_PSEUDO_CLASSES = (3, 11)
    cfg.STAGE4.RARE_THING_PSEUDO_CLASSES = (2,)

    ids = resolve_stage4_class_ids(cfg, thing_pseudo_classes=(65, 66), stuff_pseudo_classes=(0, 1))

    assert ids.rare_stuff_targets == (3, 11)
    assert ids.rare_thing_targets == (2,)


def test_eqlv2_loss_is_finite_with_empty_and_background_samples():
    loss_fn = EQLv2Loss(num_classes=3)
    empty = loss_fn(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))
    assert torch.isfinite(empty)

    scores = torch.randn(6, 4, requires_grad=True)
    gt = torch.tensor([0, 1, 3, 3, 2, 3])
    loss_vec = loss_fn(scores, gt, reduction="none")
    assert loss_vec.shape == (6,)
    loss = loss_vec.mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad is not None


def test_seesaw_loss_is_finite_with_background_samples():
    loss_fn = SeesawSoftmaxLoss(num_classes=3)
    scores = torch.randn(5, 4, requires_grad=True)
    gt = torch.tensor([0, 3, 1, 3, 2])
    loss = loss_fn(scores, gt)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad is not None
