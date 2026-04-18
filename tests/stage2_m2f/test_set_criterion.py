from __future__ import annotations

import torch

from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.set_criterion import SetCriterion


def test_set_criterion_returns_expected_keys(tiny_gt_panoptic: list[dict]) -> None:
    torch.manual_seed(0)
    B, Q, K, H, W = 2, 4, 20, 64, 128
    outputs = {
        "pred_logits": torch.randn(B, Q, K, requires_grad=True),
        "pred_masks": torch.randn(B, Q, H, W, requires_grad=True),
        "aux_outputs": [
            {"pred_logits": torch.randn(B, Q, K, requires_grad=True),
             "pred_masks": torch.randn(B, Q, H, W, requires_grad=True)}
            for _ in range(2)
        ],
    }
    matcher = HungarianMatcher(num_points=128)
    crit = SetCriterion(
        num_classes=K - 1,  # K=20 channels == num_classes + 1 (last = no-object)
        matcher=matcher,
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
        eos_coef=0.1,
        losses=("labels", "masks"),
        num_points=128,
    )
    losses = crit(outputs, tiny_gt_panoptic)
    # Main + 2 aux decoders: at least loss_ce, loss_mask, loss_dice + _0, _1 suffixes.
    assert "loss_ce" in losses and "loss_mask" in losses and "loss_dice" in losses
    assert "loss_ce_0" in losses and "loss_ce_1" in losses
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"
        assert v.requires_grad, f"{k} has no grad"
