from __future__ import annotations

import torch

from cups.model.modeling.mask2former.matcher import HungarianMatcher


def test_matcher_exact_assignment_on_identity() -> None:
    """When Q=N_gt=3 and predictions match gt exactly, indices should be identity."""
    torch.manual_seed(0)
    B, Q, K, H, W = 1, 3, 20, 16, 32
    pred_logits = torch.full((B, Q, K), -10.0)
    pred_masks = torch.full((B, Q, H, W), -10.0)
    gt_labels = torch.tensor([2, 5, 7], dtype=torch.long)
    gt_masks = torch.zeros(3, H, W, dtype=torch.bool)
    for i, c in enumerate(gt_labels.tolist()):
        pred_logits[0, i, c] = 10.0
        pred_masks[0, i] = 10.0
        gt_masks[i, 4 * i : 4 * i + 5, 4 * i : 4 * i + 5] = True
        pred_masks[0, i] = (gt_masks[i].float() * 20.0 - 10.0)

    matcher = HungarianMatcher(
        cost_class=1.0, cost_mask=5.0, cost_dice=5.0, num_points=256
    )
    targets = [{"labels": gt_labels, "masks": gt_masks}]
    outputs = {"pred_logits": pred_logits, "pred_masks": pred_masks}
    indices = matcher(outputs, targets)
    src, tgt = indices[0]
    # Identity assignment (src_i == tgt_i) up to permutation.
    assert sorted(src.tolist()) == [0, 1, 2]
    assert sorted(tgt.tolist()) == [0, 1, 2]
    # Each src_i must map to the tgt_i with the same label.
    for s, t in zip(src.tolist(), tgt.tolist()):
        assert s == t
