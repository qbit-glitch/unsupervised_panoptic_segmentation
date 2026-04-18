from __future__ import annotations

import torch

from cups.losses.self_train import filter_by_confidence


def test_filter_keeps_high_confidence_only() -> None:
    labels = torch.tensor([2, 5, 7, 3], dtype=torch.long)
    scores = torch.tensor([0.99, 0.50, 0.97, 0.30], dtype=torch.float64)
    masks = torch.zeros(4, 8, 8, dtype=torch.bool)
    masks[0, 0:3, 0:3] = True
    masks[1, 3:6, 3:6] = True
    masks[2, 5:8, 5:8] = True
    masks[3, 0:2, 6:8] = True
    kept = filter_by_confidence({"labels": labels, "masks": masks, "scores": scores}, threshold=0.95)
    assert kept["labels"].tolist() == [2, 7]
    assert kept["masks"].shape == (2, 8, 8)
    assert kept["scores"].tolist() == [0.99, 0.97]


def test_filter_empty_when_all_below_threshold() -> None:
    labels = torch.tensor([0], dtype=torch.long)
    scores = torch.tensor([0.1])
    masks = torch.ones(1, 4, 4, dtype=torch.bool)
    kept = filter_by_confidence({"labels": labels, "masks": masks, "scores": scores}, threshold=0.5)
    assert kept["labels"].numel() == 0
    assert kept["masks"].shape == (0, 4, 4)
