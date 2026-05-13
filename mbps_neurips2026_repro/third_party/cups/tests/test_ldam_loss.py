import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_LOSS_PATH = Path(__file__).resolve().parents[1] / "cups" / "losses" / "long_tail.py"
spec = importlib.util.spec_from_file_location("long_tail", _LOSS_PATH)
loss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss_module)
LDAMSemanticLoss = loss_module.LDAMSemanticLoss


def test_ldam_zero_margin_matches_scaled_cross_entropy():
    logits = torch.randn(2, 3, 2, 2)
    targets = torch.tensor([[[0, 1], [2, 255]], [[1, 2], [0, 1]]])
    ldam = LDAMSemanticLoss(num_classes=3, class_freq=(1.0, 1.0, 1.0), max_margin=0.0, s=30.0, ignore_index=255)

    expected = F.cross_entropy(30.0 * logits, targets, ignore_index=255)
    actual = ldam(logits, targets)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_ldam_assigns_larger_margin_to_rarer_class():
    ldam = LDAMSemanticLoss(num_classes=3, class_freq=(100.0, 10.0, 1.0), max_margin=0.5, s=1.0)

    assert ldam.margins[2] > ldam.margins[1] > ldam.margins[0]
    assert torch.isclose(ldam.margins.max(), torch.tensor(0.5))


def test_ldam_matches_hand_computed_three_class_example():
    logits = torch.tensor([[[[1.0]], [[0.2]], [[-0.5]]]])
    targets = torch.tensor([[[1]]])
    ldam = LDAMSemanticLoss(num_classes=3, class_freq=(16.0, 1.0, 16.0), max_margin=0.4, s=2.0)

    adjusted = logits.clone()
    adjusted[:, 1] -= ldam.margins[1]
    expected = F.cross_entropy(2.0 * adjusted, targets)
    actual = ldam(logits, targets)

    assert torch.allclose(actual, expected, atol=1e-6)
