from __future__ import annotations

import torch

from cups.model.modeling.mask2former.swa import average_state_dicts


def test_swa_averages_three_state_dicts() -> None:
    sd1 = {"w": torch.tensor([1.0, 2.0])}
    sd2 = {"w": torch.tensor([3.0, 4.0])}
    sd3 = {"w": torch.tensor([5.0, 6.0])}
    avg = average_state_dicts([sd1, sd2, sd3])
    assert torch.allclose(avg["w"], torch.tensor([3.0, 4.0]))
