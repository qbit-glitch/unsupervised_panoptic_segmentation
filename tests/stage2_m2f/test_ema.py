from __future__ import annotations

import torch

from cups.model.modeling.mask2former.ema import EMAModel


def test_ema_update_rule() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    ema = EMAModel(model, decay=0.9)
    # Snapshot initial weights.
    w0 = model.weight.detach().clone()
    # Modify model weights.
    with torch.no_grad():
        model.weight.copy_(w0 + 1.0)
    ema.update(model)
    # EMA should be 0.9 * w0 + 0.1 * (w0 + 1.0) = w0 + 0.1
    expected = w0 + 0.1
    assert torch.allclose(ema.shadow["weight"], expected, atol=1e-6)


def test_ema_load_into_model() -> None:
    model = torch.nn.Linear(4, 2)
    ema = EMAModel(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)
    ema.copy_to(model)
    # After copy_to, model weights == ema.shadow (initial random weights).
    assert not torch.all(model.weight == 0.0)
