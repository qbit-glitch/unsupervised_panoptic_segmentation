"""Stochastic Weight Averaging utility (G2): mean over a list of state dicts."""
from __future__ import annotations

from typing import Dict, List

import torch

__all__ = ["average_state_dicts"]


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    assert state_dicts, "average_state_dicts requires at least one dict"
    out: Dict[str, torch.Tensor] = {}
    keys = state_dicts[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        out[k] = stacked.mean(0).to(state_dicts[0][k].dtype)
    return out
