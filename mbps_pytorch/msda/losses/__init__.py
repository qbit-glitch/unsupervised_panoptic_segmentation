"""MSDA loss function registry."""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_loss(name: str):
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def create_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    import inspect
    cls = LOSS_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)


from .depth_contrastive import DepthContrastiveLoss  # noqa: E402, F401
from .stego import StegoCorrespondenceLoss  # noqa: E402, F401
from .swav_sinkhorn import SwAVSinkhornLoss  # noqa: E402, F401
from .hybrid import HybridLoss  # noqa: E402, F401

__all__ = [
    "LOSS_REGISTRY",
    "DepthContrastiveLoss",
    "HybridLoss",
    "StegoCorrespondenceLoss",
    "SwAVSinkhornLoss",
    "create_loss",
    "register_loss",
]
