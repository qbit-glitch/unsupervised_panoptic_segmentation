"""MSDA architecture registry."""

from __future__ import annotations

from typing import Dict, Type

from .base import AdapterConfig, BaseAdapter

ARCH_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_arch(name: str):
    def decorator(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
        ARCH_REGISTRY[name] = cls
        return cls
    return decorator


def create_adapter(name: str, cfg: AdapterConfig) -> BaseAdapter:
    if name not in ARCH_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{name}'. Available: {list(ARCH_REGISTRY.keys())}"
        )
    return ARCH_REGISTRY[name](cfg)


from .conv_adapter import ConvAdapter  # noqa: E402, F401
from .transformer_pyramid import TransformerPyramid  # noqa: E402, F401
from .slot_adapter import SlotAdapter  # noqa: E402, F401
from .conv_transformer import ConvTransformerAdapter  # noqa: E402, F401

__all__ = [
    "ARCH_REGISTRY",
    "AdapterConfig",
    "BaseAdapter",
    "ConvAdapter",
    "TransformerPyramid",
    "SlotAdapter",
    "ConvTransformerAdapter",
    "create_adapter",
    "register_arch",
]
