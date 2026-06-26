#!/usr/bin/env python3
"""Stage interfaces + backend registry (factory pattern).

A backend is selected by string in PipelineConfig (semantic_backend /
instance_backend), so swapping the dummy for the real INSID3 / SAM3 wrapper is a
one-line config change. Concrete backends register themselves on import.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

import numpy as np

from ..config import PipelineConfig
from ..schemas import InstanceResult, SemanticResult

__all__ = [
    "SemanticAnnotator",
    "InstanceAnnotator",
    "register_semantic",
    "register_instance",
    "build_semantic",
    "build_instance",
]


class SemanticAnnotator(ABC):
    """Produces a dense semantic label map for one RGB frame."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def predict(self, image: np.ndarray) -> SemanticResult:
        """image: HxWx3 uint8 RGB -> SemanticResult."""
        raise NotImplementedError


class InstanceAnnotator(ABC):
    """Produces thing-instance masks for one RGB frame (and optionally tracks)."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def predict(self, image: np.ndarray) -> InstanceResult:
        raise NotImplementedError

    def track(self, images, results):
        """Optional: assign stable track_ids across a frame list in place.

        Default no-op; SAM3/SAM2 backends override with real memory propagation.
        """
        return results


_SEMANTIC: Dict[str, Type[SemanticAnnotator]] = {}
_INSTANCE: Dict[str, Type[InstanceAnnotator]] = {}


def register_semantic(name: str) -> Callable:
    def deco(cls: Type[SemanticAnnotator]) -> Type[SemanticAnnotator]:
        _SEMANTIC[name] = cls
        return cls
    return deco


def register_instance(name: str) -> Callable:
    def deco(cls: Type[InstanceAnnotator]) -> Type[InstanceAnnotator]:
        _INSTANCE[name] = cls
        return cls
    return deco


def build_semantic(cfg: PipelineConfig) -> SemanticAnnotator:
    _import_backends()
    if cfg.semantic_backend not in _SEMANTIC:
        raise KeyError(f"unknown semantic_backend '{cfg.semantic_backend}'; "
                       f"have {sorted(_SEMANTIC)}")
    return _SEMANTIC[cfg.semantic_backend](cfg)


def build_instance(cfg: PipelineConfig) -> InstanceAnnotator:
    _import_backends()
    if cfg.instance_backend not in _INSTANCE:
        raise KeyError(f"unknown instance_backend '{cfg.instance_backend}'; "
                       f"have {sorted(_INSTANCE)}")
    return _INSTANCE[cfg.instance_backend](cfg)


def _import_backends() -> None:
    """Trigger self-registration (kept lazy to avoid importing torch eagerly)."""
    from . import instances_sam3, semantic_insid3  # noqa: F401
