#!/usr/bin/env python3
"""Shared data contracts for the auto-annotation pipeline.

These lightweight containers are the *only* thing the stages agree on, so a real
INSID3 / SAM3 backend and the dummy backend are interchangeable. Arrays follow
numpy HxW conventions; instance masks are boolean HxW.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = [
    "SemanticResult",
    "InstanceMask",
    "InstanceResult",
    "PanopticResult",
]


@dataclass
class SemanticResult:
    """Dense per-pixel semantic prediction.

    Attributes:
        label_map: int32 HxW, values are taxonomy class ids (stuff or thing).
        confidence: float32 HxW in [0, 1], per-pixel max class prob/similarity.
            Used by the QA stage. None if the backend cannot provide it.
        margin: float32 HxW, top1-minus-top2 score per pixel (optional).
    """

    label_map: np.ndarray
    confidence: Optional[np.ndarray] = None
    margin: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple:
        return self.label_map.shape


@dataclass
class InstanceMask:
    """One thing-instance mask."""

    mask: np.ndarray  # bool HxW
    class_id: int
    score: float
    track_id: Optional[int] = None  # stable id across video frames


@dataclass
class InstanceResult:
    """All thing-instances in a single frame."""

    instances: List[InstanceMask] = field(default_factory=list)

    def filter_by_score(self, thr: float) -> "InstanceResult":
        return InstanceResult([i for i in self.instances if i.score >= thr])


@dataclass
class PanopticResult:
    """COCO-style panoptic map + segment metadata.

    pan_map encodes panoptic_id = class_id * label_divisor + instance_id.
    Stuff segments use instance_id = 0.
    """

    pan_map: np.ndarray  # int32 HxW
    segments_info: List[dict]
    label_divisor: int = 1000
