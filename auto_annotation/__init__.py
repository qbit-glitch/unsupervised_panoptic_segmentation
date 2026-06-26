#!/usr/bin/env python3
"""auto_annotation: foundation-model auto-labeling for monocular driving video.

Pipeline (see README.md): video -> keyframes -> INSID3 semantics + SAM3 instances
-> SAM-snap boundaries -> panoptic merge -> QA routing (auto train / human val).
"""

from .config import PipelineConfig
from .pipeline import run
from .schemas import InstanceMask, InstanceResult, PanopticResult, SemanticResult

__all__ = [
    "PipelineConfig",
    "run",
    "SemanticResult",
    "InstanceResult",
    "InstanceMask",
    "PanopticResult",
]
