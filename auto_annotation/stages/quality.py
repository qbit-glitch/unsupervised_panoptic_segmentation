#!/usr/bin/env python3
"""Label-quality scoring + human-review routing.

Implements a model-agnostic, cleanlab-style image score (Lad & Mueller, ICML 2023,
arXiv:2307.05080): a soft-minimum (low quantile) of the per-pixel confidence of the
ASSIGNED label flags frames likely to be mislabeled. Complemented by a U2PL-style
(CVPR 2022) low-confidence pixel fraction. The split decides routing:
  - val / test -> always human-verify (these become trusted GT)
  - train      -> auto-accept unless quality is low -> route to human review
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import PipelineConfig
from ..schemas import SemanticResult

logger = logging.getLogger(__name__)

__all__ = ["FrameQuality", "score_frame", "needs_review"]


@dataclass(frozen=True)
class FrameQuality:
    image_score: float          # soft-min confidence (higher = cleaner)
    low_conf_frac: float        # fraction of pixels below 0.5 confidence
    route_to_human: bool
    reason: str


def score_frame(semantic: SemanticResult, cfg: PipelineConfig) -> FrameQuality:
    conf: Optional[np.ndarray] = semantic.confidence
    if conf is None:
        # no confidence available -> conservatively route everything
        return FrameQuality(0.0, 1.0, True, "no-confidence-from-backend")

    flat = conf.reshape(-1).astype(np.float32)
    image_score = float(np.quantile(flat, cfg.low_conf_quantile))  # soft-min proxy
    low_conf_frac = float((flat < 0.5).mean())
    route, reason = needs_review(image_score, low_conf_frac, cfg)
    return FrameQuality(image_score, low_conf_frac, route, reason)


def needs_review(image_score: float, low_conf_frac: float,
                 cfg: PipelineConfig) -> tuple:
    """Return (route_to_human, reason)."""
    if cfg.split in ("val", "test"):
        return True, "val/test-always-verified"
    if image_score < cfg.route_threshold:
        return True, f"low-quality({image_score:.2f}<{cfg.route_threshold})"
    if low_conf_frac > cfg.low_conf_pixel_frac_thr:
        return True, f"too-many-uncertain-px({low_conf_frac:.2f})"
    return False, "auto-accept"
