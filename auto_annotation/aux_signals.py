#!/usr/bin/env python3
"""Depth + cross-cue agreement helpers (importable by notebook and scripts).

Depth-Anything-V2 monocular depth + an edge-agreement QA map that flags where the
panoptic likely errs (depth discontinuity with no panoptic boundary = MISSED split).
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ["DepthRunner", "edges", "depth_edges", "agreement_rgb"]


class DepthRunner:
    def __init__(self, device: str = "cpu",
                 model: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=model, device=device)
        logger.info("Depth-Anything-V2 ready")

    def predict(self, pil: Image.Image) -> np.ndarray:
        d = np.array(self.pipe(pil)["depth"], dtype=np.float32)
        return (d - d.min()) / (np.ptp(d) + 1e-6)   # 0..1 (near..far)


def edges(label_map: np.ndarray, k: int = 3) -> np.ndarray:
    """Boundary pixels of a label map (max != min in a k×k window)."""
    import cv2
    lm = label_map.astype(np.float32)
    mx = cv2.dilate(lm, np.ones((k, k), np.uint8))
    mn = -cv2.dilate(-lm, np.ones((k, k), np.uint8))
    return mx != mn


def depth_edges(depth: np.ndarray, thr: float = 0.06) -> np.ndarray:
    import cv2
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy) > thr


def agreement_rgb(pan_edge: np.ndarray, dep_edge: np.ndarray, shape: tuple) -> np.ndarray:
    """green=depth-supported boundary, yellow=weak, RED=missed split (review)."""
    import cv2
    de = cv2.dilate(dep_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pe = cv2.dilate(pan_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = np.zeros((*shape, 3), np.uint8)
    out[pe & de] = (0, 220, 0)
    out[pe & ~de] = (230, 230, 0)
    out[de & ~pe] = (255, 0, 0)
    return out
