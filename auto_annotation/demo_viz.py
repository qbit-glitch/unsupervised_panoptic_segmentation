#!/usr/bin/env python3
"""Visualization helpers for the panoptic demo (colorize + overlay + grids)."""

import colorsys
from typing import Dict, List

import numpy as np

from .demo_panoptic import IDX2COLOR, IDX2NAME, VOID

__all__ = [
    "colorize_semantic", "colorize_instances", "colorize_panoptic",
    "overlay", "legend_handles",
]


def colorize_semantic(sem: np.ndarray) -> np.ndarray:
    h, w = sem.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in IDX2COLOR.items():
        out[sem == idx] = color
    return out


def _distinct_color(i: int) -> tuple:
    """Evenly-spaced hues so adjacent instances are easy to tell apart."""
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def colorize_instances(insts: List[dict], shape: tuple) -> np.ndarray:
    out = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for i, inst in enumerate(sorted(insts, key=lambda x: -x["mask"].sum())):
        out[inst["mask"]] = _distinct_color(i)
    return out


def colorize_panoptic(pan: np.ndarray, label_divisor: int) -> np.ndarray:
    h, w = pan.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    cls = (pan // label_divisor).astype(np.int64)
    inst = (pan % label_divisor).astype(np.int64)
    for idx, color in IDX2COLOR.items():
        sel = cls == idx
        if not sel.any():
            continue
        base = np.array(color, dtype=np.int16)
        jit = ((inst[sel] * 47) % 70 - 35).astype(np.int16)  # per-instance shade
        out[sel] = np.clip(base[None, :] + jit[:, None], 0, 255).astype(np.uint8)
    return out


def overlay(img: np.ndarray, color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    m = color.any(axis=2)
    out = img.copy()
    out[m] = (alpha * color[m] + (1 - alpha) * img[m]).astype(np.uint8)
    return out


def legend_handles(present_idx: List[int]):
    """matplotlib Patch handles for the classes present (for a shared legend)."""
    from matplotlib.patches import Patch
    handles = []
    for idx in sorted(set(present_idx)):
        if idx == VOID:
            continue
        c = np.array(IDX2COLOR[idx]) / 255.0
        handles.append(Patch(facecolor=c, edgecolor="k", label=IDX2NAME[idx]))
    return handles
