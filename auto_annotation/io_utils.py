#!/usr/bin/env python3
"""Video decoding, keyframe sampling, and label-map I/O.

Uses tf-free OpenCV for video and PIL/numpy for masks. Panoptic maps are saved as
both a raw int32 .npy (lossless) and a colorized .png (for eyeballing).
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from .schemas import PanopticResult
from .taxonomy import CLASSES, VOID_ID

logger = logging.getLogger(__name__)

__all__ = [
    "extract_keyframes",
    "save_semantic_png",
    "save_instance_png",
    "save_panoptic",
    "colorize_panoptic",
]


def extract_keyframes(video_path: Path, frames_dir: Path, stride: int,
                      max_frames: int = 0) -> List[Path]:
    """Decode a video and write every `stride`-th frame as a PNG.

    Returns the list of written frame paths (sorted). Raises FileNotFoundError if
    the video cannot be opened.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV could not open: {video_path}")

    written: List[Path] = []
    idx, kept = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                out = frames_dir / f"frame_{idx:08d}.png"
                cv2.imwrite(str(out), frame)  # BGR on disk; readers must convert
                written.append(out)
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    logger.info("Extracted %d keyframes (stride=%d) from %s", len(written), stride,
                video_path.name)
    return sorted(written)


def save_semantic_png(label_map: np.ndarray, path: Path) -> None:
    """Save a uint8 semantic label map (class ids), losslessly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_map.astype(np.uint8)).save(path)


def save_instance_png(pan: PanopticResult, path: Path) -> None:
    """Save a uint16 instance-id map (0 = stuff/void) for quick inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    inst = (pan.pan_map % pan.label_divisor).astype(np.uint16)
    Image.fromarray(inst).save(path)


def save_panoptic(pan: PanopticResult, stem: str, out_dir: Path) -> None:
    """Persist the panoptic map (.npy lossless + colorized .png)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}_panoptic.npy", pan.pan_map.astype(np.int32))
    color = colorize_panoptic(pan)
    Image.fromarray(color).save(out_dir / f"{stem}_panoptic_color.png")


def colorize_panoptic(pan: PanopticResult) -> np.ndarray:
    """Render a panoptic map to RGB using taxonomy colors (instances dithered)."""
    h, w = pan.pan_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    class_map = (pan.pan_map // pan.label_divisor).astype(np.int64)
    inst_map = (pan.pan_map % pan.label_divisor).astype(np.int64)
    for cid, entry in CLASSES.items():
        sel = class_map == cid
        if not sel.any():
            continue
        base = np.array(entry.color, dtype=np.int16)
        # vary brightness slightly per instance so adjacent things differ
        jitter = ((inst_map[sel] * 37) % 60 - 30).astype(np.int16)
        out[sel] = np.clip(base[None, :] + jitter[:, None], 0, 255).astype(np.uint8)
    out[class_map == VOID_ID] = 0
    return out
