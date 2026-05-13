"""Find color-diverse crops across all 80 rendered Ours overlay images.

For each image folder, slide a window over the overlay PNG, compute color
diversity (number of distinct colors with at least min_pct pixel coverage),
and rank candidate crops globally. Used to pick supplementary panels that
demonstrate fine-grained class transitions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/notebooks/qualitative_results_filled")
DATASETS = ["cityscapes", "kitti", "mapillary", "waymo"]
CROP = 256
STRIDE = 128
MIN_PCT = 0.05  # a "distinct color" must cover >=5% of crop pixels
TOP_K_GLOBAL = 24


@dataclass(frozen=True)
class CropCandidate:
    dataset: str
    image_id: str
    x: int
    y: int
    diversity: int
    instance_count: int
    n_colors: int


def _quantize(rgb: np.ndarray, levels: int = 6) -> np.ndarray:
    """Quantize RGB to a coarse palette so categorical overlay colors group well."""
    bin_size = 256 // levels
    q = (rgb.astype(np.int32) // bin_size).clip(0, levels - 1)
    code = q[..., 0] * (levels * levels) + q[..., 1] * levels + q[..., 2]
    return code


def _crop_score(overlay: np.ndarray, ins: np.ndarray, x: int, y: int) -> Tuple[int, int, int]:
    """Return (diversity, instance_count, n_colors) for a single crop.

    diversity = number of distinct quantized colors covering >= MIN_PCT pixels.
    instance_count = unique instance IDs (excl. 0/255) in the crop.
    n_colors = total distinct quantized colors (no min coverage).
    """
    crop_overlay = overlay[y : y + CROP, x : x + CROP]
    code = _quantize(crop_overlay)
    flat = code.ravel()
    counts = np.bincount(flat)
    total = flat.size
    n_colors = int((counts > 0).sum())
    diversity = int((counts >= total * MIN_PCT).sum())

    crop_ins = ins[y : y + CROP, x : x + CROP]
    uniq = np.unique(crop_ins)
    uniq = uniq[(uniq != 0) & (uniq != 255) & (uniq != -1)]
    instance_count = int(len(uniq))
    return diversity, instance_count, n_colors


def _scan_image(dataset: str, image_id: str) -> List[CropCandidate]:
    folder = ROOT / dataset / image_id
    overlay_path = folder / "ours_overlay.png"
    npz_path = folder / "ours_raw_panoptic.npz"
    if not overlay_path.exists() or not npz_path.exists():
        logger.warning("Skipping %s/%s: missing files", dataset, image_id)
        return []
    overlay = np.array(Image.open(overlay_path))
    if overlay.ndim == 3 and overlay.shape[-1] == 4:
        overlay = overlay[..., :3]
    H, W = overlay.shape[:2]
    if H < CROP or W < CROP:
        return []

    raw = np.load(npz_path)
    ins = raw["instance_id"].astype(np.int32)
    if ins.shape != (H, W):
        # Resize via nearest if mismatched
        ins_img = Image.fromarray(ins.astype(np.int32), mode="I")
        ins = np.array(ins_img.resize((W, H), Image.NEAREST), dtype=np.int32)

    out: List[CropCandidate] = []
    for y in range(0, H - CROP + 1, STRIDE):
        for x in range(0, W - CROP + 1, STRIDE):
            div, inst, n_cols = _crop_score(overlay, ins, x, y)
            out.append(
                CropCandidate(
                    dataset=dataset,
                    image_id=image_id,
                    x=x,
                    y=y,
                    diversity=div,
                    instance_count=inst,
                    n_colors=n_cols,
                )
            )
    return out


def main() -> None:
    all_candidates: List[CropCandidate] = []
    for ds in DATASETS:
        ds_dir = ROOT / ds
        if not ds_dir.exists():
            continue
        for image_id in sorted(os.listdir(ds_dir)):
            if not (ds_dir / image_id).is_dir():
                continue
            cands = _scan_image(ds, image_id)
            all_candidates.extend(cands)
        logger.info("Scanned %s (cumulative candidates: %d)", ds, len(all_candidates))

    # Rank: primary by diversity, secondary by instance_count, tertiary by n_colors
    all_candidates.sort(key=lambda c: (c.diversity, c.instance_count, c.n_colors), reverse=True)

    # De-duplicate: at most 2 crops per image_id (avoid hogging from one scene)
    selected: List[CropCandidate] = []
    per_image_count: dict = {}
    for c in all_candidates:
        key = (c.dataset, c.image_id)
        if per_image_count.get(key, 0) >= 2:
            continue
        # Check spatial overlap: skip if heavily overlaps an already-picked crop in same image
        overlap = False
        for s in selected:
            if s.dataset == c.dataset and s.image_id == c.image_id:
                if abs(s.x - c.x) < CROP // 2 and abs(s.y - c.y) < CROP // 2:
                    overlap = True
                    break
        if overlap:
            continue
        selected.append(c)
        per_image_count[key] = per_image_count.get(key, 0) + 1
        if len(selected) >= TOP_K_GLOBAL:
            break

    # Save report
    out_path = ROOT / "_supplementary" / "diverse_crops_top.json"
    payload = [
        {
            "dataset": c.dataset,
            "image_id": c.image_id,
            "x": c.x,
            "y": c.y,
            "diversity": c.diversity,
            "instance_count": c.instance_count,
            "n_colors": c.n_colors,
        }
        for c in selected
    ]
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote top-%d candidates to %s", len(selected), out_path)

    # Also export thumbnail PNGs to inspect
    inspect_dir = ROOT / "_supplementary" / "candidate_thumbs"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(selected):
        overlay = np.array(Image.open(ROOT / c.dataset / c.image_id / "ours_overlay.png"))
        original = np.array(Image.open(ROOT / c.dataset / c.image_id / "original.png"))
        crop_o = overlay[c.y : c.y + CROP, c.x : c.x + CROP]
        crop_r = original[c.y : c.y + CROP, c.x : c.x + CROP]
        side = np.concatenate([crop_r, crop_o], axis=1)
        Image.fromarray(side).save(
            inspect_dir / f"{i:02d}_{c.dataset}_{c.image_id[:30]}_x{c.x}_y{c.y}_div{c.diversity}_inst{c.instance_count}.png"
        )
    logger.info("Wrote candidate thumbnails to %s", inspect_dir)


if __name__ == "__main__":
    main()
