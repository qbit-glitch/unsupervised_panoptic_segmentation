import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mbps_pytorch.sweep_depthpro import (
    evaluate_panoptic_single, STUFF_IDS, THING_IDS, NUM_CLASSES,
)
from mbps_pytorch.premise_check_geometry_affinity import load_gt
from mbps_pytorch.ga_uniap.config import Phase0Config


def load_gt_pair(stem: str, city: str, cfg: Phase0Config) -> Tuple[np.ndarray, np.ndarray]:
    gt_sem = load_gt(stem, city, cfg.split).astype(np.uint8)  # (512,1024) trainID
    inst_path = (cfg.data_root / "gtFine" / cfg.split / city
                 / f"{stem}_gtFine_instanceIds.png")
    inst = np.array(Image.open(inst_path)).astype(np.int32)   # (1024,2048)
    inst = np.array(Image.fromarray(inst.astype(np.int32)).resize(
        (cfg.work_w, cfg.work_h), Image.NEAREST)).astype(np.int32)
    return gt_sem, inst


def _upsample(mask_grid: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    img = Image.fromarray(mask_grid.astype(np.uint8)).resize((W, H), Image.NEAREST)
    return np.array(img).astype(bool)


def masks_to_panoptic(masks: np.ndarray, gt_sem: np.ndarray
                      ) -> Tuple[np.ndarray, List]:
    """Oracle GT-majority labels (diagnostic upper bound on grouping quality)."""
    H, W = gt_sem.shape
    pred_sem = np.full((H, W), 255, np.uint8)
    pred_inst: List[Tuple[np.ndarray, int, float]] = []
    for k in range(masks.shape[0]):
        m = _upsample(masks[k], (H, W))
        vals = gt_sem[m]
        vals = vals[vals != 255]
        if vals.size == 0:
            continue
        cls = int(np.bincount(vals, minlength=NUM_CLASSES).argmax())
        if cls in THING_IDS:
            pred_inst.append((m, cls, 1.0))
        elif cls in STUFF_IDS:
            pred_sem[m] = cls
    return pred_sem, pred_inst


def score_image(masks: np.ndarray, gt_sem: np.ndarray, gt_inst: np.ndarray,
                cfg: Phase0Config):
    """-> (tp, fp, fn, iou_sum) per-class accumulator arrays (length NUM_CLASSES)."""
    pred_sem, pred_inst = masks_to_panoptic(masks, gt_sem)
    tp, fp, fn, iou_sum, _ = evaluate_panoptic_single(
        pred_sem, pred_inst, gt_sem, gt_inst, (cfg.work_h, cfg.work_w))
    return tp, fp, fn, iou_sum
