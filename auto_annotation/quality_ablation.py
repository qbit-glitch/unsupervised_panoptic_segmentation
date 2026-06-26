#!/usr/bin/env python3
"""Quality-ablation building blocks: ensemble stuff, SAM3 tiling, CRF refine.

Cumulative levers on top of the baseline (Mask2Former-Mapillary stuff + SAM3 things):
  +ensemble : per-pixel majority of {M2F-Mapillary, M2F-Cityscapes, EoMT-Cityscapes}
  +tiling   : SAM3 on full image + overlapping tiles -> recover small/crowded instances
  +crf      : dense-CRF boundary sharpening guided by RGB

All stuff models output Cityscapes train-ids 0..18, which equal our palette idx order
(see demo_panoptic._CS), so no remap needed. Build all transformers models BEFORE SAM3.
"""

import logging
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from .demo_panoptic import IDX2LABELID, STUFF_ORDER, NAME2IDX, VOID, DemoConfig

logger = logging.getLogger(__name__)

STUFF_IDX = {NAME2IDX[n] for n in STUFF_ORDER}


# ----------------------------- domain-matched stuff models --------------------
class M2FCityscapesRunner:
    REPO = "facebook/mask2former-swin-large-cityscapes-semantic"

    def __init__(self, long_side: int = 1024, device: str = "cpu"):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(self.REPO)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.REPO).eval().to(device)
        self.long_side = long_side
        logger.info("Mask2Former-Cityscapes ready")

    @torch.no_grad()
    def predict(self, pil: Image.Image) -> np.ndarray:
        W, H = pil.size
        inp = self.proc(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        sem = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0]
        return sem.cpu().numpy().astype(np.int32)   # trainId == our idx


class EoMTCityscapesRunner:
    REPO = "tue-mps/cityscapes_semantic_eomt_large_1024"

    def __init__(self, device: str = "cpu"):
        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(self.REPO)
        self.model = AutoModelForUniversalSegmentation.from_pretrained(self.REPO).eval().to(device)
        logger.info("EoMT-Cityscapes ready")

    @torch.no_grad()
    def predict(self, pil: Image.Image) -> np.ndarray:
        W, H = pil.size
        inp = self.proc(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        sem = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0]
        return sem.cpu().numpy().astype(np.int32)


def ensemble_stuff(sem_maps: List[np.ndarray]) -> np.ndarray:
    """Per-pixel majority vote over stuff classes; ties -> first map's label."""
    h, w = sem_maps[0].shape
    stack = np.stack(sem_maps, 0)                       # (M,H,W)
    out = sem_maps[0].copy()
    # vote only where at least 2 models agree on a stuff class
    for y in range(0, h, 1):
        pass
    # vectorised majority: count per pixel via bincount over small label set
    M = stack.shape[0]
    flat = stack.reshape(M, -1)
    res = out.reshape(-1).copy()
    # candidate labels present anywhere
    for lab in np.unique(stack):
        if lab not in STUFF_IDX:
            continue
        votes = (flat == lab).sum(0)
        res[votes >= 2] = lab          # 2/3 agreement wins
    return res.reshape(h, w)


# --------------------------------- SAM3 tiling --------------------------------
def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def sam3_tiled(sam3, pil: Image.Image, prompts: List[str], thr: float,
               tiles: int = 2, overlap: float = 0.2, name_to_idx=None) -> List[dict]:
    """Run SAM3 on full image + a tiles×tiles overlapping grid; NMS-merge masks.

    name_to_idx: optional dict mapping prompt strings (incl. synonyms like
    'scooter'->motorcycle idx) to class idx; defaults to exact NAME2IDX lookup.
    """
    idx_of = (lambda p: name_to_idx[p]) if name_to_idx else (lambda p: NAME2IDX[p])
    W, H = pil.size
    dets = [{"mask": d["mask"], "class_idx": idx_of(d["prompt"]), "score": d["score"]}
            for d in sam3.predict(pil, prompts)]
    tw, th = int(W / tiles * (1 + overlap)), int(H / tiles * (1 + overlap))
    for iy in range(tiles):
        for ix in range(tiles):
            x0 = int(ix * W / tiles); y0 = int(iy * H / tiles)
            x1 = min(W, x0 + tw); y1 = min(H, y0 + th)
            crop = pil.crop((x0, y0, x1, y1))
            for d in sam3.predict(crop, prompts):
                full = np.zeros((H, W), bool)
                full[y0:y1, x0:x1] = d["mask"]
                dets.append({"mask": full, "class_idx": idx_of(d["prompt"]),
                             "score": d["score"] * 0.95})  # slight tile penalty
    # NMS dedup
    dets.sort(key=lambda d: -d["score"])
    kept: List[dict] = []
    for d in dets:
        if all(_mask_iou(d["mask"], k["mask"]) < 0.5 for k in kept
               if k["class_idx"] == d["class_idx"]):
            if d["mask"].sum() >= 200:
                kept.append(d)
    return kept


# ----------------------------------- CRF --------------------------------------
def crf_refine(rgb: np.ndarray, sem: np.ndarray, n_iter: int = 5) -> np.ndarray:
    """Dense-CRF sharpen a class map (idx 0..18, VOID->own label) guided by RGB."""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels
    except ImportError:                      # pydensecrf can fail to build (e.g. Kaggle)
        logger.warning("pydensecrf unavailable -> skipping CRF refine")
        return sem
    h, w = sem.shape
    work = sem.copy()
    work[work == VOID] = 19                  # void as an extra label
    n_labels = 20
    d = dcrf.DenseCRF2D(w, h, n_labels)
    U = unary_from_labels(work.astype(np.int32), n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=13, rgbim=np.ascontiguousarray(rgb), compat=8)
    Q = d.inference(n_iter)
    out = np.argmax(np.array(Q), axis=0).reshape(h, w).astype(np.int32)
    out[out == 19] = VOID
    return out
