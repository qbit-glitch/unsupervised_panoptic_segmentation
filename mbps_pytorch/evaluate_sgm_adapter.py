"""Evaluate the SGM adapter on Cityscapes val with COCO-style instance metrics.

For each val image:
1. Load DINOv2 patch features + DepthPro depth.
2. Forward the trained SGM adapter -> per-thing-class foreground probability
   ``(8, H, W) in [0, 1]``.
3. Threshold + connected components per class -> instance proposals
   ``(mask, class, score)``.
4. Aggregate per-class predictions across the val split.

Compute COCO-style instance-segmentation metrics:

* **AP** -- mean Average Precision across IoU thresholds ``[0.5, 0.55, ..., 0.95]``
  averaged across thing classes.
* **AP50**, **AP75** -- AP at IoU = 0.50 and 0.75.
* **APs / APm / APl** -- AP across small / medium / large GT instances.
* **AR1 / AR10 / AR100** -- max recall with the top-k predictions per image,
  averaged across IoU thresholds and classes.

These are the standard instance-segmentation metrics from the COCO benchmark;
they are more reliable than PQ for class-imbalanced thing-only evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from mbps_pytorch.generate_depth_guided_instances import DEFAULT_THING_IDS, WORK_H, WORK_W
from mbps_pytorch.models.instance.sgm_adapter import SGMAdapter, SGMAdapterConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

THING_TRAIN_IDS = tuple(sorted(DEFAULT_THING_IDS))  # (11, 12, 13, 14, 15, 16, 17, 18)
CS_NAMES = {
    11: "person", 12: "rider", 13: "car", 14: "truck",
    15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}
CS_THING_CLASS_IDS = {
    11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33,
}

# COCO area conventions (px^2 at this resolution).
AREA_SMALL = 32 ** 2
AREA_MEDIUM = 96 ** 2

# COCO IoU sweep
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 10 thresholds
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)  # 101-point interpolation
MAX_DETS_LIST = (1, 10, 100)


# ---------------------------------------------------------------------------
# Adapter forward
# ---------------------------------------------------------------------------


def _load_adapter(ckpt_path: Path, device: torch.device) -> SGMAdapter:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = state.get("adapter_config", {})
    cfg = SGMAdapterConfig(**cfg_dict)
    adapter = SGMAdapter(cfg)
    adapter.load_state_dict(state["state_dict"])
    adapter.to(device).eval()
    return adapter


def _reshape_dino_to_grid(dino: np.ndarray) -> np.ndarray:
    if dino.ndim == 3:
        return dino
    n, d = dino.shape
    h_p = 32
    w_p = n // h_p
    if h_p * w_p != n:
        side = int(np.sqrt(n))
        if side * side == n:
            return dino.reshape(side, side, d)
        raise ValueError(f"Cannot reshape DINO of shape {dino.shape}")
    return dino.reshape(h_p, w_p, d)


def _patch_depth(depth_full: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:
    return F.adaptive_avg_pool2d(depth_full[None, None], (h_p, w_p)).reshape(-1)


def predict_instances(
    adapter: SGMAdapter,
    dino_patch: np.ndarray,
    depth_full: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    min_area: int = 200,
) -> List[Tuple[np.ndarray, int, float]]:
    """Return per-image instance proposals ``[(mask, train_id, score), ...]``."""
    dino_grid = _reshape_dino_to_grid(dino_patch.astype(np.float32))
    h_p, w_p, _ = dino_grid.shape
    dino_t = torch.from_numpy(dino_grid.reshape(h_p * w_p, -1))[None].to(device)
    depth_t = torch.from_numpy(depth_full.astype(np.float32)).to(device)
    depth_patch = _patch_depth(depth_t, h_p, w_p)[None]
    with torch.no_grad():
        m_tilde = adapter(dino_t, depth_patch, out_hw=(WORK_H, WORK_W)).squeeze(0)
    m_np = m_tilde.cpu().numpy()  # (T, H, W)
    proposals: list = []
    for ti, train_id in enumerate(THING_TRAIN_IDS):
        fg = m_np[ti] > threshold
        if not fg.any():
            continue
        labeled, n_cc = ndimage.label(fg)
        for cc in range(1, n_cc + 1):
            mask = labeled == cc
            area = int(mask.sum())
            if area < min_area:
                continue
            score = float(m_np[ti][mask].mean())
            proposals.append((mask, train_id, score))
    return proposals


def extract_gt_things(
    instance_ids: np.ndarray,
    work_hw: Tuple[int, int],
    min_area: int = 100,
) -> List[Tuple[np.ndarray, int]]:
    """Convert Cityscapes ``*_instanceIds.png`` to ``[(mask, train_id), ...]``."""
    h_native, w_native = instance_ids.shape
    if (h_native, w_native) != work_hw:
        inst_pil = Image.fromarray(instance_ids.astype(np.int32))
        inst_pil = inst_pil.resize((work_hw[1], work_hw[0]), Image.NEAREST)
        instance_ids = np.array(inst_pil)
    class_id_to_train_id = {v: k for k, v in CS_THING_CLASS_IDS.items()}
    out: list = []
    for uid in np.unique(instance_ids):
        if uid < 1000:
            continue
        cls_id = int(uid) // 1000
        train_id = class_id_to_train_id.get(cls_id)
        if train_id is None:
            continue
        mask = instance_ids == uid
        if int(mask.sum()) < min_area:
            continue
        out.append((mask, train_id))
    return out


# ---------------------------------------------------------------------------
# COCO-style AP / AR
# ---------------------------------------------------------------------------


def _iou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Pairwise IoU between binary masks. preds: (P, H, W); gts: (G, H, W)."""
    if preds.shape[0] == 0 or gts.shape[0] == 0:
        return np.zeros((preds.shape[0], gts.shape[0]), dtype=np.float32)
    p_flat = preds.reshape(preds.shape[0], -1).astype(np.bool_)
    g_flat = gts.reshape(gts.shape[0], -1).astype(np.bool_)
    inter = (p_flat.astype(np.int64) @ g_flat.astype(np.int64).T).astype(np.float32)
    p_area = p_flat.sum(axis=1, keepdims=True).astype(np.float32)
    g_area = g_flat.sum(axis=1, keepdims=True).astype(np.float32).T
    union = p_area + g_area - inter
    return np.where(union > 0, inter / union, 0.0)


def _coco_ap_for_class(
    image_preds: List[List[Tuple[np.ndarray, float]]],
    image_gts: List[List[np.ndarray]],
    iou_thresholds: np.ndarray,
    recall_thresholds: np.ndarray,
    area_range: Tuple[float, float] = (0.0, float("inf")),
    max_dets: int = 100,
) -> Tuple[float, float, float]:
    """COCO-style per-class AP + AR.

    Args:
        image_preds: list (over images) of list of ``(mask, score)`` for this class.
        image_gts: list (over images) of list of ``mask`` for this class.

    Returns:
        ``(AP_meanIoU, AP50, AR_maxDets)``.
    """
    n_imgs = len(image_preds)
    if n_imgs == 0:
        return 0.0, 0.0, 0.0
    T = len(iou_thresholds)
    # Filter and sort preds per image by descending score, cap at max_dets
    filtered_preds: list = []
    filtered_scores: list = []
    img_idx_per_det: list = []
    for i in range(n_imgs):
        plist = image_preds[i]
        if not plist:
            continue
        scores = np.array([s for _m, s in plist], dtype=np.float32)
        order = np.argsort(-scores)[:max_dets]
        for j in order:
            mask, s = plist[j]
            filtered_preds.append(mask)
            filtered_scores.append(s)
            img_idx_per_det.append(i)
    # Build GTs with area filtering
    gt_count_in_range = 0
    valid_gt_per_image: list = []
    for i in range(n_imgs):
        glist = image_gts[i]
        valids: list = []
        for j, m in enumerate(glist):
            a = float(m.sum())
            in_range = (a >= area_range[0]) and (a < area_range[1])
            valids.append(in_range)
            if in_range:
                gt_count_in_range += 1
        valid_gt_per_image.append(valids)
    if gt_count_in_range == 0:
        return float("nan"), float("nan"), float("nan")

    # For each prediction, compute matched flag per IoU threshold
    n_det = len(filtered_preds)
    tp = np.zeros((T, n_det), dtype=np.float32)
    fp = np.zeros((T, n_det), dtype=np.float32)
    gt_matched = [
        [[False] * len(image_gts[i]) for _ in range(T)] for i in range(n_imgs)
    ]
    # Sort detections globally by score descending
    g_scores = np.array(filtered_scores, dtype=np.float32)
    g_order = np.argsort(-g_scores)
    for rank, det_idx in enumerate(g_order):
        pmask = filtered_preds[det_idx]
        img = img_idx_per_det[det_idx]
        # Skip detection if it falls outside the area range (use pred area as proxy)
        p_area = float(pmask.sum())
        det_in_range = (p_area >= area_range[0]) and (p_area < area_range[1])
        gts_here = image_gts[img]
        valids = valid_gt_per_image[img]
        if not gts_here:
            for t in range(T):
                if det_in_range:
                    fp[t, rank] = 1.0
            continue
        gts_arr = np.stack(gts_here, axis=0)
        ious = _iou_matrix(pmask[None], gts_arr).reshape(-1)
        for t, iou_t in enumerate(iou_thresholds):
            best_iou = iou_t
            best_g = -1
            for gj in range(len(gts_here)):
                if gt_matched[img][t][gj] or not valids[gj]:
                    continue
                if ious[gj] >= best_iou:
                    best_iou = ious[gj]
                    best_g = gj
            if best_g >= 0:
                gt_matched[img][t][best_g] = True
                if det_in_range:
                    tp[t, rank] = 1.0
            else:
                if det_in_range:
                    fp[t, rank] = 1.0
    # P-R per IoU threshold
    ap = np.zeros(T, dtype=np.float32)
    ar = np.zeros(T, dtype=np.float32)
    for t in range(T):
        cum_tp = np.cumsum(tp[t])
        cum_fp = np.cumsum(fp[t])
        recall = cum_tp / max(gt_count_in_range, 1)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        # Make precision monotonically decreasing for AP interpolation
        for i in range(len(precision) - 1, 0, -1):
            if precision[i] > precision[i - 1]:
                precision[i - 1] = precision[i]
        # 101-point interpolation
        ap_t = 0.0
        for r in recall_thresholds:
            mask_ge = recall >= r
            p_at = float(precision[mask_ge].max()) if mask_ge.any() else 0.0
            ap_t += p_at / len(recall_thresholds)
        ap[t] = ap_t
        ar[t] = float(recall[-1]) if len(recall) else 0.0
    return float(ap.mean()), float(ap[0]), float(ar.mean())


def aggregate_coco(
    per_image: List[Dict[int, Dict[str, List]]],
) -> Dict[str, float]:
    """Compute AP / AP50 / AP75 / APs/m/l / AR1/10/100, plus per-class APs."""
    classes = THING_TRAIN_IDS
    # Reorganise: per-class lists indexed by image
    n_imgs = len(per_image)
    preds_by_cls = {c: [[] for _ in range(n_imgs)] for c in classes}
    gts_by_cls = {c: [[] for _ in range(n_imgs)] for c in classes}
    for i, rec in enumerate(per_image):
        for c in classes:
            preds_by_cls[c][i] = rec[c]["preds"]
            gts_by_cls[c][i] = rec[c]["gts"]

    summary: Dict[str, float] = {}
    aps_05_95: list = []
    aps_50: list = []
    aps_75: list = []
    aps_s: list = []
    aps_m: list = []
    aps_l: list = []
    ars_1: list = []
    ars_10: list = []
    ars_100: list = []
    for c in classes:
        # AP at IoU=0.5 (special) and overall
        ap_avg, ap50, _ar = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], IOU_THRESHOLDS, RECALL_THRESHOLDS, max_dets=100,
        )
        # AP75 specifically
        _, ap75_only, _ = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], np.array([0.75]), RECALL_THRESHOLDS, max_dets=100,
        )
        # Area-binned
        ap_small, _a50, _ar = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], IOU_THRESHOLDS, RECALL_THRESHOLDS,
            area_range=(0.0, AREA_SMALL), max_dets=100,
        )
        ap_med, _a50, _ar = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], IOU_THRESHOLDS, RECALL_THRESHOLDS,
            area_range=(AREA_SMALL, AREA_MEDIUM), max_dets=100,
        )
        ap_large, _a50, _ar = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], IOU_THRESHOLDS, RECALL_THRESHOLDS,
            area_range=(AREA_MEDIUM, float("inf")), max_dets=100,
        )
        # AR at max_dets
        _ap_avg, _ap50, ar_100 = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], IOU_THRESHOLDS, RECALL_THRESHOLDS,
            max_dets=100,
        )
        _ap_avg, _ap50, ar_10 = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], IOU_THRESHOLDS, RECALL_THRESHOLDS,
            max_dets=10,
        )
        _ap_avg, _ap50, ar_1 = _coco_ap_for_class(
            preds_by_cls[c], gts_by_cls[c], IOU_THRESHOLDS, RECALL_THRESHOLDS,
            max_dets=1,
        )
        name = CS_NAMES[c]
        summary[f"AP_{name}"] = ap_avg
        summary[f"AP50_{name}"] = ap50
        summary[f"AP75_{name}"] = ap75_only
        summary[f"APs_{name}"] = ap_small
        summary[f"APm_{name}"] = ap_med
        summary[f"APl_{name}"] = ap_large
        summary[f"AR1_{name}"] = ar_1
        summary[f"AR10_{name}"] = ar_10
        summary[f"AR100_{name}"] = ar_100
        for value, lst in (
            (ap_avg, aps_05_95), (ap50, aps_50), (ap75_only, aps_75),
            (ap_small, aps_s), (ap_med, aps_m), (ap_large, aps_l),
            (ar_1, ars_1), (ar_10, ars_10), (ar_100, ars_100),
        ):
            if value == value:  # not NaN
                lst.append(value)
    summary["AP"] = float(np.mean(aps_05_95)) if aps_05_95 else 0.0
    summary["AP50"] = float(np.mean(aps_50)) if aps_50 else 0.0
    summary["AP75"] = float(np.mean(aps_75)) if aps_75 else 0.0
    summary["APs"] = float(np.mean(aps_s)) if aps_s else 0.0
    summary["APm"] = float(np.mean(aps_m)) if aps_m else 0.0
    summary["APl"] = float(np.mean(aps_l)) if aps_l else 0.0
    summary["AR1"] = float(np.mean(ars_1)) if ars_1 else 0.0
    summary["AR10"] = float(np.mean(ars_10)) if ars_10 else 0.0
    summary["AR100"] = float(np.mean(ars_100)) if ars_100 else 0.0
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    adapter = _load_adapter(Path(args.checkpoint).expanduser(), device)
    logger.info("Loaded adapter from %s; params=%d", args.checkpoint, adapter.num_parameters())
    root = Path(args.cityscapes_root).expanduser()
    img_dir = root / "leftImg8bit" / "val"
    depth_dir = root / args.depth_subdir / "val"
    dino_dir = root / args.dino_subdir / "val"
    gt_dir = root / "gtFine" / "val"

    stems: list[Tuple[str, str]] = []
    for sem_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
        city = sem_path.parent.name
        stem = sem_path.stem.replace("_leftImg8bit", "")
        stems.append((city, stem))
    if args.max_images > 0:
        stems = stems[: args.max_images]
    logger.info("Evaluating on %d val images", len(stems))

    # Per-image, per-class container of preds and GTs
    per_image: List[Dict[int, Dict[str, List]]] = []
    n_pred_total = 0
    n_gt_total = 0
    for city, stem in tqdm(stems, desc="eval"):
        depth_path = depth_dir / city / f"{stem}.npy"
        if not depth_path.exists():
            depth_path = depth_dir / city / f"{stem}_leftImg8bit.npy"
        dino_path = dino_dir / city / f"{stem}_leftImg8bit.npy"
        if not dino_path.exists():
            dino_path = dino_dir / city / f"{stem}.npy"
        gt_path = gt_dir / city / f"{stem}_gtFine_instanceIds.png"
        if not (depth_path.exists() and dino_path.exists() and gt_path.exists()):
            continue

        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (WORK_H, WORK_W):
            depth = np.array(
                Image.fromarray(depth).resize((WORK_W, WORK_H), Image.BILINEAR)
            ).astype(np.float32)
        dino_np = np.load(dino_path).astype(np.float32)

        pred = predict_instances(
            adapter, dino_np, depth, device,
            threshold=args.fg_threshold, min_area=args.min_area,
        )
        gt = extract_gt_things(
            np.array(Image.open(gt_path)),
            work_hw=(WORK_H, WORK_W), min_area=args.gt_min_area,
        )
        n_pred_total += len(pred)
        n_gt_total += len(gt)
        rec: Dict[int, Dict[str, List]] = {
            c: {"preds": [], "gts": []} for c in THING_TRAIN_IDS
        }
        for mask, cls, score in pred:
            rec[cls]["preds"].append((mask, score))
        for mask, cls in gt:
            rec[cls]["gts"].append(mask)
        per_image.append(rec)

    summary = aggregate_coco(per_image)
    summary["n_images"] = len(per_image)
    summary["n_predictions_total"] = n_pred_total
    summary["n_gt_things_total"] = n_gt_total
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SGM adapter (AP/AR) on Cityscapes val")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cityscapes_root", type=str, required=True)
    p.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    p.add_argument("--dino_subdir", type=str, default="dinov2_features")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--min_area", type=int, default=200)
    p.add_argument("--gt_min_area", type=int, default=100)
    p.add_argument("--output_json", type=str, default="results/sgm_adapter_eval.json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    t0 = time.time()
    summary = evaluate(args)
    summary["wall_clock_s"] = time.time() - t0
    out = Path(args.output_json).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    logger.info("=== SUMMARY ===")
    logger.info("  AP    (IoU 0.5:0.95, classes mean) = %.4f", summary["AP"])
    logger.info("  AP50  = %.4f", summary["AP50"])
    logger.info("  AP75  = %.4f", summary["AP75"])
    logger.info("  APs / APm / APl = %.4f / %.4f / %.4f",
                summary["APs"], summary["APm"], summary["APl"])
    logger.info("  AR1 / AR10 / AR100 = %.4f / %.4f / %.4f",
                summary["AR1"], summary["AR10"], summary["AR100"])
    for c in THING_TRAIN_IDS:
        name = CS_NAMES[c]
        logger.info("  AP_%-11s = %.4f", name, summary[f"AP_{name}"])
    logger.info("predictions=%d  gt_things=%d  images=%d  wall=%.1fs",
                summary["n_predictions_total"], summary["n_gt_things_total"],
                summary["n_images"], summary["wall_clock_s"])
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
