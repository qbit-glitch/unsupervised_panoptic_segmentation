#!/usr/bin/env python3
"""Probe unMORE-style center-boundary reasoning on a small Cityscapes slice.

This is a training-free diagnostic, not an official unMORE reproduction. The
official unMORE objectness checkpoints are not vendored in this repo, so this
probe uses the local SAM fine-mask proposals as object candidates and scores
them with simple center/boundary priors before evaluating against the current
depth-derived instance pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, sobel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.sweep_depthpro import (  # noqa: E402
    CLASS_NAMES,
    CS_ID_TO_TRAIN,
    NUM_CLASSES,
    THING_IDS,
    compute_pq_from_accumulators,
    depth_guided_instances,
    evaluate_panoptic_single,
    resize_nearest,
)


FINE_CLASS_TO_TRAINID = {
    0: 11,  # person
    1: 18,  # bicycle
    2: 17,  # motorcycle
    3: 12,  # rider
    6: 14,  # truck
    7: 15,  # bus
    8: 16,  # train
    12: 13,  # car
}


def remap_gt_to_trainids(gt_raw: np.ndarray) -> np.ndarray:
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


def gt_thing_instances(gt_inst_map: np.ndarray) -> dict[int, list[np.ndarray]]:
    instances: dict[int, list[np.ndarray]] = {cid: [] for cid in THING_IDS}
    for uid in np.unique(gt_inst_map):
        if uid < 1000:
            continue
        raw_cls = int(uid) // 1000
        train_id = CS_ID_TO_TRAIN.get(raw_cls)
        if train_id not in THING_IDS:
            continue
        mask = gt_inst_map == uid
        if int(mask.sum()) < 10:
            continue
        instances[int(train_id)].append(mask)
    return instances


def load_cluster_lut(centroids_path: Path) -> np.ndarray:
    data = np.load(str(centroids_path))
    cluster_to_class = data["cluster_to_class"]
    lut = np.full(256, 255, dtype=np.uint8)
    for cid, tid in enumerate(cluster_to_class):
        lut[cid] = int(tid)
    return lut


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    return mask & ~eroded


def dominant_thing_class(pred_sem: np.ndarray, mask: np.ndarray) -> tuple[int | None, float]:
    labels = pred_sem[mask]
    labels = labels[np.isin(labels, list(THING_IDS))]
    if labels.size == 0:
        return None, 0.0
    counts = np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES)
    cls = int(np.argmax(counts))
    return cls, float(counts[cls] / max(mask.sum(), 1))


def load_depth_instances(
    npz_path: Path,
    pred_sem: np.ndarray,
    *,
    min_area: int,
    min_thing_frac: float,
) -> list[tuple[np.ndarray, int, float]]:
    data = np.load(str(npz_path), allow_pickle=False)
    masks = data["masks"]
    h = int(data["h_patches"]) if "h_patches" in data.files else pred_sem.shape[0]
    w = int(data["w_patches"]) if "w_patches" in data.files else pred_sem.shape[1]
    scores = data["scores"] if "scores" in data.files else np.ones((len(masks),), dtype=np.float32)
    num_valid = int(data["num_valid"]) if "num_valid" in data.files else len(masks)

    instances: list[tuple[np.ndarray, int, float]] = []
    for idx in range(min(num_valid, len(masks))):
        mask = masks[idx]
        if mask.ndim == 1:
            mask = mask.reshape(h, w)
        mask = mask.astype(bool)
        if mask.shape != pred_sem.shape:
            mask = resize_nearest(mask.astype(np.uint8), pred_sem.shape).astype(bool)
        if int(mask.sum()) < min_area:
            continue
        cls, thing_frac = dominant_thing_class(pred_sem, mask)
        if cls is None or thing_frac < min_thing_frac:
            continue
        instances.append((mask, cls, float(scores[idx])))

    instances.sort(key=lambda item: -float(item[0].sum()))
    return instances


def depth_edge_map(depth_path: Path, hw: tuple[int, int], quantile: float = 0.86) -> np.ndarray | None:
    if not depth_path.exists():
        return None
    depth = np.load(str(depth_path)).astype(np.float64)
    if depth.shape != hw:
        depth = np.array(Image.fromarray(depth).resize((hw[1], hw[0]), Image.BILINEAR))
    gx = sobel(depth, axis=1)
    gy = sobel(depth, axis=0)
    mag = np.sqrt(gx * gx + gy * gy)
    thresh = np.quantile(mag, quantile)
    return mag >= thresh


def semantic_boundary_map(pred_sem: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(pred_sem, dtype=bool)
    edges[:, 1:] |= pred_sem[:, 1:] != pred_sem[:, :-1]
    edges[1:, :] |= pred_sem[1:, :] != pred_sem[:-1, :]
    return ndimage.binary_dilation(edges, iterations=1)


def center_score(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    max_dist = float(distance_transform_edt(mask).max())
    equivalent_radius = float(np.sqrt(area / np.pi))
    return float(np.clip(max_dist / (equivalent_radius + 1e-6), 0.0, 1.0))


def boundary_score(mask: np.ndarray, support_edges: np.ndarray | None) -> float:
    boundary = mask_boundary(mask)
    denom = int(boundary.sum())
    if denom == 0:
        return 0.0
    if support_edges is None:
        return 0.0
    return float((boundary & support_edges).sum() / denom)


def load_sam_reasoning_instances(
    sam_path: Path,
    pred_sem: np.ndarray,
    *,
    support_edges: np.ndarray | None,
    min_area: int,
    min_thing_frac: float,
    min_score: float,
) -> list[tuple[np.ndarray, int, float]]:
    data = np.load(str(sam_path), allow_pickle=False)
    masks = data["masks"].astype(bool)
    iou_scores = data["iou_scores"] if "iou_scores" in data.files else np.ones(len(masks))
    class_labels = (
        data["class_labels"].astype(np.int64)
        if "class_labels" in data.files
        else np.full((len(masks),), -1, dtype=np.int64)
    )

    candidates: list[tuple[np.ndarray, int, float]] = []
    for idx, mask in enumerate(masks):
        if mask.shape != pred_sem.shape:
            mask = resize_nearest(mask.astype(np.uint8), pred_sem.shape).astype(bool)
        if int(mask.sum()) < min_area:
            continue

        sem_cls, thing_frac = dominant_thing_class(pred_sem, mask)
        sam_cls = FINE_CLASS_TO_TRAINID.get(int(class_labels[idx]))
        cls = sem_cls if sem_cls is not None else sam_cls
        if cls is None or cls not in THING_IDS or thing_frac < min_thing_frac:
            continue

        c_score = center_score(mask)
        b_score = boundary_score(mask, support_edges)
        sam_score = float(iou_scores[idx])
        score = 0.55 * sam_score + 0.25 * c_score + 0.20 * b_score
        if score < min_score:
            continue
        candidates.append((mask, int(cls), float(score)))

    return greedy_non_overlapping(candidates)


def greedy_non_overlapping(
    candidates: Iterable[tuple[np.ndarray, int, float]],
    *,
    max_overlap: float = 0.55,
) -> list[tuple[np.ndarray, int, float]]:
    accepted: list[tuple[np.ndarray, int, float]] = []
    occupied = None
    for mask, cls, score in sorted(candidates, key=lambda item: -item[2]):
        if occupied is None:
            occupied = np.zeros(mask.shape, dtype=bool)
        overlap = float((mask & occupied).sum() / max(mask.sum(), 1))
        if overlap > max_overlap:
            continue
        trimmed = mask & ~occupied
        if trimmed.sum() < 10:
            continue
        accepted.append((trimmed, cls, score))
        occupied |= trimmed
    return accepted


def hybrid_instances(
    sam_instances: list[tuple[np.ndarray, int, float]],
    depth_instances: list[tuple[np.ndarray, int, float]],
    *,
    min_residual_area: int,
) -> list[tuple[np.ndarray, int, float]]:
    merged: list[tuple[np.ndarray, int, float]] = []
    occupied = None
    for mask, cls, score in sorted(sam_instances, key=lambda item: -item[2]):
        if occupied is None:
            occupied = np.zeros(mask.shape, dtype=bool)
        trimmed = mask & ~occupied
        if trimmed.sum() >= min_residual_area:
            merged.append((trimmed, cls, score + 1.0))
            occupied |= trimmed

    for mask, cls, score in depth_instances:
        if occupied is None:
            occupied = np.zeros(mask.shape, dtype=bool)
        residual = mask & ~occupied
        if residual.sum() >= min_residual_area:
            merged.append((residual, cls, score))
            occupied |= residual
    return merged


def update_accumulators(acc: dict, pred_sem: np.ndarray, instances, gt_sem, gt_inst_map, eval_hw) -> None:
    tp, fp, fn, iou_s, n_inst = evaluate_panoptic_single(
        pred_sem, instances, gt_sem, gt_inst_map, eval_hw
    )
    acc["tp"] += tp
    acc["fp"] += fp
    acc["fn"] += fn
    acc["iou"] += iou_s
    acc["n_inst"] += n_inst
    acc["n_images"] += 1


def record_ap_inputs(ap_store: dict, image_id: str, instances, gt_instances: dict[int, list[np.ndarray]]) -> None:
    for cls in sorted(THING_IDS):
        ap_store["gt"][cls][image_id] = gt_instances.get(cls, [])
    for mask, cls, score in instances:
        if cls in THING_IDS:
            ap_store["pred"][int(cls)].append({
                "image_id": image_id,
                "score": float(score),
                "mask": mask.astype(bool),
            })


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    if inter == 0:
        return 0.0
    union = int((a | b).sum())
    return float(inter / max(union, 1))


def average_precision_at_threshold(preds: list[dict], gt_by_image: dict[str, list[np.ndarray]],
                                   iou_threshold: float) -> float | None:
    n_gt = sum(len(masks) for masks in gt_by_image.values())
    if n_gt == 0:
        return None
    if not preds:
        return 0.0

    matched = {image_id: np.zeros(len(masks), dtype=bool) for image_id, masks in gt_by_image.items()}
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)

    for idx, pred in enumerate(sorted(preds, key=lambda item: -item["score"])):
        gt_masks = gt_by_image.get(pred["image_id"], [])
        if not gt_masks:
            fp[idx] = 1.0
            continue
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt_mask in enumerate(gt_masks):
            if matched[pred["image_id"]][gt_idx]:
                continue
            iou = mask_iou(pred["mask"], gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            tp[idx] = 1.0
            matched[pred["image_id"]][best_gt] = True
        else:
            fp[idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    # COCO-style 101-point interpolation.
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        valid = precision[recall >= r]
        ap += float(valid.max()) if valid.size else 0.0
    return ap / 101.0


def compute_mask_ap(ap_store: dict) -> dict:
    thresholds = [round(float(t), 2) for t in np.arange(0.50, 0.96, 0.05)]
    per_class = {}
    all_ap = []
    all_ap50 = []
    all_ap75 = []

    for cls in sorted(THING_IDS):
        preds = ap_store["pred"][cls]
        gt_by_image = ap_store["gt"][cls]
        class_aps = []
        for thr in thresholds:
            ap = average_precision_at_threshold(preds, gt_by_image, thr)
            if ap is not None:
                class_aps.append((thr, ap))
        if not class_aps:
            continue
        ap_map = {thr: ap for thr, ap in class_aps}
        ap_mean = float(np.mean([ap for _, ap in class_aps]))
        ap50 = float(ap_map.get(0.5, 0.0))
        ap75 = float(ap_map.get(0.75, 0.0))
        all_ap.append(ap_mean)
        all_ap50.append(ap50)
        all_ap75.append(ap75)
        n_gt = sum(len(masks) for masks in gt_by_image.values())
        per_class[CLASS_NAMES[cls]] = {
            "AP": round(ap_mean * 100, 2),
            "AP50": round(ap50 * 100, 2),
            "AP75": round(ap75 * 100, 2),
            "n_gt": int(n_gt),
            "n_pred": int(len(preds)),
        }

    return {
        "AP": round(float(np.mean(all_ap)) * 100, 2) if all_ap else 0.0,
        "AP50": round(float(np.mean(all_ap50)) * 100, 2) if all_ap50 else 0.0,
        "AP75": round(float(np.mean(all_ap75)) * 100, 2) if all_ap75 else 0.0,
        "per_class": per_class,
    }


def aggregate_thing_metrics(acc: dict) -> dict:
    pqs, sqs, rqs = [], [], []
    for cid in sorted(THING_IDS):
        tp = acc["tp"][cid]
        fp = acc["fp"][cid]
        fn = acc["fn"][cid]
        if tp + fp + fn <= 0:
            continue
        denom = tp + 0.5 * fp + 0.5 * fn
        sq = acc["iou"][cid] / (tp + 1e-8) if tp > 0 else 0.0
        rq = tp / denom if denom > 0 else 0.0
        pqs.append(sq * rq)
        sqs.append(sq)
        rqs.append(rq)
    return {
        "PQ": round(float(np.mean(pqs)) * 100, 2) if pqs else 0.0,
        "SQ": round(float(np.mean(sqs)) * 100, 2) if sqs else 0.0,
        "RQ": round(float(np.mean(rqs)) * 100, 2) if rqs else 0.0,
    }


def thing_summary(metrics: dict, acc: dict) -> dict:
    thing_agg = aggregate_thing_metrics(acc)
    out = {
        "PQ_things": thing_agg["PQ"],
        "RQ_things": thing_agg["RQ"],
        "SQ_things": thing_agg["SQ"],
        "avg_instances": metrics.get("avg_instances", 0.0),
        "per_class": {},
    }
    for cid in sorted(THING_IDS):
        out["per_class"][CLASS_NAMES[cid]] = metrics["per_class"][CLASS_NAMES[cid]]
    return out


def discover_sam_slice(root: Path, split: str, sam_subdir: str, max_images: int | None):
    sam_root = root / sam_subdir / split
    paths = sorted(sam_root.rglob("*_fine_masks.npz"))
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_raw_k80")
    parser.add_argument("--centroids_path", default=None)
    parser.add_argument("--depth_instance_subdir", default="pseudo_instance_depthpro")
    parser.add_argument("--sam_subdir", default="sam_fine_masks_sam3")
    parser.add_argument("--depth_subdir", default="depth_depthpro")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--min_thing_frac", type=float, default=0.10)
    parser.add_argument("--sam_min_score", type=float, default=0.35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    eval_hw = (512, 1024)
    centroids_path = Path(args.centroids_path) if args.centroids_path else (
        root / args.semantic_subdir / "kmeans_centroids.npz"
    )
    lut = load_cluster_lut(centroids_path)

    sam_paths = discover_sam_slice(root, args.split, args.sam_subdir, args.max_images)
    if not sam_paths:
        raise FileNotFoundError(f"No SAM masks under {root / args.sam_subdir / args.split}")

    accs = {
        "current_depth_instances": {
            "tp": np.zeros(NUM_CLASSES),
            "fp": np.zeros(NUM_CLASSES),
            "fn": np.zeros(NUM_CLASSES),
            "iou": np.zeros(NUM_CLASSES),
            "n_inst": 0,
            "n_images": 0,
        },
        "depth_sobel_tau020_min1000": {
            "tp": np.zeros(NUM_CLASSES),
            "fp": np.zeros(NUM_CLASSES),
            "fn": np.zeros(NUM_CLASSES),
            "iou": np.zeros(NUM_CLASSES),
            "n_inst": 0,
            "n_images": 0,
        },
        "depth_sobel_tau001_min1000": {
            "tp": np.zeros(NUM_CLASSES),
            "fp": np.zeros(NUM_CLASSES),
            "fn": np.zeros(NUM_CLASSES),
            "iou": np.zeros(NUM_CLASSES),
            "n_inst": 0,
            "n_images": 0,
        },
        "sam_center_boundary_proxy": {
            "tp": np.zeros(NUM_CLASSES),
            "fp": np.zeros(NUM_CLASSES),
            "fn": np.zeros(NUM_CLASSES),
            "iou": np.zeros(NUM_CLASSES),
            "n_inst": 0,
            "n_images": 0,
        },
        "hybrid_sam_then_depth": {
            "tp": np.zeros(NUM_CLASSES),
            "fp": np.zeros(NUM_CLASSES),
            "fn": np.zeros(NUM_CLASSES),
            "iou": np.zeros(NUM_CLASSES),
            "n_inst": 0,
            "n_images": 0,
        },
    }
    per_image = []
    missing = []
    ap_stores = {
        name: {
            "gt": {cid: {} for cid in THING_IDS},
            "pred": {cid: [] for cid in THING_IDS},
        }
        for name in accs
    }

    for sam_path in tqdm(sam_paths, desc="unMORE-style probe"):
        city = sam_path.parent.name
        stem = sam_path.name.replace("_fine_masks.npz", "")
        sem_path = root / args.semantic_subdir / args.split / city / f"{stem}.png"
        depth_inst_path = root / args.depth_instance_subdir / args.split / city / f"{stem}.npz"
        depth_path = root / args.depth_subdir / args.split / city / f"{stem}.npy"
        gt_label_path = root / "gtFine" / args.split / city / f"{stem}_gtFine_labelIds.png"
        gt_inst_path = root / "gtFine" / args.split / city / f"{stem}_gtFine_instanceIds.png"

        required = [sem_path, depth_inst_path, gt_label_path, gt_inst_path]
        if any(not p.exists() for p in required):
            missing.append({
                "stem": stem,
                "missing": [str(p) for p in required if not p.exists()],
            })
            continue

        pred_raw = np.array(Image.open(sem_path), dtype=np.uint8)
        pred_sem = lut[pred_raw]
        if pred_sem.shape != eval_hw:
            pred_sem = resize_nearest(pred_sem, eval_hw)

        gt_sem = remap_gt_to_trainids(np.array(Image.open(gt_label_path), dtype=np.uint8))
        if gt_sem.shape != eval_hw:
            gt_sem = resize_nearest(gt_sem, eval_hw)
        gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst.shape != eval_hw:
            gt_inst = np.array(Image.fromarray(gt_inst).resize((eval_hw[1], eval_hw[0]), Image.NEAREST))
        gt_instances = gt_thing_instances(gt_inst)

        sem_edges = semantic_boundary_map(pred_sem)
        depth_edges = depth_edge_map(depth_path, eval_hw)
        support_edges = sem_edges if depth_edges is None else (sem_edges | depth_edges)

        depth_instances = load_depth_instances(
            depth_inst_path,
            pred_sem,
            min_area=args.min_area,
            min_thing_frac=args.min_thing_frac,
        )
        sam_instances = load_sam_reasoning_instances(
            sam_path,
            pred_sem,
            support_edges=support_edges,
            min_area=args.min_area,
            min_thing_frac=args.min_thing_frac,
            min_score=args.sam_min_score,
        )
        hybrid = hybrid_instances(
            sam_instances,
            depth_instances,
            min_residual_area=args.min_area,
        )
        if depth_path.exists():
            depth = np.load(str(depth_path))
            if depth.shape != eval_hw:
                depth = np.array(
                    Image.fromarray(depth).resize((eval_hw[1], eval_hw[0]), Image.BILINEAR)
                )
            depth_sobel020 = depth_guided_instances(
                pred_sem,
                depth,
                THING_IDS,
                grad_threshold=0.20,
                min_area=1000,
                dilation_iters=3,
                depth_blur_sigma=0.0,
            )
            depth_sobel001 = depth_guided_instances(
                pred_sem,
                depth,
                THING_IDS,
                grad_threshold=0.01,
                min_area=1000,
                dilation_iters=3,
                depth_blur_sigma=0.0,
            )
        else:
            depth_sobel020 = []
            depth_sobel001 = []

        update_accumulators(
            accs["current_depth_instances"], pred_sem, depth_instances, gt_sem, gt_inst, eval_hw
        )
        record_ap_inputs(ap_stores["current_depth_instances"], stem, depth_instances, gt_instances)
        update_accumulators(
            accs["depth_sobel_tau020_min1000"], pred_sem, depth_sobel020, gt_sem, gt_inst, eval_hw
        )
        record_ap_inputs(ap_stores["depth_sobel_tau020_min1000"], stem, depth_sobel020, gt_instances)
        update_accumulators(
            accs["depth_sobel_tau001_min1000"], pred_sem, depth_sobel001, gt_sem, gt_inst, eval_hw
        )
        record_ap_inputs(ap_stores["depth_sobel_tau001_min1000"], stem, depth_sobel001, gt_instances)
        update_accumulators(
            accs["sam_center_boundary_proxy"], pred_sem, sam_instances, gt_sem, gt_inst, eval_hw
        )
        record_ap_inputs(ap_stores["sam_center_boundary_proxy"], stem, sam_instances, gt_instances)
        update_accumulators(
            accs["hybrid_sam_then_depth"], pred_sem, hybrid, gt_sem, gt_inst, eval_hw
        )
        record_ap_inputs(ap_stores["hybrid_sam_then_depth"], stem, hybrid, gt_instances)
        per_image.append({
            "stem": stem,
            "depth_instances": len(depth_instances),
            "depth_sobel_tau020_instances": len(depth_sobel020),
            "depth_sobel_tau001_instances": len(depth_sobel001),
            "sam_instances": len(sam_instances),
            "hybrid_instances": len(hybrid),
        })

    methods = {}
    for name, acc in accs.items():
        metrics = compute_pq_from_accumulators(acc["tp"], acc["fp"], acc["fn"], acc["iou"])
        metrics["avg_instances"] = round(float(acc["n_inst"] / max(acc["n_images"], 1)), 2)
        metrics["n_images"] = int(acc["n_images"])
        methods[name] = thing_summary(metrics, acc)
        methods[name]["mask_ap"] = compute_mask_ap(ap_stores[name])

    result = {
        "note": (
            "Training-free proxy for unMORE-style center-boundary reasoning. "
            "Official unMORE checkpoints were unavailable locally; SAM fine masks "
            "provide the object candidates, scored by center compactness and boundary support."
        ),
        "config": {
            "cityscapes_root": str(root),
            "split": args.split,
            "semantic_subdir": args.semantic_subdir,
            "centroids_path": str(centroids_path),
            "depth_instance_subdir": args.depth_instance_subdir,
            "sam_subdir": args.sam_subdir,
            "depth_subdir": args.depth_subdir,
            "min_area": args.min_area,
            "min_thing_frac": args.min_thing_frac,
            "sam_min_score": args.sam_min_score,
            "num_sam_files": len(sam_paths),
            "missing": missing,
        },
        "methods": methods,
        "per_image_counts": per_image,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["methods"], indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
