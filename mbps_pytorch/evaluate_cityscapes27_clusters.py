#!/usr/bin/env python3
"""Evaluate raw cluster PNGs against the Cityscapes-27 label space.

Cityscapes-27 here follows the CUPS/CAUSE convention: raw Cityscapes label IDs
7..33 are mapped to class IDs 0..26. Cluster IDs are mapped to the 27 classes
only for metric reporting. For overclustering, the default mapping is
many-to-one majority vote, so multiple clusters can map to the same class.

Usage:
    python3 mbps_pytorch/evaluate_cityscapes27_clusters.py \
        --pred_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_anyup_64x128_k80/val \
        --gt_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/gtFine/val \
        --num_clusters 80 \
        --output results/dinov3_anyup_k80_cityscapes27_val.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

IGNORE_LABEL = 255
NUM_CLASSES = 27
CLASS_NAMES = [
    "road",
    "sidewalk",
    "parking",
    "rail_track",
    "building",
    "wall",
    "fence",
    "guard_rail",
    "bridge",
    "tunnel",
    "pole",
    "polegroup",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
    "motorcycle",
    "bicycle",
]
THING_IDS = set(range(17, 27))
STUFF_IDS = set(range(0, 17))


def remap_labelids_to_cityscapes27(gt: np.ndarray) -> np.ndarray:
    out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
    valid = (gt >= 7) & (gt <= 33)
    out[valid] = (gt[valid] - 7).astype(np.uint8)
    return out


def resize_nearest(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def discover_pairs(pred_dir: Path, gt_dir: Path):
    pairs = []
    for pred_path in sorted(pred_dir.rglob("*.png")):
        rel = pred_path.relative_to(pred_dir)
        base = str(rel).replace("_leftImg8bit.png", "").replace(".png", "")
        gt_path = gt_dir / (base + "_gtFine_labelIds.png")
        if gt_path.exists():
            pairs.append((pred_path, gt_path))
    return pairs


def build_confusion(pairs, num_clusters: int, eval_hw: tuple[int, int]) -> np.ndarray:
    conf = np.zeros((num_clusters, NUM_CLASSES), dtype=np.int64)
    for pred_path, gt_path in tqdm(pairs, desc="Building cluster/class table"):
        pred = np.array(Image.open(pred_path), dtype=np.uint8)
        gt_raw = np.array(Image.open(gt_path), dtype=np.uint8)
        gt = remap_labelids_to_cityscapes27(gt_raw)

        if pred.shape != eval_hw:
            pred = resize_nearest(pred, eval_hw)
        if gt.shape != eval_hw:
            gt = resize_nearest(gt, eval_hw)

        valid = (gt != IGNORE_LABEL) & (pred < num_clusters)
        joint = pred[valid].astype(np.int64) * NUM_CLASSES + gt[valid].astype(np.int64)
        counts = np.bincount(joint, minlength=num_clusters * NUM_CLASSES)
        conf += counts.reshape(num_clusters, NUM_CLASSES)
    return conf


def mapping_from_conf(conf: np.ndarray, mode: str) -> np.ndarray:
    num_clusters = conf.shape[0]
    lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    if mode == "majority":
        support = conf.sum(axis=1)
        active = support > 0
        lut[:num_clusters][active] = np.argmax(conf[active], axis=1).astype(np.uint8)
        return lut

    row_ind, col_ind = linear_sum_assignment(-conf)
    for r, c in zip(row_ind, col_ind):
        if conf[r, c] > 0:
            lut[r] = c
    return lut


def evaluate(pairs, lut: np.ndarray, eval_hw: tuple[int, int]):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for pred_path, gt_path in tqdm(pairs, desc="Evaluating mapped labels"):
        pred_raw = np.array(Image.open(pred_path), dtype=np.uint8)
        gt_raw = np.array(Image.open(gt_path), dtype=np.uint8)
        gt = remap_labelids_to_cityscapes27(gt_raw)

        if pred_raw.shape != eval_hw:
            pred_raw = resize_nearest(pred_raw, eval_hw)
        if gt.shape != eval_hw:
            gt = resize_nearest(gt, eval_hw)

        pred = lut[pred_raw]
        valid = (gt != IGNORE_LABEL) & (pred != IGNORE_LABEL)
        joint = gt[valid].astype(np.int64) * NUM_CLASSES + pred[valid].astype(np.int64)
        counts = np.bincount(joint, minlength=NUM_CLASSES * NUM_CLASSES)
        conf += counts.reshape(NUM_CLASSES, NUM_CLASSES)

    per_class = {}
    ious = []
    stuff_ious = []
    thing_ious = []
    for c, name in enumerate(CLASS_NAMES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else 0.0
        per_class[name] = round(iou * 100, 3)
        if conf[c, :].sum() > 0:
            ious.append(iou)
            if c in THING_IDS:
                thing_ious.append(iou)
            else:
                stuff_ious.append(iou)

    total = conf.sum()
    acc = float(np.trace(conf) / total) if total else 0.0
    return {
        "mIoU": round(float(np.mean(ious)) * 100, 3) if ious else 0.0,
        "mIoU_stuff": round(float(np.mean(stuff_ious)) * 100, 3) if stuff_ious else 0.0,
        "mIoU_things": round(float(np.mean(thing_ious)) * 100, 3) if thing_ious else 0.0,
        "pixel_accuracy": round(acc * 100, 3),
        "per_class_iou": per_class,
        "confusion": conf.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw clusters on Cityscapes-27")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--num_clusters", type=int, default=80)
    parser.add_argument("--mapping", choices=["majority", "hungarian"], default="majority")
    parser.add_argument("--eval_size", nargs=2, type=int, default=[512, 1024], metavar=("H", "W"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    eval_hw = (args.eval_size[0], args.eval_size[1])
    pairs = discover_pairs(pred_dir, gt_dir)
    print(f"Found {len(pairs)} prediction/GT pairs")
    if not pairs:
        raise SystemExit("No matching pairs found")

    cluster_conf = build_confusion(pairs, args.num_clusters, eval_hw)
    lut = mapping_from_conf(cluster_conf, args.mapping)
    mapped = int((lut[: args.num_clusters] != IGNORE_LABEL).sum())
    print(f"Mapped {mapped}/{args.num_clusters} clusters using {args.mapping}")

    results = evaluate(pairs, lut, eval_hw)
    results.update(
        {
            "num_images": len(pairs),
            "num_clusters": args.num_clusters,
            "mapping_mode": args.mapping,
            "cluster_to_cityscapes27": {
                str(i): (None if lut[i] == IGNORE_LABEL else int(lut[i]))
                for i in range(args.num_clusters)
            },
            "cluster_to_name": {
                str(i): (None if lut[i] == IGNORE_LABEL else CLASS_NAMES[int(lut[i])])
                for i in range(args.num_clusters)
            },
            "cluster_class_support": cluster_conf.tolist(),
        }
    )

    print("\nCityscapes-27 semantic metrics")
    print(f"  mIoU:       {results['mIoU']:.3f}")
    print(f"  mIoU stuff: {results['mIoU_stuff']:.3f}")
    print(f"  mIoU thing: {results['mIoU_things']:.3f}")
    print(f"  pAcc:       {results['pixel_accuracy']:.3f}")
    print("  Per-class IoU:")
    for name, iou in sorted(results["per_class_iou"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {name:15s} {iou:7.3f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
