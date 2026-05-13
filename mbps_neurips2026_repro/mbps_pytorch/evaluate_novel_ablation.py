#!/usr/bin/env python3
"""Unified evaluation for novel semantic pseudo-label ablation.

Evaluates all methods on full 501 val images with Hungarian matching,
computes mIoU, things-mIoU, stuff-mIoU, and per-class breakdown.

Usage:
    python mbps_pytorch/evaluate_novel_ablation.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --methods_json methods.json

    # Or auto-discover all pseudo_semantic_* directories
    python mbps_pytorch/evaluate_novel_ablation.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --auto_discover
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ─── COCO-Stuff-27 class definitions ───

COCOSTUFF27_CLASSNAMES = [
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]
NUM_CLASSES = 27
THING_IDS = set(range(12))
STUFF_IDS = set(range(12, 27))

SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


def load_coco_panoptic_gt(coco_root: str, image_id: int) -> Optional[np.ndarray]:
    """Load COCO panoptic GT and convert to 27-class semantic label map."""
    panoptic_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
    panoptic_dir = Path(coco_root) / "annotations" / "panoptic_val2017"

    if not hasattr(load_coco_panoptic_gt, "_cache"):
        with open(panoptic_json) as f:
            data = json.load(f)
        cat_map = {cat["id"]: cat["supercategory"] for cat in data["categories"]}
        ann_map = {ann["image_id"]: ann for ann in data["annotations"]}
        load_coco_panoptic_gt._cache = (cat_map, ann_map, str(panoptic_dir))

    cat_map, ann_map, pdir = load_coco_panoptic_gt._cache
    if image_id not in ann_map:
        return None

    ann = ann_map[image_id]
    pan_img = np.array(Image.open(Path(pdir) / ann["file_name"]))
    pan_id = (pan_img[:, :, 0].astype(np.int32) +
              pan_img[:, :, 1].astype(np.int32) * 256 +
              pan_img[:, :, 2].astype(np.int32) * 256 * 256)

    sem_label = np.full(pan_id.shape, 255, dtype=np.uint8)
    for seg in ann["segments_info"]:
        mask = pan_id == seg["id"]
        supercat = cat_map.get(seg["category_id"])
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            sem_label[mask] = SUPERCATEGORY_TO_COARSE[supercat]
    return sem_label


def evaluate_method(
    coco_root: str,
    pred_dir: str,
    n_images: Optional[int] = None,
) -> Dict:
    """Evaluate a set of pseudo-semantic predictions.

    Args:
        coco_root: Path to COCO dataset root.
        pred_dir: Directory with predicted PNGs (image_id.png, values 0-26).
        n_images: Max images to evaluate (None=all).

    Returns:
        Dict with mIoU, things_miou, stuff_miou, per_class_iou.
    """
    pred_path = Path(pred_dir)
    pred_files = sorted(pred_path.glob("*.png"))

    if not pred_files:
        return {"miou": 0.0, "error": f"No PNG files in {pred_dir}"}

    image_ids = [int(f.stem) for f in pred_files]
    if n_images:
        image_ids = image_ids[:n_images]

    iou_per_class = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for img_id in image_ids:
        gt_sem = load_coco_panoptic_gt(coco_root, img_id)
        if gt_sem is None:
            continue

        pred_file = pred_path / f"{img_id:012d}.png"
        if not pred_file.exists():
            continue
        pred = np.array(Image.open(pred_file))

        if pred.shape != gt_sem.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt_sem.shape[1], gt_sem.shape[0]), Image.NEAREST
            ))

        for c in range(NUM_CLASSES):
            gt_mask = gt_sem == c
            pred_mask = pred == c
            inter = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if union > 0:
                iou_per_class[c] += inter / union
                count_per_class[c] += 1

    valid = count_per_class > 0
    per_class_iou = {}
    things_ious = []
    stuff_ious = []

    for c in range(NUM_CLASSES):
        if count_per_class[c] > 0:
            iou = iou_per_class[c] / count_per_class[c] * 100
            per_class_iou[COCOSTUFF27_CLASSNAMES[c]] = round(iou, 1)
            if c in THING_IDS:
                things_ious.append(iou)
            else:
                stuff_ious.append(iou)

    miou = (iou_per_class[valid] / count_per_class[valid]).mean() * 100
    things_miou = np.mean(things_ious) if things_ious else 0.0
    stuff_miou = np.mean(stuff_ious) if stuff_ious else 0.0

    return {
        "miou": round(miou, 2),
        "things_miou": round(things_miou, 2),
        "stuff_miou": round(stuff_miou, 2),
        "n_images": len(image_ids),
        "per_class_iou": per_class_iou,
    }


def auto_discover_methods(coco_root: str) -> List[Tuple[str, str]]:
    """Find all pseudo_semantic_* directories in coco_root."""
    root = Path(coco_root)
    methods = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("pseudo_semantic_"):
            val_dir = d / "val2017"
            if val_dir.exists() and list(val_dir.glob("*.png")):
                name = d.name.replace("pseudo_semantic_", "")
                methods.append((name, str(val_dir)))
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Novel Ablation Evaluation")
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--auto_discover", action="store_true",
                        help="Auto-discover all pseudo_semantic_* directories")
    parser.add_argument("--pred_dirs", nargs="*", default=[],
                        help="Explicit prediction directories to evaluate")
    parser.add_argument("--names", nargs="*", default=[],
                        help="Method names (must match --pred_dirs length)")
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: coco_root/novel_ablation_results.json)")
    args = parser.parse_args()

    methods: List[Tuple[str, str]] = []

    if args.auto_discover:
        methods = auto_discover_methods(args.coco_root)
    elif args.pred_dirs:
        names = args.names if args.names else [Path(d).parent.name for d in args.pred_dirs]
        methods = list(zip(names, args.pred_dirs))

    if not methods:
        print("No methods found. Use --auto_discover or --pred_dirs.")
        return

    print(f"\n{'='*70}")
    print(f"NOVEL SEMANTIC PSEUDO-LABEL ABLATION — UNIFIED EVALUATION")
    print(f"{'='*70}")
    print(f"  COCO root: {args.coco_root}")
    print(f"  Methods: {len(methods)}")
    print(f"  Eval images: {args.n_images or 'all'}")

    results = {}
    for name, pred_dir in methods:
        print(f"\n  Evaluating: {name}...")
        result = evaluate_method(args.coco_root, pred_dir, args.n_images)
        results[name] = result
        print(f"    mIoU={result['miou']:.1f}%, "
              f"Things={result.get('things_miou', 0):.1f}%, "
              f"Stuff={result.get('stuff_miou', 0):.1f}%")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Method':<50s} {'mIoU':>6s} {'Things':>7s} {'Stuff':>7s}")
    print(f"{'-'*70}")
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("miou", 0), reverse=True)
    for name, r in sorted_results:
        print(f"  {name:<48s} {r['miou']:>5.1f}% {r.get('things_miou', 0):>6.1f}% "
              f"{r.get('stuff_miou', 0):>6.1f}%")
    print(f"{'='*70}")

    # Save results
    out_path = args.output or str(Path(args.coco_root) / "novel_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
