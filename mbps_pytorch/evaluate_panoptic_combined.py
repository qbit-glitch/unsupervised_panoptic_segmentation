"""Evaluate panoptic quality: semantic pseudo-labels + depth-guided instance splitting.

Uses the SAME depth_guided_instances() logic from sweep_depthpro.py that produced
PQ=28.40 / PQ_things=23.35 with DepthPro depth.

Pipeline:
  1. Load semantic PNGs (19 Cityscapes trainIDs) — already mapped, no cluster LUT needed
  2. Load DepthPro depth NPY → Sobel gradients → split thing CCs at depth boundaries
  3. Build panoptic map: stuff from semantics, things from depth-guided CCs
  4. Evaluate PQ/SQ/RQ against Cityscapes GT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.sweep_depthpro import (
    depth_guided_instances,
    evaluate_panoptic_single,
    cc_only_instances,
    THING_IDS,
    STUFF_IDS,
    NUM_CLASSES,
    CLASS_NAMES,
)

# Alias for readability
CS_TRAINID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IGNORE_LABEL = 255

# Cityscapes labelId → trainId
CITYSCAPES_ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}


def resize_nearest(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize 2D array with nearest-neighbor interpolation."""
    return np.array(
        Image.fromarray(arr).resize((hw[1], hw[0]), Image.NEAREST),
        dtype=arr.dtype,
    )


def remap_gt_to_trainid(gt: np.ndarray) -> np.ndarray:
    """Remap Cityscapes labelId GT to trainId format."""
    out = np.full_like(gt, IGNORE_LABEL)
    for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        out[gt == label_id] = train_id
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate panoptic: semantic PNGs + depth-guided instance splitting"
    )
    parser.add_argument("--sem_dir", required=True,
                        help="Semantic prediction PNGs (city/stem.png), 19-class trainIDs")
    parser.add_argument("--depth_subdir", default="depth_depthpro",
                        help="Depth map subdir under cityscapes_root")
    parser.add_argument("--cityscapes_root", required=True,
                        help="Cityscapes root directory")
    parser.add_argument("--gt_dir", default=None,
                        help="GT dir (default: cityscapes_root/gtFine/val)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    # DepthPro winning params
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Depth gradient threshold (default: 0.01, DepthPro winner)")
    parser.add_argument("--min_area", type=int, default=1000,
                        help="Min instance area (default: 1000)")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Depth Gaussian blur sigma (default: 0.0)")
    parser.add_argument("--dilation", type=int, default=3,
                        help="Boundary reclamation dilation iterations (default: 3)")
    parser.add_argument("--no_depth", action="store_true",
                        help="CC-only baseline (no depth-guided splitting)")
    args = parser.parse_args()

    sem_path = Path(args.sem_dir)
    cs_root = Path(args.cityscapes_root)
    depth_root = cs_root / args.depth_subdir / "val"
    gt_path = Path(args.gt_dir) if args.gt_dir else cs_root / "gtFine" / "val"

    eval_hw = (512, 1024)
    H, W = eval_hw

    sem_files = sorted(sem_path.rglob("*.png"))
    logger.info("Found %d semantic predictions", len(sem_files))
    logger.info("Depth-guided splitting: tau=%.3f, min_area=%d, sigma=%.1f, dil=%d",
                args.tau, args.min_area, args.sigma, args.dilation)
    if args.no_depth:
        logger.info("CC-only mode (no depth-guided splitting)")

    # Accumulators
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    sem_inter = np.zeros(NUM_CLASSES)
    sem_union = np.zeros(NUM_CLASSES)
    total_correct = 0
    total_pixels = 0
    total_instances = 0

    for sem_file in tqdm(sem_files, desc="Evaluating"):
        city = sem_file.parent.name
        stem = sem_file.stem  # e.g., frankfurt_000000_000294

        # Load semantic prediction (already 19-class trainIDs)
        pred_sem = np.array(Image.open(sem_file), dtype=np.uint8)
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        # Generate instances via depth-guided splitting
        if not args.no_depth:
            depth_path = depth_root / city / f"{stem}.npy"
            if not depth_path.exists():
                # Try with _leftImg8bit suffix
                depth_path = depth_root / city / f"{stem}_leftImg8bit.npy"
            if depth_path.exists():
                depth = np.load(str(depth_path))
                if depth.shape != (H, W):
                    depth = np.array(
                        Image.fromarray(depth).resize((W, H), Image.BILINEAR)
                    )
                instances = depth_guided_instances(
                    pred_sem, depth, THING_IDS,
                    grad_threshold=args.tau,
                    min_area=args.min_area,
                    dilation_iters=args.dilation,
                    depth_blur_sigma=args.sigma,
                )
            else:
                logger.warning("No depth for %s, falling back to CC", stem)
                instances = cc_only_instances(pred_sem, THING_IDS, min_area=10)
        else:
            instances = cc_only_instances(pred_sem, THING_IDS, min_area=10)

        total_instances += len(instances)

        # Load GT
        gt_label_path = gt_path / city / f"{stem}_gtFine_labelIds.png"
        gt_inst_path = gt_path / city / f"{stem}_gtFine_instanceIds.png"
        gt_trainid_path = gt_path / city / f"{stem}_gtFine_labelTrainIds.png"

        if gt_trainid_path.exists():
            gt_sem = np.array(Image.open(gt_trainid_path), dtype=np.uint8)
        elif gt_label_path.exists():
            gt_sem = remap_gt_to_trainid(np.array(Image.open(gt_label_path), dtype=np.uint8))
        else:
            logger.warning("No GT for %s", stem)
            continue

        if gt_sem.shape != (H, W):
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                Image.fromarray(gt_inst_map).resize((W, H), Image.NEAREST)
            )

        # Semantic mIoU
        valid = gt_sem != IGNORE_LABEL
        for c in range(NUM_CLASSES):
            pred_c = pred_sem == c
            gt_c = gt_sem == c
            sem_inter[c] += (pred_c & gt_c & valid).sum()
            sem_union[c] += ((pred_c | gt_c) & valid).sum()
        total_correct += ((pred_sem == gt_sem) & valid).sum()
        total_pixels += valid.sum()

        # Panoptic eval (uses sweep_depthpro's evaluate_panoptic_single)
        tp, fp, fn, iou_s, n_inst = evaluate_panoptic_single(
            pred_sem, instances, gt_sem, gt_inst_map, eval_hw
        )
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        iou_acc += iou_s

    # --- Aggregate semantic ---
    iou = sem_inter / (sem_union + 1e-10)
    valid_classes = sem_union > 0
    miou = iou[valid_classes].mean()
    pixel_acc = total_correct / (total_pixels + 1e-10)

    # --- Aggregate panoptic ---
    def _agg(class_ids: set[int]) -> dict:
        active = [c for c in class_ids if tp_acc[c] + fn_acc[c] + fp_acc[c] > 0]
        if not active:
            return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "n_classes": 0}
        pqs, sqs, rqs = [], [], []
        for c in active:
            denom = tp_acc[c] + 0.5 * fp_acc[c] + 0.5 * fn_acc[c]
            pqs.append(iou_acc[c] / denom if denom > 0 else 0.0)
            sqs.append(iou_acc[c] / tp_acc[c] if tp_acc[c] > 0 else 0.0)
            rqs.append(tp_acc[c] / denom if denom > 0 else 0.0)
        return {
            "PQ": round(float(np.mean(pqs)) * 100, 2),
            "SQ": round(float(np.mean(sqs)) * 100, 2),
            "RQ": round(float(np.mean(rqs)) * 100, 2),
            "n_classes": len(active),
        }

    # --- Print results ---
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    mode = "CC-only" if args.no_depth else f"depth-guided (tau={args.tau}, sigma={args.sigma})"
    print(f"\n--- Semantic ({len(sem_files)} images) ---")
    print(f"  mIoU:           {miou * 100:.2f}%")
    print(f"  Pixel Accuracy: {pixel_acc * 100:.2f}%")
    print(f"\n  Per-class IoU:")
    for c in range(NUM_CLASSES):
        print(f"    {CS_TRAINID_TO_NAME[c]:15s}: {iou[c] * 100:.2f}%")

    panoptic_all = _agg(set(range(NUM_CLASSES)))
    panoptic_stuff = _agg(STUFF_IDS)
    panoptic_things = _agg(THING_IDS)

    print(f"\n--- Panoptic ({mode}) ---")
    print(f"  Instances/image: {total_instances / len(sem_files):.1f}")
    print(f"  ALL       PQ={panoptic_all['PQ']:.2f}%  SQ={panoptic_all['SQ']:.2f}%  "
          f"RQ={panoptic_all['RQ']:.2f}%  ({panoptic_all['n_classes']} classes)")
    print(f"  STUFF     PQ={panoptic_stuff['PQ']:.2f}%  SQ={panoptic_stuff['SQ']:.2f}%  "
          f"RQ={panoptic_stuff['RQ']:.2f}%  ({panoptic_stuff['n_classes']} classes)")
    print(f"  THINGS    PQ={panoptic_things['PQ']:.2f}%  SQ={panoptic_things['SQ']:.2f}%  "
          f"RQ={panoptic_things['RQ']:.2f}%  ({panoptic_things['n_classes']} classes)")

    print(f"\n  Per-class:")
    per_class = {}
    for c in range(NUM_CLASSES):
        name = CS_TRAINID_TO_NAME[c]
        tp_c, fp_c, fn_c = tp_acc[c], fp_acc[c], fn_acc[c]
        iou_c = iou_acc[c]
        denom = tp_c + 0.5 * fp_c + 0.5 * fn_c
        pq_val = iou_c / denom * 100 if denom > 0 else 0.0
        sq_val = iou_c / tp_c * 100 if tp_c > 0 else 0.0
        rq_val = tp_c / denom * 100 if denom > 0 else 0.0
        kind = "thing" if c in THING_IDS else "stuff"
        print(f"    {name:15s} [{kind:5s}]  PQ={pq_val:6.2f}  SQ={sq_val:6.2f}  "
              f"RQ={rq_val:6.2f}  TP={int(tp_c):5d}  FP={int(fp_c):5d}  FN={int(fn_c):5d}")
        per_class[name] = {
            "PQ": round(pq_val, 2), "SQ": round(sq_val, 2), "RQ": round(rq_val, 2),
            "TP": int(tp_c), "FP": int(fp_c), "FN": int(fn_c), "type": kind,
        }

    print("=" * 70)

    if args.output:
        results = {
            "semantic": {
                "mIoU": round(float(miou) * 100, 2),
                "pixel_accuracy": round(float(pixel_acc) * 100, 2),
                "per_class_iou": {
                    CS_TRAINID_TO_NAME[c]: round(float(iou[c]) * 100, 2)
                    for c in range(NUM_CLASSES)
                },
            },
            "panoptic": {
                "all": panoptic_all, "stuff": panoptic_stuff, "things": panoptic_things,
                "per_class": per_class,
            },
            "config": {
                "sem_dir": args.sem_dir,
                "mode": mode,
                "tau": args.tau, "min_area": args.min_area,
                "sigma": args.sigma, "dilation": args.dilation,
                "num_images": len(sem_files),
                "avg_instances_per_image": round(total_instances / len(sem_files), 1),
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
