#!/usr/bin/env python3
"""Evaluate pseudo-labels with Hungarian cluster-to-trainID mapping.

CUPS pseudo-labels use raw cluster IDs (0-79). This script computes
an optimal global mapping to Cityscapes trainIDs via Hungarian algorithm,
then evaluates panoptic quality.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mbps_pytorch.evaluation.hungarian_matching import hungarian_match, compute_miou
from mbps_pytorch.sweep_depthpro import (
    depth_guided_instances,
    evaluate_panoptic_single,
    cc_only_instances,
    THING_IDS,
    NUM_CLASSES,
)

IGNORE_LABEL = 255

CITYSCAPES_ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}


def resize_nearest(arr, hw):
    return np.array(Image.fromarray(arr).resize((hw[1], hw[0]), Image.NEAREST), dtype=arr.dtype)


def load_gt(city, stem, gt_path, eval_hw=(512, 1024)):
    H, W = eval_hw
    gt_trainid_path = gt_path / city / f"{stem}_gtFine_labelTrainIds.png"
    gt_label_path = gt_path / city / f"{stem}_gtFine_labelIds.png"
    gt_inst_path = gt_path / city / f"{stem}_gtFine_instanceIds.png"

    if gt_trainid_path.exists():
        gt_sem = np.array(Image.open(gt_trainid_path), dtype=np.uint8)
        if gt_sem.shape != (H, W):
            gt_sem = resize_nearest(gt_sem, eval_hw)
    elif gt_label_path.exists():
        gt_raw = np.array(Image.open(gt_label_path), dtype=np.uint8)
        if gt_raw.shape != (H, W):
            gt_raw = resize_nearest(gt_raw, eval_hw)
        gt_sem = np.full((H, W), IGNORE_LABEL, dtype=np.uint8)
        for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
            gt_sem[gt_raw == label_id] = train_id
    else:
        return None, None

    # Already resized above

    gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
    if gt_inst.shape != (H, W):
        gt_inst = np.array(Image.fromarray(gt_inst).resize((W, H), Image.NEAREST))

    return gt_sem, gt_inst


def compute_global_mapping(sem_files, gt_path, num_clusters=80, max_images=500):
    """Compute global cluster-to-trainID mapping using all val images."""
    all_pred = []
    all_gt = []

    for sem_file in tqdm(sem_files[:max_images], desc="Computing global mapping"):
        # Extract city from filename stem (CUPS-format flat dirs)
        stem = sem_file.stem.replace("_semantic", "").replace("_leftImg8bit", "")
        city = stem.split('_')[0]

        gt_sem, _ = load_gt(city, stem, gt_path)
        if gt_sem is None:
            continue

        pred_sem = np.array(Image.open(sem_file), dtype=np.uint8)
        if pred_sem.shape != gt_sem.shape:
            pred_sem = resize_nearest(pred_sem, gt_sem.shape)

        valid = gt_sem != IGNORE_LABEL
        all_pred.append(pred_sem[valid].flatten())
        all_gt.append(gt_sem[valid].flatten())

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)

    # Only keep valid trainIDs (0-18)
    valid_gt = (all_gt >= 0) & (all_gt < NUM_CLASSES)
    all_pred = all_pred[valid_gt]
    all_gt = all_gt[valid_gt]

    mapping, acc = hungarian_match(all_pred, all_gt, num_clusters, NUM_CLASSES)
    print(f"Global mapping accuracy: {acc*100:.2f}%")
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem_dir", type=Path, required=True)
    parser.add_argument("--cityscapes_root", type=Path, required=True)
    parser.add_argument("--depth_subdir", default="depth_depthpro")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_clusters", type=int, default=80)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--no_depth", action="store_true")
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    gt_path = cs_root / "gtFine" / args.split
    depth_root = cs_root / args.depth_subdir / args.split

    sem_files = sorted(args.sem_dir.rglob("*_semantic.png"))
    if args.split == "val":
        val_cities = {"frankfurt", "lindau", "munster"}
        sem_files = [f for f in sem_files if any(c in f.name for c in val_cities)]
    print(f"Found {len(sem_files)} semantic files")

    if len(sem_files) == 0:
        print("No semantic files found!")
        sys.exit(1)

    # Compute global mapping
    mapping = compute_global_mapping(sem_files, gt_path, args.num_clusters, args.max_images)
    print(f"Mapping: {mapping}")

    # Evaluate
    eval_hw = (512, 1024)
    H, W = eval_hw

    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    sem_inter = np.zeros(NUM_CLASSES)
    sem_union = np.zeros(NUM_CLASSES)
    total_correct = 0
    total_pixels = 0

    for sem_file in tqdm(sem_files, desc="Evaluating"):
        stem_full = sem_file.stem.replace("_semantic", "")
        stem = stem_full.replace("_leftImg8bit", "")
        city = stem.split('_')[0]

        pred_sem_raw = np.array(Image.open(sem_file), dtype=np.uint8)
        if pred_sem_raw.shape != (H, W):
            pred_sem_raw = resize_nearest(pred_sem_raw, eval_hw)

        # Apply mapping
        pred_sem = np.full_like(pred_sem_raw, IGNORE_LABEL)
        for pred_c, gt_c in mapping.items():
            pred_sem[pred_sem_raw == pred_c] = gt_c

        # Instances
        if not args.no_depth:
            depth_path = depth_root / city / f"{stem}.npy"
            if not depth_path.exists():
                depth_path = depth_root / city / f"{stem}_leftImg8bit.npy"
            if depth_path.exists():
                depth = np.load(str(depth_path))
                if depth.shape != (H, W):
                    depth = np.array(Image.fromarray(depth).resize((W, H), Image.BILINEAR))
                instances = depth_guided_instances(
                    pred_sem, depth, THING_IDS,
                    grad_threshold=args.tau,
                    min_area=args.min_area,
                    dilation_iters=args.dilation,
                )
            else:
                instances = cc_only_instances(pred_sem, THING_IDS, min_area=10)
        else:
            instances = cc_only_instances(pred_sem, THING_IDS, min_area=10)

        gt_sem, gt_inst = load_gt(city, stem, gt_path, eval_hw)
        if gt_sem is None:
            continue

        valid = gt_sem != IGNORE_LABEL
        for c in range(NUM_CLASSES):
            pred_c = pred_sem == c
            gt_c = gt_sem == c
            sem_inter[c] += (pred_c & gt_c & valid).sum()
            sem_union[c] += ((pred_c | gt_c) & valid).sum()
        total_correct += ((pred_sem == gt_sem) & valid).sum()
        total_pixels += valid.sum()

        tp, fp, fn, iou_s, n_inst = evaluate_panoptic_single(
            pred_sem, instances, gt_sem, gt_inst, eval_hw
        )
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        iou_acc += iou_s

    # Compute metrics
    pq_per_class = np.zeros(NUM_CLASSES)
    sq_per_class = np.zeros(NUM_CLASSES)
    rq_per_class = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        tp = tp_acc[c]
        fp = fp_acc[c]
        fn = fn_acc[c]
        iou = iou_acc[c]
        if tp + fp + fn > 0:
            sq = iou / tp if tp > 0 else 0
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            pq = sq * rq
            pq_per_class[c] = pq
            sq_per_class[c] = sq
            rq_per_class[c] = rq

    stuff_mask = np.array([c in {0,1,2,3,4,5,6,7,8,9,10} for c in range(NUM_CLASSES)])
    things_mask = ~stuff_mask

    valid_classes = (tp_acc + fp_acc + fn_acc) > 0

    results = {
        "mapping": {int(k): int(v) for k, v in mapping.items()},
        "semantic": {
            "mIoU": float(np.mean(sem_inter[valid_classes] / (sem_union[valid_classes] + 1e-10))) * 100,
            "pixel_accuracy": float(total_correct / (total_pixels + 1e-10)) * 100,
        },
        "panoptic": {
            "all": {
                "PQ": float(np.mean(pq_per_class[valid_classes])) * 100,
                "SQ": float(np.mean(sq_per_class[valid_classes])) * 100,
                "RQ": float(np.mean(rq_per_class[valid_classes])) * 100,
            },
            "stuff": {
                "PQ": float(np.mean(pq_per_class[stuff_mask & valid_classes])) * 100,
                "SQ": float(np.mean(sq_per_class[stuff_mask & valid_classes])) * 100,
                "RQ": float(np.mean(rq_per_class[stuff_mask & valid_classes])) * 100,
            },
            "things": {
                "PQ": float(np.mean(pq_per_class[things_mask & valid_classes])) * 100,
                "SQ": float(np.mean(sq_per_class[things_mask & valid_classes])) * 100,
                "RQ": float(np.mean(rq_per_class[things_mask & valid_classes])) * 100,
            },
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
