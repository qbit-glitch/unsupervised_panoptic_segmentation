#!/usr/bin/env python3
"""Diagnose depth-guided instance splitting: oracle ceiling, edge alignment, failure taxonomy.

Phase 0 of the monocular depth ablation study. Answers:
  1. Oracle ceiling: what PQ_things can a perfect edge map achieve?
  2. Edge alignment: how well do depth Sobel edges match GT instance boundaries?
  3. Failure taxonomy: why do person instances fail? (co-planar, over-split, semantic miss, etc.)

Usage:
    python mbps_pytorch/diagnose_depth_splitting.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --depth_subdir depth_spidepth \
        --semantic_subdir pseudo_semantic_mapped_k80

    # Quick test on 10 images:
    python mbps_pytorch/diagnose_depth_splitting.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --max_images 10

    # Compare multiple depth sources:
    python mbps_pytorch/diagnose_depth_splitting.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --depth_subdir depth_spidepth depth_dav3
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel, binary_dilation
from tqdm import tqdm

# ─── Cityscapes Constants ───

CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}
NUM_CLASSES = 19
STUFF_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
WORK_H, WORK_W = 512, 1024
BOUNDARY_TOLERANCE_PX = 3  # Pixels tolerance for edge alignment


def remap_gt_to_trainids(gt_raw):
    out = np.full_like(gt_raw, 255, dtype=np.uint8)
    for cs_id, train_id in CS_ID_TO_TRAIN.items():
        out[gt_raw == cs_id] = train_id
    return out


def resize_nearest(arr, hw):
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def resize_bilinear(arr, hw):
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


# ─── Core Analysis Functions ───

def compute_gt_instance_boundaries(gt_inst_map):
    """Extract binary boundary map from GT instance IDs.

    A pixel is a boundary if any of its 4-connected neighbors has a different instance ID.
    Returns binary mask (H, W) where True = boundary pixel.
    """
    h, w = gt_inst_map.shape
    boundary = np.zeros((h, w), dtype=bool)

    # Shift in 4 directions and compare
    boundary[:-1, :] |= gt_inst_map[:-1, :] != gt_inst_map[1:, :]   # down
    boundary[1:, :]  |= gt_inst_map[1:, :]  != gt_inst_map[:-1, :]  # up
    boundary[:, :-1] |= gt_inst_map[:, :-1] != gt_inst_map[:, 1:]   # right
    boundary[:, 1:]  |= gt_inst_map[:, 1:]  != gt_inst_map[:, :-1]  # left

    return boundary


def compute_gt_thing_boundaries(gt_inst_map, gt_sem):
    """Extract boundaries ONLY between different thing instances of the same class.

    These are the boundaries that depth splitting must detect.
    Excludes stuff-thing boundaries and different-class boundaries.
    """
    h, w = gt_inst_map.shape
    boundary = np.zeros((h, w), dtype=bool)

    for dy, dx in [(0, 1), (1, 0)]:
        # Compare adjacent pixels
        y1, y2 = slice(None, -dy or None), slice(dy or None, None)
        x1, x2 = slice(None, -dx or None), slice(dx or None, None)

        a_inst = gt_inst_map[y1, x1]
        b_inst = gt_inst_map[y2, x2]

        a_cls = gt_sem[y1, x1]
        b_cls = gt_sem[y2, x2]

        # Same class, different instance → boundary we need to detect
        same_class = (a_cls == b_cls) & (a_cls != 255)
        diff_instance = a_inst != b_inst
        is_thing = np.isin(a_cls, list(THING_IDS))

        edge = same_class & diff_instance & is_thing
        boundary[y1, x1] |= edge
        boundary[y2, x2] |= edge

    return boundary


def compute_depth_edges(depth, grad_threshold=0.20, depth_blur_sigma=1.0):
    """Compute binary depth edge map using Sobel gradients."""
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return grad_mag > grad_threshold, grad_mag


def oracle_edges_from_gt(gt_inst_map, gt_sem):
    """Create perfect edge map from GT: edges between different same-class instances."""
    return compute_gt_thing_boundaries(gt_inst_map, gt_sem)


def dilate_mask(mask, px):
    """Dilate a binary mask by px pixels."""
    if px <= 0:
        return mask
    return binary_dilation(mask, iterations=px)


def compute_edge_alignment(depth_edges, gt_boundaries, tolerance_px=BOUNDARY_TOLERANCE_PX):
    """Compute edge recall, precision, and F1 with pixel tolerance.

    Recall: fraction of GT boundary pixels within tolerance of a depth edge
    Precision: fraction of depth edge pixels within tolerance of a GT boundary
    """
    if gt_boundaries.sum() == 0 and depth_edges.sum() == 0:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0,
                "gt_boundary_px": 0, "depth_edge_px": 0}
    if gt_boundaries.sum() == 0:
        return {"recall": 1.0, "precision": 0.0, "f1": 0.0,
                "gt_boundary_px": 0, "depth_edge_px": int(depth_edges.sum())}
    if depth_edges.sum() == 0:
        return {"recall": 0.0, "precision": 1.0, "f1": 0.0,
                "gt_boundary_px": int(gt_boundaries.sum()), "depth_edge_px": 0}

    # Dilate both for tolerance
    gt_dilated = dilate_mask(gt_boundaries, tolerance_px)
    depth_dilated = dilate_mask(depth_edges, tolerance_px)

    # Recall: GT boundary pixels covered by dilated depth edges
    recall = float(np.sum(gt_boundaries & depth_dilated)) / float(gt_boundaries.sum())

    # Precision: depth edge pixels covered by dilated GT boundaries
    precision = float(np.sum(depth_edges & gt_dilated)) / float(depth_edges.sum())

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "gt_boundary_px": int(gt_boundaries.sum()),
        "depth_edge_px": int(depth_edges.sum()),
    }


def compute_coplanar_separation_rate(gt_inst_map, gt_sem, depth_edges, tolerance_px=3):
    """For adjacent same-class GT instance pairs, compute fraction with a depth edge between them.

    Returns per-class rates and overall rate.
    """
    per_class = defaultdict(lambda: {"total_pairs": 0, "separated": 0})

    for cls in sorted(THING_IDS):
        cls_mask = gt_sem == cls
        if cls_mask.sum() == 0:
            continue

        # Get all unique GT instance IDs for this class
        unique_insts = []
        for uid in np.unique(gt_inst_map[cls_mask]):
            if uid < 1000:
                continue
            raw_cls = uid // 1000
            if raw_cls not in CS_ID_TO_TRAIN:
                continue
            if CS_ID_TO_TRAIN[raw_cls] != cls:
                continue
            unique_insts.append(uid)

        if len(unique_insts) < 2:
            continue

        # Check each pair of instances for adjacency and depth edge separation
        inst_masks = {}
        for uid in unique_insts:
            inst_masks[uid] = gt_inst_map == uid

        for i, uid_a in enumerate(unique_insts):
            for uid_b in unique_insts[i+1:]:
                mask_a = inst_masks[uid_a]
                mask_b = inst_masks[uid_b]

                # Check adjacency: dilate A, check overlap with B
                a_dilated = dilate_mask(mask_a, tolerance_px)
                if not np.any(a_dilated & mask_b):
                    continue  # Not adjacent

                per_class[cls]["total_pairs"] += 1

                # Check if there's a depth edge between them
                # Look at the boundary region between the two instances
                boundary_region = a_dilated & dilate_mask(mask_b, tolerance_px)
                if boundary_region.sum() > 0 and depth_edges[boundary_region].sum() > 0:
                    edge_density = depth_edges[boundary_region].mean()
                    if edge_density > 0.1:  # At least 10% of boundary region has edges
                        per_class[cls]["separated"] += 1

    results = {}
    total_pairs = 0
    total_separated = 0
    for cls in sorted(THING_IDS):
        data = per_class[cls]
        if data["total_pairs"] > 0:
            rate = data["separated"] / data["total_pairs"]
            results[CLASS_NAMES[cls]] = {
                "total_pairs": data["total_pairs"],
                "separated": data["separated"],
                "separation_rate": round(rate, 4),
            }
            total_pairs += data["total_pairs"]
            total_separated += data["separated"]

    overall_rate = total_separated / max(total_pairs, 1)
    results["_overall"] = {
        "total_pairs": total_pairs,
        "separated": total_separated,
        "separation_rate": round(overall_rate, 4),
    }
    return results


# ─── Instance Generation (reused from sweep_k50_spidepth.py) ───

def depth_guided_instances(semantic, depth_edges, thing_ids=THING_IDS,
                           min_area=100, dilation_iters=3):
    """Split thing regions using pre-computed depth edge map."""
    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue
        split_mask = cls_mask & ~depth_edges
        labeled, n_cc = ndimage.label(split_mask)

        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                cc_list.append((cc_id, cc_mask, area))
        cc_list.sort(key=lambda x: -x[2])

        for cc_id, cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = binary_dilation(cc_mask, iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask
            final_area = float(final_mask.sum())
            if final_area < min_area:
                continue
            assigned |= final_mask
            instances.append((final_mask, cls, final_area))

    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]
    return instances


# ─── Panoptic Evaluation (from sweep_k50_spidepth.py) ───

def evaluate_panoptic_single(pred_sem, pred_instances, gt_sem, gt_inst_map, eval_hw):
    """Evaluate one image. Returns per-class (tp, fp, fn, iou) arrays."""
    H, W = eval_hw

    pred_pan = np.zeros((H, W), dtype=np.int32)
    pred_segments = {}
    next_id = 1

    for cls in STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pred_pan[mask] = next_id
        pred_segments[next_id] = cls
        next_id += 1

    for mask, cls, score in pred_instances:
        if cls not in THING_IDS:
            continue
        new_pixels = mask & (pred_pan == 0)
        if new_pixels.sum() < 10:
            continue
        pred_pan[new_pixels] = next_id
        pred_segments[next_id] = cls
        next_id += 1

    gt_pan = np.zeros((H, W), dtype=np.int32)
    gt_segments = {}
    gt_next_id = 1

    for cls in STUFF_IDS:
        mask = gt_sem == cls
        if mask.sum() < 64:
            continue
        gt_pan[mask] = gt_next_id
        gt_segments[gt_next_id] = cls
        gt_next_id += 1

    for uid in np.unique(gt_inst_map):
        if uid < 1000:
            continue
        raw_cls = uid // 1000
        if raw_cls not in CS_ID_TO_TRAIN:
            continue
        train_id = CS_ID_TO_TRAIN[raw_cls]
        if train_id not in THING_IDS:
            continue
        mask = gt_inst_map == uid
        if mask.sum() < 10:
            continue
        gt_pan[mask] = gt_next_id
        gt_segments[gt_next_id] = train_id
        gt_next_id += 1

    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)
    iou_sum = np.zeros(NUM_CLASSES)
    matched_pred = set()

    gt_by_cat = defaultdict(list)
    for seg_id, cat in gt_segments.items():
        gt_by_cat[cat].append(seg_id)
    pred_by_cat = defaultdict(list)
    for seg_id, cat in pred_segments.items():
        pred_by_cat[cat].append(seg_id)

    for cat in range(NUM_CLASSES):
        gt_segs = gt_by_cat.get(cat, [])
        pred_segs = pred_by_cat.get(cat, [])
        if not gt_segs and not pred_segs:
            continue

        for gt_id in gt_segs:
            gt_mask = gt_pan == gt_id
            best_iou = 0.0
            best_pred = None
            for pred_id in pred_segs:
                if pred_id in matched_pred:
                    continue
                pred_mask = pred_pan == pred_id
                inter = np.sum(gt_mask & pred_mask)
                union = np.sum(gt_mask | pred_mask)
                if union == 0:
                    continue
                iou_val = inter / union
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred = pred_id
            if best_iou > 0.5 and best_pred is not None:
                tp[cat] += 1
                iou_sum[cat] += best_iou
                matched_pred.add(best_pred)
            else:
                fn[cat] += 1

        for pred_id in pred_segs:
            if pred_id not in matched_pred:
                fp[cat] += 1

    return tp, fp, fn, iou_sum


def compute_pq(tp, fp, fn, iou_sum):
    """Compute PQ/SQ/RQ from accumulators."""
    all_pq, stuff_pq, thing_pq = [], [], []
    per_class = {}

    for c in range(NUM_CLASSES):
        t, f_p, f_n = tp[c], fp[c], fn[c]
        if t + f_p + f_n > 0:
            sq = iou_sum[c] / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0

        per_class[CLASS_NAMES[c]] = {
            "PQ": round(pq * 100, 2), "SQ": round(sq * 100, 2),
            "RQ": round(rq * 100, 2), "TP": int(t), "FP": int(f_p), "FN": int(f_n),
        }
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            if c in STUFF_IDS:
                stuff_pq.append(pq)
            else:
                thing_pq.append(pq)

    return {
        "PQ": round(float(np.mean(all_pq)) * 100, 2) if all_pq else 0.0,
        "PQ_stuff": round(float(np.mean(stuff_pq)) * 100, 2) if stuff_pq else 0.0,
        "PQ_things": round(float(np.mean(thing_pq)) * 100, 2) if thing_pq else 0.0,
        "per_class": per_class,
    }


# ─── Failure Taxonomy ───

def classify_person_failures(gt_inst_map, gt_sem, pred_sem, depth_edges,
                             instances, eval_hw):
    """Classify each person GT instance into: matched, co-planar_merge, over_split,
    semantic_miss, edge_obliteration.

    Returns dict with counts per category.
    """
    H, W = eval_hw
    person_cls = 11  # trainID for person

    # Build predicted instance map for persons
    pred_person_masks = []
    for mask, cls, score in instances:
        if cls == person_cls:
            pred_person_masks.append(mask)

    # Get GT person instances
    gt_person_insts = []
    for uid in np.unique(gt_inst_map):
        if uid < 1000:
            continue
        raw_cls = uid // 1000
        if raw_cls not in CS_ID_TO_TRAIN:
            continue
        if CS_ID_TO_TRAIN[raw_cls] != person_cls:
            continue
        inst_mask = gt_inst_map == uid
        if inst_mask.sum() < 10:
            continue
        gt_person_insts.append((uid, inst_mask))

    results = {
        "total_gt": len(gt_person_insts),
        "matched": 0,
        "co_planar_merge": 0,
        "over_split": 0,
        "semantic_miss": 0,
        "edge_obliteration": 0,
    }

    for uid, gt_mask in gt_person_insts:
        gt_area = gt_mask.sum()

        # Check 1: Is the GT instance covered by person semantics?
        semantic_overlap = np.sum(gt_mask & (pred_sem == person_cls)) / gt_area
        if semantic_overlap < 0.3:
            results["semantic_miss"] += 1
            continue

        # Check 2: Is the GT instance obliterated by depth edges?
        gt_minus_edges = gt_mask & ~depth_edges
        if gt_minus_edges.sum() < 50:  # Almost entirely consumed by edges
            results["edge_obliteration"] += 1
            continue

        # Check 3: Does any predicted instance match (IoU > 0.5)?
        matched = False
        for pred_mask in pred_person_masks:
            inter = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)
            if union > 0 and inter / union > 0.5:
                matched = True
                break

        if matched:
            results["matched"] += 1
            continue

        # Check 4: Is this a co-planar merge or over-split?
        # Co-planar merge: GT instance shares a predicted CC with another GT instance
        # Over-split: GT instance is split into multiple small predicted instances
        fragments = 0
        total_overlap = 0
        for pred_mask in pred_person_masks:
            overlap = np.sum(gt_mask & pred_mask)
            if overlap > 0:
                fragments += 1
                total_overlap += overlap

        if fragments > 1:
            results["over_split"] += 1
        else:
            # Single or no matching prediction → co-planar merge
            results["co_planar_merge"] += 1

    return results


# ─── Data Discovery ───

def discover_files(cityscapes_root, split, semantic_subdir, depth_subdir):
    """Find matching (semantic, depth, gt_label, gt_instance) file paths."""
    root = Path(cityscapes_root)
    sem_dir = root / semantic_subdir / split
    depth_dir = root / depth_subdir / split
    gt_dir = root / "gtFine" / split

    gt_label_files = sorted(gt_dir.rglob("*_gtFine_labelIds.png"))
    files = []

    for gt_label_path in gt_label_files:
        rel = gt_label_path.relative_to(gt_dir)
        base = str(rel).replace("_gtFine_labelIds.png", "")
        city = base.split("/")[0]
        stem = base.split("/")[-1]

        gt_inst_path = gt_dir / (base + "_gtFine_instanceIds.png")
        if not gt_inst_path.exists():
            continue

        # Try both naming conventions for semantic
        sem_path = sem_dir / city / f"{stem}.png"
        if not sem_path.exists():
            sem_path = sem_dir / city / f"{stem}_leftImg8bit.png"
        if not sem_path.exists():
            continue

        # Try both naming conventions for depth
        depth_path = depth_dir / city / f"{stem}.npy"
        if not depth_path.exists():
            depth_path = depth_dir / city / f"{stem}_leftImg8bit.npy"
        if not depth_path.exists():
            # Allow running without depth (oracle mode only)
            depth_path = None

        files.append((sem_path, depth_path, gt_label_path, gt_inst_path))

    return files


# ─── Main Analysis ───

def run_analysis(cityscapes_root, split, semantic_subdir, depth_subdirs,
                 grad_threshold, min_area, max_images=None):
    """Run all three diagnostic analyses."""
    root = Path(cityscapes_root)
    eval_hw = (WORK_H, WORK_W)

    # Load cluster → trainID mapping (for overclustered semantics)
    centroids_path = root / semantic_subdir / "kmeans_centroids.npz"
    if centroids_path.exists():
        data = np.load(centroids_path)
        cluster_to_class = data["cluster_to_class"]
        k = len(cluster_to_class)
        lut = np.full(256, 255, dtype=np.uint8)
        for cid, tid in enumerate(cluster_to_class):
            lut[cid] = int(tid)
        print(f"  Loaded k={k} cluster→trainID mapping from {centroids_path}")
    else:
        lut = None
        print(f"  No centroids found — assuming semantic labels are already trainIDs")

    # Discover files using first depth subdir
    primary_depth = depth_subdirs[0] if depth_subdirs else "depth_spidepth"
    files = discover_files(root, split, semantic_subdir, primary_depth)
    print(f"  Found {len(files)} images in {split} split")

    if max_images:
        files = files[:max_images]
        print(f"  Limited to {max_images} images")

    # ═══════════════════════════════════════════════
    # Analysis 1: Oracle Depth Experiment
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 1: ORACLE DEPTH EXPERIMENT")
    print(f"{'='*70}")
    print("  Using GT instance boundaries as perfect depth edges...")

    oracle_tp = np.zeros(NUM_CLASSES)
    oracle_fp = np.zeros(NUM_CLASSES)
    oracle_fn = np.zeros(NUM_CLASSES)
    oracle_iou = np.zeros(NUM_CLASSES)

    for sem_path, depth_path, gt_label_path, gt_inst_path in tqdm(files, desc="Oracle"):
        # Load semantic
        pred_raw = np.array(Image.open(sem_path))
        pred_sem = lut[pred_raw] if lut is not None else pred_raw
        if pred_sem.shape != eval_hw:
            pred_sem = resize_nearest(pred_sem, eval_hw)

        # Load GT
        gt_raw = np.array(Image.open(gt_label_path))
        gt_sem = remap_gt_to_trainids(gt_raw)
        if gt_sem.shape != eval_hw:
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != eval_hw:
            gt_inst_map = resize_nearest(gt_inst_map.astype(np.uint16), eval_hw).astype(np.int32)

        # Oracle edges: perfect boundaries between same-class GT instances
        oracle_edges = oracle_edges_from_gt(gt_inst_map, gt_sem)

        # Generate instances using oracle edges
        instances = depth_guided_instances(
            pred_sem, oracle_edges, THING_IDS,
            min_area=min_area, dilation_iters=3,
        )

        tp, fp, fn, iou_s = evaluate_panoptic_single(
            pred_sem, instances, gt_sem, gt_inst_map, eval_hw,
        )
        oracle_tp += tp
        oracle_fp += fp
        oracle_fn += fn
        oracle_iou += iou_s

    oracle_metrics = compute_pq(oracle_tp, oracle_fp, oracle_fn, oracle_iou)
    print(f"\n  Oracle PQ:        {oracle_metrics['PQ']:6.2f}")
    print(f"  Oracle PQ_stuff:  {oracle_metrics['PQ_stuff']:6.2f}")
    print(f"  Oracle PQ_things: {oracle_metrics['PQ_things']:6.2f}")
    print(f"\n  Per-class (things only):")
    for cls in sorted(THING_IDS):
        name = CLASS_NAMES[cls]
        v = oracle_metrics["per_class"][name]
        if v["TP"] + v["FP"] + v["FN"] > 0:
            print(f"    {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                  f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

    # ═══════════════════════════════════════════════
    # Analysis 2: Edge-Boundary Alignment (per depth source)
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 2: EDGE-BOUNDARY ALIGNMENT")
    print(f"{'='*70}")

    depth_metrics = {}

    for depth_subdir in depth_subdirs:
        print(f"\n  --- Depth source: {depth_subdir} (tau={grad_threshold}) ---")

        # Re-discover files with this depth source
        depth_files = discover_files(root, split, semantic_subdir, depth_subdir)
        if max_images:
            depth_files = depth_files[:max_images]

        align_all = defaultdict(lambda: {"recall": [], "precision": [], "f1": []})
        coplanar_all = defaultdict(lambda: {"total_pairs": 0, "separated": 0})
        edge_densities = []

        # Also compute PQ with this depth source
        dep_tp = np.zeros(NUM_CLASSES)
        dep_fp = np.zeros(NUM_CLASSES)
        dep_fn = np.zeros(NUM_CLASSES)
        dep_iou = np.zeros(NUM_CLASSES)

        for sem_path, depth_path, gt_label_path, gt_inst_path in tqdm(
            depth_files, desc=f"Alignment ({depth_subdir})"
        ):
            if depth_path is None:
                continue

            # Load semantic
            pred_raw = np.array(Image.open(sem_path))
            pred_sem = lut[pred_raw] if lut is not None else pred_raw
            if pred_sem.shape != eval_hw:
                pred_sem = resize_nearest(pred_sem, eval_hw)

            # Load depth
            depth = np.load(str(depth_path))
            if depth.shape != eval_hw:
                depth = resize_bilinear(depth, eval_hw)

            # Load GT
            gt_raw = np.array(Image.open(gt_label_path))
            gt_sem = remap_gt_to_trainids(gt_raw)
            if gt_sem.shape != eval_hw:
                gt_sem = resize_nearest(gt_sem, eval_hw)

            gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
            if gt_inst_map.shape != eval_hw:
                gt_inst_map = resize_nearest(gt_inst_map.astype(np.uint16), eval_hw).astype(np.int32)

            # Compute depth edges
            depth_edges, grad_mag = compute_depth_edges(
                depth, grad_threshold=grad_threshold,
            )
            edge_densities.append(float(depth_edges.mean()))

            # GT instance boundaries (same-class only)
            gt_boundaries = compute_gt_thing_boundaries(gt_inst_map, gt_sem)

            # Overall alignment
            align = compute_edge_alignment(depth_edges, gt_boundaries)
            align_all["_overall"]["recall"].append(align["recall"])
            align_all["_overall"]["precision"].append(align["precision"])
            align_all["_overall"]["f1"].append(align["f1"])

            # Per-class alignment
            for cls in sorted(THING_IDS):
                cls_gt_boundaries = gt_boundaries & (gt_sem == cls)
                cls_depth_edges = depth_edges  # depth edges are not class-specific
                if cls_gt_boundaries.sum() == 0:
                    continue
                cls_align = compute_edge_alignment(cls_depth_edges, cls_gt_boundaries)
                align_all[CLASS_NAMES[cls]]["recall"].append(cls_align["recall"])
                align_all[CLASS_NAMES[cls]]["precision"].append(cls_align["precision"])
                align_all[CLASS_NAMES[cls]]["f1"].append(cls_align["f1"])

            # Co-planar separation
            coplanar = compute_coplanar_separation_rate(
                gt_inst_map, gt_sem, depth_edges,
            )
            for key, val in coplanar.items():
                coplanar_all[key]["total_pairs"] += val["total_pairs"]
                coplanar_all[key]["separated"] += val["separated"]

            # PQ evaluation
            instances = depth_guided_instances(
                pred_sem, depth_edges, THING_IDS,
                min_area=min_area, dilation_iters=3,
            )
            tp, fp, fn, iou_s = evaluate_panoptic_single(
                pred_sem, instances, gt_sem, gt_inst_map, eval_hw,
            )
            dep_tp += tp
            dep_fp += fp
            dep_fn += fn
            dep_iou += iou_s

        dep_pq = compute_pq(dep_tp, dep_fp, dep_fn, dep_iou)

        # Print results
        print(f"\n  Edge density: {np.mean(edge_densities):.4f} ({np.mean(edge_densities)*100:.2f}%)")
        print(f"\n  PQ={dep_pq['PQ']:6.2f}  PQ_stuff={dep_pq['PQ_stuff']:6.2f}  "
              f"PQ_things={dep_pq['PQ_things']:6.2f}")

        print(f"\n  Edge-Boundary Alignment (tolerance={BOUNDARY_TOLERANCE_PX}px):")
        print(f"  {'Class':15s} {'Recall':>8s} {'Precision':>10s} {'F1':>6s}  {'N':>5s}")
        print(f"  {'-'*50}")
        for key in ["_overall"] + [CLASS_NAMES[c] for c in sorted(THING_IDS)]:
            if key in align_all and align_all[key]["recall"]:
                r = np.mean(align_all[key]["recall"])
                p = np.mean(align_all[key]["precision"])
                f = np.mean(align_all[key]["f1"])
                n = len(align_all[key]["recall"])
                label = "OVERALL" if key == "_overall" else key
                print(f"  {label:15s} {r:8.4f} {p:10.4f} {f:6.4f}  {n:5d}")

        print(f"\n  Co-planar Separation Rate:")
        print(f"  {'Class':15s} {'Pairs':>6s} {'Separated':>10s} {'Rate':>8s}")
        print(f"  {'-'*45}")
        for key in ["_overall"] + [CLASS_NAMES[c] for c in sorted(THING_IDS)]:
            if key in coplanar_all and coplanar_all[key]["total_pairs"] > 0:
                d = coplanar_all[key]
                rate = d["separated"] / d["total_pairs"]
                label = "OVERALL" if key == "_overall" else key
                print(f"  {label:15s} {d['total_pairs']:6d} {d['separated']:10d} {rate:8.4f}")

        # Per-class PQ for things
        print(f"\n  Per-class PQ (things):")
        for cls in sorted(THING_IDS):
            name = CLASS_NAMES[cls]
            v = dep_pq["per_class"][name]
            if v["TP"] + v["FP"] + v["FN"] > 0:
                print(f"    {name:15s}: PQ={v['PQ']:5.1f}  SQ={v['SQ']:5.1f}  "
                      f"RQ={v['RQ']:5.1f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

        depth_metrics[depth_subdir] = {
            "PQ": dep_pq["PQ"],
            "PQ_stuff": dep_pq["PQ_stuff"],
            "PQ_things": dep_pq["PQ_things"],
            "edge_density": round(float(np.mean(edge_densities)), 6),
            "alignment": {
                key: {
                    "recall": round(float(np.mean(vals["recall"])), 4),
                    "precision": round(float(np.mean(vals["precision"])), 4),
                    "f1": round(float(np.mean(vals["f1"])), 4),
                }
                for key, vals in align_all.items()
                if vals["recall"]
            },
            "coplanar": {
                key: {
                    "total_pairs": d["total_pairs"],
                    "separated": d["separated"],
                    "rate": round(d["separated"] / max(d["total_pairs"], 1), 4),
                }
                for key, d in coplanar_all.items()
                if d["total_pairs"] > 0
            },
            "per_class": dep_pq["per_class"],
        }

    # ═══════════════════════════════════════════════
    # Analysis 3: Person Failure Taxonomy
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 3: PERSON FAILURE TAXONOMY")
    print(f"{'='*70}")

    # Use first depth source
    primary_files = discover_files(root, split, semantic_subdir, depth_subdirs[0])
    if max_images:
        primary_files = primary_files[:max_images]

    taxonomy_totals = defaultdict(int)

    for sem_path, depth_path, gt_label_path, gt_inst_path in tqdm(
        primary_files, desc="Person taxonomy"
    ):
        if depth_path is None:
            continue

        pred_raw = np.array(Image.open(sem_path))
        pred_sem = lut[pred_raw] if lut is not None else pred_raw
        if pred_sem.shape != eval_hw:
            pred_sem = resize_nearest(pred_sem, eval_hw)

        depth = np.load(str(depth_path))
        if depth.shape != eval_hw:
            depth = resize_bilinear(depth, eval_hw)

        gt_raw = np.array(Image.open(gt_label_path))
        gt_sem = remap_gt_to_trainids(gt_raw)
        if gt_sem.shape != eval_hw:
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(Image.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != eval_hw:
            gt_inst_map = resize_nearest(gt_inst_map.astype(np.uint16), eval_hw).astype(np.int32)

        depth_edges, _ = compute_depth_edges(depth, grad_threshold=grad_threshold)
        instances = depth_guided_instances(
            pred_sem, depth_edges, THING_IDS,
            min_area=min_area, dilation_iters=3,
        )

        tax = classify_person_failures(
            gt_inst_map, gt_sem, pred_sem, depth_edges, instances, eval_hw,
        )
        for k, v in tax.items():
            taxonomy_totals[k] += v

    total = taxonomy_totals["total_gt"]
    print(f"\n  Total person GT instances: {total}")
    if total > 0:
        for cat in ["matched", "co_planar_merge", "over_split", "semantic_miss", "edge_obliteration"]:
            count = taxonomy_totals[cat]
            pct = count / total * 100
            print(f"    {cat:25s}: {count:5d} ({pct:5.1f}%)")

    # ═══════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Oracle ceiling (perfect edges): PQ_things = {oracle_metrics['PQ_things']:.2f}")
    for depth_subdir in depth_subdirs:
        dm = depth_metrics.get(depth_subdir, {})
        print(f"  {depth_subdir:25s}: PQ_things = {dm.get('PQ_things', 'N/A')}")
    gap = oracle_metrics["PQ_things"] - depth_metrics.get(depth_subdirs[0], {}).get("PQ_things", 0)
    print(f"\n  Gap (oracle - current): {gap:.2f} PQ_things")
    if gap > 5:
        print(f"  → LARGE gap: Better depth edges CAN significantly improve PQ_things")
    elif gap > 2:
        print(f"  → MODERATE gap: Some room for depth model improvement")
    else:
        print(f"  → SMALL gap: Depth edges are NOT the bottleneck — algorithm or semantics limits PQ")

    # Save results
    output = {
        "config": {
            "cityscapes_root": str(cityscapes_root),
            "split": split,
            "semantic_subdir": semantic_subdir,
            "depth_subdirs": depth_subdirs,
            "grad_threshold": grad_threshold,
            "min_area": min_area,
            "max_images": max_images,
        },
        "oracle": {
            "PQ": oracle_metrics["PQ"],
            "PQ_stuff": oracle_metrics["PQ_stuff"],
            "PQ_things": oracle_metrics["PQ_things"],
            "per_class": oracle_metrics["per_class"],
        },
        "depth_sources": depth_metrics,
        "person_taxonomy": dict(taxonomy_totals),
    }

    output_path = Path(cityscapes_root) / f"diagnose_depth_splitting_{split}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose depth-guided instance splitting: oracle, alignment, taxonomy"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_mapped_k80",
                        help="Semantic pseudo-label subdirectory")
    parser.add_argument("--depth_subdir", type=str, nargs="+",
                        default=["depth_spidepth"],
                        help="Depth map subdirectory(ies) to analyze. "
                             "Multiple values compare across depth sources.")
    parser.add_argument("--grad_threshold", type=float, default=0.20,
                        help="Depth gradient threshold for Sobel edge detection (default: 0.20)")
    parser.add_argument("--min_area", type=int, default=1000,
                        help="Minimum instance area in pixels (default: 1000)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit to first N images (for testing)")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"DEPTH SPLITTING DIAGNOSTIC — Phase 0")
    print(f"{'='*70}")
    print(f"  Root:      {args.cityscapes_root}")
    print(f"  Split:     {args.split}")
    print(f"  Semantics: {args.semantic_subdir}")
    print(f"  Depth:     {args.depth_subdir}")
    print(f"  Threshold: {args.grad_threshold}")
    print(f"  Min area:  {args.min_area}")

    t0 = time.time()
    run_analysis(
        cityscapes_root=args.cityscapes_root,
        split=args.split,
        semantic_subdir=args.semantic_subdir,
        depth_subdirs=args.depth_subdir,
        grad_threshold=args.grad_threshold,
        min_area=args.min_area,
        max_images=args.max_images,
    )
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
