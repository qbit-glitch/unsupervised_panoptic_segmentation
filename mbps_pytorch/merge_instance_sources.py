#!/usr/bin/env python3
"""Merge instance NPZ files from multiple detectors into a single set.

Concatenates masks from all sources, sorted by score descending.
Optionally applies IoU-based NMS to remove near-duplicate detections.

Usage:
    python mbps_pytorch/merge_instance_sources.py \
        --sources pseudo_instances_cuvler/val pseudo_instances_cutler/val \
        --output pseudo_instances_ensemble/val
"""

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def _resize_masks(masks, target_h, target_w):
    """Resize boolean masks to target resolution using nearest neighbor."""
    if masks.shape[1] == target_h and masks.shape[2] == target_w:
        return masks
    M = masks.shape[0]
    resized = np.zeros((M, target_h, target_w), dtype=bool)
    for i in range(M):
        m_img = Image.fromarray(masks[i].astype(np.uint8) * 255)
        resized[i] = np.array(m_img.resize((target_w, target_h), Image.NEAREST)) > 127
    return resized


def merge_npz(npz_paths, nms_iou_thresh=None, target_hw=None):
    """Merge masks from multiple NPZ files, sorted by score.

    Handles resolution differences by resizing all masks to a common resolution.
    """
    all_masks = []
    all_scores = []
    all_boxes = []

    for p in npz_paths:
        if not os.path.exists(p):
            continue
        data = np.load(p)
        n = int(data["num_valid"]) if "num_valid" in data else data["masks"].shape[0]
        if n == 0:
            continue
        all_masks.append(data["masks"][:n])
        all_scores.append(data["scores"][:n])
        if "boxes" in data:
            all_boxes.append(data["boxes"][:n])
        else:
            all_boxes.append(np.zeros((n, 4), dtype=np.float32))

    if not all_masks:
        h, w = target_hw if target_hw else (1024, 2048)
        return np.zeros((0, h, w), dtype=bool), np.array([], dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    # Determine target resolution (use largest or explicit target)
    if target_hw:
        tgt_h, tgt_w = target_hw
    else:
        tgt_h = max(m.shape[1] for m in all_masks)
        tgt_w = max(m.shape[2] for m in all_masks)

    # Resize all to common resolution
    all_masks = [_resize_masks(m, tgt_h, tgt_w) for m in all_masks]

    masks = np.concatenate(all_masks, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((len(masks), 4), dtype=np.float32)

    # Sort by score descending
    order = np.argsort(-scores)
    masks = masks[order]
    scores = scores[order]
    boxes = boxes[order]

    # Optional mask-IoU NMS
    if nms_iou_thresh is not None and len(masks) > 1:
        keep = []
        areas = masks.sum(axis=(1, 2)).astype(float)
        for i in range(len(masks)):
            skip = False
            for j in keep:
                intersection = (masks[i] & masks[j]).sum()
                union = areas[i] + areas[j] - intersection
                if union > 0 and intersection / union > nms_iou_thresh:
                    skip = True
                    break
            if not skip:
                keep.append(i)
        masks = masks[keep]
        scores = scores[keep]
        boxes = boxes[keep]

    return masks, scores, boxes


def gap_fill_merge(primary_path, supplement_path, coverage_thresh=0.3, target_hw=None):
    """Merge primary + supplement instances with coverage-based gap filling.

    Keep ALL primary instances. For each supplement instance, only add it if
    less than coverage_thresh fraction of its pixels are already covered by
    any primary instance. This avoids duplicates while filling gaps.

    Args:
        primary_path: path to primary NPZ file (e.g., CuVLER)
        supplement_path: path to supplement NPZ file (e.g., DINOv3 CC)
        coverage_thresh: max fraction of supplement mask covered by primary
                         masks to still be added (lower = stricter)
        target_hw: (h, w) target resolution, or None for auto

    Returns:
        (masks, scores, boxes) merged arrays
    """
    # Load primary
    p_masks = np.zeros((0, 512, 1024), dtype=bool)
    p_scores = np.array([], dtype=np.float32)
    p_boxes = np.zeros((0, 4), dtype=np.float32)

    if os.path.exists(primary_path):
        data = np.load(primary_path)
        n = int(data["num_valid"]) if "num_valid" in data else data["masks"].shape[0]
        if n > 0:
            p_masks = data["masks"][:n]
            p_scores = data["scores"][:n]
            p_boxes = data["boxes"][:n] if "boxes" in data else np.zeros((n, 4), dtype=np.float32)

    # Load supplement
    s_masks = np.zeros((0, 512, 1024), dtype=bool)
    s_scores = np.array([], dtype=np.float32)

    if os.path.exists(supplement_path):
        data = np.load(supplement_path)
        n = int(data["num_valid"]) if "num_valid" in data else data["masks"].shape[0]
        if n > 0:
            s_masks = data["masks"][:n]
            s_scores = data["scores"][:n]

    if len(p_masks) == 0 and len(s_masks) == 0:
        h, w = target_hw if target_hw else (512, 1024)
        return np.zeros((0, h, w), dtype=bool), np.array([], dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    # Determine target resolution
    if target_hw:
        tgt_h, tgt_w = target_hw
    else:
        all_shapes = []
        if len(p_masks) > 0:
            all_shapes.append((p_masks.shape[1], p_masks.shape[2]))
        if len(s_masks) > 0:
            all_shapes.append((s_masks.shape[1], s_masks.shape[2]))
        tgt_h = max(s[0] for s in all_shapes)
        tgt_w = max(s[1] for s in all_shapes)

    # Resize to common resolution
    if len(p_masks) > 0:
        p_masks = _resize_masks(p_masks, tgt_h, tgt_w)
    if len(s_masks) > 0:
        s_masks = _resize_masks(s_masks, tgt_h, tgt_w)

    # Build primary coverage map (union of all primary masks)
    primary_coverage = np.zeros((tgt_h, tgt_w), dtype=bool)
    for m in p_masks:
        primary_coverage |= m

    # Filter supplement: only add masks with low coverage by primary
    keep_supplement = []
    for i in range(len(s_masks)):
        s_area = s_masks[i].sum()
        if s_area == 0:
            continue
        covered = (s_masks[i] & primary_coverage).sum()
        coverage_frac = covered / s_area
        if coverage_frac < coverage_thresh:
            keep_supplement.append(i)

    # Merge: primary first (higher priority), then filtered supplement
    out_masks = list(p_masks)
    out_scores = list(p_scores)
    out_boxes = list(p_boxes)

    for i in keep_supplement:
        out_masks.append(s_masks[i])
        # Scale supplement scores below primary scores
        out_scores.append(s_scores[i] * 0.5)
        out_boxes.append(np.zeros(4, dtype=np.float32))

    if not out_masks:
        return np.zeros((0, tgt_h, tgt_w), dtype=bool), np.array([], dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    masks = np.stack(out_masks, axis=0)
    scores = np.array(out_scores, dtype=np.float32)
    boxes = np.stack(out_boxes, axis=0)

    # Sort by score descending
    order = np.argsort(-scores)
    return masks[order], scores[order], boxes[order]


def main():
    parser = argparse.ArgumentParser("Merge instance NPZ files from multiple sources")
    parser.add_argument("--sources", nargs="+",
                        help="Input directories for concatenation mode")
    parser.add_argument("--primary", type=str, default=None,
                        help="Primary source directory for gap-fill mode (e.g., CuVLER)")
    parser.add_argument("--supplement", type=str, default=None,
                        help="Supplement source directory for gap-fill mode (e.g., DINOv3 CC)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--nms_iou", type=float, default=None,
                        help="Mask IoU threshold for NMS (e.g., 0.5). None=no NMS.")
    parser.add_argument("--coverage_thresh", type=float, default=0.3,
                        help="Gap-fill: max coverage fraction to add supplement mask (default 0.3)")
    args = parser.parse_args()

    # Determine mode
    gap_fill = args.primary is not None and args.supplement is not None
    if not gap_fill and args.sources is None:
        parser.error("Either --sources or (--primary + --supplement) required")

    if gap_fill:
        # Gap-fill mode: primary + supplement with coverage filtering
        # Discover all NPZ files from both sources
        all_files = set()
        for src in [args.primary, args.supplement]:
            if not os.path.isdir(src):
                continue
            for city in sorted(os.listdir(src)):
                city_dir = os.path.join(src, city)
                if not os.path.isdir(city_dir):
                    continue
                for fname in sorted(os.listdir(city_dir)):
                    if fname.endswith(".npz"):
                        all_files.add((city, fname))

        print(f"Gap-fill merge: primary={args.primary}")
        print(f"  supplement={args.supplement}")
        print(f"  coverage_thresh={args.coverage_thresh}")
        print(f"  {len(all_files)} images")

        os.makedirs(args.output, exist_ok=True)
        total_primary = 0
        total_supplement = 0

        for city, fname in tqdm(sorted(all_files), desc="Gap-fill merge"):
            p_path = os.path.join(args.primary, city, fname)
            s_path = os.path.join(args.supplement, city, fname)

            masks, scores, boxes = gap_fill_merge(
                p_path, s_path, coverage_thresh=args.coverage_thresh
            )

            # Count primary vs supplement
            n_primary = 0
            if os.path.exists(p_path):
                d = np.load(p_path)
                n_primary = int(d["num_valid"]) if "num_valid" in d else d["masks"].shape[0]
            n_total = len(masks)
            n_supp = n_total - min(n_primary, n_total)
            total_primary += min(n_primary, n_total)
            total_supplement += n_supp

            out_city = os.path.join(args.output, city)
            os.makedirs(out_city, exist_ok=True)
            np.savez_compressed(
                os.path.join(out_city, fname),
                masks=masks, scores=scores, boxes=boxes,
                num_valid=len(masks),
            )

        total = total_primary + total_supplement
        avg = total / max(len(all_files), 1)
        print(f"Done. Total: {total} masks ({total_primary} primary + "
              f"{total_supplement} supplement), avg {avg:.1f}/image")

    else:
        # Original concatenation mode
        all_files = {}
        for src in args.sources:
            for city in sorted(os.listdir(src)):
                city_dir = os.path.join(src, city)
                if not os.path.isdir(city_dir):
                    continue
                for fname in sorted(os.listdir(city_dir)):
                    if fname.endswith(".npz"):
                        key = (city, fname)
                        if key not in all_files:
                            all_files[key] = []
                        all_files[key].append(os.path.join(city_dir, fname))

        print(f"Merging {len(args.sources)} sources, {len(all_files)} images")
        if args.nms_iou is not None:
            print(f"  NMS IoU threshold: {args.nms_iou}")

        os.makedirs(args.output, exist_ok=True)
        total_merged = 0

        for (city, fname), paths in tqdm(sorted(all_files.items()), desc="Merging"):
            masks, scores, boxes = merge_npz(paths, args.nms_iou)

            out_city = os.path.join(args.output, city)
            os.makedirs(out_city, exist_ok=True)
            np.savez_compressed(
                os.path.join(out_city, fname),
                masks=masks, scores=scores, boxes=boxes,
                num_valid=len(masks),
            )
            total_merged += len(masks)

        avg = total_merged / max(len(all_files), 1)
        print(f"Done. Total masks: {total_merged}, avg {avg:.1f}/image")


if __name__ == "__main__":
    main()
