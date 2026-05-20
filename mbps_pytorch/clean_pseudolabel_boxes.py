"""Clean up CUPS-format pseudo-label instance maps for tighter, more box-like targets.

Reads `<name>_instance.png` files from `--input_dir`, applies per-instance:
  1. Largest connected component per instance ID (splits multi-blob IDs)
  2. Drops fragments with area < `--min_area`
  3. Morphological closing (k=`--close_kernel`) to fill aliasing gaps
  4. Renumbers remaining instance IDs contiguously starting from `--first_instance_id`
Copies `<name>_semantic.png` and `<name>.pt` unchanged.

All operations are deterministic geometric transforms on the existing pseudo-label
pixels. No ground-truth label is read at any point — fully unsupervised.

Usage:
  python mbps_pytorch/clean_pseudolabel_boxes.py \
      --input_dir  /home/santosh/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc \
      --output_dir /home/santosh/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc_clean \
      --min_area 800 --close_kernel 3 --workers 8
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def _clean_instance_map(
    inst: np.ndarray,
    min_area: int,
    close_kernel: int,
    first_instance_id: int,
) -> Tuple[np.ndarray, dict]:
    """Apply cleanup ops to a single instance map. Returns (cleaned, stats)."""
    stats = {
        "in_instances": 0,
        "out_instances": 0,
        "dropped_too_small": 0,
        "split_multi_cc": 0,
        "split_keep_largest": 0,
        "mean_box_tightness_in": 0.0,
        "mean_box_tightness_out": 0.0,
    }
    H, W = inst.shape
    out = np.zeros_like(inst, dtype=inst.dtype)

    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    if close_kernel > 0:
        close_struct = np.ones((close_kernel, close_kernel), dtype=bool)
    else:
        close_struct = None

    in_tight, in_count = 0.0, 0
    out_tight, out_count = 0.0, 0

    uniq_ids = [v for v in np.unique(inst) if v > 0]
    stats["in_instances"] = len(uniq_ids)

    next_id = first_instance_id

    for inst_id in uniq_ids:
        mask = (inst == inst_id)
        if not mask.any():
            continue

        # Input box tightness for stats
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size > 0 and cols.size > 0:
            box_area_in = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
            in_tight += float(mask.sum()) / float(box_area_in)
            in_count += 1

        # Largest connected component
        cc, n_cc = ndimage.label(mask, structure=struct)
        if n_cc == 0:
            stats["dropped_too_small"] += 1
            continue
        if n_cc > 1:
            stats["split_multi_cc"] += 1
            sizes = ndimage.sum(mask, cc, range(1, n_cc + 1))
            largest = int(np.argmax(sizes)) + 1
            mask = (cc == largest)
            stats["split_keep_largest"] += 1

        # Drop tiny
        if int(mask.sum()) < min_area:
            stats["dropped_too_small"] += 1
            continue

        # Morphological close (deterministic)
        if close_struct is not None:
            mask = ndimage.binary_closing(mask, structure=close_struct)

        # Re-confirm post-close largest CC (closing can merge with nearby strays)
        cc2, n_cc2 = ndimage.label(mask, structure=struct)
        if n_cc2 > 1:
            sizes = ndimage.sum(mask, cc2, range(1, n_cc2 + 1))
            largest = int(np.argmax(sizes)) + 1
            mask = (cc2 == largest)

        if int(mask.sum()) < min_area:
            stats["dropped_too_small"] += 1
            continue

        # Output box tightness
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size > 0 and cols.size > 0:
            box_area_out = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
            out_tight += float(mask.sum()) / float(box_area_out)
            out_count += 1

        out[mask] = next_id
        next_id += 1

    stats["out_instances"] = next_id - first_instance_id
    stats["mean_box_tightness_in"] = (in_tight / in_count) if in_count else 0.0
    stats["mean_box_tightness_out"] = (out_tight / out_count) if out_count else 0.0
    return out, stats


def _process_one(
    inst_path: Path,
    in_root: Path,
    out_root: Path,
    min_area: int,
    close_kernel: int,
    first_instance_id: int,
) -> dict:
    rel = inst_path.relative_to(in_root)
    out_inst_path = out_root / rel
    out_inst_path.parent.mkdir(parents=True, exist_ok=True)

    inst = np.array(Image.open(inst_path))
    if inst.ndim != 2:
        inst = inst[..., 0]
    in_dtype = inst.dtype
    cleaned, stats = _clean_instance_map(
        inst.astype(np.int32), min_area, close_kernel, first_instance_id
    )

    # Preserve dtype where possible
    if cleaned.max() <= np.iinfo(in_dtype).max:
        cleaned = cleaned.astype(in_dtype)
    elif cleaned.max() <= np.iinfo(np.uint16).max:
        cleaned = cleaned.astype(np.uint16)
    else:
        cleaned = cleaned.astype(np.int32)

    Image.fromarray(cleaned).save(out_inst_path)

    stem = inst_path.name.replace("_instance.png", "")
    for sibling_suffix in ("_semantic.png", ".pt"):
        src = inst_path.parent / (stem + sibling_suffix)
        if src.exists():
            dst = out_root / src.relative_to(in_root)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    stats["file"] = str(rel)
    return stats


def _aggregate(stats_list):
    agg = {
        "files": len(stats_list),
        "in_instances": 0,
        "out_instances": 0,
        "dropped_too_small": 0,
        "split_multi_cc": 0,
        "mean_box_tightness_in": 0.0,
        "mean_box_tightness_out": 0.0,
    }
    tight_in_sum, tight_in_n = 0.0, 0
    tight_out_sum, tight_out_n = 0.0, 0
    for s in stats_list:
        agg["in_instances"] += s["in_instances"]
        agg["out_instances"] += s["out_instances"]
        agg["dropped_too_small"] += s["dropped_too_small"]
        agg["split_multi_cc"] += s["split_multi_cc"]
        if s["in_instances"] > 0:
            tight_in_sum += s["mean_box_tightness_in"] * s["in_instances"]
            tight_in_n += s["in_instances"]
        if s["out_instances"] > 0:
            tight_out_sum += s["mean_box_tightness_out"] * s["out_instances"]
            tight_out_n += s["out_instances"]
    agg["mean_box_tightness_in"] = (tight_in_sum / tight_in_n) if tight_in_n else 0.0
    agg["mean_box_tightness_out"] = (tight_out_sum / tight_out_n) if tight_out_n else 0.0
    agg["multi_cc_rate_in"] = (agg["split_multi_cc"] / max(1, agg["in_instances"]))
    agg["mean_inst_per_img_in"] = agg["in_instances"] / max(1, agg["files"])
    agg["mean_inst_per_img_out"] = agg["out_instances"] / max(1, agg["files"])
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--min_area", type=int, default=800,
                   help="Drop instances with mask area below this (px).")
    p.add_argument("--close_kernel", type=int, default=3,
                   help="Morphological close kernel size (0 to disable).")
    p.add_argument("--first_instance_id", type=int, default=1,
                   help="Starting instance ID after renumbering. CUPS uses 1.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="0=all; else process first N.")
    args = p.parse_args()

    in_root: Path = args.input_dir
    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    inst_paths = sorted(in_root.rglob("*_instance.png"))
    if args.limit:
        inst_paths = inst_paths[: args.limit]
    print(f"[clean] {len(inst_paths)} instance files in {in_root}")
    print(f"[clean] writing to {out_root}")
    print(f"[clean] min_area={args.min_area} close_kernel={args.close_kernel} workers={args.workers}")

    stats_list = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _process_one,
                ip, in_root, out_root,
                args.min_area, args.close_kernel, args.first_instance_id,
            )
            for ip in inst_paths
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            stats_list.append(fut.result())
            if i % 200 == 0:
                print(f"[clean] {i}/{len(inst_paths)} done")

    agg = _aggregate(stats_list)
    print()
    print("=" * 60)
    print("AGGREGATE BOX-CLEANUP STATS")
    print("=" * 60)
    print(f"  files processed             : {agg['files']}")
    print(f"  in  instances               : {agg['in_instances']:>8d}  (mean/img {agg['mean_inst_per_img_in']:.2f})")
    print(f"  out instances               : {agg['out_instances']:>8d}  (mean/img {agg['mean_inst_per_img_out']:.2f})")
    print(f"  dropped (too small)         : {agg['dropped_too_small']:>8d}")
    print(f"  split multi-CC (had > 1)    : {agg['split_multi_cc']:>8d}  (rate {agg['multi_cc_rate_in']*100:5.2f}%)")
    print()
    print(f"  mean box tightness (in)     : {agg['mean_box_tightness_in']:.4f}  (mask_area / box_area)")
    print(f"  mean box tightness (out)    : {agg['mean_box_tightness_out']:.4f}  (mask_area / box_area)")
    delta = agg["mean_box_tightness_out"] - agg["mean_box_tightness_in"]
    pct = (delta / agg["mean_box_tightness_in"] * 100) if agg["mean_box_tightness_in"] else 0.0
    print(f"  tightness Δ                 : {delta:+.4f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
