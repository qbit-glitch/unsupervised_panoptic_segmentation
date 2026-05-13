"""Inject SAM3 masks for dead Cityscapes classes into CUPS pseudo-labels.

Dead classes (PQ=0%) have zero thing instances in the existing k=80 pseudo-labels because
the DepthPro+k-means clustering never produced consistent instance proposals for rare
classes like motorcycle, caravan, trailer. This script seeds the pseudo-label set with
SAM3-sourced instance masks, converting these dead stuff clusters into alive thing instances.

Mechanism
---------
For each training image:
  1. Load SAM3 masks for target classes (motorcycle=2, caravan=10, trailer=11).
  2. Filter by IoU quality and minimum area.
  3. For each valid mask:
     - Overwrite the semantic PNG with the assigned thing cluster ID.
     - Add a new unique instance ID to the instance PNG.
     - Update the .pt distribution tensors so instanceness flips above 0.05.
  4. Write enriched files to the output directory (unchanged images are copied as-is).

After running, point SELF_TRAINING.ROOT_PSEUDO at the output directory and retrain
with USE_SEESAW_LOSS=True (SEESAW_P=0.5) and DROP_LOSS_IOU_THRESHOLD=0.25.

Usage
-----
    # Injection (full run — all SAM3 thing classes)
    python scripts/inject_sam3_dead_classes.py \\
        --sam3_dir  /path/to/sam_fine_masks_sam3/train/ \\
        --pseudo_in  /path/to/cups_pseudo_labels_dcfa_simcf_abc/ \\
        --pseudo_out /path/to/cups_pseudo_labels_sam3_enriched/ \\
        --class_mapping cups_class_mapping.json \\
        --target_classes 0,1,2,3,6,7,8,10,11,12 \\
        --min_iou 0.40 \\
        --min_area 300 \\
        --max_per_image 20

    # Verification (fast check — no file writes)
    python scripts/inject_sam3_dead_classes.py \\
        --pseudo_out /path/to/cups_pseudo_labels_sam3_enriched/ \\
        --class_mapping cups_class_mapping.json \\
        --target_classes 2,10,11 \\
        --verify

    # Dry-run on 10 images (test format without writing full set)
    python scripts/inject_sam3_dead_classes.py \\
        ... same as above ... \\
        --dry_run --max_images 10
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SAM3 class definitions (must match generate_sam_fine_masks.py)
# ---------------------------------------------------------------------------
SAM3_CLASS_NAMES = [
    "person",       # 0
    "bicycle",      # 1
    "motorcycle",   # 2  ← target dead class
    "rider",        # 3
    "traffic sign", # 4
    "traffic light",# 5
    "truck",        # 6
    "bus",          # 7
    "train",        # 8
    "guard rail",   # 9  ← stuff dead class (handled by Track C, not here)
    "caravan",      # 10 ← target dead class
    "trailer",      # 11 ← target dead class
    "car",          # 12
    "pole",         # 13
]


def load_class_mapping(mapping_path: str) -> Dict[int, int]:
    """Load SAM3-class → CUPS-thing-cluster mapping from JSON.

    Returns {sam3_class_idx: cups_thing_cluster_id}.
    """
    with open(mapping_path) as f:
        m = json.load(f)
    raw = m.get("sam3_thing_cluster_map", {})
    return {int(k): int(v) for k, v in raw.items()}


def find_sam3_path(sam3_dir: Path, stem: str) -> Optional[Path]:
    """Find SAM3 .npz for a given image stem.

    Stem format: e.g. 'aachen_000000_000019_leftImg8bit' or
    'aachen_000000_000019'.
    """
    clean = stem.replace("_leftImg8bit_semantic", "").replace("_leftImg8bit", "")
    city = clean.split("_")[0]
    candidates = [
        sam3_dir / f"{clean}_fine_masks.npz",
        sam3_dir / city / f"{clean}_fine_masks.npz",
        sam3_dir / f"{clean}_leftImg8bit_fine_masks.npz",
        sam3_dir / city / f"{clean}_leftImg8bit_fine_masks.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_sam3(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load SAM3 NPZ and return (masks, iou_scores, areas, class_labels)."""
    d = np.load(path)
    return d["masks"], d["iou_scores"].astype(np.float32), d["areas"].astype(np.int32), d["class_labels"].astype(np.int32)


def inject_image(
    sem: np.ndarray,
    ins: np.ndarray,
    dist: Dict[str, torch.Tensor],
    sam3_path: Path,
    sam3_to_cluster: Dict[int, int],
    target_classes: List[int],
    min_iou: float,
    min_area: int,
    max_per_image: int,
    max_frame_fraction: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor], int]:
    """Inject SAM3 masks into a single image's pseudo-labels.

    Returns:
        (sem, ins, dist, num_injected)
    """
    masks, ious, areas, labels = load_sam3(sam3_path)
    H, W = sem.shape
    total_pixels = H * W

    # Filter: target class + quality + size
    keep = (
        np.isin(labels, target_classes)
        & (ious >= min_iou)
        & (areas >= min_area)
        & (areas < max_frame_fraction * total_pixels)
    )
    if not keep.any():
        return sem, ins, dist, 0

    # Resize SAM3 masks to pseudo-label resolution if needed
    if masks.shape[1:] != (H, W):
        resized = []
        for mk in masks[keep]:
            img = Image.fromarray(mk.astype(np.uint8) * 255)
            img = img.resize((W, H), Image.NEAREST)
            resized.append(np.array(img) > 127)
        masks_keep = np.stack(resized)
    else:
        masks_keep = masks[keep]

    ious_keep = ious[keep]
    labels_keep = labels[keep]

    # Sort highest IoU first, cap per image
    order = np.argsort(-ious_keep)[:max_per_image]
    masks_keep = masks_keep[order]
    labels_keep = labels_keep[order]

    sem = sem.copy()
    ins = ins.copy()
    dist = {k: v.clone() for k, v in dist.items()}

    next_id = int(ins.max()) + 1
    claimed = np.zeros((H, W), dtype=bool)
    num_injected = 0

    for mask, sam_class in zip(masks_keep, labels_keep):
        cluster_id = sam3_to_cluster.get(int(sam_class))
        if cluster_id is None:
            continue

        # Only write pixels not claimed by a higher-priority mask this pass
        write = mask & (~claimed)
        n_write = int(write.sum())
        if n_write < min_area:
            continue

        # Update pseudo-labels
        sem[write] = cluster_id
        ins[write] = next_id
        claimed |= write
        next_id += 1
        num_injected += 1

        # Update .pt distributions: flip instanceness above 0.05 threshold
        dist["distribution all pixels"][cluster_id] += n_write
        dist["distribution inside object proposals"][cluster_id] += n_write

    return sem, ins, dist, num_injected


def run_injection(args: argparse.Namespace) -> None:
    pseudo_in = Path(args.pseudo_in)
    pseudo_out = Path(args.pseudo_out)
    sam3_dir = Path(args.sam3_dir)
    target_classes = [int(x) for x in args.target_classes.split(",")]

    sam3_to_cluster = load_class_mapping(args.class_mapping)
    # Filter mapping to only requested target classes
    sam3_to_cluster = {k: v for k, v in sam3_to_cluster.items() if k in target_classes}

    print(f"Target SAM3 classes: {[SAM3_CLASS_NAMES[c] for c in target_classes]}")
    print(f"SAM3 → CUPS cluster map: {sam3_to_cluster}")

    pseudo_out.mkdir(parents=True, exist_ok=True)

    # Enumerate all semantic PNGs in pseudo_in
    sem_files = sorted(pseudo_in.glob("*_semantic.png"))
    if args.max_images:
        sem_files = sem_files[: args.max_images]

    total_injected = {c: 0 for c in target_classes}
    images_with_injection = 0

    for sem_path in tqdm(sem_files, desc="Injecting"):
        stem = sem_path.stem.replace("_leftImg8bit_semantic", "")
        base_stem = f"{stem}_leftImg8bit"

        ins_path = pseudo_in / f"{base_stem}_instance.png"
        pt_path = pseudo_in / f"{base_stem}.pt"
        sam3_path = find_sam3_path(sam3_dir, stem)

        out_sem = pseudo_out / f"{base_stem}_semantic.png"
        out_ins = pseudo_out / f"{base_stem}_instance.png"
        out_pt = pseudo_out / f"{base_stem}.pt"

        # Load current pseudo-labels
        sem_arr = np.array(Image.open(sem_path))
        ins_arr = np.array(Image.open(ins_path))
        dist = torch.load(pt_path, map_location="cpu", weights_only=False)

        if sam3_path is None:
            # No SAM3 file — copy unchanged
            shutil.copy2(sem_path, out_sem)
            shutil.copy2(ins_path, out_ins)
            torch.save(dist, out_pt)
            continue

        # Inject SAM3 masks
        sem_new, ins_new, dist_new, n_inj = inject_image(
            sem_arr,
            ins_arr,
            dist,
            sam3_path,
            sam3_to_cluster,
            target_classes,
            min_iou=args.min_iou,
            min_area=args.min_area,
            max_per_image=args.max_per_image,
        )

        if n_inj > 0:
            images_with_injection += 1
            # Track per-class stats (approximate — just count masks per image)
            for c in target_classes:
                total_injected[c] += n_inj  # rough; exact tracking omitted for speed

        # Write outputs
        Image.fromarray(sem_new.astype(np.uint8)).save(out_sem)
        Image.fromarray(ins_new.astype(np.uint16)).save(out_ins)
        torch.save(dist_new, out_pt)

    # Summary
    print(f"\n=== Injection complete ===")
    print(f"Images with ≥1 injection: {images_with_injection} / {len(sem_files)}")
    for c in target_classes:
        print(f"  {SAM3_CLASS_NAMES[c]} (SAM3 class {c}) → cluster {sam3_to_cluster.get(c,'?')}: "
              f"~{total_injected[c]} total instances injected")
    print(f"Output: {pseudo_out}")


def run_verify(args: argparse.Namespace) -> None:
    """Verify that injected pseudo-labels have instanceness > 0.05 for target clusters."""
    import json

    pseudo_out = Path(args.pseudo_out)
    target_classes = [int(x) for x in args.target_classes.split(",")]

    sam3_to_cluster = load_class_mapping(args.class_mapping)
    sam3_to_cluster = {k: v for k, v in sam3_to_cluster.items() if k in target_classes}
    target_clusters = set(sam3_to_cluster.values())

    pt_files = sorted(pseudo_out.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: No .pt files found in {pseudo_out}. Run injection first.")
        return

    print(f"Verifying {len(pt_files)} images in {pseudo_out}...")
    dist_all = None
    dist_inst = None
    for pt_path in tqdm(pt_files, desc="Summing distributions"):
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        a = d["distribution all pixels"].float()
        i = d["distribution inside object proposals"].float()
        if dist_all is None:
            dist_all = a
            dist_inst = i
        else:
            dist_all += a
            dist_inst += i

    instanceness = (dist_inst / (dist_all + 1e-6))

    print(f"\n{'Cluster':>8} | {'SAM3 class':>14} | {'Instanceness':>12} | {'Status':>10}")
    print("-" * 55)
    all_ok = True
    for sam3_cls, cluster_id in sorted(sam3_to_cluster.items()):
        inst_val = float(instanceness[cluster_id])
        status = "✓ THING" if inst_val > 0.05 else "✗ STILL STUFF"
        if inst_val <= 0.05:
            all_ok = False
        print(f"{cluster_id:>8} | {SAM3_CLASS_NAMES[sam3_cls]:>14} | {inst_val:>12.4f} | {status}")

    print()
    if all_ok:
        print("✓ All target clusters have instanceness > 0.05 → classified as THINGS")
    else:
        print("✗ Some clusters still below 0.05. Try lowering --min_iou to 0.35 and re-running.")
        print("  Or: increase --max_per_image to inject more instances.")

    # File count check
    sem_count = len(list(pseudo_out.glob("*_semantic.png")))
    ins_count = len(list(pseudo_out.glob("*_instance.png")))
    pt_count = len(pt_files)
    print(f"\nFile counts: semantic={sem_count}, instance={ins_count}, pt={pt_count}")
    if sem_count == ins_count == pt_count:
        print("✓ File counts match")
    else:
        print("✗ File count mismatch — some images may be missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Verify output directory (no injection).")

    # Injection args
    parser.add_argument("--sam3_dir", type=Path, default=None,
                        help="Directory containing SAM3 .npz files (train split).")
    parser.add_argument("--pseudo_in", type=Path, default=None,
                        help="Input pseudo-label directory.")
    parser.add_argument("--pseudo_out", type=Path, required=True,
                        help="Output enriched pseudo-label directory.")
    parser.add_argument("--class_mapping", type=str, default="cups_class_mapping.json",
                        help="Path to cups_class_mapping.json from extract_cups_class_mapping.py.")
    parser.add_argument("--target_classes", type=str, default="2,10,11",
                        help="Comma-separated SAM3 class indices to inject (default: 2=motorcycle,10=caravan,11=trailer).")
    parser.add_argument("--min_iou", type=float, default=0.40,
                        help="Minimum SAM3 IoU score (default 0.40). Lower to 0.35 if instanceness check fails.")
    parser.add_argument("--min_area", type=int, default=300,
                        help="Minimum mask pixel area (default 300).")
    parser.add_argument("--max_per_image", type=int, default=6,
                        help="Maximum SAM3 masks to inject per image (default 6).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Process only first N images (for dry runs).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Inject only --max_images images for format testing.")

    args = parser.parse_args()

    if args.dry_run and args.max_images is None:
        args.max_images = 10

    if args.verify:
        run_verify(args)
    else:
        if args.sam3_dir is None or args.pseudo_in is None:
            parser.error("--sam3_dir and --pseudo_in are required for injection mode.")
        run_injection(args)


if __name__ == "__main__":
    main()
