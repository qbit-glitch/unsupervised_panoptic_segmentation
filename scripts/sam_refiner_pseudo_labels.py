"""SAMRefiner: offline refinement of CUPS pseudo-label instance masks using SAM3.

For each pseudo-label instance mask, finds the best-overlapping SAM3 mask
and replaces the instance with the refined SAM3 boundary if IoU > --min_iou.
This improves instance boundary quality without changing class assignments.

Inspired by:
  SAMRefiner: Taming Segment Anything Model for Universal Mask Refinement
  Hu et al., ICLR 2025.

Ablation target: improve PQ_things by sharpening instance boundaries for
co-planar objects (especially persons and cars merged by Sobel+CC splitting).

Usage
-----
# Dry run on 20 images (verify format, no writes)
python scripts/sam_refiner_pseudo_labels.py \\
    --sam3_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/sam_fine_masks_sam3/train/ \\
    --instance_in /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc/ \\
    --instance_out /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_sam3_refined/ \\
    --min_iou 0.50 --max_images 20 --dry_run

# Full run
python scripts/sam_refiner_pseudo_labels.py \\
    --sam3_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/sam_fine_masks_sam3/train/ \\
    --instance_in /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc/ \\
    --instance_out /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_sam3_refined/ \\
    --min_iou 0.50

# Verify statistics on output directory (no writes)
python scripts/sam_refiner_pseudo_labels.py \\
    --instance_out /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_sam3_refined/ \\
    --verify
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# SAM3 I/O helpers (shared with inject_sam3_dead_classes.py)
# ---------------------------------------------------------------------------

def find_sam3_path(sam3_dir: Path, stem: str) -> Optional[Path]:
    """Find SAM3 .npz for a given image stem."""
    clean = stem.replace("_leftImg8bit_semantic", "").replace("_leftImg8bit_instance", "").replace("_leftImg8bit", "")
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
    """Load SAM3 NPZ → (masks, iou_scores, areas, class_labels)."""
    d = np.load(path)
    return (
        d["masks"],
        d["iou_scores"].astype(np.float32),
        d["areas"].astype(np.int32),
        d["class_labels"].astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Core refinement logic
# ---------------------------------------------------------------------------

def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection) / (float(union) + 1e-6)


def _resize_masks_to(masks: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize SAM3 masks from (N, H', W') to (N, H, W) via nearest-neighbour."""
    if masks.shape[1:] == (H, W):
        return masks
    out = []
    for mk in masks:
        img = Image.fromarray(mk.astype(np.uint8) * 255)
        img = img.resize((W, H), Image.NEAREST)
        out.append(np.array(img) > 127)
    return np.stack(out)


def refine_image(
    ins_arr: np.ndarray,
    sam3_path: Path,
    min_iou: float = 0.50,
    min_area: int = 100,
    max_sam3_masks: int = 40,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Refine a single image's instance map using SAM3.

    For each instance ID, finds the SAM3 mask with the highest IoU. If that
    IoU exceeds ``min_iou``, replaces the instance's pixel footprint with the
    SAM3 mask's footprint (preserving the instance ID).

    Args:
        ins_arr: uint16 (H, W) instance PNG (0=background, >0=instance IDs).
        sam3_path: Path to the .npz file with SAM3 masks.
        min_iou: Minimum IoU to accept a SAM3 mask as a refinement.
        min_area: Minimum pixel area of original instance to attempt refinement.
        max_sam3_masks: Only consider the ``max_sam3_masks`` highest-IoU SAM3
            masks to keep runtime manageable.

    Returns:
        (refined_ins, stats) where stats is a dict with counts.
    """
    H, W = ins_arr.shape
    masks_sam, ious_sam, areas_sam, labels_sam = load_sam3(sam3_path)

    # Resize SAM3 masks to pseudo-label resolution
    masks_sam = _resize_masks_to(masks_sam, H, W)

    # Sort SAM3 masks by IoU descending, cap to max_sam3_masks
    order = np.argsort(-ious_sam)[:max_sam3_masks]
    masks_sam = masks_sam[order]

    ins_out = ins_arr.copy()
    unique_ids = np.unique(ins_arr)
    unique_ids = unique_ids[unique_ids > 0]  # skip background

    n_refined = 0
    n_skipped_area = 0
    n_skipped_iou = 0

    claimed = np.zeros((H, W), dtype=bool)  # track SAM3 masks already used

    for inst_id in unique_ids:
        orig_mask = ins_arr == inst_id
        area = int(orig_mask.sum())

        if area < min_area:
            n_skipped_area += 1
            continue

        # Find best-overlapping SAM3 mask (not yet claimed)
        best_iou = 0.0
        best_idx = -1
        for j, sam_mask in enumerate(masks_sam):
            if claimed[sam_mask].mean() > 0.5:
                continue  # this SAM3 mask is mostly taken
            iou = _compute_iou(orig_mask, sam_mask)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou < min_iou or best_idx < 0:
            n_skipped_iou += 1
            continue

        # Replace original instance footprint with SAM3 refined mask
        refined_mask = masks_sam[best_idx]
        ins_out[orig_mask] = 0           # clear original pixels
        ins_out[refined_mask] = inst_id  # write refined pixels
        claimed |= refined_mask
        n_refined += 1

    stats = {
        "n_instances": len(unique_ids),
        "n_refined": n_refined,
        "n_skipped_area": n_skipped_area,
        "n_skipped_iou": n_skipped_iou,
    }
    return ins_out, stats


# ---------------------------------------------------------------------------
# Main runners
# ---------------------------------------------------------------------------

def run_refine(args: argparse.Namespace) -> None:
    """Run SAMRefiner on all images in pseudo_in → pseudo_out."""
    instance_in = Path(args.instance_in)
    instance_out = Path(args.instance_out)
    sam3_dir = Path(args.sam3_dir)

    instance_out.mkdir(parents=True, exist_ok=True)

    # Enumerate instance PNGs
    ins_files = sorted(instance_in.glob("*_instance.png"))
    if args.max_images:
        ins_files = ins_files[: args.max_images]

    total_stats = {"n_instances": 0, "n_refined": 0, "n_skipped_area": 0, "n_skipped_iou": 0}
    images_with_no_sam3 = 0
    images_with_refinement = 0

    for ins_path in tqdm(ins_files, desc="SAMRefiner"):
        stem = ins_path.stem.replace("_instance", "")
        sem_path = instance_in / f"{stem}_semantic.png"
        pt_path = instance_in / f"{stem}.pt"
        sam3_path = find_sam3_path(sam3_dir, stem)

        out_ins = instance_out / ins_path.name
        out_sem = instance_out / f"{stem}_semantic.png"
        out_pt = instance_out / f"{stem}.pt"

        ins_arr = np.array(Image.open(ins_path))

        if sam3_path is None:
            # No SAM3 masks — copy instance unchanged
            images_with_no_sam3 += 1
            if not args.dry_run:
                shutil.copy2(ins_path, out_ins)
                if sem_path.exists():
                    shutil.copy2(sem_path, out_sem)
                if pt_path.exists():
                    shutil.copy2(pt_path, out_pt)
            continue

        # Refine instance map
        ins_refined, stats = refine_image(
            ins_arr,
            sam3_path,
            min_iou=args.min_iou,
            min_area=args.min_area,
        )

        for k in total_stats:
            total_stats[k] += stats[k]

        if stats["n_refined"] > 0:
            images_with_refinement += 1

        if not args.dry_run:
            Image.fromarray(ins_refined.astype(np.uint16)).save(out_ins)
            if sem_path.exists():
                shutil.copy2(sem_path, out_sem)
            if pt_path.exists():
                shutil.copy2(pt_path, out_pt)

    n_total = len(ins_files)
    print(f"\n=== SAMRefiner complete {'(DRY RUN)' if args.dry_run else ''} ===")
    print(f"Images processed:        {n_total}")
    print(f"Images without SAM3:     {images_with_no_sam3}")
    print(f"Images with ≥1 refined:  {images_with_refinement}")
    print(f"Total instances:         {total_stats['n_instances']}")
    print(f"  Refined (IoU≥{args.min_iou:.2f}):   {total_stats['n_refined']} "
          f"({100*total_stats['n_refined'] / max(1, total_stats['n_instances']):.1f}%)")
    print(f"  Skipped (area<{args.min_area}):  {total_stats['n_skipped_area']}")
    print(f"  Skipped (IoU<{args.min_iou:.2f}):  {total_stats['n_skipped_iou']}")
    if not args.dry_run:
        print(f"Output: {instance_out}")


def run_verify(args: argparse.Namespace) -> None:
    """Print statistics about an already-refined output directory."""
    instance_out = Path(args.instance_out)
    ins_files = sorted(instance_out.glob("*_instance.png"))
    if not ins_files:
        print(f"ERROR: No *_instance.png found in {instance_out}")
        return

    sem_count = len(list(instance_out.glob("*_semantic.png")))
    pt_count = len(list(instance_out.glob("*.pt")))

    print(f"Output directory: {instance_out}")
    print(f"  *_instance.png: {len(ins_files)}")
    print(f"  *_semantic.png: {sem_count}")
    print(f"  *.pt:           {pt_count}")
    if len(ins_files) == sem_count == pt_count:
        print("✓ File counts match")
    else:
        print("✗ File count mismatch — some files may be missing")

    # Instance count distribution
    counts: List[int] = []
    for p in tqdm(ins_files[:200], desc="Sampling"):
        arr = np.array(Image.open(p))
        counts.append(int((np.unique(arr) > 0).sum()))

    arr_counts = np.array(counts)
    print(f"\nInstances per image (sample of {len(counts)}): "
          f"mean={arr_counts.mean():.1f}, min={arr_counts.min()}, max={arr_counts.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="Print stats for output dir without running refinement.")

    parser.add_argument("--sam3_dir", type=Path, default=None,
                        help="Directory with SAM3 .npz files (train split).")
    parser.add_argument("--instance_in", type=Path, default=None,
                        help="Input pseudo-label directory (flat, *_instance.png).")
    parser.add_argument("--instance_out", type=Path, required=True,
                        help="Output refined pseudo-label directory.")
    parser.add_argument("--min_iou", type=float, default=0.50,
                        help="Min IoU to accept SAM3 mask as refinement (default 0.50).")
    parser.add_argument("--min_area", type=int, default=100,
                        help="Min pixel area of original instance to attempt refinement (default 100).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Process only first N images (dry run / debug).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Compute stats but do not write any files.")

    args = parser.parse_args()

    if args.dry_run and args.max_images is None:
        args.max_images = 20

    if args.verify:
        run_verify(args)
    else:
        if args.sam3_dir is None or args.instance_in is None:
            parser.error("--sam3_dir and --instance_in are required for refinement mode.")
        run_refine(args)


if __name__ == "__main__":
    main()
