#!/usr/bin/env python3
"""Diagnostic script for CUPS Stage-2 pseudo-label format verification.

Simulates exactly what PseudoLabelDataset does during training, without
needing the full CUPS/Detectron2 stack. Run this on the remote machine
pointing at the actual cups_pseudo_labels_v3/ directory.

Usage:
    python diagnose_cups_pseudolabels.py \
        --pseudo_dir /path/to/cups_pseudo_labels_v3 \
        --cityscapes_root /path/to/cityscapes \
        --num_samples 5

What this checks:
    1. Semantic label value range (should be 0-26 for CAUSE-27)
    2. Instance map non-zero counts (should have multiple instances per image)
    3. Thing/stuff split from .pt distribution files
    4. Whether instances survive the object_semantics_valid filter
    5. The remapped sem_seg values the model actually trains on
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


# CAUSE 27-class names (after Hungarian matching)
CAUSE27_NAMES = [
    "road", "sidewalk", "parking", "rail_track", "building", "wall", "fence",
    "guard_rail", "bridge", "tunnel", "pole", "polegroup", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky", "person", "rider",
    "car", "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle",
]

# Expected thing classes in CAUSE-27 space
EXPECTED_THING_IDS_27 = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26}
# person, rider, car, truck, bus, caravan, trailer, train, motorcycle, bicycle


def load_label_like_cups(path):
    """Simulate CUPS load_label: read_image(RGB) → [3, H, W] long tensor."""
    try:
        from torchvision.io import read_image
        from torchvision.io.image import ImageReadMode
        label = read_image(path=path, mode=ImageReadMode.RGB)
    except (RuntimeError, Exception):
        img = np.array(Image.open(path))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        else:
            img = img.transpose(2, 0, 1)
        label = torch.from_numpy(img.astype(np.int32))
    return label


def compute_thing_stuff_split(pseudo_dir, threshold=0.05):
    """Replicate PseudoLabelDataset.__init__ lines 131-142."""
    pt_files = [f for f in os.listdir(pseudo_dir) if f.endswith(".pt")]
    if not pt_files:
        print("ERROR: No .pt distribution files found!")
        return None, None, None

    tensor_files = [torch.load(os.path.join(pseudo_dir, f), map_location="cpu") for f in sorted(pt_files)]

    class_distribution_instances = torch.stack(
        [t["distribution inside object proposals"] for t in tensor_files]
    ).sum(dim=0)
    class_distribution = torch.stack(
        [t["distribution all pixels"] for t in tensor_files]
    ).sum(dim=0)

    ratio = class_distribution_instances / (class_distribution + 1e-06)
    distribution, indices = torch.sort(ratio, descending=True)
    distribution = distribution / distribution.sum()
    num_instance_pseudo_classes = (distribution > threshold).float().argmin().item()

    things_classes = tuple(indices[:num_instance_pseudo_classes].tolist())
    stuff_classes = tuple(indices[num_instance_pseudo_classes:].tolist())

    return things_classes, stuff_classes, {
        "ratio": ratio,
        "class_distribution": class_distribution,
        "class_distribution_instances": class_distribution_instances,
    }


def simulate_getitem(semantic_path, instance_path, things_classes, stuff_classes, void_id=255):
    """Replicate PseudoLabelDataset.__getitem__ lines 320-414 remapping logic."""
    # Load labels (same as CUPS)
    semantic_raw = load_label_like_cups(semantic_path)[0].long()  # [H, W]
    instance_raw = load_label_like_cups(instance_path)[0].long()  # [H, W]

    # === REMAPPING (lines 372-375) ===
    weight = torch.ones(256, dtype=torch.long) * void_id
    weight[torch.tensor(things_classes)] = 0  # thing → 0
    weight[torch.tensor(stuff_classes)] = torch.arange(len(stuff_classes), dtype=torch.long) + 1
    sem_seg = torch.embedding(weight.reshape(-1, 1), semantic_raw.unsqueeze(0).unsqueeze(0)).squeeze()

    # === INSTANCE PROCESSING (lines 384-414) ===
    # instances_to_masks
    unique_ids = instance_raw.unique(sorted=True)
    instance_ids = unique_ids[unique_ids > 0]

    if len(instance_ids) == 0:
        return {
            "semantic_raw": semantic_raw,
            "sem_seg": sem_seg,
            "instance_raw": instance_raw,
            "num_instances": 0,
            "num_valid_instances": 0,
            "object_classes_raw": [],
            "object_classes_remapped": [],
        }

    instance_masks = torch.stack([(instance_raw == iid) for iid in instance_ids])  # [N, H, W]

    # Get semantic class of each instance (line 386)
    object_semantics = (semantic_raw.unsqueeze(0) * instance_masks).amax(dim=(-1, -2))

    # Check if in things_classes (line 388)
    object_semantics_valid = torch.isin(object_semantics, torch.tensor(things_classes))

    # Remap object semantics (lines 412-414)
    weight2 = torch.zeros(len(stuff_classes) + len(things_classes), dtype=torch.long)
    weight2[torch.tensor(things_classes)] = torch.arange(len(things_classes), dtype=torch.long)

    valid_object_semantics = object_semantics[object_semantics_valid]
    if len(valid_object_semantics) > 0:
        # Check if indices are within range for the embedding
        max_idx = len(stuff_classes) + len(things_classes)
        out_of_range = (valid_object_semantics >= max_idx).any()
        if not out_of_range:
            remapped_classes = torch.embedding(weight2.reshape(-1, 1), valid_object_semantics)[..., 0]
        else:
            remapped_classes = torch.tensor([-1])  # Error marker
    else:
        remapped_classes = torch.tensor([])

    return {
        "semantic_raw": semantic_raw,
        "sem_seg": sem_seg,
        "instance_raw": instance_raw,
        "num_instances": len(instance_ids),
        "num_valid_instances": int(object_semantics_valid.sum().item()),
        "object_classes_raw": object_semantics.tolist(),
        "object_classes_valid_mask": object_semantics_valid.tolist(),
        "object_classes_remapped": remapped_classes.tolist() if len(remapped_classes) > 0 else [],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose CUPS pseudo-label format")
    parser.add_argument("--pseudo_dir", type=str, required=True,
                        help="Path to cups_pseudo_labels_v3/ directory")
    parser.add_argument("--cityscapes_root", type=str, default=None,
                        help="Path to cityscapes root (for image paths)")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Thing/stuff threshold (CUPS default: 0.05)")
    args = parser.parse_args()

    pseudo_dir = args.pseudo_dir
    if not os.path.isdir(pseudo_dir):
        print(f"ERROR: {pseudo_dir} does not exist!")
        sys.exit(1)

    # =====================================================
    # CHECK 1: List files
    # =====================================================
    all_files = os.listdir(pseudo_dir)
    semantic_files = sorted([f for f in all_files if f.endswith("_semantic.png")])
    instance_files = sorted([f for f in all_files if f.endswith("_instance.png")])
    pt_files = sorted([f for f in all_files if f.endswith(".pt")])

    print("=" * 70)
    print("CHECK 1: FILE INVENTORY")
    print("=" * 70)
    print(f"  Directory: {pseudo_dir}")
    print(f"  Semantic PNGs: {len(semantic_files)}")
    print(f"  Instance PNGs: {len(instance_files)}")
    print(f"  Distribution .pt: {len(pt_files)}")
    print(f"  Total files: {len(all_files)}")

    if len(semantic_files) == 0:
        print("  ERROR: No semantic files found!")
        sys.exit(1)

    # =====================================================
    # CHECK 2: Semantic label values
    # =====================================================
    print("\n" + "=" * 70)
    print("CHECK 2: SEMANTIC LABEL VALUES (first 3 files)")
    print("=" * 70)
    for sem_file in semantic_files[:3]:
        sem_path = os.path.join(pseudo_dir, sem_file)
        sem = np.array(Image.open(sem_path))
        unique_vals = np.unique(sem)
        print(f"  {sem_file}:")
        print(f"    Shape: {sem.shape}, dtype: {sem.dtype}")
        print(f"    Range: [{sem.min()}, {sem.max()}]")
        print(f"    Unique values: {unique_vals}")
        if sem.max() <= 18:
            print(f"    WARNING: Max value is {sem.max()} — looks like 19-class trainIDs, NOT 27-class CAUSE!")
        elif sem.max() <= 26:
            print(f"    OK: Values in 0-26 range (CAUSE 27-class)")
        else:
            print(f"    WARNING: Values exceed 26!")

    # =====================================================
    # CHECK 3: Instance map values
    # =====================================================
    print("\n" + "=" * 70)
    print("CHECK 3: INSTANCE MAP VALUES (first 3 files)")
    print("=" * 70)
    for inst_file in instance_files[:3]:
        inst_path = os.path.join(pseudo_dir, inst_file)
        inst = np.array(Image.open(inst_path))
        num_instances = inst.max()
        print(f"  {inst_file}:")
        print(f"    Shape: {inst.shape}, dtype: {inst.dtype}")
        print(f"    Range: [{inst.min()}, {inst.max()}]")
        print(f"    Num instances: {num_instances}")
        if num_instances == 0:
            print(f"    WARNING: ZERO instances in this image!")

    # =====================================================
    # CHECK 4: Distribution .pt files
    # =====================================================
    print("\n" + "=" * 70)
    print("CHECK 4: DISTRIBUTION .pt FILES (first 3)")
    print("=" * 70)
    for pt_file in pt_files[:3]:
        pt_path = os.path.join(pseudo_dir, pt_file)
        data = torch.load(pt_path, map_location="cpu")
        dist_all = data["distribution all pixels"]
        dist_inst = data["distribution inside object proposals"]
        print(f"  {pt_file}:")
        print(f"    Dist all:  bins={len(dist_all)}, nonzero={(dist_all > 0).sum().item()}")
        print(f"    Dist inst: bins={len(dist_inst)}, nonzero={(dist_inst > 0).sum().item()}")
        nonzero_inst_bins = (dist_inst > 0).nonzero().squeeze().tolist()
        if isinstance(nonzero_inst_bins, int):
            nonzero_inst_bins = [nonzero_inst_bins]
        print(f"    Nonzero inst bins: {nonzero_inst_bins}")
        for bin_idx in nonzero_inst_bins:
            if bin_idx < len(CAUSE27_NAMES):
                name = CAUSE27_NAMES[bin_idx]
                is_thing = bin_idx in EXPECTED_THING_IDS_27
                marker = "THING" if is_thing else "STUFF (!)"
                print(f"      Bin {bin_idx} ({name}): {dist_inst[bin_idx]:.0f} pixels [{marker}]")

    # =====================================================
    # CHECK 5: Thing/Stuff split
    # =====================================================
    print("\n" + "=" * 70)
    print("CHECK 5: THING/STUFF SPLIT (threshold={:.2f})".format(args.threshold))
    print("=" * 70)
    things, stuffs, info = compute_thing_stuff_split(pseudo_dir, args.threshold)
    if things is None:
        print("  ERROR: Could not compute split!")
        sys.exit(1)

    print(f"  Things classes ({len(things)}): {things}")
    print(f"  Stuff classes ({len(stuffs)}): {stuffs}")
    print(f"  Total: {len(things) + len(stuffs)}")

    # Check if expected thing classes are in things
    print(f"\n  Expected thing IDs (CAUSE-27): {sorted(EXPECTED_THING_IDS_27)}")
    found_things = set(things) & EXPECTED_THING_IDS_27
    missing_things = EXPECTED_THING_IDS_27 - set(things)
    wrong_things = set(things) - EXPECTED_THING_IDS_27
    print(f"  Correctly identified as things: {sorted(found_things)}")
    if missing_things:
        print(f"  MISSING from things (classified as stuff): {sorted(missing_things)}")
        for mid in sorted(missing_things):
            name = CAUSE27_NAMES[mid] if mid < len(CAUSE27_NAMES) else f"class_{mid}"
            ratio = info["ratio"][mid].item()
            print(f"    {mid} ({name}): instance_ratio={ratio:.6f}")
    if wrong_things:
        print(f"  WRONGLY in things (should be stuff): {sorted(wrong_things)}")
        for wid in sorted(wrong_things):
            name = CAUSE27_NAMES[wid] if wid < len(CAUSE27_NAMES) else f"class_{wid}"
            ratio = info["ratio"][wid].item()
            print(f"    {wid} ({name}): instance_ratio={ratio:.6f}")

    # Show per-class ratios
    print(f"\n  Per-class instance ratio (sorted by ratio):")
    ratios = info["ratio"]
    sorted_indices = torch.argsort(ratios, descending=True)
    for idx in sorted_indices:
        idx = idx.item()
        r = ratios[idx].item()
        name = CAUSE27_NAMES[idx] if idx < len(CAUSE27_NAMES) else f"class_{idx}"
        in_things = "THING" if idx in set(things) else "stuff"
        expected = "THING" if idx in EXPECTED_THING_IDS_27 else "stuff"
        mismatch = " *** MISMATCH ***" if in_things != expected else ""
        if r > 0 or idx in EXPECTED_THING_IDS_27:
            print(f"    {idx:2d} ({name:15s}): ratio={r:.6f}  [{in_things:5s}] expected=[{expected:5s}]{mismatch}")

    # =====================================================
    # CHECK 6: Simulate training data flow
    # =====================================================
    print("\n" + "=" * 70)
    print(f"CHECK 6: SIMULATE TRAINING DATA (first {args.num_samples} samples)")
    print("=" * 70)

    total_instances = 0
    total_valid_instances = 0

    for i, (sem_file, inst_file) in enumerate(zip(semantic_files[:args.num_samples],
                                                    instance_files[:args.num_samples])):
        sem_path = os.path.join(pseudo_dir, sem_file)
        inst_path = os.path.join(pseudo_dir, inst_file)

        result = simulate_getitem(sem_path, inst_path, things, stuffs)

        raw = result["semantic_raw"]
        seg = result["sem_seg"]

        total_instances += result["num_instances"]
        total_valid_instances += result["num_valid_instances"]

        print(f"\n  Sample {i}: {sem_file}")
        print(f"    Raw semantic unique: {raw.unique().tolist()}")
        print(f"    Remapped sem_seg unique: {seg.unique().tolist()}")
        void_pct = (seg == 255).float().mean().item() * 100
        thing_pct = (seg == 0).float().mean().item() * 100
        print(f"    sem_seg: void={void_pct:.1f}%, thing(0)={thing_pct:.1f}%, stuff={100-void_pct-thing_pct:.1f}%")
        print(f"    Instance count: {result['num_instances']} total, {result['num_valid_instances']} valid (in things)")
        print(f"    Object classes (raw CAUSE-27): {result['object_classes_raw']}")
        if 'object_classes_valid_mask' in result:
            print(f"    Valid mask: {result['object_classes_valid_mask']}")
        print(f"    Remapped classes: {result['object_classes_remapped']}")

        if result["num_instances"] > 0 and result["num_valid_instances"] == 0:
            print(f"    *** ALL INSTANCES FILTERED OUT! ***")
            print(f"    Instance semantic classes: {result['object_classes_raw']}")
            print(f"    Things classes: {things}")
            print(f"    None of the instance classes are in things_classes!")

    # =====================================================
    # SUMMARY
    # =====================================================
    n = min(args.num_samples, len(semantic_files))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Samples checked: {n}")
    print(f"  Total instances: {total_instances} ({total_instances/max(n,1):.1f}/img)")
    print(f"  Valid instances: {total_valid_instances} ({total_valid_instances/max(n,1):.1f}/img)")
    if total_instances > 0:
        survival_rate = total_valid_instances / total_instances * 100
        print(f"  Instance survival rate: {survival_rate:.1f}%")
        if survival_rate < 50:
            print(f"  WARNING: Less than 50% of instances survive the thing-class filter!")
        if survival_rate == 0:
            print(f"  CRITICAL: ZERO instances survive! Training will have NO instance targets!")

    print(f"\n  Thing/stuff split: {len(things)} thing classes, {len(stuffs)} stuff classes")
    if missing_things:
        print(f"  PROBLEM: {len(missing_things)} expected thing classes missing: {sorted(missing_things)}")
    if wrong_things:
        print(f"  PROBLEM: {len(wrong_things)} stuff classes wrongly classified as thing: {sorted(wrong_things)}")

    if total_valid_instances == 0:
        print("\n  DIAGNOSIS: Stage-2 is likely collapsing because the model receives")
        print("  ZERO valid instance targets during training. Check:")
        print("  1. Are semantic labels in 27-class CAUSE format? (not 19-class trainID)")
        print("  2. Do .pt files have nonzero bins at indices 17-26?")
        print("  3. Does the thing/stuff threshold correctly classify CAUSE 17-26 as things?")


if __name__ == "__main__":
    main()
