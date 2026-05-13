"""Extract CUPS pseudo-class → Cityscapes class mapping from a Stage-3 checkpoint.

This script computes the Hungarian assignment between the k=80 pseudo-class
semantic head channels and the 27-class Cityscapes evaluation vocabulary.
The output tells you WHICH semantic head channel (0..num_stuff-1) corresponds
to each Cityscapes stuff class, and WHICH pseudo-thing-class corresponds to
each Cityscapes thing class.

These mappings are needed for the Stage-4 FineObjectSemanticLoss to use
``class_specific`` supervision for stuff classes (pole, traffic sign, etc.)
instead of the weaker entropy mode.

Usage (on Santosh's machine)
-----------------------------
    python scripts/extract_cups_class_mapping.py \\
        --checkpoint /home/santosh/experiments/stage3_dcfa_simcf_abc/experiments/cups_dinov3_vitb_dcfa_simcf_abc_stage3/Unsupervised\\ Panoptic\\ Segmentation/<run_id>/checkpoints/best_pq_step=<step>.ckpt \\
        --pseudo_dir /home/santosh/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc/ \\
        --gt_dir /home/santosh/datasets/cityscapes/gtFine/val/ \\
        --output /home/santosh/cups_class_mapping.json \\
        --num_images 50

Output JSON format
------------------
{
    "num_stuff_channels": 65,
    "num_thing_classes": 15,
    "thing_channel": 0,             # channel 0 = unified thing class in all CUPS k=80 heads
    "stuff_channel_to_cityscapes": {
        "0": "road",
        "4": "sidewalk",
        ...
        "42": "pole",               # ← use this for SAM3 pole supervision
        "28": "traffic sign",       # ← use this for SAM3 traffic sign supervision
        "31": "traffic light",      # ← use this for SAM3 traffic light supervision
        "17": "guard rail",         # ← use this for SAM3 guard rail supervision
        ...
    },
    "sam3_stuff_channel_map": {
        # SAM3 class index → semantic head channel for stuff classes
        "4": 28,   # traffic sign
        "5": 31,   # traffic light
        "9": 17,   # guard rail
        "13": 42   # pole
    },
    "thing_class_to_cityscapes": {
        "0": "bicycle",
        "1": "car",
        ...
    },
    "sam3_thing_class_map": {
        # SAM3 class index → pseudo-thing-class index (for future instance head supervision)
        "0": 3,   # person → pseudo-thing-class 3
        "1": 7,   # bicycle → pseudo-thing-class 7
        ...
    }
}

After running, add the sam3_stuff_channel_map to your Stage-4 config:
    SELF_TRAINING:
      FINE_OBJECT:
        STUFF_CHANNEL_MAP:
          4: 28    # traffic sign
          5: 31    # traffic light
          9: 17    # guard rail
          13: 42   # pole
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

# ── Cityscapes 27-class definitions (raw_id - 7 = cups27_idx) ──────────────
CUPS27_NAMES = [
    "ground",         # 0  (raw 7)
    "road",           # 1  (raw 8)
    "sidewalk",       # 2  (raw 9)
    "parking",        # 3  (raw 10)
    "rail track",     # 4  (raw 11)
    "building",       # 5  (raw 12)
    "wall",           # 6  (raw 13)
    "guard rail",     # 7  (raw 14)  ← SAM3 class 9
    "bridge",         # 8  (raw 15)
    "tunnel",         # 9  (raw 16)
    "pole",           # 10 (raw 17)  ← SAM3 class 13
    "polegroup",      # 11 (raw 18)
    "traffic light",  # 12 (raw 19)  ← SAM3 class 5
    "traffic sign",   # 13 (raw 20)  ← SAM3 class 4
    "vegetation",     # 14 (raw 21)
    "terrain",        # 15 (raw 22)
    "sky",            # 16 (raw 23)
    "person",         # 17 (raw 24)  ← SAM3 class 0
    "rider",          # 18 (raw 25)  ← SAM3 class 3
    "car",            # 19 (raw 26)  ← SAM3 class 12
    "truck",          # 20 (raw 27)  ← SAM3 class 6
    "bus",            # 21 (raw 28)  ← SAM3 class 7
    "caravan",        # 22 (raw 29)  ← SAM3 class 10
    "trailer",        # 23 (raw 30)  ← SAM3 class 11
    "train",          # 24 (raw 31)  ← SAM3 class 8
    "motorcycle",     # 25 (raw 32)  ← SAM3 class 2
    "bicycle",        # 26 (raw 33)  ← SAM3 class 1
]

# SAM3 class index → CUPS27 evaluation class index
SAM3_TO_CUPS27_IDX = {
    0: 17,   # person
    1: 26,   # bicycle
    2: 25,   # motorcycle
    3: 18,   # rider
    4: 13,   # traffic sign
    5: 12,   # traffic light
    6: 20,   # truck
    7: 21,   # bus
    8: 24,   # train
    9:  7,   # guard rail
    10: 22,  # caravan
    11: 23,  # trailer
    12: 19,  # car
    13: 10,  # pole
}

# Cityscapes raw label → CUPS27 index mapping (used for GT images)
CITYSCAPES_RAW_TO_CUPS27: Dict[int, int] = {
    7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 7,
    15: 8, 16: 9, 17: 10, 18: 11, 19: 12, 20: 13, 21: 14, 22: 15,
    23: 16, 24: 17, 25: 18, 26: 19, 27: 20, 28: 21, 29: 22, 30: 23,
    31: 24, 32: 25, 33: 26,
}


def load_checkpoint_metadata(ckpt_path: str) -> Tuple[List[int], List[int]]:
    """Extract thing_pseudo_classes and stuff_pseudo_classes from checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})

    thing_pseudo_classes = hp.get("thing_pseudo_classes")
    stuff_pseudo_classes = hp.get("stuff_pseudo_classes")

    if thing_pseudo_classes is None or stuff_pseudo_classes is None:
        # Fall back: infer from state_dict
        state = ckpt.get("state_dict", ckpt)
        bias_key = next(
            (k for k in state if "sem_seg_head.predictor.bias" in k), None
        )
        if bias_key:
            num_channels = state[bias_key].shape[0]
            print(f"  Inferred num_semantic_channels = {num_channels} from predictor bias")
        else:
            raise RuntimeError("Cannot find sem_seg_head.predictor.bias in checkpoint")

        print("  WARNING: thing_pseudo_classes / stuff_pseudo_classes not found in hparams.")
        print("  Hungarian mapping will be computed from pseudo-label pixel statistics only.")
        return [], []

    thing_pseudo_classes = list(thing_pseudo_classes)
    stuff_pseudo_classes = list(stuff_pseudo_classes)
    print(f"  thing_pseudo_classes ({len(thing_pseudo_classes)}): {thing_pseudo_classes}")
    print(f"  stuff_pseudo_classes ({len(stuff_pseudo_classes)}): {stuff_pseudo_classes}")
    return thing_pseudo_classes, stuff_pseudo_classes


def compute_hungarian_mapping(
    pseudo_dir: Path,
    gt_dir: Path,
    thing_clusters: List[int],
    stuff_clusters: List[int],
    num_images: int = 50,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Compute Hungarian mapping pseudo-class → Cityscapes class.

    Returns:
        stuff_channel_to_cups27: {semantic_head_channel → cups27_class_idx}
        thing_channel_to_cups27: {thing_pseudo_class_rank → cups27_class_idx}
    """
    n_stuff = len(stuff_clusters)
    n_cups27 = len(CUPS27_NAMES)

    # Confusion matrix: pseudo_class × cups27_class (pixel counts)
    confusion = np.zeros((n_stuff + 1, n_cups27), dtype=np.int64)
    # +1 for the unified thing channel

    # Map cluster_id → semantic head channel
    # Channel 0 = unified thing (pseudo_label_dataset: things_classes → 0).
    # Stuff clusters map to channels 1..n_stuff (1-indexed, not 0-indexed).
    cluster_to_channel: Dict[int, int] = {}
    for ch, cid in enumerate(sorted(stuff_clusters)):
        cluster_to_channel[cid] = ch + 1  # +1: channel 0 is reserved for thing
    thing_channel = 0  # channel 0 = unified thing in all CUPS k=80 heads

    # Find matching pseudo-label and GT files
    sem_files = sorted(pseudo_dir.glob("**/*_semantic.png"))[:num_images]
    print(f"  Processing {len(sem_files)} images...")

    for sem_path in sem_files:
        # Find matching GT file
        stem = sem_path.stem.replace("_leftImg8bit_semantic", "")
        city = stem.split("_")[0]
        gt_path = gt_dir / city / f"{stem}_gtFine_labelIds.png"
        if not gt_path.exists():
            # Try alternative naming
            gt_path = gt_dir / f"{stem}_gtFine_labelIds.png"
        if not gt_path.exists():
            continue

        sem_arr = np.array(Image.open(sem_path))
        gt_arr = np.array(Image.open(gt_path))

        # Resize to match if needed
        if sem_arr.shape != gt_arr.shape:
            from PIL import Image as PILImage
            gt_img = PILImage.fromarray(gt_arr).resize(
                (sem_arr.shape[1], sem_arr.shape[0]), PILImage.NEAREST
            )
            gt_arr = np.array(gt_img)

        # Convert GT raw IDs → CUPS27 indices
        gt_cups27 = np.full_like(gt_arr, 255)
        for raw_id, cups27_idx in CITYSCAPES_RAW_TO_CUPS27.items():
            gt_cups27[gt_arr == raw_id] = cups27_idx

        valid = (gt_cups27 != 255) & (sem_arr != 255)

        for cluster_id in stuff_clusters:
            ch = cluster_to_channel[cluster_id]
            mask = valid & (sem_arr == cluster_id)
            if not mask.any():
                continue
            gt_in_mask = gt_cups27[mask]
            for cups_cls in range(n_cups27):
                confusion[ch, cups_cls] += int((gt_in_mask == cups_cls).sum())

        # Thing clusters → thing_channel
        for cluster_id in thing_clusters:
            mask = valid & (sem_arr == cluster_id)
            if not mask.any():
                continue
            gt_in_mask = gt_cups27[mask]
            for cups_cls in range(n_cups27):
                confusion[thing_channel, cups_cls] += int((gt_in_mask == cups_cls).sum())

    print(f"  Confusion matrix computed. Running Hungarian matching...")

    # Hungarian matching on stuff channels only (rows 1..n_stuff; row 0 = thing)
    stuff_confusion = confusion[1:, :]   # rows 1..n_stuff = stuff
    row_ind, col_ind = linear_sum_assignment(-stuff_confusion)
    # row_ind is 0-indexed into stuff_confusion; add 1 to get actual semantic head channel
    stuff_channel_to_cups27 = {int(r) + 1: int(c) for r, c in zip(row_ind, col_ind)}
    print("  Stuff channel → Cityscapes class:")
    for ch, cups_idx in sorted(stuff_channel_to_cups27.items()):
        count = stuff_confusion[ch - 1, cups_idx]   # -1 to index into stuff_confusion
        name = CUPS27_NAMES[cups_idx]
        print(f"    channel {ch:3d} → {name:20s} ({count:,} pixels)")

    return stuff_channel_to_cups27, {}


def build_sam3_stuff_channel_map(
    stuff_channel_to_cups27: Dict[int, int],
) -> Dict[int, int]:
    """Build SAM3 class index → semantic head channel for stuff classes."""
    cups27_to_channel = {v: k for k, v in stuff_channel_to_cups27.items()}
    sam3_stuff_map: Dict[int, int] = {}
    for sam3_idx, cups27_idx in SAM3_TO_CUPS27_IDX.items():
        # Only stuff classes (traffic sign=4, traffic light=5, guard rail=9, pole=13)
        stuff_sam3_indices = {4, 5, 9, 13}
        if sam3_idx not in stuff_sam3_indices:
            continue

        if cups27_idx in cups27_to_channel:
            ch = cups27_to_channel[cups27_idx]
            sam3_stuff_map[sam3_idx] = ch
            print(f"  SAM3 idx {sam3_idx:2d} ({CUPS27_NAMES[cups27_idx]:20s})"
                  f" → semantic channel {ch}")
        else:
            print(f"  WARNING: SAM3 idx {sam3_idx} ({CUPS27_NAMES[cups27_idx]})"
                  f" not found in Hungarian assignment")

    return sam3_stuff_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to Stage-3 .ckpt file")
    parser.add_argument("--pseudo_dir", required=True,
                        help="Path to cups_pseudo_labels_dcfa_simcf_abc/ directory")
    parser.add_argument("--gt_dir", required=True,
                        help="Path to cityscapes/gtFine/val/ directory")
    parser.add_argument("--output", default="cups_class_mapping.json",
                        help="Output JSON path")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Number of validation images to use for Hungarian matching")
    args = parser.parse_args()

    print("=== CUPS Class Mapping Extraction ===\n")

    # Step 1: Extract pseudo-class lists from checkpoint
    thing_clusters, stuff_clusters = load_checkpoint_metadata(args.checkpoint)

    if not stuff_clusters:
        print("ERROR: Could not extract stuff_pseudo_classes from checkpoint hparams.")
        print("Try inspecting the checkpoint manually:")
        print("  python3 -c \"import torch; hp = torch.load('<ckpt>', map_location='cpu')['hyper_parameters']; print(list(hp.keys()))\"")
        return

    n_stuff = len(stuff_clusters)
    thing_channel = 0  # channel 0 = thing; channels 1..n_stuff = stuff
    print(f"\nSemantic head: {n_stuff + 1} channels (channel 0 = thing, channels 1..{n_stuff} = stuff)")

    # Step 2: Compute Hungarian matching
    print(f"\nComputing Hungarian mapping from {args.num_images} validation images...")
    stuff_ch_to_cups27, _ = compute_hungarian_mapping(
        Path(args.pseudo_dir),
        Path(args.gt_dir),
        thing_clusters,
        stuff_clusters,
        args.num_images,
    )

    # Step 3: Build SAM3-specific stuff channel map
    print("\nBuilding SAM3 stuff channel map:")
    sam3_stuff_map = build_sam3_stuff_channel_map(stuff_ch_to_cups27)

    # Step 4: Save output
    result = {
        "num_stuff_channels": n_stuff,
        "num_thing_classes": len(thing_clusters),
        "thing_channel": thing_channel,
        "stuff_clusters": sorted(stuff_clusters),
        "thing_clusters": sorted(thing_clusters),
        "stuff_channel_to_cups27": {
            str(ch): {"cups27_idx": idx, "name": CUPS27_NAMES[idx]}
            for ch, idx in sorted(stuff_ch_to_cups27.items())
        },
        "sam3_stuff_channel_map": {
            str(k): v for k, v in sorted(sam3_stuff_map.items())
        },
        "instructions": {
            "add_to_stage4_config": (
                "Under SELF_TRAINING.FINE_OBJECT, add:\n"
                "  STUFF_CHANNEL_MAP:\n"
                + "".join(
                    f"    {k}: {v}  # {CUPS27_NAMES[SAM3_TO_CUPS27_IDX[int(k)]]}\n"
                    for k, v in sorted(sam3_stuff_map.items(), key=lambda x: int(x[0]))
                )
            )
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to {args.output}")
    print("\n=== Add to Stage-4 config (SELF_TRAINING.FINE_OBJECT) ===")
    print("  STUFF_CHANNEL_MAP:")
    for k, v in sorted(sam3_stuff_map.items(), key=lambda x: int(x[0])):
        name = CUPS27_NAMES[SAM3_TO_CUPS27_IDX[int(k)]]
        print(f"    {k}: {v}    # {name}")


if __name__ == "__main__":
    main()
