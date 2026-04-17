#!/usr/bin/env python3
"""Diagnose pseudo-label differences between machines.

Run this on BOTH santosh and A6000 machines, then compare the JSON output.
This checks: file counts, val city contamination, thing/stuff split,
instance stats, and semantic class distribution.

Usage:
    # On santosh:
    python scripts/diagnose_pseudo_labels.py \
        --pseudo_dir /media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020 \
        --cityscapes_root /media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes \
        --output santosh_diag.json

    # On A6000:
    python scripts/diagnose_pseudo_labels.py \
        --pseudo_dir ~/umesh/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020 \
        --cityscapes_root ~/umesh/datasets/cityscapes \
        --output a6000_diag.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


VAL_CITIES = {"frankfurt", "lindau", "munster"}
TRAIN_CITIES = {
    "aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf",
    "erfurt", "hamburg", "hanover", "jena", "krefeld", "monchengladbach",
    "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"
}


def diagnose(pseudo_dir: str, cityscapes_root: str) -> dict:
    """Run all diagnostics on a pseudo-label directory."""
    pseudo_dir = os.path.expanduser(pseudo_dir)
    cityscapes_root = os.path.expanduser(cityscapes_root)

    result = {
        "pseudo_dir": pseudo_dir,
        "cityscapes_root": cityscapes_root,
        "exists": os.path.isdir(pseudo_dir),
    }

    if not result["exists"]:
        print(f"ERROR: {pseudo_dir} does not exist!")
        return result

    # 1. File counts
    all_files = sorted(os.listdir(pseudo_dir))
    instance_files = [f for f in all_files if "_instance.png" in f]
    semantic_files = [f for f in all_files if "_semantic.png" in f]
    pt_files = [f for f in all_files if f.endswith(".pt")]

    result["file_counts"] = {
        "total": len(all_files),
        "instance_png": len(instance_files),
        "semantic_png": len(semantic_files),
        "pt_files": len(pt_files),
    }
    print(f"Files: {len(instance_files)} instance, {len(semantic_files)} semantic, {len(pt_files)} .pt")

    # 2. City breakdown — check for val city contamination
    city_counts = Counter()
    for f in instance_files:
        city = f.split("_")[0]
        city_counts[city] += 1

    val_city_files = {c: city_counts[c] for c in VAL_CITIES if c in city_counts}
    train_city_files = {c: city_counts[c] for c in city_counts if c not in VAL_CITIES}

    result["city_breakdown"] = {
        "train_cities": dict(sorted(train_city_files.items())),
        "val_cities_present": val_city_files,
        "total_train_images": sum(train_city_files.values()),
        "total_val_images": sum(val_city_files.values()),
        "val_contamination": len(val_city_files) > 0,
    }

    if val_city_files:
        print(f"WARNING: Val cities in training dir! {val_city_files}")
    else:
        print(f"OK: No val city contamination. {sum(train_city_files.values())} train images.")

    # 3. Thing/stuff split (replicates CUPS PseudoLabelDataset logic)
    print("Computing thing/stuff split from .pt files...")
    thing_stuff_threshold = 0.05  # from config
    try:
        tensor_files = []
        for p in sorted(pt_files):
            t = torch.load(os.path.join(pseudo_dir, p), weights_only=False, map_location="cpu")
            tensor_files.append(t)

        if tensor_files:
            class_dist_instances = torch.stack(
                [t["distribution inside object proposals"] for t in tensor_files]
            ).sum(dim=0)
            class_dist_all = torch.stack(
                [t["distribution all pixels"] for t in tensor_files]
            ).sum(dim=0)

            distribution, indices = torch.sort(
                class_dist_instances / (class_dist_all + 1e-06), descending=True
            )
            distribution = distribution / distribution.sum()
            num_things = int((distribution > thing_stuff_threshold).float().argmin().item())

            things_classes = sorted(indices[:num_things].tolist())
            stuff_classes = sorted(indices[num_things:].tolist())

            result["thing_stuff_split"] = {
                "num_classes": int(class_dist_all.shape[0]),
                "num_things": num_things,
                "num_stuff": len(stuff_classes),
                "things_classes": things_classes,
                "stuff_classes": stuff_classes,
                "pt_files_used": len(tensor_files),
            }
            print(f"Thing/stuff: {num_things} things, {len(stuff_classes)} stuff (from {len(tensor_files)} .pt files)")
            print(f"Things: {things_classes}")
        else:
            result["thing_stuff_split"] = {"error": "no .pt files found"}
    except Exception as e:
        result["thing_stuff_split"] = {"error": str(e)}
        print(f"ERROR computing thing/stuff: {e}")

    # 4. Instance statistics (sample first 100 images)
    print("Computing instance stats (first 100 images)...")
    instance_stats = []
    sample_files = instance_files[:100]
    for f in sample_files:
        path = os.path.join(pseudo_dir, f)
        inst = np.array(Image.open(path))
        unique_ids = np.unique(inst)
        # Filter out background (0)
        fg_ids = unique_ids[unique_ids > 0]
        areas = []
        for uid in fg_ids:
            areas.append(int((inst == uid).sum()))
        instance_stats.append({
            "file": f,
            "num_instances": len(fg_ids),
            "total_fg_pixels": int((inst > 0).sum()),
            "mean_area": float(np.mean(areas)) if areas else 0,
            "min_area": int(min(areas)) if areas else 0,
            "max_area": int(max(areas)) if areas else 0,
        })

    if instance_stats:
        avg_instances = np.mean([s["num_instances"] for s in instance_stats])
        avg_fg = np.mean([s["total_fg_pixels"] for s in instance_stats])
        empty_count = sum(1 for s in instance_stats if s["num_instances"] == 0)

        result["instance_stats"] = {
            "sample_size": len(sample_files),
            "avg_instances_per_image": round(float(avg_instances), 2),
            "avg_fg_pixels": round(float(avg_fg), 0),
            "empty_instance_maps": empty_count,
            "empty_pct": round(100 * empty_count / len(sample_files), 1),
            "per_image": instance_stats[:5],  # first 5 for inspection
        }
        print(f"Instances: avg {avg_instances:.1f}/img, {empty_count}/{len(sample_files)} empty")

    # 5. Semantic class distribution (sample first 100)
    print("Computing semantic class distribution (first 100 images)...")
    sem_sample = semantic_files[:100]
    class_pixel_counts = Counter()
    for f in sem_sample:
        path = os.path.join(pseudo_dir, f)
        sem = np.array(Image.open(path))
        unique, counts = np.unique(sem, return_counts=True)
        for u, c in zip(unique, counts):
            class_pixel_counts[int(u)] += int(c)

    total_pixels = sum(class_pixel_counts.values())
    class_distribution = {
        k: round(v / total_pixels * 100, 3) for k, v in sorted(class_pixel_counts.items())
    }
    result["semantic_distribution"] = {
        "num_unique_classes": len(class_distribution),
        "class_pixel_pct": class_distribution,
    }
    print(f"Semantic: {len(class_distribution)} unique classes")

    # 6. Check leftImg8bit_sequence exists
    seq_dir = os.path.join(cityscapes_root, "leftImg8bit_sequence")
    img_dir = os.path.join(cityscapes_root, "leftImg8bit")
    result["image_dirs"] = {
        "leftImg8bit_sequence_exists": os.path.isdir(seq_dir),
        "leftImg8bit_sequence_is_symlink": os.path.islink(seq_dir),
        "leftImg8bit_exists": os.path.isdir(img_dir),
    }
    if os.path.islink(seq_dir):
        result["image_dirs"]["leftImg8bit_sequence_target"] = os.readlink(seq_dir)
    print(f"leftImg8bit_sequence: exists={os.path.isdir(seq_dir)}, symlink={os.path.islink(seq_dir)}")

    # 7. Check if images can be found for first 5 pseudo-labels
    print("Checking image path resolution...")
    missing_images = []
    for f in instance_files[:20]:
        city = f.split("_")[0]
        img_name = f.replace("_instance.png", ".png")
        # CUPS constructs: root/leftImg8bit_sequence/train/{city}/{img_name}
        img_path_train = os.path.join(cityscapes_root, "leftImg8bit_sequence", "train", city, img_name)
        img_path_val = os.path.join(cityscapes_root, "leftImg8bit_sequence", "val", city, img_name)
        if not os.path.exists(img_path_train) and not os.path.exists(img_path_val):
            missing_images.append(f)
    result["image_resolution"] = {
        "checked": min(20, len(instance_files)),
        "missing": len(missing_images),
        "missing_files": missing_images[:5],
    }
    if missing_images:
        print(f"WARNING: {len(missing_images)}/20 images not found!")
    else:
        print("OK: All checked images found.")

    # 8. File size fingerprint (first 10 instance files)
    print("Computing file size fingerprint...")
    size_fingerprint = []
    for f in instance_files[:10]:
        path = os.path.join(pseudo_dir, f)
        size = os.path.getsize(path)
        size_fingerprint.append({"file": f, "bytes": size})
    result["size_fingerprint"] = size_fingerprint

    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose pseudo-label directory")
    parser.add_argument("--pseudo_dir", required=True, help="Path to CUPS pseudo-label directory")
    parser.add_argument("--cityscapes_root", required=True, help="Path to Cityscapes root")
    parser.add_argument("--output", default="pseudo_diag.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"=== Pseudo-Label Diagnostics ===")
    print(f"Dir: {args.pseudo_dir}")
    print()

    result = diagnose(args.pseudo_dir, args.cityscapes_root)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nDiagnostics saved to {args.output}")


if __name__ == "__main__":
    main()
