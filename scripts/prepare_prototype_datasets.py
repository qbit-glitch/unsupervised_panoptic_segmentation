#!/usr/bin/env python3
"""Download and prepare prototype datasets for MBPS.

Usage:
    # Download NYU Depth V2 (primary prototype — has real depth)
    python scripts/prepare_prototype_datasets.py nyu --data-dir data/nyu_depth_v2

    # Download PASCAL VOC 2012
    python scripts/prepare_prototype_datasets.py pascal --data-dir data/pascal_voc

    # Create 5% COCO-Stuff-27 subset (requires full COCO already downloaded)
    python scripts/prepare_prototype_datasets.py coco_subset \
        --source-dir /data/coco --output-dir data/coco_stuff27_5pct --fraction 0.05

    # Download ZoeDepth checkpoint
    python scripts/prepare_prototype_datasets.py zoedepth --output-dir checkpoints/zoedepth

    # Download all prototype datasets
    python scripts/prepare_prototype_datasets.py all --data-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  ✓ Already exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  ⬇ Downloading {desc or os.path.basename(dest)}...")
    print(f"    URL: {url}")
    print(f"    Dest: {dest}")

    try:
        urllib.request.urlretrieve(url, dest, _progress_hook)
        print()  # newline after progress
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        raise


def _progress_hook(count, block_size, total_size):
    """Progress hook for urlretrieve."""
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        mb_done = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r    [{pct:3d}%] {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)


# ─────────────────────────────────────────────────────────────
# NYU Depth V2
# ─────────────────────────────────────────────────────────────

def download_nyu_depth_v2(data_dir: str):
    """Download and extract NYU Depth V2 labeled dataset.

    Downloads the official .mat file and extracts images, depth, and labels.
    - 1,449 images total (795 train / 654 test)
    - 40 semantic classes
    - Real Kinect depth (no ZoeDepth needed!)
    """
    print("\n" + "=" * 60)
    print("  📦 NYU Depth V2 — Downloading")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)
    mat_path = os.path.join(data_dir, "nyu_depth_v2_labeled.mat")

    # Official NYU Depth V2 labeled dataset (~2.8 GB)
    # Mirror URL (original NYU server can be slow)
    urls = [
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
        "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/nyu_depth_v2_labeled.mat",
    ]

    if not os.path.exists(mat_path):
        for url in urls:
            try:
                download_file(url, mat_path, "NYU Depth V2 labeled.mat (~2.8 GB)")
                break
            except Exception:
                print(f"  ⚠ Trying next mirror...")
                continue
        else:
            print("  ✗ All mirrors failed. Trying wget...")
            subprocess.run(
                ["wget", "-c", urls[0], "-O", mat_path],
                check=True,
            )

    # Extract images, depth, and labels from .mat file
    print("\n  📂 Extracting images, depth, and labels...")
    _extract_nyu_mat(mat_path, data_dir)

    print(f"\n  ✅ NYU Depth V2 ready at: {data_dir}")


def _extract_nyu_mat(mat_path: str, output_dir: str):
    """Extract images, depth, and labels from NYU .mat file.
    
    The NYU .mat file is HDF5 v7.3 format where MATLAB stores arrays
    transposed. Typical shapes in h5py:
      images: (N, 3, W, H) — need transpose to (H, W, 3)
      depths: (N, W, H)    — need transpose to (H, W)
      labels: (N, W, H)    — need transpose to (H, W)
    """
    try:
        import h5py
    except ImportError:
        print("  Installing h5py...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py", "-q"])
        import h5py

    from PIL import Image

    images_dir = os.path.join(output_dir, "images")
    depth_dir = os.path.join(output_dir, "depth")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Check if already extracted
    existing = list(Path(images_dir).glob("*.png"))
    if len(existing) > 1000:
        print(f"  ✓ Already extracted ({len(existing)} images)")
        return

    print(f"  Loading {mat_path}...")
    with h5py.File(mat_path, "r") as f:
        print(f"  Keys: {list(f.keys())}")
        
        # Get arrays — shapes vary by .mat version
        images_ds = f["images"]
        depths_ds = f["depths"]
        labels_ds = f["labels"]
        
        print(f"  images shape: {images_ds.shape}, dtype: {images_ds.dtype}")
        print(f"  depths shape: {depths_ds.shape}, dtype: {depths_ds.dtype}")
        print(f"  labels shape: {labels_ds.shape}, dtype: {labels_ds.dtype}")
        
        # Determine number of images and array layout
        # HDF5/MATLAB v7.3: arrays are transposed
        # MATLAB stores (H, W, 3, N) → h5py reads as (N, 3, W, H)
        img_shape = images_ds.shape
        
        if len(img_shape) == 4:
            if img_shape[1] == 3:
                # (N, 3, W, H) format — HDF5 transposed
                n_images = img_shape[0]
                layout = "hdf5_transposed"
            elif img_shape[3] == 3:
                # (N, H, W, 3) format — standard
                n_images = img_shape[0]
                layout = "standard"
            elif img_shape[0] == 3:
                # (3, H, W, N) format — another transposition
                n_images = img_shape[3]
                layout = "channels_first"
            else:
                # Try to figure it out — smallest dim with value 3 is channels
                n_images = img_shape[0]
                layout = "standard"
                print(f"  ⚠ Unknown layout, assuming standard: {img_shape}")
        else:
            raise ValueError(f"Unexpected images shape: {img_shape}")
        
        print(f"  Layout: {layout}, N={n_images}")

        for idx in range(n_images):
            if layout == "hdf5_transposed":
                # (N, 3, W, H) → (H, W, 3)
                img = np.array(images_ds[idx]).transpose(2, 1, 0)  # (3, W, H) → (H, W, 3)
                depth = np.array(depths_ds[idx]).T  # (W, H) → (H, W)
                label = np.array(labels_ds[idx]).T  # (W, H) → (H, W)
            elif layout == "channels_first":
                # (3, H, W, N) → (H, W, 3)
                img = np.array(images_ds[:, :, :, idx]).transpose(1, 2, 0)
                depth = np.array(depths_ds[:, :, idx])
                label = np.array(labels_ds[:, :, idx])
            else:
                # (N, H, W, 3) — standard
                img = np.array(images_ds[idx])
                depth = np.array(depths_ds[idx])
                label = np.array(labels_ds[idx])

            name = f"{idx:05d}"
            
            # Save image
            img_uint8 = img.astype(np.uint8)
            Image.fromarray(img_uint8).save(
                os.path.join(images_dir, f"{name}.png")
            )
            
            # Save depth as numpy
            np.save(
                os.path.join(depth_dir, f"{name}.npy"),
                depth.astype(np.float32),
            )
            
            # Save label as 16-bit PNG
            label_uint16 = label.astype(np.uint16)
            Image.fromarray(label_uint16).save(
                os.path.join(labels_dir, f"{name}.png")
            )

            if (idx + 1) % 200 == 0:
                print(f"    Extracted {idx + 1}/{n_images}")

    # Create train/test split file
    # Standard NYU split: first 795 = train, last 654 = test
    split_path = os.path.join(output_dir, "splits.txt")
    with open(split_path, "w") as f:
        for idx in range(n_images):
            split = "train" if idx < 795 else "test"
            f.write(f"{idx:05d} {split}\n")

    print(f"  ✓ Extracted {n_images} images to {output_dir}")
    print(f"    Images: {images_dir} ({n_images} files)")
    print(f"    Depth:  {depth_dir} ({n_images} files)")
    print(f"    Labels: {labels_dir} ({n_images} files)")


# ─────────────────────────────────────────────────────────────
# PASCAL VOC 2012
# ─────────────────────────────────────────────────────────────

def download_pascal_voc(data_dir: str):
    """Download PASCAL VOC 2012 segmentation dataset.

    - ~1,500 train + ~1,500 val images
    - 20 classes + background = 21 classes
    - Both semantic and instance segmentation annotations
    """
    print("\n" + "=" * 60)
    print("  📦 PASCAL VOC 2012 — Downloading")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")
    voc_dir = os.path.join(data_dir, "VOCdevkit", "VOC2012")

    # Check if already extracted
    if os.path.isdir(voc_dir):
        n_imgs = len(list(Path(os.path.join(voc_dir, "JPEGImages")).glob("*.jpg")))
        if n_imgs > 0:
            print(f"  ✓ Already downloaded ({n_imgs} images)")
            return

    # Download VOC 2012 (~2 GB)
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    download_file(url, tar_path, "PASCAL VOC 2012 (~2 GB)")

    # Extract
    print("\n  📂 Extracting...")
    subprocess.run(
        ["tar", "xf", tar_path, "-C", data_dir],
        check=True,
    )

    n_imgs = len(list(Path(os.path.join(voc_dir, "JPEGImages")).glob("*.jpg")))
    n_seg = len(list(Path(os.path.join(voc_dir, "SegmentationClass")).glob("*.png")))
    n_inst = len(list(Path(os.path.join(voc_dir, "SegmentationObject")).glob("*.png")))

    print(f"\n  ✅ PASCAL VOC 2012 ready at: {voc_dir}")
    print(f"    Images: {n_imgs}")
    print(f"    Semantic labels: {n_seg}")
    print(f"    Instance labels: {n_inst}")


# ─────────────────────────────────────────────────────────────
# COCO-Stuff-27 Subset
# ─────────────────────────────────────────────────────────────

def create_coco_subset(source_dir: str, output_dir: str, fraction: float = 0.05):
    """Create a deterministic subset of COCO-Stuff-27.

    Uses symlinks to avoid duplicating data.

    Args:
        source_dir: Path to full COCO dataset.
        output_dir: Path to write the subset.
        fraction: Fraction of training data to use (0.05 = 5%).
    """
    print("\n" + "=" * 60)
    print(f"  📦 COCO-Stuff-27 ({fraction*100:.0f}% subset) — Creating")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train2017", "val2017"]:
        src_img_dir = os.path.join(source_dir, "images", split)
        src_ann_dir = os.path.join(source_dir, "annotations", f"stuff_{split}_pixelmaps")

        if not os.path.isdir(src_img_dir):
            print(f"  ⚠ Source not found: {src_img_dir}")
            print(f"    Download full COCO first to {source_dir}")
            continue

        # Collect all images
        images = sorted(Path(src_img_dir).glob("*.jpg"))
        total = len(images)

        # Deterministic subset
        rng = np.random.RandomState(42)
        n_subset = max(1, int(total * fraction))
        indices = rng.choice(total, size=n_subset, replace=False)
        selected = [images[i] for i in sorted(indices)]

        # Create output dirs
        dst_img_dir = os.path.join(output_dir, "images", split)
        dst_ann_dir = os.path.join(output_dir, "annotations", f"stuff_{split}_pixelmaps")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)

        # Symlink selected images + annotations
        linked = 0
        for img_path in selected:
            dst_img = os.path.join(dst_img_dir, img_path.name)
            if not os.path.exists(dst_img):
                os.symlink(str(img_path), dst_img)

            # Corresponding annotation
            ann_name = img_path.stem + ".png"
            src_ann = os.path.join(src_ann_dir, ann_name)
            dst_ann = os.path.join(dst_ann_dir, ann_name)
            if os.path.exists(src_ann) and not os.path.exists(dst_ann):
                os.symlink(src_ann, dst_ann)

            linked += 1

        print(f"  {split}: {linked}/{total} images selected ({fraction*100:.0f}%)")

    print(f"\n  ✅ COCO-Stuff-27 subset ready at: {output_dir}")


# ─────────────────────────────────────────────────────────────
# ZoeDepth Checkpoints
# ─────────────────────────────────────────────────────────────

def download_zoedepth(output_dir: str):
    """Download ZoeDepth pre-trained checkpoints.

    Downloads ZoeD_N (NYU-trained) and ZoeD_NK (NYU+KITTI) models.
    These are used for pre-computing depth maps on datasets
    that don't have real depth (PASCAL VOC, COCO).
    """
    print("\n" + "=" * 60)
    print("  📦 ZoeDepth — Downloading Checkpoints")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # ZoeDepth model checkpoints from official repo
    checkpoints = {
        "ZoeD_M12_N.pt": {
            "url": "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt",
            "desc": "ZoeD_M12_N (NYU indoor, 93MB)",
        },
        "ZoeD_M12_NK.pt": {
            "url": "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt",
            "desc": "ZoeD_M12_NK (NYU+KITTI indoor+outdoor, 93MB)",
        },
    }

    for filename, info in checkpoints.items():
        dest = os.path.join(output_dir, filename)
        download_file(info["url"], dest, info["desc"])

    # Also clone the ZoeDepth repo for the model code
    zoedepth_repo = os.path.join(os.path.dirname(output_dir), "refs", "zoedepth")
    if not os.path.isdir(zoedepth_repo):
        print("\n  📦 Cloning ZoeDepth repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/isl-org/ZoeDepth.git", zoedepth_repo],
            check=True,
        )
        print(f"  ✓ ZoeDepth repo cloned to: {zoedepth_repo}")
    else:
        print(f"  ✓ ZoeDepth repo already exists: {zoedepth_repo}")

    print(f"\n  ✅ ZoeDepth checkpoints ready at: {output_dir}")
    print(f"    Use with: python scripts/precompute_depth.py \\")
    print(f"      --data_dir <dataset_path> --output_dir <depth_path>")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare prototype datasets for MBPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Dataset to prepare")

    # NYU
    nyu_parser = subparsers.add_parser("nyu", help="Download NYU Depth V2")
    nyu_parser.add_argument("--data-dir", type=str, default="data/nyu_depth_v2")

    # PASCAL VOC
    pascal_parser = subparsers.add_parser("pascal", help="Download PASCAL VOC 2012")
    pascal_parser.add_argument("--data-dir", type=str, default="data/pascal_voc")

    # COCO subset
    coco_parser = subparsers.add_parser("coco_subset", help="Create COCO-Stuff-27 subset")
    coco_parser.add_argument("--source-dir", type=str, required=True, help="Full COCO path")
    coco_parser.add_argument("--output-dir", type=str, default="data/coco_stuff27_5pct")
    coco_parser.add_argument("--fraction", type=float, default=0.05)

    # ZoeDepth
    zoe_parser = subparsers.add_parser("zoedepth", help="Download ZoeDepth checkpoints")
    zoe_parser.add_argument("--output-dir", type=str, default="checkpoints/zoedepth")

    # All
    all_parser = subparsers.add_parser("all", help="Download all prototype datasets")
    all_parser.add_argument("--data-dir", type=str, default="data")
    all_parser.add_argument("--coco-source", type=str, default=None,
                           help="Full COCO path (skip COCO subset if not set)")

    args = parser.parse_args()

    if args.command == "nyu":
        download_nyu_depth_v2(args.data_dir)
    elif args.command == "pascal":
        download_pascal_voc(args.data_dir)
    elif args.command == "coco_subset":
        create_coco_subset(args.source_dir, args.output_dir, args.fraction)
    elif args.command == "zoedepth":
        download_zoedepth(args.output_dir)
    elif args.command == "all":
        download_zoedepth(os.path.join(args.data_dir, "..", "checkpoints", "zoedepth"))
        download_nyu_depth_v2(os.path.join(args.data_dir, "nyu_depth_v2"))
        download_pascal_voc(os.path.join(args.data_dir, "pascal_voc"))
        if args.coco_source:
            create_coco_subset(
                args.coco_source,
                os.path.join(args.data_dir, "coco_stuff27_5pct"),
            )
        else:
            print("\n  ⚠ Skipping COCO subset (--coco-source not provided)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
