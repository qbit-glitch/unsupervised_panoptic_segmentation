#!/usr/bin/env python3
"""Visualize CC + k=80 + DepthPro depth-guided instances for training images.

Generates side-by-side comparison:
  Col 1: RGB input
  Col 2: DepthPro depth map (viridis)
  Col 3: Depth edges (Sobel > τ)
  Col 4: Semantic pseudo-labels (k=80 → 19-class colored)
  Col 5: Instance pseudo-labels (depth-guided CC, each instance a random color)

Uses SAME parameters as SPIdepth baseline: τ=0.20, A_min=1000.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel

# Cityscapes 19-class colors (RGB)
CS_COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)

THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def colorize_semantic(sem: np.ndarray) -> np.ndarray:
    """Map trainID semantic map to RGB."""
    h, w = sem.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(19):
        rgb[sem == c] = CS_COLORS[c]
    return rgb


def colorize_instances(sem: np.ndarray, instances: list) -> np.ndarray:
    """Color each instance with a unique random color, stuff in gray."""
    h, w = sem.shape
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)  # gray background

    # Stuff classes get their semantic color (dimmed)
    for c in range(11):
        mask = sem == c
        rgb[mask] = (CS_COLORS[c] * 0.5).astype(np.uint8)

    # Each instance gets a bright random color
    rng = np.random.RandomState(42)
    for mask, cls, score in instances:
        color = rng.randint(60, 255, size=3)
        rgb[mask] = color

    return rgb


def depth_guided_instances(semantic, depth, tau=0.20, min_area=1000,
                           dilation_iters=3, depth_blur_sigma=1.0):
    """Split thing regions using depth gradient edges."""
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64), sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    depth_edges = grad_mag > tau

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(THING_IDS):
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
                dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
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
    return instances, depth_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=str,
                        default="/Users/qbit-glitch/Desktop/datasets/cityscapes")
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--output", type=str, default="figures/depthpro_instances_train.png")
    args = parser.parse_args()

    root = Path(args.cityscapes_root)

    # Pick 3 diverse training images
    samples = [
        ("aachen", "aachen_000000_000019"),
        ("bremen", "bremen_000045_000019"),
        ("cologne", "cologne_000080_000019"),
    ]

    # Load cluster → trainID mapping
    centroids = np.load(str(root / "pseudo_semantic_raw_k80" / "kmeans_centroids.npz"))
    cluster_to_class = centroids["cluster_to_class"]
    lut = np.full(256, 255, dtype=np.uint8)
    for cid, tid in enumerate(cluster_to_class):
        lut[cid] = int(tid)

    fig, axes = plt.subplots(len(samples), 5, figsize=(25, 4 * len(samples)))

    col_titles = [
        "RGB Input",
        f"DepthPro Depth",
        f"Depth Edges (τ={args.tau})",
        "Semantic (k=80 → 19cls)",
        f"Instances (τ={args.tau}, A_min={args.min_area})",
    ]

    for row, (city, stem) in enumerate(samples):
        print(f"Processing {stem}...")

        # RGB
        rgb_path = root / "leftImg8bit" / "train" / city / f"{stem}_leftImg8bit.png"
        rgb = np.array(Image.open(rgb_path))

        # Depth
        depth_path = root / args.depth_subdir / "train" / city / f"{stem}.npy"
        depth = np.load(str(depth_path))
        h, w = depth.shape[:2]

        # Semantic
        sem_path = root / "pseudo_semantic_raw_k80" / "train" / city / f"{stem}.png"
        sem_raw = np.array(Image.open(sem_path))
        sem = lut[sem_raw]
        if sem.shape != (h, w):
            sem = np.array(Image.fromarray(sem).resize((w, h), Image.NEAREST))

        # Generate instances
        instances, depth_edges = depth_guided_instances(
            sem, depth, tau=args.tau, min_area=args.min_area
        )

        n_inst = len(instances)
        areas = [float(m.sum()) for m, c, s in instances]
        small = sum(1 for a in areas if a < 1000)

        print(f"  {n_inst} instances, {small} small (<1000px)")
        for m, c, s in instances:
            print(f"    {CLASS_NAMES[c]:15s}: {int(m.sum()):6d}px")

        # Plot
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(depth, cmap="viridis")
        axes[row, 2].imshow(depth_edges, cmap="gray")
        axes[row, 3].imshow(colorize_semantic(sem))
        axes[row, 4].imshow(colorize_instances(sem, instances))

        axes[row, 0].set_ylabel(stem.split("_")[0], fontsize=12, fontweight="bold")
        axes[row, 4].text(
            0.98, 0.02, f"{n_inst} inst",
            transform=axes[row, 4].transAxes, fontsize=10,
            ha="right", va="bottom", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    for col in range(5):
        axes[0, col].set_title(col_titles[col], fontsize=11, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
