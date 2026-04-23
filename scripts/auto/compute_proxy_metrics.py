#!/usr/bin/env python3
"""Compute proxy metrics for pseudo-label sets.

Metrics:
  - SIC: Semantic-Instance Consistency (% instance pixels agreeing with majority semantic class)
  - FE: Fragmentation Efficiency (PQ_things / fragments_per_image)
  - DAS: Depth-Alignment Score (correlation between depth edges and semantic boundaries)
  - LER: Label Entropy Reduction (reduction in per-pixel label uncertainty)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm


def load_pseudolabels(pseudo_dir: Path, max_images: int = None):
    """Load semantic and instance PNGs from CUPS-format directory."""
    semantic_files = sorted(pseudo_dir.glob("*_semantic.png"))
    if max_images:
        semantic_files = semantic_files[:max_images]

    pairs = []
    for sem_path in semantic_files:
        inst_path = pseudo_dir / (sem_path.stem.replace("_semantic", "") + "_instance.png")
        if inst_path.exists():
            pairs.append((sem_path, inst_path))
    return pairs


def compute_sic(semantic: np.ndarray, instance: np.ndarray) -> float:
    """Semantic-Instance Consistency: % of instance pixels with majority semantic class."""
    total_pixels = 0
    consistent_pixels = 0

    inst_ids = np.unique(instance)
    for uid in inst_ids:
        if uid == 0:
            continue
        mask = instance == uid
        sem_in_inst = semantic[mask]
        if len(sem_in_inst) == 0:
            continue
        majority_class = np.bincount(sem_in_inst.flatten()).argmax()
        consistent = (sem_in_inst == majority_class).sum()
        consistent_pixels += consistent
        total_pixels += len(sem_in_inst)

    return consistent_pixels / (total_pixels + 1e-10) * 100


def compute_fragments(instance: np.ndarray, thing_ids: set = None) -> int:
    """Count thing-class fragments."""
    inst_ids = np.unique(instance)
    fragments = 0
    for uid in inst_ids:
        if uid == 0:
            continue
        fragments += 1
    return fragments


def compute_stuff_contamination(semantic: np.ndarray, instance: np.ndarray) -> float:
    """% of fragment pixels that belong to stuff classes."""
    stuff_ids = set(range(0, 11))
    total = 0
    stuff_pixels = 0

    inst_ids = np.unique(instance)
    for uid in inst_ids:
        if uid == 0:
            continue
        mask = instance == uid
        sem_in_inst = semantic[mask]
        total += len(sem_in_inst)
        stuff_pixels += sum(1 for s in sem_in_inst.flatten() if s in stuff_ids)

    return stuff_pixels / (total + 1e-10) * 100


def compute_ler(semantic: np.ndarray) -> float:
    """Label entropy: average per-pixel uncertainty."""
    hist = np.bincount(semantic.flatten(), minlength=256)
    probs = hist / hist.sum()
    entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)
    return entropy


def compute_das(semantic: np.ndarray, depth: np.ndarray) -> float:
    """Depth-Alignment Score: correlation between depth edges and semantic boundaries."""
    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)

    # Depth edges
    dx = sobel(depth_norm, axis=1)
    dy = sobel(depth_norm, axis=0)
    depth_edges = np.sqrt(dx**2 + dy**2)

    # Semantic boundaries (binary)
    sem_edges = np.zeros_like(semantic, dtype=bool)
    h, w = semantic.shape
    for i in range(h - 1):
        for j in range(w - 1):
            if semantic[i, j] != semantic[i, j + 1] or semantic[i, j] != semantic[i + 1, j]:
                sem_edges[i, j] = True

    # Correlation
    if depth_edges.sum() == 0 or sem_edges.sum() == 0:
        return 0.0

    # Normalize
    de = depth_edges / (depth_edges.std() + 1e-10)
    se = sem_edges.astype(float) / (sem_edges.std() + 1e-10)

    corr = (de * se).sum() / (np.sqrt((de**2).sum()) * np.sqrt((se**2).sum()) + 1e-10)
    return float(corr)


def evaluate_pseudo_dir(pseudo_dir: Path, depth_dir: Path = None, max_images: int = None) -> dict:
    pairs = load_pseudolabels(pseudo_dir, max_images)
    if not pairs:
        return {"error": "No pseudo-label pairs found"}

    sics = []
    fragment_counts = []
    contaminations = []
    entropies = []
    dases = []

    for sem_path, inst_path in tqdm(pairs, desc="Computing proxies"):
        semantic = np.array(Image.open(sem_path))
        instance = np.array(Image.open(inst_path))

        # Ensure same size
        if semantic.shape != instance.shape:
            instance = np.array(Image.fromarray(instance).resize((semantic.shape[1], semantic.shape[0]), Image.NEAREST))

        sics.append(compute_sic(semantic, instance))
        fragment_counts.append(compute_fragments(instance))
        contaminations.append(compute_stuff_contamination(semantic, instance))
        entropies.append(compute_ler(semantic))

        if depth_dir:
            stem = sem_path.stem.replace("_semantic", "")
            depth_path = depth_dir / (stem + "_depth.npy")
            if not depth_path.exists():
                depth_path = depth_dir / (stem + "_leftImg8bit_depth.npy")
            if depth_path.exists():
                depth = np.load(depth_path)
                if depth.shape != semantic.shape:
                    depth = np.array(Image.fromarray(depth).resize((semantic.shape[1], semantic.shape[0]), Image.BILINEAR))
                dases.append(compute_das(semantic, depth))

    return {
        "num_images": len(pairs),
        "SIC_mean": float(np.mean(sics)),
        "SIC_std": float(np.std(sics)),
        "fragments_per_image_mean": float(np.mean(fragment_counts)),
        "fragments_per_image_std": float(np.std(fragment_counts)),
        "stuff_contamination_mean": float(np.mean(contaminations)),
        "stuff_contamination_std": float(np.std(contaminations)),
        "LER_mean": float(np.mean(entropies)),
        "LER_std": float(np.std(entropies)),
        "DAS_mean": float(np.mean(dases)) if dases else None,
        "DAS_std": float(np.std(dases)) if dases else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_dir", type=Path, required=True)
    parser.add_argument("--depth_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    results = evaluate_pseudo_dir(args.pseudo_dir, args.depth_dir, args.max_images)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
