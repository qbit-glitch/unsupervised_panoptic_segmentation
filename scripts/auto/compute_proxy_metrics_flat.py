#!/usr/bin/env python3
"""Compute proxy metrics on flat CUPS-format pseudo-label directories."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from tqdm import tqdm


def compute_sic(semantic: np.ndarray, instance: np.ndarray) -> float:
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


def compute_fragments(instance: np.ndarray) -> int:
    return len([uid for uid in np.unique(instance) if uid != 0])


def compute_stuff_contamination(semantic: np.ndarray, instance: np.ndarray) -> float:
    stuff_ids = set(range(0, 11))
    total = 0
    stuff_pixels = 0
    for uid in np.unique(instance):
        if uid == 0:
            continue
        mask = instance == uid
        sem_in_inst = semantic[mask]
        total += len(sem_in_inst)
        stuff_pixels += sum(1 for s in sem_in_inst.flatten() if s in stuff_ids)
    return stuff_pixels / (total + 1e-10) * 100


def compute_ler(semantic: np.ndarray) -> float:
    hist = np.bincount(semantic.flatten(), minlength=256)
    probs = hist / hist.sum()
    entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)
    return entropy


def compute_das(semantic: np.ndarray, depth: np.ndarray) -> float:
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
    dx = sobel(depth_norm, axis=1)
    dy = sobel(depth_norm, axis=0)
    depth_edges = np.sqrt(dx**2 + dy**2)
    sem_edges = np.zeros_like(semantic, dtype=bool)
    h, w = semantic.shape
    for i in range(h - 1):
        for j in range(w - 1):
            if semantic[i, j] != semantic[i, j + 1] or semantic[i, j] != semantic[i + 1, j]:
                sem_edges[i, j] = True
    if depth_edges.sum() == 0 or sem_edges.sum() == 0:
        return 0.0
    de = depth_edges / (depth_edges.std() + 1e-10)
    se = sem_edges.astype(float) / (sem_edges.std() + 1e-10)
    corr = (de * se).sum() / (np.sqrt((de**2).sum()) * np.sqrt((se**2).sum()) + 1e-10)
    return float(corr)


def evaluate_pseudo_dir(pseudo_dir: Path, depth_dir: Path = None, max_images: int = 100, seed: int = 42) -> dict:
    semantic_files = sorted(pseudo_dir.glob("*_semantic.png"))
    if not semantic_files:
        return {"error": "No semantic PNGs found"}

    # Filter val cities if possible, otherwise sample from all
    val_cities = {"frankfurt", "lindau", "munster"}
    val_files = [f for f in semantic_files if any(c in f.name for c in val_cities)]
    if val_files:
        files = val_files
    else:
        files = semantic_files

    random.seed(seed)
    if max_images and len(files) > max_images:
        files = random.sample(files, max_images)

    sics = []
    fragment_counts = []
    contaminations = []
    entropies = []
    dases = []

    for sem_path in tqdm(files, desc=f"Computing proxies for {pseudo_dir.name}"):
        inst_path = pseudo_dir / (sem_path.stem.replace("_semantic", "") + "_instance.png")
        if not inst_path.exists():
            continue
        semantic = np.array(Image.open(sem_path))
        instance = np.array(Image.open(inst_path))
        if semantic.shape != instance.shape:
            instance = np.array(Image.fromarray(instance).resize((semantic.shape[1], semantic.shape[0]), Image.NEAREST))

        sics.append(compute_sic(semantic, instance))
        fragment_counts.append(compute_fragments(instance))
        contaminations.append(compute_stuff_contamination(semantic, instance))
        entropies.append(compute_ler(semantic))

        if depth_dir:
            stem = sem_path.stem.replace("_semantic", "").replace("_leftImg8bit", "")
            # Try multiple depth file naming patterns
            depth_paths = [
                depth_dir / (stem + "_leftImg8bit_depth.npy"),
                depth_dir / (stem + "_depth.npy"),
                depth_dir / (stem + ".npy"),
            ]
            dp = None
            for p in depth_paths:
                if p.exists():
                    dp = p
                    break
            if dp:
                depth = np.load(dp)
                if depth.shape != semantic.shape:
                    depth = np.array(Image.fromarray(depth).resize((semantic.shape[1], semantic.shape[0]), Image.BILINEAR))
                dases.append(compute_das(semantic, depth))

    return {
        "num_images": len(files),
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
    parser.add_argument("--max_images", type=int, default=100)
    args = parser.parse_args()

    results = evaluate_pseudo_dir(args.pseudo_dir, args.depth_dir, args.max_images)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
