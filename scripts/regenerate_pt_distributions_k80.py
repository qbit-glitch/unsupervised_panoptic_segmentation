#!/usr/bin/env python3
"""Regenerate .pt distribution files for CUPS pseudo-labels in k=80 cluster space.

Bug context:
    The pseudo-label PNGs in cups_pseudo_labels_dcfa_simcf_depthpro/ contain
    k=80 cluster IDs (max=79), but the accompanying .pt distribution arrays
    are length 27 (CAUSE-style taxonomy), produced by an earlier converter.
    This dimension mismatch causes pseudo_label_dataset.PseudoLabelDataset to
    derive a thing/stuff split with only 5+22=27 classes, training the model
    on a CAUSE-27 taxonomy instead of the planned k=80 over-clustering.

This script walks a CUPS-format pseudo-label directory, recomputes:
    - "distribution all pixels":            (K,) per-cluster pixel counts
    - "distribution inside object proposals": (K,) per-cluster pixel counts
                                              over pixels inside instance masks
in the full K=80 cluster space, and overwrites each .pt in place.

NO ground truth labels are used. Cluster IDs come from the unsupervised
k-means on DINOv3 features; instance IDs come from the depth-guided
DepthPro split.

Usage:
    python scripts/regenerate_pt_distributions_k80.py \\
        --pseudo-dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_depthpro \\
        --num-clusters 80
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_distributions(semantic: np.ndarray, instance: np.ndarray, num_clusters: int) -> dict:
    """Compute per-cluster pixel counts in the full K cluster space."""
    sem_flat = semantic.flatten().astype(np.int64)
    inst_flat = instance.flatten()

    valid = (sem_flat >= 0) & (sem_flat < num_clusters)
    dist_all = torch.zeros(num_clusters, dtype=torch.float32)
    if valid.sum() > 0:
        counts = np.bincount(sem_flat[valid], minlength=num_clusters)
        dist_all = torch.from_numpy(counts[:num_clusters].astype(np.float32))

    inside = (inst_flat > 0) & valid
    dist_inside = torch.zeros(num_clusters, dtype=torch.float32)
    if inside.sum() > 0:
        counts = np.bincount(sem_flat[inside], minlength=num_clusters)
        dist_inside = torch.from_numpy(counts[:num_clusters].astype(np.float32))

    return {
        "distribution all pixels": dist_all,
        "distribution inside object proposals": dist_inside,
    }


def _process(stem_and_dir: Tuple[str, str, int]) -> Tuple[str, int, int]:
    stem, pseudo_dir, num_clusters = stem_and_dir
    pseudo = Path(pseudo_dir)
    sem_path = pseudo / f"{stem}_semantic.png"
    inst_path = pseudo / f"{stem}_instance.png"
    pt_path = pseudo / f"{stem}.pt"

    semantic = np.array(Image.open(sem_path))
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    instance = np.array(Image.open(inst_path))
    if instance.ndim == 3:
        instance = instance[..., 0]

    stats = compute_distributions(semantic, instance, num_clusters)
    torch.save(stats, str(pt_path))

    n_active = int((stats["distribution all pixels"] > 0).sum())
    n_inside = int((stats["distribution inside object proposals"] > 0).sum())
    return stem, n_active, n_inside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-dir", type=Path, required=True,
                        help="CUPS-format pseudo-label root (flat dir with *_semantic.png, *_instance.png, *.pt)")
    parser.add_argument("--num-clusters", type=int, default=80,
                        help="Target cluster count K (default 80)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pseudo_dir = args.pseudo_dir.expanduser()
    sem_paths = sorted(pseudo_dir.glob("*_semantic.png"))
    stems = [p.name.replace("_semantic.png", "") for p in sem_paths]
    logger.info("Found %d pseudo-labels in %s", len(stems), pseudo_dir)

    payloads = [(stem, str(pseudo_dir), args.num_clusters) for stem in stems]
    n_active_total = np.zeros(args.num_clusters, dtype=np.int64)
    n_inside_total = np.zeros(args.num_clusters, dtype=np.int64)

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for stem, n_active, n_inside in tqdm(
            executor.map(_process, payloads), total=len(payloads), desc="regen-pt"
        ):
            # Re-load to aggregate global cluster activity for the summary.
            stats = torch.load(pseudo_dir / f"{stem}.pt", weights_only=False)
            n_active_total += (stats["distribution all pixels"] > 0).numpy().astype(np.int64)
            n_inside_total += (stats["distribution inside object proposals"] > 0).numpy().astype(np.int64)

    logger.info("Cluster activity across dataset:")
    n_seen = int((n_active_total > 0).sum())
    n_inside_seen = int((n_inside_total > 0).sum())
    logger.info("  %d / %d clusters appear in semantic PNGs", n_seen, args.num_clusters)
    logger.info("  %d / %d clusters appear inside instance proposals", n_inside_seen, args.num_clusters)


if __name__ == "__main__":
    main()
