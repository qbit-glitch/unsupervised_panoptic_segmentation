#!/usr/bin/env python3
"""Recursive Deep Spectral Clustering for pseudo-label generation.

Two-phase approach:
  Phase 1: Per-image recursive normalized cut on DINOv2 affinity graphs.
           Discovers adaptive segments (small objects get own segments).
  Phase 2: Global clustering of segment features via spherical k-means.
           Maps per-image segments to globally consistent cluster IDs.

Based on "Recursive Deep Spectral Clustering" (NeurIPS 2024).

Usage:
    python mbps_pytorch/generate_recursive_dsc_ablation.py \
        --cityscapes_root /data/cityscapes \
        --feat_subdir dinov3_features_vitl16 \
        --k 100 --seed 42
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.linalg import eigh
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from mbps_pytorch.generate_clustering_ablation import (
    PATCH_GRIDS,
    assign_clusters_cosine,
    compute_cluster_stats,
    detect_feat_dims,
    find_feature_files,
    fit_spherical_kmeans,
    load_features_normalized,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUT_H, OUT_W = 512, 1024


def normalized_cut_cost(
    affinity: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray
) -> float:
    """Compute the normalized cut cost between partition A and B."""
    cut_ab = affinity[np.ix_(mask_a, mask_b)].sum()
    assoc_a = affinity[mask_a, :].sum()
    assoc_b = affinity[mask_b, :].sum()
    if assoc_a < 1e-10 or assoc_b < 1e-10:
        return float("inf")
    return cut_ab / assoc_a + cut_ab / assoc_b


def recursive_ncut(
    features: np.ndarray,
    indices: np.ndarray,
    full_affinity: np.ndarray,
    ncut_threshold: float = 0.03,
    min_segment_size: int = 16,
    max_depth: int = 8,
    depth: int = 0,
) -> List[np.ndarray]:
    """Recursively bipartition patches using normalized cuts.

    Args:
        features: (N_sub, D) L2-normalized features for current partition.
        indices: (N_sub,) indices into the original patch array.
        full_affinity: (N_total, N_total) precomputed affinity matrix.
        ncut_threshold: stop if NCut cost exceeds this.
        min_segment_size: don't split segments smaller than this.
        max_depth: maximum recursion depth.
        depth: current depth.

    Returns:
        List of index arrays, one per discovered segment.
    """
    n = len(indices)
    if n < min_segment_size * 2 or depth >= max_depth:
        return [indices]

    sub_affinity = full_affinity[np.ix_(indices, indices)]

    d = sub_affinity.sum(axis=1)
    d_inv_sqrt = 1.0 / (np.sqrt(d) + 1e-8)
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * sub_affinity) * d_inv_sqrt[None, :]

    try:
        _, eigvecs = eigh(L_sym, subset_by_index=[1, 1])
    except Exception:
        return [indices]

    fiedler = eigvecs[:, 0]

    mask_a = fiedler <= 0
    mask_b = fiedler > 0

    n_a = mask_a.sum()
    n_b = mask_b.sum()
    if n_a < min_segment_size or n_b < min_segment_size:
        return [indices]

    cost = normalized_cut_cost(sub_affinity, mask_a, mask_b)
    if cost > ncut_threshold:
        return [indices]

    idx_a = indices[mask_a]
    idx_b = indices[mask_b]

    segments_a = recursive_ncut(
        features[mask_a], idx_a, full_affinity,
        ncut_threshold, min_segment_size, max_depth, depth + 1,
    )
    segments_b = recursive_ncut(
        features[mask_b], idx_b, full_affinity,
        ncut_threshold, min_segment_size, max_depth, depth + 1,
    )

    return segments_a + segments_b


def segment_single_image(
    features: np.ndarray,
    ncut_threshold: float = 0.03,
    min_segment_size: int = 16,
    max_depth: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run recursive NCut on one image's features.

    Args:
        features: (N_patches, D) L2-normalized.

    Returns:
        segment_ids: (N_patches,) per-patch segment assignment.
        segment_features: (n_segments, D) mean feature per segment, L2-normalized.
    """
    n = features.shape[0]

    affinity = features @ features.T
    np.clip(affinity, 0, None, out=affinity)
    np.fill_diagonal(affinity, 0.0)

    all_indices = np.arange(n)
    segments = recursive_ncut(
        features, all_indices, affinity,
        ncut_threshold, min_segment_size, max_depth,
    )

    segment_ids = np.zeros(n, dtype=np.int32)
    segment_features = []
    for seg_id, seg_indices in enumerate(segments):
        segment_ids[seg_indices] = seg_id
        mean_feat = features[seg_indices].mean(axis=0)
        mean_feat /= np.linalg.norm(mean_feat) + 1e-8
        segment_features.append(mean_feat)

    return segment_ids, np.array(segment_features, dtype=np.float32)


def phase1_per_image_segmentation(
    files: List[Dict],
    ncut_threshold: float = 0.03,
    min_segment_size: int = 16,
    max_depth: int = 8,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Phase 1: per-image recursive NCut segmentation.

    Returns:
        per_image_segments: list of (N_patches,) segment ID arrays.
        all_segment_features: (total_segments, D) concatenated segment features.
    """
    per_image_segments = []
    all_segment_features = []
    segment_counts = []

    for entry in tqdm(files, desc="Phase 1: recursive NCut"):
        feat = load_features_normalized(entry["feat"])
        seg_ids, seg_feats = segment_single_image(
            feat, ncut_threshold, min_segment_size, max_depth
        )
        per_image_segments.append(seg_ids)
        all_segment_features.append(seg_feats)
        segment_counts.append(len(seg_feats))

    all_features = np.concatenate(all_segment_features, axis=0)
    counts_arr = np.array(segment_counts)
    logger.info(
        f"Phase 1 done: {len(files)} images, "
        f"{all_features.shape[0]} total segments, "
        f"segments/image: min={counts_arr.min()}, "
        f"median={np.median(counts_arr):.0f}, max={counts_arr.max()}"
    )
    return per_image_segments, all_features


def phase2_global_clustering(
    segment_features: np.ndarray,
    k: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Phase 2: cluster segment features globally."""
    logger.info(f"Phase 2: spherical k-means on {segment_features.shape[0]} segments, k={k}")
    result = fit_spherical_kmeans(segment_features, k=k, seed=seed, refine_iters=20)
    return result["centers"]


def assign_recursive_labels(
    files: List[Dict],
    per_image_segments: List[np.ndarray],
    global_centers: np.ndarray,
    all_segment_features: np.ndarray,
    segment_offsets: List[int],
    output_dir: Path,
    split: str,
) -> None:
    """Map per-image segment IDs to global cluster IDs and save as PNGs."""
    # First assign all segment features to global clusters
    sims = all_segment_features @ global_centers.T
    segment_to_global = sims.argmax(axis=1).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(tqdm(files, desc=f"Assigning {split}")):
        seg_ids = per_image_segments[i]
        offset = segment_offsets[i]
        n_segs = seg_ids.max() + 1

        global_ids = segment_to_global[offset:offset + n_segs]
        label_flat = global_ids[seg_ids]

        n_patches = len(seg_ids)
        fh, fw = PATCH_GRIDS.get(n_patches, detect_feat_dims(entry["feat"]))
        label_2d = label_flat.reshape(fh, fw)
        label_full = np.array(
            Image.fromarray(label_2d).resize((OUT_W, OUT_H), Image.NEAREST)
        )

        city_dir = output_dir / split / entry["city"]
        city_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_full).save(str(city_dir / f"{entry['stem']}.png"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursive Deep Spectral Clustering for pseudo-labels"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--feat_subdir", type=str, default="dinov3_features_vitl16")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--ncut_threshold", type=float, default=1.0)
    parser.add_argument("--min_segment_size", type=int, default=16)
    parser.add_argument("--max_recursion_depth", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    out_subdir = f"pseudo_semantic_raw_dinov3_k{args.k}_recursive_dsc_vitl16"
    out_dir = root / out_subdir

    t0 = time.time()

    # Phase 1 + 2 on train
    train_files = find_feature_files(root, "train", args.feat_subdir)
    logger.info(f"Found {len(train_files)} train images")

    train_segments, train_seg_features = phase1_per_image_segmentation(
        train_files, args.ncut_threshold, args.min_segment_size, args.max_recursion_depth
    )

    # Compute segment offsets for mapping
    train_seg_offsets = []
    offset = 0
    for seg_ids in train_segments:
        train_seg_offsets.append(offset)
        offset += seg_ids.max() + 1

    global_centers = phase2_global_clustering(train_seg_features, args.k, args.seed)

    fit_time = time.time() - t0
    logger.info(f"Fitting done in {fit_time:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / "centroids.npz"), centers=global_centers)
    logger.info(f"Saved centroids to {out_dir / 'centroids.npz'}")

    # Assign train
    if "train" in args.splits:
        assign_recursive_labels(
            train_files, train_segments, global_centers,
            train_seg_features, train_seg_offsets, out_dir, "train",
        )

    # For val: must re-run Phase 1 per-image NCut
    if "val" in args.splits:
        val_files = find_feature_files(root, "val", args.feat_subdir)
        logger.info(f"Found {len(val_files)} val images")
        val_segments, val_seg_features = phase1_per_image_segmentation(
            val_files, args.ncut_threshold, args.min_segment_size,
            args.max_recursion_depth,
        )
        val_seg_offsets = []
        offset = 0
        for seg_ids in val_segments:
            val_seg_offsets.append(offset)
            offset += seg_ids.max() + 1
        assign_recursive_labels(
            val_files, val_segments, global_centers,
            val_seg_features, val_seg_offsets, out_dir, "val",
        )

    # Stats using the global centers (cosine assignment on raw features)
    logger.info("Computing cluster statistics...")
    stats = compute_cluster_stats(train_files[:200], global_centers)
    stats["fit_time_seconds"] = fit_time
    stats["method"] = "recursive_dsc"
    stats["k"] = args.k
    stats["ncut_threshold"] = args.ncut_threshold
    stats["total_segments_train"] = len(train_seg_features)
    stats["mean_segments_per_image"] = float(
        np.mean([s.max() + 1 for s in train_segments])
    )
    with open(str(out_dir / "cluster_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        f"Stats: entropy={stats['entropy']:.3f}, gini={stats['gini']:.3f}, "
        f"empty={stats['empty_clusters']}, "
        f"mean_segments/image={stats['mean_segments_per_image']:.1f}"
    )

    logger.info(f"All done -> {out_dir}")


if __name__ == "__main__":
    main()
