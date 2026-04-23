#!/usr/bin/env python3
"""SIMCF: Semantic-Instance Mutual Consistency Filtering.

Three-step pseudo-label refinement exploiting cross-modal consistency
between independently-generated semantic and instance labels.

Step A -- Instance validates semantics:
    Within each instance, majority-vote the semantic class, fix outlier pixels.
Step B -- Semantics validate instances:
    Merge adjacent same-class instances with high DINOv3 feature similarity.
Step C -- Depth validates semantics:
    Mask pixels whose depth deviates >3-sigma from per-class depth profile.

Usage:
    python scripts/refine_simcf.py \
        --input_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020 \
        --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_simcf_abc \
        --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
        --cityscapes_root ~/Desktop/datasets/cityscapes \
        --steps A,B,C
"""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
NUM_CLASSES = 19


def _extract_city(cups_stem: str) -> str:
    """Extract city name from CUPS stem like 'aachen_000000_000019_leftImg8bit'."""
    match = re.match(r"^(.+?)_\d{6}_\d{6}_leftImg8bit$", cups_stem)
    if match:
        return match.group(1)
    return cups_stem.split("_")[0]


def _base_stem(cups_stem: str) -> str:
    """Strip _leftImg8bit suffix: 'aachen_000000_000019_leftImg8bit' -> 'aachen_000000_000019'."""
    return cups_stem.replace("_leftImg8bit", "")


def list_cups_images(input_dir: Path) -> list:
    """List all CUPS image stems from semantic PNGs."""
    stems = []
    for p in sorted(input_dir.glob("*_semantic.png")):
        stem = p.name.replace("_semantic.png", "")
        stems.append(stem)
    return stems


# ---------------------------------------------------------------------------
# Step A: Instance validates semantics
# ---------------------------------------------------------------------------

def step_a(semantic: np.ndarray, instance: np.ndarray,
           cluster_to_class: np.ndarray, num_clusters: int) -> int:
    """Fix semantic labels using per-instance majority vote.

    Args:
        semantic: (H, W) uint8 cluster IDs, modified in-place.
        instance: (H, W) uint16 instance IDs.
        cluster_to_class: (K,) cluster -> trainID mapping.
        num_clusters: number of clusters (e.g. 80).

    Returns:
        Number of pixels changed.
    """
    n_changed = 0
    for iid in np.unique(instance):
        if iid == 0:
            continue
        ys, xs = np.where(instance == iid)
        if len(ys) == 0:
            continue

        clusters = semantic[ys, xs]
        train_ids = cluster_to_class[clusters]

        # Majority trainID within this instance
        tid_counts = np.bincount(train_ids[train_ids < NUM_CLASSES], minlength=NUM_CLASSES)
        if tid_counts.sum() == 0:
            continue
        majority_tid = tid_counts.argmax()

        # Find inconsistent pixels (only valid trainIDs, not unmapped 255)
        inconsistent = (train_ids != majority_tid) & (train_ids < NUM_CLASSES)
        if not inconsistent.any():
            continue

        # Best cluster: most frequent cluster that maps to majority_tid within this instance
        consistent = ~inconsistent
        if not consistent.any():
            continue
        consistent_clusters = clusters[consistent]
        best_cluster = np.bincount(consistent_clusters, minlength=num_clusters).argmax()

        # Fix inconsistent pixels
        semantic[ys[inconsistent], xs[inconsistent]] = best_cluster
        n_changed += int(inconsistent.sum())

    return n_changed


# ---------------------------------------------------------------------------
# Step B: Semantics validate instances
# ---------------------------------------------------------------------------

def step_b(semantic: np.ndarray, instance: np.ndarray,
           features: np.ndarray, cluster_to_class: np.ndarray,
           sim_threshold: float = 0.85, dilate_px: int = 3) -> tuple:
    """Merge adjacent same-class instances with high DINOv3 feature similarity.

    Args:
        semantic: (H, W) uint8 cluster IDs.
        instance: (H, W) uint16 instance IDs, modified in-place.
        features: (N_patches, D) L2-normalized DINOv3 features.
        cluster_to_class: (K,) cluster -> trainID mapping.
        sim_threshold: cosine similarity threshold for merging.
        dilate_px: dilation pixels for adjacency detection.

    Returns:
        (modified_instance, n_merges)
    """
    # Downsample instance to feature resolution for feature extraction
    inst_small = np.array(
        Image.fromarray(instance).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1)

    instance_ids = np.unique(instance)
    instance_ids = instance_ids[instance_ids > 0]
    if len(instance_ids) < 2:
        return instance, 0

    # Per-instance class (majority vote) and mean feature
    mapped_sem = cluster_to_class[semantic]
    inst_class = {}
    inst_feat = {}

    for iid in instance_ids:
        # Class from full-resolution semantic
        mask = instance == iid
        tids = mapped_sem[mask]
        valid = tids[tids < NUM_CLASSES]
        if len(valid) == 0:
            continue
        inst_class[iid] = int(np.bincount(valid, minlength=NUM_CLASSES).argmax())

        # Mean feature from patch resolution
        mask_s = inst_small == iid
        if not mask_s.any():
            continue
        patches = feat_2d[mask_s]
        feat = patches.mean(axis=0)
        norm = np.linalg.norm(feat) + 1e-8
        inst_feat[iid] = feat / norm

    # Build adjacency at full resolution
    struct = ndimage.generate_binary_structure(2, 1)
    adjacency = set()
    for iid in instance_ids:
        if iid not in inst_class:
            continue
        mask = instance == iid
        dilated = ndimage.binary_dilation(mask, structure=struct, iterations=dilate_px)
        border = dilated & ~mask
        for nb in np.unique(instance[border]):
            if nb == 0 or nb == iid or nb not in inst_class:
                continue
            adjacency.add((min(iid, nb), max(iid, nb)))

    # Filter: same class + high cosine similarity
    merge_pairs = []
    for i, j in adjacency:
        if inst_class.get(i) != inst_class.get(j):
            continue
        if i not in inst_feat or j not in inst_feat:
            continue
        sim = float(np.dot(inst_feat[i], inst_feat[j]))
        if sim > sim_threshold:
            merge_pairs.append((i, j))

    if not merge_pairs:
        return instance, 0

    # Union-find
    parent = {iid: iid for iid in instance_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in merge_pairs:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Renumber instances contiguously
    new_instance = np.zeros_like(instance)
    root_to_new = {}
    next_id = 1
    for iid in sorted(instance_ids):
        root = find(iid)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        new_instance[instance == iid] = root_to_new[root]

    n_merges = len(instance_ids) - len(root_to_new)
    return new_instance, n_merges


# ---------------------------------------------------------------------------
# Step C: Depth validates semantics
# ---------------------------------------------------------------------------

def compute_depth_stats(stems: list, input_dir: Path, depth_dir: Path,
                        cluster_to_class: np.ndarray) -> tuple:
    """First pass: compute per-class depth mean and std across all images.

    Returns:
        (class_mean, class_std) each shape (NUM_CLASSES,)
    """
    logger.info("Step C first pass: computing per-class depth statistics...")
    class_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_sum_sq = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    for cups_stem in tqdm(stems, desc="Depth stats"):
        city = _extract_city(cups_stem)
        base = _base_stem(cups_stem)

        sem = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        mapped = cluster_to_class[sem].astype(np.int64)
        sem_h, sem_w = sem.shape

        depth_path = depth_dir / "train" / city / f"{base}.npy"
        if not depth_path.exists():
            depth_path = depth_dir / "train" / city / f"{cups_stem}.npy"
        if not depth_path.exists():
            continue

        depth = np.load(str(depth_path)).astype(np.float64)
        if depth.shape != (sem_h, sem_w):
            depth = np.array(
                Image.fromarray(depth.astype(np.float32)).resize(
                    (sem_w, sem_h), Image.BILINEAR
                )
            ).astype(np.float64)

        for cls in range(NUM_CLASSES):
            mask = mapped == cls
            if not mask.any():
                continue
            vals = depth[mask]
            class_sum[cls] += vals.sum()
            class_sum_sq[cls] += (vals ** 2).sum()
            class_count[cls] += len(vals)

    safe_count = np.maximum(class_count, 1)
    class_mean = class_sum / safe_count
    class_var = class_sum_sq / safe_count - class_mean ** 2
    class_std = np.sqrt(np.maximum(class_var, 0.0))

    for cls in range(NUM_CLASSES):
        if class_count[cls] > 0:
            logger.info(f"  class {cls:2d}: mean={class_mean[cls]:.3f}, "
                        f"std={class_std[cls]:.3f}, n={class_count[cls]}")
    return class_mean, class_std


def step_c(semantic: np.ndarray, depth: np.ndarray,
           cluster_to_class: np.ndarray,
           class_mean: np.ndarray, class_std: np.ndarray,
           sigma_threshold: float = 3.0) -> int:
    """Mask pixels whose depth deviates from class profile. Modifies semantic in-place.

    Returns:
        Number of pixels masked.
    """
    mapped = cluster_to_class[semantic].astype(np.int64)
    n_masked = 0
    for cls in range(NUM_CLASSES):
        if class_std[cls] < 1e-6:
            continue
        mask = mapped == cls
        if not mask.any():
            continue
        deviation = np.abs(depth[mask] - class_mean[cls])
        outlier = deviation > sigma_threshold * class_std[cls]
        if outlier.any():
            ys, xs = np.where(mask)
            semantic[ys[outlier], xs[outlier]] = 255
            n_masked += int(outlier.sum())
    return n_masked


# ---------------------------------------------------------------------------
# Distribution computation (from convert_to_cups_format.py)
# ---------------------------------------------------------------------------

def compute_distributions(semantic: np.ndarray, instance: np.ndarray,
                          num_classes: int) -> dict:
    """Compute per-class pixel distributions for CUPS thing/stuff split."""
    sem_flat = semantic.flatten().astype(np.int64)
    inst_flat = instance.flatten()

    valid = sem_flat < num_classes
    dist_all = torch.zeros(num_classes, dtype=torch.float32)
    if valid.sum() > 0:
        counts = np.bincount(sem_flat[valid], minlength=num_classes)
        dist_all = torch.from_numpy(counts[:num_classes].astype(np.float32))

    inside = (inst_flat > 0) & valid
    dist_inside = torch.zeros(num_classes, dtype=torch.float32)
    if inside.sum() > 0:
        counts = np.bincount(sem_flat[inside], minlength=num_classes)
        dist_inside = torch.from_numpy(counts[:num_classes].astype(np.float32))

    return {
        "distribution all pixels": dist_all,
        "distribution inside object proposals": dist_inside,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SIMCF pseudo-label refinement")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="CUPS flat label directory (source)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for refined labels")
    parser.add_argument("--centroids_path", type=str, required=True,
                        help="Path to kmeans_centroids.npz with cluster_to_class")
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Cityscapes root (for features and depth)")
    parser.add_argument("--steps", type=str, default="A,B,C",
                        help="Comma-separated steps to apply (default: A,B,C)")
    parser.add_argument("--features_subdir", type=str, default="dinov3_features",
                        help="Feature directory under cityscapes_root")
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro",
                        help="Depth directory under cityscapes_root")
    parser.add_argument("--sim_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for Step B merging")
    parser.add_argument("--sigma_threshold", type=float, default=3.0,
                        help="Sigma threshold for Step C depth outlier masking")
    parser.add_argument("--num_clusters", type=int, default=80,
                        help="Number of k-means clusters")
    args = parser.parse_args()

    steps = set(args.steps.upper().split(","))
    logger.info(f"SIMCF steps: {sorted(steps)}")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    cs_root = Path(args.cityscapes_root).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cluster-to-class mapping (extend to 256 for ignore label safety)
    data = np.load(args.centroids_path)
    _raw_c2c = data["cluster_to_class"].astype(np.uint8)
    cluster_to_class = np.full(256, 255, dtype=np.uint8)
    cluster_to_class[:len(_raw_c2c)] = _raw_c2c
    num_clusters = args.num_clusters
    logger.info(f"Loaded cluster_to_class: {num_clusters} clusters -> {NUM_CLASSES} classes")

    stems = list_cups_images(input_dir)
    logger.info(f"Found {len(stems)} images in {input_dir}")

    feat_dir = cs_root / args.features_subdir
    depth_dir = cs_root / args.depth_subdir

    # Step C first pass: compute depth stats
    class_mean, class_std = None, None
    if "C" in steps:
        class_mean, class_std = compute_depth_stats(
            stems, input_dir, depth_dir, cluster_to_class
        )

    # Main loop
    t0 = time.time()
    total_a_changed = 0
    total_b_merges = 0
    total_c_masked = 0
    total_pixels = 0
    n_processed = 0

    for cups_stem in tqdm(stems, desc="SIMCF"):
        city = _extract_city(cups_stem)
        base = _base_stem(cups_stem)

        semantic = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        instance = np.array(Image.open(input_dir / f"{cups_stem}_instance.png"))
        total_pixels += semantic.shape[0] * semantic.shape[1]

        # Step A
        if "A" in steps:
            n_changed = step_a(semantic, instance, cluster_to_class, num_clusters)
            total_a_changed += n_changed

        # Step B
        if "B" in steps:
            feat_path = feat_dir / "train" / city / f"{cups_stem}.npy"
            if feat_path.exists():
                features = np.load(str(feat_path)).astype(np.float32)
                norms = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
                features = features / norms
                instance, n_merges = step_b(
                    semantic, instance, features, cluster_to_class,
                    sim_threshold=args.sim_threshold,
                )
                total_b_merges += n_merges
            else:
                logger.warning(f"Features not found: {feat_path}")

        # Step C
        if "C" in steps:
            depth_path = depth_dir / "train" / city / f"{base}.npy"
            if not depth_path.exists():
                depth_path = depth_dir / "train" / city / f"{cups_stem}.npy"
            if depth_path.exists():
                depth = np.load(str(depth_path)).astype(np.float64)
                sem_h, sem_w = semantic.shape
                if depth.shape != (sem_h, sem_w):
                    depth = np.array(
                        Image.fromarray(depth.astype(np.float32)).resize(
                            (sem_w, sem_h), Image.BILINEAR
                        )
                    ).astype(np.float64)
                n_masked = step_c(
                    semantic, depth, cluster_to_class,
                    class_mean, class_std, args.sigma_threshold,
                )
                total_c_masked += n_masked

        # Regenerate .pt
        stats = compute_distributions(semantic, instance, num_clusters)

        # Save
        Image.fromarray(semantic.astype(np.uint8)).save(
            str(output_dir / f"{cups_stem}_semantic.png")
        )
        Image.fromarray(instance.astype(np.uint16)).save(
            str(output_dir / f"{cups_stem}_instance.png")
        )
        torch.save(stats, str(output_dir / f"{cups_stem}.pt"))
        n_processed += 1

    elapsed = time.time() - t0

    logger.info(f"\n{'='*60}")
    logger.info(f"SIMCF complete in {elapsed:.1f}s ({n_processed} images)")
    if "A" in steps:
        logger.info(f"  Step A: {total_a_changed} pixels changed "
                     f"({100*total_a_changed/max(total_pixels,1):.2f}%)")
    if "B" in steps:
        logger.info(f"  Step B: {total_b_merges} instance merges "
                     f"({total_b_merges/max(n_processed,1):.1f}/img)")
    if "C" in steps:
        logger.info(f"  Step C: {total_c_masked} pixels masked "
                     f"({100*total_c_masked/max(total_pixels,1):.2f}%)")
    logger.info(f"Output: {output_dir} ({n_processed * 3} files)")

    # Sanity checks
    n_files = len(list(output_dir.glob("*")))
    logger.info(f"\nSanity: {n_files} files (expected {n_processed * 3})")
    if "C" in steps:
        ignore_pct = 100 * total_c_masked / max(total_pixels, 1)
        if ignore_pct > 15:
            logger.warning(f"  High ignore rate: {ignore_pct:.1f}% (threshold: 15%)")


if __name__ == "__main__":
    main()
