#!/usr/bin/env python3
"""CN-SIMCF: Cluster-Native Semantic-Instance Mutual Consistency Filtering.

Forked from refine_simcf.py. Operates on cluster IDs (80) end-to-end instead
of mapping to trainIDs (19) before filtering. Eliminates the chokepoint where
cluster_to_class static argmax erases rare clusters.

Three steps (cluster-native):
  Step A: per-instance majority vote in CLUSTER space (not class space).
          Reassigns minority pixels to majority cluster. Preserves rare clusters
          that map to the same trainID as the majority cluster.
  Step B: merge adjacent instances ONLY if they share the same majority cluster
          (not just same class) AND have feature similarity > 0.92 (raised from
          0.85 because cluster identity is stricter than class identity).
  Step C: per-cluster depth profile (80 means/stds). Clusters with <500 pixels
          get 5σ tolerance instead of 3σ to preserve rare-cluster signal.

Optional: --protected_mask_dir can be used with two policies:
  all: legacy behavior; protected pixels affect Steps A/B/C.
  rare_core: Step A is unprotected, Step B only blocks merges whose boundary
             crosses a high-confidence rare core, and Step C protects only
             those rare-core pixels from outlier masking.

Usage:
    python scripts/refine_cn_simcf.py \
        --input_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_adapter_V3_tau020 \
        --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf \
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
    match = re.match(r"^(.+?)_\d{6}_\d{6}_leftImg8bit$", cups_stem)
    if match:
        return match.group(1)
    return cups_stem.split("_")[0]


def _base_stem(cups_stem: str) -> str:
    return cups_stem.replace("_leftImg8bit", "")


def _resolve_depth_path(depth_dir: Path, split: str, city: str, cups_stem: str) -> Path | None:
    base = _base_stem(cups_stem)
    candidates = [
        depth_dir / split / city / f"{base}.npy",
        depth_dir / split / city / f"{base}_leftImg8bit.npy",
        depth_dir / split / city / f"{cups_stem}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def list_cups_images(input_dir: Path) -> list:
    stems = []
    for p in sorted(input_dir.glob("*_semantic.png")):
        stem = p.name.replace("_semantic.png", "")
        stems.append(stem)
    return stems


# ---------------------------------------------------------------------------
# Step A (cluster-native): per-instance majority vote in CLUSTER space
# ---------------------------------------------------------------------------

def step_a(semantic: np.ndarray, instance: np.ndarray,
           cluster_to_class: np.ndarray, num_clusters: int,
           protected_mask: np.ndarray | None = None) -> int:
    """CN-SIMCF Step A: per-instance majority vote in CLUSTER space.

    Reassigns minority pixels to the majority cluster (not class).
    Preserves rare clusters mapping to same trainID as the majority cluster.

    If protected_mask is provided, protected pixels are NEVER overwritten.
    """
    n_changed = 0
    for iid in np.unique(instance):
        if iid == 0:
            continue
        ys, xs = np.where(instance == iid)
        if len(ys) == 0:
            continue

        clusters = semantic[ys, xs]
        valid_clusters = clusters[clusters < num_clusters]
        if len(valid_clusters) == 0:
            continue

        # Majority CLUSTER (not class) within instance
        cluster_counts = np.bincount(valid_clusters, minlength=num_clusters)
        majority_cluster = int(cluster_counts.argmax())
        majority_tid = int(cluster_to_class[majority_cluster])
        if majority_tid >= NUM_CLASSES:
            continue

        # Reassign only pixels whose cluster maps to a DIFFERENT trainID than the majority
        train_ids = cluster_to_class[clusters]
        inconsistent = (train_ids != majority_tid) & (train_ids < NUM_CLASSES)

        # Honor protected mask: never overwrite protected pixels
        if protected_mask is not None:
            inst_protected = protected_mask[ys, xs]
            inconsistent = inconsistent & (~inst_protected)

        if not inconsistent.any():
            continue

        semantic[ys[inconsistent], xs[inconsistent]] = majority_cluster
        n_changed += int(inconsistent.sum())

    return n_changed


# ---------------------------------------------------------------------------
# Step B (cluster-native): merge instances only if SAME CLUSTER + sim > 0.92
# ---------------------------------------------------------------------------

def step_b(semantic: np.ndarray, instance: np.ndarray,
           features: np.ndarray, num_clusters: int,
           sim_threshold: float = 0.92, dilate_px: int = 3,
           protected_mask: np.ndarray | None = None,
           protected_merge_mode: str = "any_overlap",
           core_boundary_px: int = 8) -> tuple:
    """CN-SIMCF Step B: merge adjacent instances ONLY if same majority CLUSTER + sim > 0.92.

    Threshold raised from 0.85 -> 0.92 because cluster identity is stricter
    than class identity (multiple clusters map to one class).

    If protected_merge_mode is "any_overlap", instances overlapping protected
    pixels are not merged. If it is "boundary_core", only pairs whose merge
    corridor crosses a protected rare core are blocked.
    """
    inst_small = np.array(
        Image.fromarray(instance).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1)

    instance_ids = np.unique(instance)
    instance_ids = instance_ids[instance_ids > 0]
    if len(instance_ids) < 2:
        return instance, 0, 0

    inst_majority_cluster = {}
    inst_feat = {}
    inst_protected_overlap = {}
    n_core_blocked = 0

    for iid in instance_ids:
        mask = instance == iid
        clusters = semantic[mask]
        valid = clusters[clusters < num_clusters]
        if len(valid) == 0:
            continue
        inst_majority_cluster[iid] = int(np.bincount(valid, minlength=num_clusters).argmax())

        mask_s = inst_small == iid
        if not mask_s.any():
            continue
        patches = feat_2d[mask_s]
        feat = patches.mean(axis=0)
        norm = np.linalg.norm(feat) + 1e-8
        inst_feat[iid] = feat / norm

        if protected_mask is not None and protected_merge_mode == "any_overlap":
            inst_protected_overlap[iid] = bool(protected_mask[mask].any())

    struct = ndimage.generate_binary_structure(2, 1)
    adjacency = set()
    for iid in instance_ids:
        if iid not in inst_majority_cluster:
            continue
        mask = instance == iid
        dilated = ndimage.binary_dilation(mask, structure=struct, iterations=dilate_px)
        border = dilated & ~mask
        for nb in np.unique(instance[border]):
            if nb == 0 or nb == iid or nb not in inst_majority_cluster:
                continue
            adjacency.add((min(iid, nb), max(iid, nb)))

    merge_pairs = []
    for i, j in adjacency:
        # CRITICAL: require same CLUSTER (not just same class)
        if inst_majority_cluster.get(i) != inst_majority_cluster.get(j):
            continue
        if i not in inst_feat or j not in inst_feat:
            continue
        # Honor protected mask according to the active policy.
        if protected_mask is not None:
            if protected_merge_mode == "any_overlap":
                if inst_protected_overlap.get(i) or inst_protected_overlap.get(j):
                    continue
            elif protected_merge_mode == "boundary_core":
                if _merge_crosses_protected_core(
                    instance, i, j, protected_mask, core_boundary_px
                ):
                    n_core_blocked += 1
                    continue
            else:
                raise ValueError(f"Unknown protected_merge_mode: {protected_merge_mode}")
        sim = float(np.dot(inst_feat[i], inst_feat[j]))
        if sim > sim_threshold:
            merge_pairs.append((i, j))

    if not merge_pairs:
        return instance, 0, n_core_blocked

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
    return new_instance, n_merges, n_core_blocked


def _merge_crosses_protected_core(instance: np.ndarray, iid_a: int, iid_b: int,
                                  core_mask: np.ndarray,
                                  boundary_px: int) -> bool:
    """Return True when an instance pair's merge corridor intersects a rare core."""
    mask_a = instance == iid_a
    mask_b = instance == iid_b
    pair_mask = mask_a | mask_b
    if not core_mask[pair_mask].any():
        return False

    struct = ndimage.generate_binary_structure(2, 1)
    band_a = ndimage.binary_dilation(
        mask_a, structure=struct, iterations=max(boundary_px, 1)
    )
    band_b = ndimage.binary_dilation(
        mask_b, structure=struct, iterations=max(boundary_px, 1)
    )
    merge_corridor = (band_a & band_b) | (band_a & mask_b) | (band_b & mask_a)
    return bool((merge_corridor & core_mask).any())


# ---------------------------------------------------------------------------
# Step C (cluster-native): per-cluster depth profile, looser σ for rare clusters
# ---------------------------------------------------------------------------

def compute_depth_stats_per_cluster(stems: list, input_dir: Path, depth_dir: Path,
                                     num_clusters: int) -> tuple:
    """Per-cluster depth statistics (80 means/stds, not 19 per-class)."""
    logger.info("CN-SIMCF Step C first pass: per-CLUSTER depth statistics...")
    cluster_sum = np.zeros(num_clusters, dtype=np.float64)
    cluster_sum_sq = np.zeros(num_clusters, dtype=np.float64)
    cluster_count = np.zeros(num_clusters, dtype=np.int64)

    for cups_stem in tqdm(stems, desc="Depth stats per-cluster"):
        city = _extract_city(cups_stem)
        sem = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        sem_h, sem_w = sem.shape

        depth_path = _resolve_depth_path(depth_dir, "train", city, cups_stem)
        if depth_path is None:
            continue
        depth = np.load(str(depth_path)).astype(np.float64)
        if depth.shape != (sem_h, sem_w):
            depth = np.array(
                Image.fromarray(depth.astype(np.float32)).resize(
                    (sem_w, sem_h), Image.BILINEAR)
            ).astype(np.float64)

        for cl in range(num_clusters):
            mask = sem == cl
            if not mask.any():
                continue
            vals = depth[mask]
            cluster_sum[cl] += vals.sum()
            cluster_sum_sq[cl] += (vals ** 2).sum()
            cluster_count[cl] += len(vals)

    safe_count = np.maximum(cluster_count, 1)
    cluster_mean = cluster_sum / safe_count
    cluster_var = cluster_sum_sq / safe_count - cluster_mean ** 2
    cluster_std = np.sqrt(np.maximum(cluster_var, 0.0))

    n_active = int((cluster_count > 0).sum())
    n_rare = int(((cluster_count < 500) & (cluster_count > 0)).sum())
    logger.info(f"  Active clusters: {n_active}/{num_clusters}, rare (<500 px): {n_rare}")
    for cl in range(num_clusters):
        if cluster_count[cl] > 0 and cluster_count[cl] < 500:
            logger.info(f"    rare cluster {cl}: n={cluster_count[cl]}, mean={cluster_mean[cl]:.3f}, std={cluster_std[cl]:.3f}")

    return cluster_mean, cluster_std, cluster_count


def step_c(semantic: np.ndarray, depth: np.ndarray,
           cluster_mean: np.ndarray, cluster_std: np.ndarray,
           cluster_count: np.ndarray, num_clusters: int,
           sigma_common: float = 3.0, sigma_rare: float = 5.0,
           rare_count_threshold: int = 500,
           protected_mask: np.ndarray | None = None) -> int:
    """Per-cluster outlier masking. Rare clusters get looser σ.

    If protected_mask provided, protected pixels are never marked as outliers.
    """
    n_masked = 0
    for cl in range(num_clusters):
        if cluster_std[cl] < 1e-6 or cluster_count[cl] == 0:
            continue
        mask = semantic == cl
        if not mask.any():
            continue
        sigma = sigma_rare if cluster_count[cl] < rare_count_threshold else sigma_common
        deviation = np.abs(depth[mask] - cluster_mean[cl])
        outlier = deviation > sigma * cluster_std[cl]
        if not outlier.any():
            continue
        ys, xs = np.where(mask)
        # Honor protected mask
        if protected_mask is not None:
            protected_in_outlier = protected_mask[ys[outlier], xs[outlier]]
            outlier[outlier] = ~protected_in_outlier
        if not outlier.any():
            continue
        # Reset ys/xs after possible mutation
        outlier_mask_in_cluster = np.zeros(len(ys), dtype=bool)
        # outlier was a boolean array of length sum(mask), we need to translate
        # Re-derive: ys/xs are the (y,x) of cluster pixels. outlier is bool array
        # Actually safer to recompute:
        deviation_full = np.abs(depth[ys, xs] - cluster_mean[cl])
        outlier_full = deviation_full > sigma * cluster_std[cl]
        if protected_mask is not None:
            outlier_full = outlier_full & (~protected_mask[ys, xs])
        if not outlier_full.any():
            continue
        semantic[ys[outlier_full], xs[outlier_full]] = 255
        n_masked += int(outlier_full.sum())
    return n_masked


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def compute_distributions(semantic: np.ndarray, instance: np.ndarray,
                          num_clusters: int) -> dict:
    sem_flat = semantic.flatten().astype(np.int64)
    inst_flat = instance.flatten()

    valid = sem_flat < num_clusters
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_protected_mask(pm_dir: Path, split: str, city: str, stem: str,
                          target_shape: tuple) -> np.ndarray | None:
    """Load a binary protected mask, upsample to image resolution."""
    base_stem = _base_stem(stem)
    stem_variants = [stem]
    if base_stem != stem:
        stem_variants.append(base_stem)
    else:
        stem_variants.append(f"{stem}_leftImg8bit")

    candidates = [
        path
        for s in stem_variants
        for path in (
            pm_dir / split / city / f"{s}.png",
            pm_dir / city / f"{s}.png",
            pm_dir / f"{s}.png",
        )
    ]
    for p in candidates:
        if p.exists():
            mask = np.array(Image.open(p)) > 0
            if mask.shape != target_shape:
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255)
                    .resize((target_shape[1], target_shape[0]), Image.NEAREST)
                ) > 0
            return mask
    return None


def _make_rare_core_mask(protected_mask: np.ndarray,
                         erode_px: int = 8,
                         min_area_px: int = 64) -> np.ndarray:
    """Shrink a broad rare-protection mask into high-confidence cores."""
    core = protected_mask.astype(bool)
    if erode_px > 0:
        core = ndimage.binary_erosion(core, iterations=erode_px)
    if min_area_px <= 1:
        return core

    labeled, n_cc = ndimage.label(core)
    if n_cc == 0:
        return core
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros(n_cc + 1, dtype=bool)
    keep[1:] = sizes[1:] >= min_area_px
    return keep[labeled]


def main():
    parser = argparse.ArgumentParser(description="CN-SIMCF: Cluster-Native SIMCF")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--centroids_path", type=str, required=True)
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--steps", type=str, default="A,B,C")
    parser.add_argument("--features_subdir", type=str, default="dinov3_features")
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--sim_threshold", type=float, default=0.92,
                        help="Cosine similarity threshold for Step B (raised vs SIMCF=0.85)")
    parser.add_argument("--sigma_common", type=float, default=3.0)
    parser.add_argument("--sigma_rare", type=float, default=5.0)
    parser.add_argument("--rare_count_threshold", type=int, default=500)
    parser.add_argument("--num_clusters", type=int, default=80)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--protected_mask_dir", type=str, default=None,
                        help="Optional rare-pixel protected mask directory")
    parser.add_argument("--protected_mask_policy", type=str, default="all",
                        choices=["all", "rare_core"],
                        help="all=legacy Steps A/B/C protection; rare_core=protect "
                             "only rare cores in Step C and Step B merge corridors")
    parser.add_argument("--rare_core_erode_px", type=int, default=8,
                        help="Full-resolution erosion radius for rare-core masks")
    parser.add_argument("--rare_core_min_area_px", type=int, default=64,
                        help="Drop rare-core connected components smaller than this")
    parser.add_argument("--rare_core_boundary_px", type=int, default=8,
                        help="Step B merge corridor radius for rare-core crossing checks")
    args = parser.parse_args()

    steps = set(args.steps.upper().split(","))
    logger.info(f"CN-SIMCF steps: {sorted(steps)}, sim={args.sim_threshold}")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    cs_root = Path(args.cityscapes_root).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cluster_to_class (still needed for Step A's "majority class" check)
    data = np.load(args.centroids_path)
    _raw_c2c = data["cluster_to_class"].astype(np.uint8)
    cluster_to_class = np.full(256, 255, dtype=np.uint8)
    cluster_to_class[:len(_raw_c2c)] = _raw_c2c
    num_clusters = args.num_clusters
    logger.info(f"Loaded cluster_to_class: {num_clusters} clusters -> "
                f"{len(np.unique(_raw_c2c))} active trainIDs")

    stems = list_cups_images(input_dir)
    logger.info(f"Found {len(stems)} images in {input_dir}")

    feat_dir = cs_root / args.features_subdir
    depth_dir = cs_root / args.depth_subdir

    pm_dir = Path(args.protected_mask_dir).expanduser() if args.protected_mask_dir else None
    if pm_dir:
        logger.info(f"Using protected masks from {pm_dir}")
        logger.info(f"Protection policy: {args.protected_mask_policy}")

    # Step C first pass: per-cluster depth stats
    cluster_mean, cluster_std, cluster_count = (None, None, None)
    if "C" in steps:
        cluster_mean, cluster_std, cluster_count = compute_depth_stats_per_cluster(
            stems, input_dir, depth_dir, num_clusters
        )

    t0 = time.time()
    total_a_changed = 0
    total_b_merges = 0
    total_b_core_blocked = 0
    total_c_masked = 0
    total_pixels = 0
    total_protected_pixels = 0
    total_rare_core_pixels = 0
    n_masks_loaded = 0
    n_processed = 0

    for cups_stem in tqdm(stems, desc="CN-SIMCF"):
        city = _extract_city(cups_stem)

        semantic = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        instance = np.array(Image.open(input_dir / f"{cups_stem}_instance.png"))
        total_pixels += semantic.shape[0] * semantic.shape[1]

        protected_mask = None
        if pm_dir:
            protected_mask = _load_protected_mask(
                pm_dir, args.split, city, cups_stem, semantic.shape
            )
        rare_core_mask = None
        if protected_mask is not None:
            n_masks_loaded += 1
            total_protected_pixels += int(protected_mask.sum())
            if args.protected_mask_policy == "rare_core":
                rare_core_mask = _make_rare_core_mask(
                    protected_mask,
                    args.rare_core_erode_px,
                    args.rare_core_min_area_px,
                )
                total_rare_core_pixels += int(rare_core_mask.sum())

        if args.protected_mask_policy == "all":
            step_a_mask = protected_mask
            step_b_mask = protected_mask
            step_b_merge_mode = "any_overlap"
            step_c_mask = protected_mask
        else:
            step_a_mask = None
            step_b_mask = rare_core_mask
            step_b_merge_mode = "boundary_core"
            step_c_mask = rare_core_mask

        if "A" in steps:
            n_changed = step_a(semantic, instance, cluster_to_class,
                               num_clusters, step_a_mask)
            total_a_changed += n_changed

        if "B" in steps:
            feat_path = feat_dir / args.split / city / f"{cups_stem}.npy"
            if feat_path.exists():
                features = np.load(str(feat_path)).astype(np.float32)
                norms = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
                features = features / norms
                instance, n_merges, n_core_blocked = step_b(
                    semantic, instance, features, num_clusters,
                    sim_threshold=args.sim_threshold,
                    protected_mask=step_b_mask,
                    protected_merge_mode=step_b_merge_mode,
                    core_boundary_px=args.rare_core_boundary_px,
                )
                total_b_merges += n_merges
                total_b_core_blocked += n_core_blocked
            else:
                logger.warning(f"Features not found: {feat_path}")

        if "C" in steps:
            depth_path = _resolve_depth_path(depth_dir, args.split, city, cups_stem)
            if depth_path is not None:
                depth = np.load(str(depth_path)).astype(np.float64)
                sem_h, sem_w = semantic.shape
                if depth.shape != (sem_h, sem_w):
                    depth = np.array(
                        Image.fromarray(depth.astype(np.float32)).resize(
                            (sem_w, sem_h), Image.BILINEAR)
                    ).astype(np.float64)
                n_masked = step_c(
                    semantic, depth, cluster_mean, cluster_std, cluster_count,
                    num_clusters, args.sigma_common, args.sigma_rare,
                    args.rare_count_threshold, step_c_mask,
                )
                total_c_masked += n_masked

        stats = compute_distributions(semantic, instance, num_clusters)

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
    logger.info(f"CN-SIMCF complete in {elapsed:.1f}s ({n_processed} images)")
    if pm_dir:
        logger.info(f"  Protected masks loaded: {n_masks_loaded}/{n_processed}")
        logger.info(f"  Protected mask pixels: {total_protected_pixels} "
                    f"({100*total_protected_pixels/max(total_pixels,1):.2f}%)")
        if args.protected_mask_policy == "rare_core":
            logger.info(f"  Rare-core pixels: {total_rare_core_pixels} "
                        f"({100*total_rare_core_pixels/max(total_pixels,1):.2f}%)")
    if "A" in steps:
        logger.info(f"  Step A: {total_a_changed} pixels changed "
                    f"({100*total_a_changed/max(total_pixels,1):.2f}%)")
    if "B" in steps:
        logger.info(f"  Step B: {total_b_merges} instance merges "
                    f"({total_b_merges/max(n_processed,1):.1f}/img)")
        if pm_dir and args.protected_mask_policy == "rare_core":
            logger.info(f"  Step B: {total_b_core_blocked} merge candidates blocked "
                        f"by rare-core corridor checks")
    if "C" in steps:
        logger.info(f"  Step C: {total_c_masked} pixels masked "
                    f"({100*total_c_masked/max(total_pixels,1):.2f}%)")
    logger.info(f"Output: {output_dir} ({n_processed * 3} files)")


if __name__ == "__main__":
    main()
