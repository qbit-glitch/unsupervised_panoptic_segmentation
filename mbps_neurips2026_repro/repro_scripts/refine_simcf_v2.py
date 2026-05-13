#!/usr/bin/env python3
"""SIMCF-v2: Enhanced pseudo-label refinement with 5 novel passes.

Extends SIMCF-ABC (Steps A-C) with Steps D-H for improved pseudo-label
quality. Typically applied on top of SIMCF-ABC output.

Steps:
  D -- SDAIR: Spectral Depth-Aware Instance Refinement (splitting)
  E -- WBIM:  Wasserstein Barycentric Instance Merging
  F -- ITCBS: Info-Theoretic Cluster Boundary Sharpening
  G -- DCCPR: Depth-Conditioned Class Prior Regularization
  H -- GSID:  Grassmannian Subspace Instance Discrimination

Usage:
    python scripts/refine_simcf_v2.py \
        --input_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_simcf_abc \
        --output_dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_simcf_v2_D \
        --centroids_path ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
        --cityscapes_root ~/Desktop/datasets/cityscapes \
        --steps D
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent))
from refine_simcf import compute_distributions
from simcf_v2 import step_d, step_e, step_f, step_g, step_h
from simcf_v2.dccpr import fit_gmm_1d

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
NUM_CLASSES = 19


# ---------- Utility ----------

def _extract_city(cups_stem: str) -> str:
    match = re.match(r"^(.+?)_\d{6}_\d{6}_leftImg8bit$", cups_stem)
    return match.group(1) if match else cups_stem.split("_")[0]


def _base_stem(cups_stem: str) -> str:
    return cups_stem.replace("_leftImg8bit", "")


def _list_cups_images(input_dir: Path) -> list:
    stems = []
    for p in sorted(input_dir.glob("*_semantic.png")):
        stems.append(p.name.replace("_semantic.png", ""))
    return stems


# ---------- Global First Passes ----------

def _compute_global_depth_gmms(
    stems: list, input_dir: Path, depth_dir: Path,
    cluster_to_class: np.ndarray, n_components: int = 3,
    max_samples: int = 100000,
) -> tuple:
    """First pass: fit per-class depth GMMs and class priors."""
    logger.info("Global pass: fitting per-class depth GMMs...")
    depth_samples = {c: [] for c in range(NUM_CLASSES)}
    class_pixel_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    for cups_stem in tqdm(stems, desc="Depth GMMs"):
        city = _extract_city(cups_stem)
        base = _base_stem(cups_stem)
        sem = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        mapped = cluster_to_class[sem].astype(np.int64)
        sem_h, sem_w = sem.shape

        depth_path = _find_depth(depth_dir, city, base, cups_stem)
        if depth_path is None:
            continue
        depth = _load_depth(depth_path, sem_h, sem_w)

        for cls in range(NUM_CLASSES):
            mask = mapped == cls
            count = int(mask.sum())
            if count == 0:
                continue
            class_pixel_count[cls] += count
            if len(depth_samples[cls]) < max_samples:
                vals = depth[mask]
                n_take = min(len(vals), max_samples - len(depth_samples[cls]))
                if n_take < len(vals):
                    idx = np.random.choice(len(vals), n_take, replace=False)
                    vals = vals[idx]
                depth_samples[cls].extend(vals.tolist())

    depth_gmms = []
    for cls in range(NUM_CLASSES):
        samples = np.array(depth_samples[cls])
        if len(samples) > 0:
            w, m, v = fit_gmm_1d(samples, n_components)
        else:
            w, m, v = np.array([1.0]), np.array([0.5]), np.array([1.0])
        depth_gmms.append((w, m, v))
        if len(samples) > 0:
            logger.info(
                f"  class {cls:2d}: {len(samples):>7d} samples, "
                f"GMM means=[{', '.join(f'{x:.3f}' for x in m)}]"
            )

    total = max(class_pixel_count.sum(), 1)
    class_priors = class_pixel_count.astype(np.float64) / total
    return depth_gmms, class_priors


def _compute_global_feature_stats(
    stems: list, input_dir: Path, feat_dir: Path,
    cluster_to_class: np.ndarray, max_images: int = 300,
) -> tuple:
    """Compute per-class feature mean and variance across images."""
    logger.info("Global pass: computing per-class feature statistics...")
    D = None
    class_feat_sum = None
    class_feat_sq_sum = None
    class_feat_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    for cups_stem in tqdm(stems[:max_images], desc="Feature stats"):
        city = _extract_city(cups_stem)
        feat_path = feat_dir / "train" / city / f"{cups_stem}.npy"
        if not feat_path.exists():
            continue

        features = np.load(str(feat_path)).astype(np.float64)
        if D is None:
            D = features.shape[1]
            class_feat_sum = np.zeros((NUM_CLASSES, D), dtype=np.float64)
            class_feat_sq_sum = np.zeros((NUM_CLASSES, D), dtype=np.float64)

        feat_2d = features.reshape(FEAT_H, FEAT_W, -1)
        sem = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        sem_small = np.array(
            Image.fromarray(sem).resize((FEAT_W, FEAT_H), Image.NEAREST)
        )
        mapped_small = cluster_to_class[sem_small]

        for cls in range(NUM_CLASSES):
            mask = mapped_small == cls
            if not mask.any():
                continue
            vecs = feat_2d[mask]
            class_feat_sum[cls] += vecs.sum(axis=0)
            class_feat_sq_sum[cls] += (vecs**2).sum(axis=0)
            class_feat_count[cls] += len(vecs)

    safe_count = np.maximum(class_feat_count, 1)
    class_feat_mean = class_feat_sum / safe_count[:, None]
    class_feat_var = (
        class_feat_sq_sum / safe_count[:, None] - class_feat_mean**2
    ).mean(axis=1)
    class_feat_var = np.maximum(class_feat_var, 1e-6)

    return class_feat_mean, class_feat_var


def _compute_global_class_centroids(
    stems: list, input_dir: Path, feat_dir: Path,
    cluster_to_class: np.ndarray, max_images: int = 500,
) -> np.ndarray:
    """Compute per-class mean feature centroids across images.

    Returns:
        (NUM_CLASSES, D) centroid matrix.
    """
    logger.info("Global pass: computing per-class feature centroids...")
    D = None
    class_sum = None
    class_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    for cups_stem in tqdm(stems[:max_images], desc="Centroids"):
        city = _extract_city(cups_stem)
        feat_path = feat_dir / "train" / city / f"{cups_stem}.npy"
        if not feat_path.exists():
            continue

        features = np.load(str(feat_path)).astype(np.float64)
        if D is None:
            D = features.shape[1]
            class_sum = np.zeros((NUM_CLASSES, D), dtype=np.float64)

        feat_2d = features.reshape(FEAT_H, FEAT_W, -1)
        sem = np.array(Image.open(input_dir / f"{cups_stem}_semantic.png"))
        sem_small = np.array(
            Image.fromarray(sem).resize((FEAT_W, FEAT_H), Image.NEAREST)
        )
        mapped = cluster_to_class[sem_small]

        for cls in range(NUM_CLASSES):
            mask = mapped == cls
            if mask.any():
                class_sum[cls] += feat_2d[mask].sum(axis=0)
                class_count[cls] += int(mask.sum())

    safe = np.maximum(class_count, 1)
    return class_sum / safe[:, None]


# ---------- I/O Helpers ----------

def _find_depth(
    depth_dir: Path, city: str, base: str, cups_stem: str
) -> Path | None:
    p = depth_dir / "train" / city / f"{base}.npy"
    if p.exists():
        return p
    p = depth_dir / "train" / city / f"{cups_stem}.npy"
    return p if p.exists() else None


def _load_depth(path: Path, target_h: int, target_w: int) -> np.ndarray:
    depth = np.load(str(path)).astype(np.float64)
    if depth.shape != (target_h, target_w):
        depth = np.array(
            Image.fromarray(depth.astype(np.float32)).resize(
                (target_w, target_h), Image.BILINEAR
            )
        ).astype(np.float64)
    return depth


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="SIMCF-v2 pseudo-label refinement"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--centroids_path", type=str, required=True)
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument(
        "--steps", type=str, default="D",
        help="Comma-separated v2 steps: D,E,F,G,H",
    )
    parser.add_argument("--features_subdir", type=str, default="dinov3_features")
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--num_clusters", type=int, default=80)
    # Step D params
    parser.add_argument("--sigma_f", type=float, default=0.3)
    parser.add_argument("--sigma_d", type=float, default=0.1)
    parser.add_argument("--fiedler_threshold", type=float, default=0.15)
    # Step E params
    parser.add_argument("--sw_threshold", type=float, default=0.3)
    parser.add_argument("--n_projections", type=int, default=50)
    # Step F params
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min_gain", type=float, default=0.5)
    # Step G params
    parser.add_argument("--sigma_threshold", type=float, default=3.0)
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--gmm_components", type=int, default=3)
    # Step H params
    parser.add_argument("--subspace_rank", type=int, default=5)
    parser.add_argument("--grass_threshold", type=float, default=0.8)
    args = parser.parse_args()

    steps = set(args.steps.upper().split(","))
    valid_steps = {"D", "E", "F", "G", "H"}
    unknown = steps - valid_steps
    if unknown:
        parser.error(f"Unknown steps: {unknown}. Valid: {valid_steps}")
    if "E" in steps and "H" in steps:
        parser.error(
            "Steps E (WBIM) and H (GSID) are mutually exclusive merge "
            "strategies. Choose one."
        )
    logger.info(f"SIMCF-v2 steps: {sorted(steps)}")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    cs_root = Path(args.cityscapes_root).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cluster-to-class mapping
    data = np.load(args.centroids_path)
    _raw_c2c = data["cluster_to_class"].astype(np.uint8)
    cluster_to_class = np.full(256, 255, dtype=np.uint8)
    cluster_to_class[: len(_raw_c2c)] = _raw_c2c

    stems = _list_cups_images(input_dir)
    logger.info(f"Found {len(stems)} images in {input_dir}")

    feat_dir = cs_root / args.features_subdir
    depth_dir = cs_root / args.depth_subdir
    needs_features = bool(steps & {"D", "E", "F", "G", "H"})
    needs_depth = bool(steps & {"D", "G"})

    # Global first passes
    depth_gmms, class_priors = None, None
    class_feat_mean, class_feat_var = None, None
    class_centroids = None

    if "G" in steps:
        depth_gmms, class_priors = _compute_global_depth_gmms(
            stems, input_dir, depth_dir, cluster_to_class,
            n_components=args.gmm_components,
        )
        class_feat_mean, class_feat_var = _compute_global_feature_stats(
            stems, input_dir, feat_dir, cluster_to_class,
        )

    if "F" in steps:
        class_centroids = _compute_global_class_centroids(
            stems, input_dir, feat_dir, cluster_to_class,
        )

    # Per-image processing
    t0 = time.time()
    stats = {s: 0 for s in "d_splits e_merges f_changed g_reassigned g_masked h_merges".split()}
    total_pixels = 0
    n_processed = 0

    for cups_stem in tqdm(stems, desc="SIMCF-v2"):
        city = _extract_city(cups_stem)
        base = _base_stem(cups_stem)

        semantic = np.array(
            Image.open(input_dir / f"{cups_stem}_semantic.png")
        )
        instance = np.array(
            Image.open(input_dir / f"{cups_stem}_instance.png")
        )
        total_pixels += semantic.shape[0] * semantic.shape[1]
        sem_h, sem_w = semantic.shape

        # Load features
        features = None
        if needs_features:
            feat_path = feat_dir / "train" / city / f"{cups_stem}.npy"
            if feat_path.exists():
                features = np.load(str(feat_path)).astype(np.float32)
                norms = (
                    np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
                )
                features = features / norms
            else:
                logger.warning(f"Features not found: {feat_path}")
                features = None

        # Load depth
        depth = None
        if needs_depth:
            depth_path = _find_depth(depth_dir, city, base, cups_stem)
            if depth_path is not None:
                depth = _load_depth(depth_path, sem_h, sem_w)

        # Step D: SDAIR — instance splitting (things only)
        if "D" in steps and features is not None and depth is not None:
            instance, n = step_d(
                semantic, instance, features, depth,
                cluster_to_class=cluster_to_class,
                sigma_f=args.sigma_f, sigma_d=args.sigma_d,
                fiedler_threshold=args.fiedler_threshold,
            )
            stats["d_splits"] += n

        # Step E: WBIM — Wasserstein merging (mutually exclusive with H)
        if "E" in steps and features is not None:
            instance, n = step_e(
                semantic, instance, features, cluster_to_class,
                n_projections=args.n_projections,
                sw_threshold=args.sw_threshold,
            )
            stats["e_merges"] += n

        # Step H: GSID — Grassmannian merging (mutually exclusive with E)
        if "H" in steps and features is not None:
            instance, n = step_h(
                semantic, instance, features, cluster_to_class,
                subspace_rank=args.subspace_rank,
                grass_threshold=args.grass_threshold,
            )
            stats["h_merges"] += n

        # Step F: ITCBS — boundary sharpening
        if "F" in steps and features is not None and class_centroids is not None:
            n = step_f(
                semantic, features, cluster_to_class, class_centroids,
                num_clusters=args.num_clusters,
                temperature=args.temperature, min_gain=args.min_gain,
            )
            stats["f_changed"] += n

        # Step G: DCCPR — Bayesian reassignment
        if "G" in steps and depth is not None and features is not None:
            n_re, n_ma = step_g(
                semantic, depth, features, cluster_to_class,
                depth_gmms, class_feat_mean, class_feat_var, class_priors,
                num_clusters=args.num_clusters,
                sigma_threshold=args.sigma_threshold,
                min_confidence=args.min_confidence,
            )
            stats["g_reassigned"] += n_re
            stats["g_masked"] += n_ma

        # Regenerate .pt distributions
        dist = compute_distributions(semantic, instance, args.num_clusters)

        # Save
        Image.fromarray(semantic.astype(np.uint8)).save(
            str(output_dir / f"{cups_stem}_semantic.png")
        )
        Image.fromarray(instance.astype(np.uint16)).save(
            str(output_dir / f"{cups_stem}_instance.png")
        )
        torch.save(dist, str(output_dir / f"{cups_stem}.pt"))
        n_processed += 1

    elapsed = time.time() - t0
    tp = max(total_pixels, 1)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SIMCF-v2 complete in {elapsed:.1f}s ({n_processed} images)")
    if "D" in steps:
        logger.info(
            f"  Step D (SDAIR):  {stats['d_splits']} splits "
            f"({stats['d_splits'] / max(n_processed, 1):.2f}/img)"
        )
    if "E" in steps:
        logger.info(
            f"  Step E (WBIM):   {stats['e_merges']} merges "
            f"({stats['e_merges'] / max(n_processed, 1):.2f}/img)"
        )
    if "F" in steps:
        logger.info(
            f"  Step F (ITCBS):  {stats['f_changed']} pixels changed "
            f"({100 * stats['f_changed'] / tp:.3f}%)"
        )
    if "G" in steps:
        logger.info(
            f"  Step G (DCCPR):  {stats['g_reassigned']} reassigned, "
            f"{stats['g_masked']} masked "
            f"({100 * (stats['g_reassigned'] + stats['g_masked']) / tp:.3f}%)"
        )
    if "H" in steps:
        logger.info(
            f"  Step H (GSID):   {stats['h_merges']} merges "
            f"({stats['h_merges'] / max(n_processed, 1):.2f}/img)"
        )
    logger.info(f"Output: {output_dir} ({n_processed * 3} files)")


if __name__ == "__main__":
    main()
