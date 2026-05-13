#!/usr/bin/env python3
"""Bootstrap unsupervised re-mapping of cluster_to_class.

Refits the cluster -> trainID lookup using the current pseudo-labels'
DINOv3 feature + DepthPro depth signatures. NO GT IS USED.

Algorithm:
    1. For each cluster c, compute mean DINOv3 feature f_c and mean depth d_c.
    2. From the CURRENT cluster_to_class mapping, build per-class prototypes
       (weighted average of cluster signatures within each class).
    3. For each cluster, score against ACTIVE class prototypes (cosine sim
       in feature space).
    4. Reassign cluster -> trainID if best score is > 0.05 above current.
    5. Save new cluster_to_class.

Currently DEAD trainIDs (no clusters assigned) cannot be re-acquired by this
script alone — they need T3's protected mask for that. This script's job is
to (a) shift miss-mapped clusters within active classes and (b) reveal which
classes are dead.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 19
FEAT_H, FEAT_W = 32, 64
CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]


def collect_cluster_signatures(pseudo_dir: Path, feat_dir: Path, depth_dir: Path,
                               num_clusters: int, split: str = "train"):
    """For each cluster: mean DINOv3 feature (768D) and mean depth."""
    cluster_feat_sum = np.zeros((num_clusters, 768), dtype=np.float64)
    cluster_depth_sum = np.zeros(num_clusters, dtype=np.float64)
    cluster_count = np.zeros(num_clusters, dtype=np.int64)

    stems = sorted([p.name.replace("_semantic.png", "")
                    for p in pseudo_dir.glob("*_semantic.png")])
    logger.info(f"Collecting signatures from {len(stems)} pseudo-labels")

    for stem in tqdm(stems, desc="Signatures"):
        # Extract city from CUPS stem
        import re
        m = re.match(r"^(.+?)_\d{6}_\d{6}_leftImg8bit$", stem)
        city = m.group(1) if m else stem.split("_")[0]

        sem = np.array(Image.open(pseudo_dir / f"{stem}_semantic.png"))

        feat_path = feat_dir / split / city / f"{stem}.npy"
        if not feat_path.exists():
            continue
        feat = np.load(str(feat_path)).astype(np.float32)
        if feat.ndim == 2:
            feat_2d = feat.reshape(FEAT_H, FEAT_W, -1)
        else:
            feat_2d = feat

        # Depth path resolution (handle both stem variants)
        base_stem = stem.replace("_leftImg8bit", "")
        depth_candidates = [
            depth_dir / split / city / f"{base_stem}.npy",
            depth_dir / split / city / f"{base_stem}_leftImg8bit.npy",
            depth_dir / split / city / f"{stem}.npy",
        ]
        depth_path = None
        for p in depth_candidates:
            if p.exists():
                depth_path = p
                break
        if depth_path is None:
            continue
        depth = np.load(str(depth_path)).astype(np.float32)

        # Resize semantic to feature resolution for consistent indexing
        sem_small = np.array(Image.fromarray(sem).resize((FEAT_W, FEAT_H), Image.NEAREST))
        depth_small = np.array(
            Image.fromarray(depth).resize((FEAT_W, FEAT_H), Image.BILINEAR)
        )

        for cl in range(num_clusters):
            mask = sem_small == cl
            if not mask.any():
                continue
            cluster_feat_sum[cl] += feat_2d[mask].sum(axis=0)
            cluster_depth_sum[cl] += float(depth_small[mask].sum())
            cluster_count[cl] += int(mask.sum())

    safe_count = np.maximum(cluster_count, 1)
    cluster_feat_mean = cluster_feat_sum / safe_count[:, None]
    cluster_depth_mean = cluster_depth_sum / safe_count
    feat_norms = np.linalg.norm(cluster_feat_mean, axis=1, keepdims=True) + 1e-8
    cluster_feat_unit = cluster_feat_mean / feat_norms
    return cluster_feat_unit, cluster_depth_mean, cluster_count


def refit_cluster_to_class(cluster_feat: np.ndarray, cluster_depth: np.ndarray,
                            cluster_count: np.ndarray, current_c2c: np.ndarray,
                            min_cluster_pixels: int = 100,
                            improvement_threshold: float = 0.05) -> np.ndarray:
    """Refit cluster -> trainID via per-class prototypes from current mapping.

    Step 1: Build per-class feature prototypes (weighted by cluster size).
    Step 2: For each cluster, score against ALL ACTIVE class prototypes.
    Step 3: Reassign if best active score is > current + improvement_threshold.
    """
    num_clusters = cluster_feat.shape[0]
    new_c2c = current_c2c.copy()

    # Per-class prototypes
    class_feat = np.zeros((NUM_CLASSES, cluster_feat.shape[1]), dtype=np.float64)
    class_weight = np.zeros(NUM_CLASSES, dtype=np.float64)
    for cl in range(num_clusters):
        if cluster_count[cl] < min_cluster_pixels:
            continue
        tid = int(current_c2c[cl])
        if tid >= NUM_CLASSES:
            continue
        w = float(cluster_count[cl])
        class_feat[tid] += cluster_feat[cl] * w
        class_weight[tid] += w

    safe_w = np.maximum(class_weight, 1.0)
    class_feat /= safe_w[:, None]
    norms = np.linalg.norm(class_feat, axis=1, keepdims=True) + 1e-8
    class_feat_unit = class_feat / norms

    dead_classes = [c for c in range(NUM_CLASSES) if class_weight[c] == 0]
    active_classes = [c for c in range(NUM_CLASSES) if class_weight[c] > 0]
    logger.info(f"Active classes: {len(active_classes)} - {[CITYSCAPES_CLASS_NAMES[c] for c in active_classes]}")
    logger.info(f"Dead classes: {len(dead_classes)} - {[CITYSCAPES_CLASS_NAMES[c] for c in dead_classes]}")

    n_remapped = 0
    n_unchanged = 0
    n_skipped = 0
    for cl in range(num_clusters):
        if cluster_count[cl] < min_cluster_pixels:
            n_skipped += 1
            continue
        feat_sims = cluster_feat[cl] @ class_feat_unit.T  # (NUM_CLASSES,)
        feat_sims[class_weight == 0] = -1.0  # mask out dead classes
        best_active = int(np.argmax(feat_sims))
        best_active_sim = float(feat_sims[best_active])

        cur_tid = int(current_c2c[cl])
        if cur_tid < NUM_CLASSES and class_weight[cur_tid] > 0:
            cur_sim = float(feat_sims[cur_tid])
            if best_active_sim - cur_sim > improvement_threshold:
                new_c2c[cl] = best_active
                logger.info(
                    f"  cluster {cl} (n={cluster_count[cl]}): "
                    f"{CITYSCAPES_CLASS_NAMES[cur_tid]} (sim={cur_sim:.3f}) "
                    f"-> {CITYSCAPES_CLASS_NAMES[best_active]} (sim={best_active_sim:.3f})"
                )
                n_remapped += 1
            else:
                n_unchanged += 1
        else:
            new_c2c[cl] = best_active
            n_remapped += 1

    logger.info(f"Remapped {n_remapped}, unchanged {n_unchanged}, skipped (rare) {n_skipped}")
    return new_c2c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_dir", required=True,
                        help="Current pseudo-label dir (CUPS flat format)")
    parser.add_argument("--feat_dir", required=True,
                        help="DINOv3 features root (contains <split>/<city>/*.npy)")
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--current_centroids", required=True)
    parser.add_argument("--output_centroids", required=True)
    parser.add_argument("--num_clusters", type=int, default=80)
    parser.add_argument("--split", default="train")
    parser.add_argument("--improvement_threshold", type=float, default=0.05)
    args = parser.parse_args()

    data = np.load(args.current_centroids)
    centers = data["centers"] if "centers" in data else (data["centroids"] if "centroids" in data else None)
    current_c2c = data["cluster_to_class"].astype(np.uint8)

    cluster_feat, cluster_depth, cluster_count = collect_cluster_signatures(
        Path(args.pseudo_dir).expanduser(),
        Path(args.feat_dir).expanduser(),
        Path(args.depth_dir).expanduser(),
        args.num_clusters,
        args.split,
    )

    new_c2c = refit_cluster_to_class(
        cluster_feat, cluster_depth, cluster_count, current_c2c,
        improvement_threshold=args.improvement_threshold,
    )

    out = {"cluster_to_class": new_c2c}
    if centers is not None:
        out["centers"] = centers
    out["cluster_feat_mean"] = cluster_feat.astype(np.float32)
    out["cluster_depth_mean"] = cluster_depth.astype(np.float32)
    out["cluster_count"] = cluster_count
    output_path = Path(args.output_centroids).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)
    logger.info(f"Saved new centroids to {output_path}")


if __name__ == "__main__":
    main()
