"""Phase-B POC: cluster-reorg diagnostic on existing k=80 centroids.

Loads weights/kmeans_centroids_k80_santosh.npz and reports:
    1. cluster_to_class mapping summary (which classes are vacant?)
    2. pairwise cosine similarity statistics (well-separated or collapsed?)
    3. simulated rare-mode identification + hierarchical merge outcome

This is a sanity check before paying for NeCo training. If the existing
centroids already show clear separation between rare-class and frequent-
class regions, NeCo will help further. If not, we likely need a different
intervention.

Pixel counts are not available locally, so we use the inverse of the per-
class cluster count as a proxy for population (classes with fewer clusters
are likely the lower-population / more compressed ones).

Usage:
    python scripts/stage4_cluster_poc.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.stage4 import hierarchical_merge, identify_rare_modes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CENTROIDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "weights"
    / "kmeans_centroids_k80_santosh.npz"
)

# Cityscapes 19-class trainID names (per cityscapesScripts)
TRAINID_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


def _cosine_similarity_matrix(centroids: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(centroids, dim=-1)
    return normalized @ normalized.T


def main() -> None:
    if not CENTROIDS_PATH.exists():
        raise FileNotFoundError(f"missing centroids file: {CENTROIDS_PATH}")

    data = np.load(CENTROIDS_PATH)
    centroids_np = data["centroids"]  # (80, 90)
    cluster_to_class_np = data["cluster_to_class"]  # (80,)

    centroids = torch.from_numpy(centroids_np).float()
    cluster_to_class = torch.from_numpy(cluster_to_class_np.astype(np.int64))
    k, d = centroids.shape

    logger.info("=" * 64)
    logger.info("Phase-B POC: Cluster Reorg Diagnostic")
    logger.info("=" * 64)
    logger.info("Centroids:        shape=%s, dtype=%s", tuple(centroids.shape), centroids.dtype)
    logger.info("cluster_to_class: shape=%s", tuple(cluster_to_class.shape))

    # ------------------------------------------------------------------
    # Section 1 — class coverage
    # ------------------------------------------------------------------
    logger.info("\n[1] Class coverage (19-class TrainID space)")
    counts_per_class: dict[int, int] = {}
    for c in cluster_to_class.tolist():
        counts_per_class[c] = counts_per_class.get(c, 0) + 1
    vacant_classes = [
        i for i, name in enumerate(TRAINID_NAMES) if i not in counts_per_class
    ]
    logger.info("    Total classes covered:  %d / 19", len(counts_per_class))
    logger.info(
        "    VACANT classes:         %s",
        [f"{i} {TRAINID_NAMES[i]}" for i in vacant_classes],
    )
    underrepresented = sorted(
        [(c, n) for c, n in counts_per_class.items() if n == 1],
        key=lambda x: x[0],
    )
    logger.info(
        "    Single-cluster classes: %s",
        [f"{c} {TRAINID_NAMES[c]}" for c, _ in underrepresented],
    )

    # ------------------------------------------------------------------
    # Section 2 — pairwise similarity geometry
    # ------------------------------------------------------------------
    logger.info("\n[2] Pairwise centroid cosine similarity")
    sim = _cosine_similarity_matrix(centroids)
    sim.fill_diagonal_(-float("inf"))
    upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    pair_sims = sim[upper]
    pair_sims = pair_sims[torch.isfinite(pair_sims)]
    logger.info("    %.4f  min", pair_sims.min().item())
    logger.info("    %.4f  median", pair_sims.median().item())
    logger.info("    %.4f  mean", pair_sims.mean().item())
    logger.info("    %.4f  max", pair_sims.max().item())
    high_sim_pairs = (pair_sims > 0.95).sum().item()
    logger.info(
        "    %d  high-similarity pairs (cos > 0.95) — candidates for merging",
        high_sim_pairs,
    )

    # Find single-cluster classes (most isolated in semantic space — these are
    # the closest analogs to "rare modes" we will protect during merging).
    single_cluster_class_ids = [c for c, n in counts_per_class.items() if n == 1]
    rare_centroid_ids = [
        i for i in range(k) if cluster_to_class[i].item() in single_cluster_class_ids
    ]
    logger.info("\n[3] Rare-mode candidates (single-cluster classes)")
    logger.info(
        "    %d centroid ids mapped to single-cluster classes: %s",
        len(rare_centroid_ids),
        rare_centroid_ids,
    )
    for rid in rare_centroid_ids:
        class_id = cluster_to_class[rid].item()
        # Find this centroid's similarity to its 3 nearest neighbors
        nbr_sims, nbr_ids = torch.topk(sim[rid], k=3)
        nbr_classes = [
            f"{i.item()}({TRAINID_NAMES[cluster_to_class[i].item()]})"
            for i in nbr_ids
        ]
        nbr_sim_str = ", ".join(
            f"{s:.3f}@cls{c}" for s, c in zip(nbr_sims.tolist(), nbr_classes)
        )
        logger.info(
            "    centroid %2d  cls=%-15s  top-3 similar: %s",
            rid,
            TRAINID_NAMES[class_id],
            nbr_sim_str,
        )

    # ------------------------------------------------------------------
    # Section 4 — simulate hierarchical merge with rare-mode freezing
    # ------------------------------------------------------------------
    logger.info("\n[4] Simulated hierarchical merge: k=80 -> k=40 with rare-mode freeze")
    # Without real pixel counts, use the per-class cluster count as a proxy:
    # each centroid's "population" = number of centroids sharing its class.
    # Centroids belonging to single-cluster classes therefore have proxy=1 (smallest).
    proxy_counts = torch.tensor(
        [counts_per_class[cluster_to_class[i].item()] for i in range(k)],
        dtype=torch.float,
    )
    n_freeze = len(rare_centroid_ids)
    rare_via_heuristic = identify_rare_modes(proxy_counts, n_freeze=n_freeze)
    logger.info(
        "    rare modes from proxy counts (n=%d): %s",
        n_freeze,
        sorted(rare_via_heuristic),
    )
    overlap = set(rare_via_heuristic) & set(rare_centroid_ids)
    logger.info(
        "    overlap with single-cluster centroids: %d / %d",
        len(overlap),
        len(rare_centroid_ids),
    )

    target_k = 40
    merged_centroids, mapping = hierarchical_merge(
        centroids,
        proxy_counts,
        target_k=target_k,
        frozen_indices=rare_via_heuristic,
    )
    logger.info(
        "    merged centroid shape:  %s",
        tuple(merged_centroids.shape),
    )
    logger.info(
        "    mapping [0:10]:  %s",
        mapping[:10].tolist(),
    )
    # Check: every frozen centroid kept its own new id (no other old centroid maps to it)
    surviving_frozen_ok = True
    for old_id in rare_via_heuristic:
        new_id = mapping[old_id].item()
        siblings = (mapping == new_id).sum().item()
        if siblings != 1:
            surviving_frozen_ok = False
            logger.warning(
                "    [!] frozen old=%d new=%d has %d siblings (expected 1)",
                old_id,
                new_id,
                siblings,
            )
    logger.info(
        "    frozen centroids survived alone:  %s",
        surviving_frozen_ok,
    )

    # ------------------------------------------------------------------
    # Section 5 — Decision
    # ------------------------------------------------------------------
    logger.info("\n[5] Decision")
    median_sim = pair_sims.median().item()
    if median_sim < 0.20:
        verdict = "GO — centroids are well-separated; NeCo sharpening should help."
    elif median_sim < 0.50:
        verdict = "GO with caution — centroids are moderately separated."
    else:
        verdict = (
            "PIVOT — centroids are highly collapsed (median sim too high); "
            "NeCo alone may not separate dead-class candidates."
        )
    logger.info("    %s", verdict)
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
