#!/usr/bin/env python3
"""SDCluster: Prototype constraint + semantic consistency for overclustering.

Iterative clustering with explicit dead-prototype detection and re-initialization.
Unlike k-means, monitors per-cluster assignment counts and re-initializes dead
prototypes by sampling from high-uncertainty or furthest-point regions.

Based on "SDCluster: A clustering based self-supervised pre-training method
for semantic segmentation" (ISPRS 2025).

Usage:
    python mbps_pytorch/generate_sdcluster_ablation.py \
        --cityscapes_root /data/cityscapes \
        --feat_subdir dinov3_features_vitl16 \
        --k 100 --seed 42 --epochs 30
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from mbps_pytorch.generate_clustering_ablation import (
    assign_clusters_cosine,
    compute_cluster_stats,
    find_feature_files,
    fit_spherical_kmeans,
    load_and_subsample_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def reinit_dead_prototypes(
    prototypes: torch.Tensor,
    features: torch.Tensor,
    assignments: torch.Tensor,
    counts: torch.Tensor,
    dead_threshold: float,
    strategy: str = "furthest_point",
) -> int:
    """Detect and re-initialize dead prototypes.

    Args:
        prototypes: (K, D) current prototype vectors.
        features: (N, D) feature matrix.
        assignments: (N,) current assignments.
        counts: (K,) per-cluster counts.
        dead_threshold: fraction below which a prototype is dead.
        strategy: "furthest_point" | "perturb_largest" | "sample_uncertain".

    Returns:
        Number of re-initialized prototypes.
    """
    K = prototypes.shape[0]
    N = features.shape[0]
    threshold_count = int(dead_threshold * N)
    dead_mask = counts < threshold_count
    n_dead = dead_mask.sum().item()

    if n_dead == 0:
        return 0

    dead_indices = torch.where(dead_mask)[0]
    alive_mask = ~dead_mask

    if not alive_mask.any():
        return 0

    if strategy == "furthest_point":
        sim = features @ prototypes[alive_mask].T
        max_sim = sim.max(dim=1).values
        candidate_indices = torch.argsort(max_sim)[:n_dead * 10]
        for i, dead_idx in enumerate(dead_indices):
            if i < len(candidate_indices):
                feat_idx = candidate_indices[i]
                noise = torch.randn_like(prototypes[0]) * 0.005
                prototypes[dead_idx] = F.normalize(
                    features[feat_idx] + noise, dim=-1
                )

    elif strategy == "perturb_largest":
        alive_counts = counts[alive_mask]
        largest_alive = torch.where(alive_mask)[0][alive_counts.argmax()]
        members = features[assignments == largest_alive.item()]
        if len(members) == 0:
            return 0
        dists = 1.0 - (members @ prototypes[largest_alive:largest_alive+1].T).squeeze()
        far_order = torch.argsort(dists, descending=True)
        for i, dead_idx in enumerate(dead_indices):
            idx = far_order[min(i, len(far_order) - 1)]
            noise = torch.randn_like(prototypes[0]) * 0.01 * (i + 1)
            prototypes[dead_idx] = F.normalize(members[idx] + noise, dim=-1)

    elif strategy == "sample_uncertain":
        sim = features @ prototypes.T
        top2 = sim.topk(2, dim=1).values
        uncertainty = 1.0 - (top2[:, 0] - top2[:, 1])
        uncertain_order = torch.argsort(uncertainty, descending=True)
        for i, dead_idx in enumerate(dead_indices):
            if i < len(uncertain_order):
                feat_idx = uncertain_order[i]
                noise = torch.randn_like(prototypes[0]) * 0.005
                prototypes[dead_idx] = F.normalize(
                    features[feat_idx] + noise, dim=-1
                )

    return n_dead


def spatial_consistency_loss(
    assignments: torch.Tensor,
    prototypes: torch.Tensor,
    features: torch.Tensor,
    h_patches: int,
    w_patches: int,
    sample_size: int = 2000,
) -> torch.Tensor:
    """Encourage spatially adjacent patches to share cluster assignments.

    Samples patch pairs and penalizes different assignments when features
    are similar.
    """
    N = h_patches * w_patches
    n_images = features.shape[0] // N

    total_loss = torch.tensor(0.0, device=features.device)
    rng = torch.Generator(device=features.device)

    for img_idx in range(min(n_images, 10)):
        start = img_idx * N
        end = start + N
        img_feat = features[start:end]
        img_assign = assignments[start:end]

        rows = torch.arange(h_patches, device=features.device)
        cols = torch.arange(w_patches, device=features.device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        flat_r = grid_r.reshape(-1)
        flat_c = grid_c.reshape(-1)

        idx_a = torch.randint(0, N, (sample_size,), device=features.device)

        dr = torch.randint(-1, 2, (sample_size,), device=features.device)
        dc = torch.randint(-1, 2, (sample_size,), device=features.device)
        nr = (flat_r[idx_a] + dr).clamp(0, h_patches - 1)
        nc = (flat_c[idx_a] + dc).clamp(0, w_patches - 1)
        idx_b = nr * w_patches + nc

        feat_sim = (img_feat[idx_a] * img_feat[idx_b]).sum(dim=1)
        same_cluster = (img_assign[idx_a] == img_assign[idx_b]).float()

        total_loss += ((1.0 - same_cluster) * feat_sim.clamp(min=0)).mean()

    return total_loss / min(n_images, 10)


def within_cluster_coherence_loss(
    features: torch.Tensor,
    assignments: torch.Tensor,
    prototypes: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Minimize within-cluster feature variance (tighter clusters)."""
    sim = (features * prototypes[assignments]).sum(dim=1)
    return (1.0 - sim).mean()


def sdcluster_train(
    X: np.ndarray,
    k: int = 100,
    seed: int = 42,
    epochs: int = 30,
    dead_threshold: float = 0.005,
    reinit_strategy: str = "furthest_point",
    reinit_interval: int = 3,
    coherence_weight: float = 0.3,
    lr: float = 5e-3,
    batch_size: int = 8192,
    device: str = "auto",
) -> Dict:
    """SDCluster iterative clustering with dead-prototype recovery."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    input_dim = X.shape[1]
    N = X.shape[0]
    logger.info(
        f"SDCluster: N={N}, D={input_dim}, K={k}, epochs={epochs}, "
        f"dead_threshold={dead_threshold}, reinit={reinit_strategy}, device={device}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Warm-starting prototypes from spherical k-means...")
    init_result = fit_spherical_kmeans(X, k=k, seed=seed, refine_iters=20)
    proto_init = torch.from_numpy(init_result["centers"]).float()

    prototypes = torch.nn.Parameter(proto_init.to(device))
    optimizer = torch.optim.Adam([prototypes], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tensor = torch.from_numpy(X).float().to(device)

    total_reinits = 0

    for epoch in range(epochs):
        t0 = time.time()

        with torch.no_grad():
            proto_normed = F.normalize(prototypes, dim=-1)
            sim = X_tensor @ proto_normed.T
            assignments = sim.argmax(dim=1)
            counts = torch.bincount(assignments, minlength=k)
            empty = (counts == 0).sum().item()
            min_count = counts[counts > 0].min().item() if (counts > 0).any() else 0

        if epoch > 0 and epoch % reinit_interval == 0:
            with torch.no_grad():
                n_reinit = reinit_dead_prototypes(
                    prototypes.data, X_tensor, assignments, counts,
                    dead_threshold, reinit_strategy,
                )
                if n_reinit > 0:
                    total_reinits += n_reinit
                    logger.info(f"  Re-initialized {n_reinit} dead prototypes")

        perm = torch.randperm(N, device=device)
        epoch_losses = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            batch_x = X_tensor[idx]
            batch_assign = assignments[idx]

            proto_normed = F.normalize(prototypes, dim=-1)

            loss_coherence = within_cluster_coherence_loss(
                batch_x, batch_assign, proto_normed, k
            )

            sim_batch = batch_x @ proto_normed.T
            log_probs = F.log_softmax(sim_batch / 0.1, dim=1)
            loss_assign = F.nll_loss(log_probs, batch_assign)

            loss = loss_assign + coherence_weight * loss_coherence

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([prototypes], max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()

        entropy = -(counts.float() / counts.sum()).clamp(min=1e-30).log().mul(
            counts.float() / counts.sum()
        ).sum().item() / np.log(k)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1}/{epochs}: loss={np.mean(epoch_losses):.4f}, "
            f"empty={empty}, min_count={min_count}, entropy={entropy:.3f}, "
            f"total_reinits={total_reinits}, time={elapsed:.1f}s"
        )

    with torch.no_grad():
        final_centers = F.normalize(prototypes, dim=-1).cpu().numpy()

    return {
        "centers": final_centers,
        "method": "sdcluster",
        "total_reinits": total_reinits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDCluster prototype-constrained overclustering"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--feat_subdir", type=str, default="dinov3_features_vitl16")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_frac", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dead_threshold", type=float, default=0.005)
    parser.add_argument(
        "--reinit_strategy", type=str, default="furthest_point",
        choices=["furthest_point", "perturb_largest", "sample_uncertain"],
    )
    parser.add_argument("--reinit_interval", type=int, default=3)
    parser.add_argument("--coherence_weight", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    out_subdir = f"pseudo_semantic_raw_dinov3_k{args.k}_sdcluster_vitl16"
    out_dir = root / out_subdir

    train_files = find_feature_files(root, "train", args.feat_subdir)
    logger.info(f"Found {len(train_files)} train images")

    t0 = time.time()
    X = load_and_subsample_features(train_files, args.sample_frac, args.seed)

    result = sdcluster_train(
        X,
        k=args.k,
        seed=args.seed,
        epochs=args.epochs,
        dead_threshold=args.dead_threshold,
        reinit_strategy=args.reinit_strategy,
        reinit_interval=args.reinit_interval,
        coherence_weight=args.coherence_weight,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    fit_time = time.time() - t0
    logger.info(f"SDCluster done in {fit_time:.1f}s")

    centers = result["centers"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / "centroids.npz"), centers=centers)
    logger.info(f"Saved centroids to {out_dir / 'centroids.npz'}")

    for split in args.splits:
        files = find_feature_files(root, split, args.feat_subdir)
        logger.info(f"Assigning {split}: {len(files)} images")
        assign_clusters_cosine(files, centers, out_dir, split)

    logger.info("Computing cluster statistics...")
    stats = compute_cluster_stats(train_files[:200], centers)
    stats["fit_time_seconds"] = fit_time
    stats["method"] = "sdcluster"
    stats["k"] = args.k
    stats["epochs"] = args.epochs
    stats["total_reinits"] = result["total_reinits"]
    with open(str(out_dir / "cluster_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        f"Stats: entropy={stats['entropy']:.3f}, gini={stats['gini']:.3f}, "
        f"empty={stats['empty_clusters']}, reinits={result['total_reinits']}"
    )

    logger.info(f"All done -> {out_dir}")


if __name__ == "__main__":
    main()
