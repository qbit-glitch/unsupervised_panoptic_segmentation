#!/usr/bin/env python3
"""PPAP: Progressive Proxy Anchor Propagation for overclustering.

Replaces k-means with learnable proxy anchors that seek distinct semantic
modes regardless of feature-space density. Proxies are pulled toward
nearby features and pushed away from other proxies, preventing rare-class
absorption by dominant clusters.

Based on "Progressive Proxy Anchor Propagation for Unsupervised Semantic
Segmentation" (ECCV 2024).

Usage:
    python mbps_pytorch/generate_ppap_ablation.py \
        --cityscapes_root /data/cityscapes \
        --feat_subdir dinov3_features_vitl16 \
        --k 100 --seed 42 --epochs 20
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


def proxy_nca_loss(
    features: torch.Tensor,
    proxies: torch.Tensor,
    assignments: torch.Tensor,
    temperature: float = 0.1,
    margin: float = 0.1,
) -> torch.Tensor:
    """Proxy-NCA+ loss: pull features toward assigned proxy, push away others.

    Args:
        features: (N, D) L2-normalized feature vectors.
        proxies: (K, D) L2-normalized proxy vectors.
        assignments: (N,) hard assignments to proxies.
        temperature: softmax temperature.
        margin: additive margin for positive proxy similarity.
    """
    sim = features @ proxies.T / temperature  # (N, K)

    pos_mask = F.one_hot(assignments, num_classes=proxies.shape[0]).float()
    neg_mask = 1.0 - pos_mask

    pos_sim = (sim + margin) * pos_mask + (-1e9) * neg_mask
    neg_sim = sim * neg_mask + (-1e9) * pos_mask

    loss_pull = -torch.logsumexp(pos_sim, dim=1).mean()
    loss_push = torch.logsumexp(neg_sim, dim=1).mean()

    return loss_pull + loss_push


def proxy_diversity_loss(proxies: torch.Tensor) -> torch.Tensor:
    """Penalize proxies that collapse to the same location."""
    sim = proxies @ proxies.T
    K = proxies.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=proxies.device)
    off_diag = sim[mask]
    return off_diag.clamp(min=0).mean()


def progressive_proxy_refinement(
    X: np.ndarray,
    k: int = 100,
    seed: int = 42,
    epochs: int = 20,
    lr_proxy: float = 1e-3,
    temperature: float = 0.1,
    margin: float = 0.1,
    batch_size: int = 8192,
    diversity_weight: float = 0.1,
    progressive_schedule: bool = True,
    device: str = "auto",
) -> Dict:
    """PPAP-style proxy anchor learning on pre-extracted features."""
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
        f"PPAP: N={N}, D={input_dim}, K={k}, epochs={epochs}, "
        f"lr={lr_proxy}, temp={temperature}, device={device}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Warm-starting proxies from spherical k-means...")
    init_result = fit_spherical_kmeans(X, k=k, seed=seed, refine_iters=20)
    proxy_init = torch.from_numpy(init_result["centers"]).float()

    proxies = torch.nn.Parameter(proxy_init.to(device))
    optimizer = torch.optim.Adam([proxies], lr=lr_proxy)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tensor = torch.from_numpy(X).float().to(device)

    for epoch in range(epochs):
        t0 = time.time()

        with torch.no_grad():
            proxies_normed = F.normalize(proxies, dim=-1)
            sim = X_tensor @ proxies_normed.T
            assignments = sim.argmax(dim=1)

        counts = torch.bincount(assignments, minlength=k)
        empty = (counts == 0).sum().item()
        min_count = counts[counts > 0].min().item() if (counts > 0).any() else 0

        if progressive_schedule:
            progress = epoch / max(epochs - 1, 1)
            curr_temp = temperature * (1.0 + 2.0 * (1.0 - progress))
            curr_margin = margin * progress
        else:
            curr_temp = temperature
            curr_margin = margin

        perm = torch.randperm(N, device=device)
        epoch_losses = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            batch_x = X_tensor[idx]
            batch_assign = assignments[idx]

            proxies_normed = F.normalize(proxies, dim=-1)

            loss_nca = proxy_nca_loss(
                batch_x, proxies_normed, batch_assign,
                temperature=curr_temp, margin=curr_margin,
            )
            loss_div = proxy_diversity_loss(proxies_normed)
            loss = loss_nca + diversity_weight * loss_div

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([proxies], max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()

        if empty > 0 and epoch > 0 and epoch % 5 == 0:
            with torch.no_grad():
                proxies_normed = F.normalize(proxies, dim=-1)
                sim_all = X_tensor @ proxies_normed.T
                new_assignments = sim_all.argmax(dim=1)
                new_counts = torch.bincount(new_assignments, minlength=k)

                dead_mask = new_counts == 0
                if dead_mask.any():
                    alive_mask = ~dead_mask
                    alive_counts = new_counts[alive_mask]
                    largest_idx = torch.where(alive_mask)[0][alive_counts.argmax()]

                    members = X_tensor[new_assignments == largest_idx.item()]
                    if len(members) > 0:
                        member_dists = torch.cdist(
                            members[:min(1000, len(members))],
                            proxies_normed[largest_idx:largest_idx+1]
                        ).squeeze()
                        far_idx = member_dists.argmax()
                        dead_indices = torch.where(dead_mask)[0]
                        n_reinit = min(len(dead_indices), len(members))
                        for i in range(n_reinit):
                            noise = torch.randn_like(proxies[0]) * 0.01
                            proxies.data[dead_indices[i]] = F.normalize(
                                members[far_idx] + noise * (i + 1), dim=-1
                            )
                        logger.info(
                            f"  Re-initialized {n_reinit} dead proxies from "
                            f"cluster {largest_idx.item()} (size={new_counts[largest_idx].item()})"
                        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1}/{epochs}: loss={np.mean(epoch_losses):.4f}, "
            f"empty={empty}, min_count={min_count}, "
            f"temp={curr_temp:.3f}, margin={curr_margin:.3f}, "
            f"time={elapsed:.1f}s"
        )

    with torch.no_grad():
        final_centers = F.normalize(proxies, dim=-1).cpu().numpy()

    return {"centers": final_centers, "method": "ppap"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPAP proxy anchor overclustering for dead-class recovery"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--feat_subdir", type=str, default="dinov3_features_vitl16")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_frac", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_proxy", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    out_subdir = f"pseudo_semantic_raw_dinov3_k{args.k}_ppap_vitl16"
    out_dir = root / out_subdir

    train_files = find_feature_files(root, "train", args.feat_subdir)
    logger.info(f"Found {len(train_files)} train images")

    t0 = time.time()
    X = load_and_subsample_features(train_files, args.sample_frac, args.seed)

    result = progressive_proxy_refinement(
        X,
        k=args.k,
        seed=args.seed,
        epochs=args.epochs,
        lr_proxy=args.lr_proxy,
        temperature=args.temperature,
        margin=args.margin,
        batch_size=args.batch_size,
        diversity_weight=args.diversity_weight,
        device=args.device,
    )
    fit_time = time.time() - t0
    logger.info(f"PPAP done in {fit_time:.1f}s")

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
    stats["method"] = "ppap"
    stats["k"] = args.k
    stats["epochs"] = args.epochs
    with open(str(out_dir / "cluster_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        f"Stats: entropy={stats['entropy']:.3f}, gini={stats['gini']:.3f}, "
        f"empty={stats['empty_clusters']}"
    )

    logger.info(f"All done -> {out_dir}")


if __name__ == "__main__":
    main()
