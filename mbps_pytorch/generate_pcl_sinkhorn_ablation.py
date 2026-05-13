#!/usr/bin/env python3
"""PCL + Sinkhorn: Prototypical Contrastive Learning with equipartition.

Iteratively refines cluster assignments via an EM loop:
  1. Sinkhorn-balanced assignment (guarantees balanced clusters)
  2. Compute per-cluster prototypes in projection space
  3. Train projection head with ProtoNCE contrastive loss
  4. Re-cluster using updated projections

Uses existing ViT-L/16 features. Sinkhorn forces rare classes to get
dedicated clusters instead of being swallowed by dominant ones.

Usage:
    python mbps_pytorch/generate_pcl_sinkhorn_ablation.py \
        --cityscapes_root /data/cityscapes \
        --feat_subdir dinov3_features_vitl16 \
        --k 100 --seed 42 --em_iterations 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from mbps_pytorch.generate_clustering_ablation import (
    assign_clusters_cosine,
    compute_cluster_stats,
    find_feature_files,
    fit_spherical_kmeans,
    load_and_subsample_features,
    load_features_normalized,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUT_H, OUT_W = 512, 1024


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


def proto_nce_loss(
    z: torch.Tensor,
    prototypes: torch.Tensor,
    assignments: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """ProtoNCE: contrastive loss using cluster prototypes as anchors.

    Args:
        z: projected features (N, D_proj), L2-normalized.
        prototypes: per-cluster prototypes (K, D_proj), L2-normalized.
        assignments: cluster assignments (N,) int.
        temperature: softmax temperature.
    """
    logits = z @ prototypes.T / temperature  # (N, K)
    return F.cross_entropy(logits, assignments)


def sinkhorn_assign_torch(
    features: torch.Tensor,
    centers: torch.Tensor,
    n_iters: int = 5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Sinkhorn equipartition on GPU/MPS. Returns hard assignments."""
    N = features.shape[0]
    K = centers.shape[0]

    logits = features @ centers.T / temperature
    logits -= logits.max(dim=1, keepdim=True).values
    Q = torch.exp(logits)
    Q /= Q.sum() + 1e-30

    for _ in range(n_iters):
        Q /= Q.sum(dim=0, keepdim=True) + 1e-30
        Q /= K
        Q /= Q.sum(dim=1, keepdim=True) + 1e-30
        Q /= N

    return Q.argmax(dim=1)


def compute_prototypes(
    z: torch.Tensor, assignments: torch.Tensor, k: int
) -> torch.Tensor:
    """Mean of projected features per cluster, L2-normalized."""
    prototypes = torch.zeros(k, z.shape[1], device=z.device)
    for c in range(k):
        mask = assignments == c
        if mask.sum() > 0:
            prototypes[c] = z[mask].mean(dim=0)
    return F.normalize(prototypes, dim=-1)


def pcl_sinkhorn_em(
    X: np.ndarray,
    k: int = 100,
    seed: int = 42,
    em_iterations: int = 10,
    projection_dim: int = 128,
    hidden_dim: int = 256,
    contrastive_temp: float = 0.2,
    contrastive_lr: float = 1e-3,
    contrastive_epochs: int = 3,
    contrastive_batch_size: int = 8192,
    sinkhorn_temp: float = 0.1,
    sinkhorn_iters: int = 5,
    device: str = "auto",
) -> Dict:
    """Run PCL + Sinkhorn EM loop."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    input_dim = X.shape[1]
    logger.info(
        f"PCL+Sinkhorn: N={X.shape[0]}, D={input_dim}, K={k}, "
        f"EM_iters={em_iterations}, proj_dim={projection_dim}, device={device}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Initializing centers with spherical k-means...")
    init_result = fit_spherical_kmeans(X, k=k, seed=seed, refine_iters=20)
    centers_np = init_result["centers"].copy()

    proj_head = ProjectionHead(input_dim, hidden_dim, projection_dim).to(device)
    optimizer = torch.optim.Adam(proj_head.parameters(), lr=contrastive_lr)

    X_tensor = torch.from_numpy(X).float().to(device)
    centers_tensor = torch.from_numpy(centers_np).float().to(device)

    for em_iter in range(em_iterations):
        t0 = time.time()

        # E-step: Sinkhorn assignment in original feature space
        with torch.no_grad():
            assignments = sinkhorn_assign_torch(
                X_tensor, centers_tensor, sinkhorn_iters, sinkhorn_temp
            )

        counts = torch.bincount(assignments, minlength=k)
        empty = (counts == 0).sum().item()
        entropy = -(counts.float() / counts.sum()).clamp(min=1e-30).log().mul(
            counts.float() / counts.sum()
        ).sum().item() / np.log(k)

        # Compute prototypes in projection space
        with torch.no_grad():
            z_all = proj_head(X_tensor)
            prototypes = compute_prototypes(z_all, assignments, k)

        # M-step: train projection head with ProtoNCE
        proj_head.train()
        N = X_tensor.shape[0]
        epoch_losses = []

        for epoch in range(contrastive_epochs):
            perm = torch.randperm(N, device=device)
            batch_losses = []

            for start in range(0, N, contrastive_batch_size):
                end = min(start + contrastive_batch_size, N)
                idx = perm[start:end]
                batch_x = X_tensor[idx]
                batch_assign = assignments[idx]

                z = proj_head(batch_x)
                loss = proto_nce_loss(z, prototypes.detach(), batch_assign, contrastive_temp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        proj_head.eval()

        # Update prototypes after training
        with torch.no_grad():
            z_all = proj_head(X_tensor)
            prototypes = compute_prototypes(z_all, assignments, k)

        # Re-estimate centers in original feature space from Sinkhorn assignments
        centers_np = np.zeros((k, input_dim), dtype=np.float32)
        assignments_np = assignments.cpu().numpy()
        for c in range(k):
            mask = assignments_np == c
            if mask.sum() > 0:
                centers_np[c] = X[mask].mean(axis=0)
        norms = np.linalg.norm(centers_np, axis=1, keepdims=True) + 1e-8
        centers_np = centers_np / norms
        centers_tensor = torch.from_numpy(centers_np).float().to(device)

        elapsed = time.time() - t0
        logger.info(
            f"EM iter {em_iter+1}/{em_iterations}: "
            f"ProtoNCE={epoch_losses[-1]:.4f}, "
            f"entropy={entropy:.3f}, empty={empty}, "
            f"time={elapsed:.1f}s"
        )

        if em_iter > 2 and epoch_losses[-1] > 10.0:
            logger.warning("ProtoNCE loss too high, stopping early")
            break

    return {
        "centers": centers_np,
        "method": "pcl_sinkhorn",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCL + Sinkhorn overclustering for dead-class recovery"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--feat_subdir", type=str, default="dinov3_features_vitl16")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_frac", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    # EM loop
    parser.add_argument("--em_iterations", type=int, default=10)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    # Contrastive
    parser.add_argument("--contrastive_temp", type=float, default=0.2)
    parser.add_argument("--contrastive_lr", type=float, default=1e-3)
    parser.add_argument("--contrastive_epochs", type=int, default=3)
    parser.add_argument("--contrastive_batch_size", type=int, default=8192)
    # Sinkhorn
    parser.add_argument("--sinkhorn_temp", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iters", type=int, default=5)
    # Device
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    out_subdir = f"pseudo_semantic_raw_dinov3_k{args.k}_pcl_sinkhorn_vitl16"
    out_dir = root / out_subdir

    train_files = find_feature_files(root, "train", args.feat_subdir)
    logger.info(f"Found {len(train_files)} train images")

    t0 = time.time()
    X = load_and_subsample_features(train_files, args.sample_frac, args.seed)

    result = pcl_sinkhorn_em(
        X,
        k=args.k,
        seed=args.seed,
        em_iterations=args.em_iterations,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        contrastive_temp=args.contrastive_temp,
        contrastive_lr=args.contrastive_lr,
        contrastive_epochs=args.contrastive_epochs,
        contrastive_batch_size=args.contrastive_batch_size,
        sinkhorn_temp=args.sinkhorn_temp,
        sinkhorn_iters=args.sinkhorn_iters,
        device=args.device,
    )
    fit_time = time.time() - t0
    logger.info(f"PCL+Sinkhorn done in {fit_time:.1f}s")

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
    stats["method"] = "pcl_sinkhorn"
    stats["k"] = args.k
    stats["em_iterations"] = args.em_iterations
    with open(str(out_dir / "cluster_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        f"Stats: entropy={stats['entropy']:.3f}, gini={stats['gini']:.3f}, "
        f"empty={stats['empty_clusters']}"
    )

    logger.info(f"All done -> {out_dir}")


if __name__ == "__main__":
    main()
