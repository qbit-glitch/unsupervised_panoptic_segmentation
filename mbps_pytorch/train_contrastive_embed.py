#!/usr/bin/env python3
"""Train a contrastive projection head for instance decomposition.

Learns a compact 128-dim embedding from [DINOv2 + depth + position] where
same-instance patches cluster together. Uses InfoNCE loss with depth-based
pair mining: positives share a depth plane, negatives cross depth edges.

Usage:
    python mbps_pytorch/train_contrastive_embed.py \
        --cityscapes_root /path/to/cityscapes \
        --run_id 1 --epochs 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Reuse eval infrastructure from ablation script
sys.path.insert(0, str(Path(__file__).parent))
from ablate_instance_methods import (
    CS_ID_TO_TRAIN, NUM_CLASSES, STUFF_IDS,
    discover_files, remap_gt_to_trainids, resize_nearest,
    evaluate_panoptic_single, compute_pq_from_accumulators,
    NEEDS_FEATURES,
)
from instance_methods import METHODS
from instance_methods.utils import load_features

logger = logging.getLogger(__name__)

# Cityscapes thing class trainIDs
THING_IDS = set(range(11, 19))
FEAT_H, FEAT_W, FEAT_DIM = 32, 64, 768


# ─── Model ───

class ContrastiveProjectionHead(nn.Module):
    """MLP projection: [DINOv2(768) + depth(1) + pos(2)] -> 128-dim L2-norm."""

    def __init__(self, input_dim: int = 771, hidden_dim: int = 512,
                 embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


# ─── Dataset ───

class ContrastiveInstanceDataset(Dataset):
    """Loads pre-extracted features + depth + semantics for contrastive training.

    Each sample yields the patch-level input vectors for one image,
    plus semantic labels at patch resolution for pair mining.
    """

    def __init__(self, cityscapes_root: str, split: str = "train",
                 semantic_subdir: str = "pseudo_semantic_raw_dinov3_k80",
                 depth_subdir: str = "depth_spidepth",
                 feature_subdir: str = "dinov2_features",
                 centroids_path: str = None,
                 depth_weight: float = 2.0,
                 pos_weight: float = 0.5):
        self.root = Path(cityscapes_root)
        self.depth_weight = depth_weight
        self.pos_weight = pos_weight

        # Load cluster-to-trainID mapping
        if centroids_path is None:
            centroids_path = (
                self.root / "pseudo_semantic_raw_k80" / "kmeans_centroids.npz"
            )
        cent = np.load(str(centroids_path))
        c2c = cent["cluster_to_class"]
        self.k_to_trainid = np.full(256, 255, dtype=np.uint8)
        for k in range(len(c2c)):
            self.k_to_trainid[k] = int(c2c[k])

        # Discover files
        feat_dir = self.root / feature_subdir / split
        sem_dir = self.root / semantic_subdir / split
        depth_dir = self.root / depth_subdir / split

        self.samples: List[Tuple[Path, Path, Path]] = []
        for feat_path in sorted(feat_dir.rglob("*.npy")):
            city = feat_path.parent.name
            stem = feat_path.stem.replace("_leftImg8bit", "")

            sem_path = sem_dir / city / f"{stem}.png"
            depth_path = depth_dir / city / f"{stem}.npy"
            if sem_path.exists() and depth_path.exists():
                self.samples.append((feat_path, depth_path, sem_path))

        logger.info(f"ContrastiveInstanceDataset: {len(self.samples)} images "
                     f"({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat_path, depth_path, sem_path = self.samples[idx]

        # Features: (2048, 768) float16 -> (32, 64, 768) float32
        feats = np.load(str(feat_path)).astype(np.float32)
        feats = feats.reshape(FEAT_H, FEAT_W, FEAT_DIM)

        # Depth: (512, 1024) float32 -> downsample to (32, 64)
        from PIL import Image as PILImage
        depth_full = np.load(str(depth_path))
        depth = np.array(
            PILImage.fromarray(depth_full).resize(
                (FEAT_W, FEAT_H), PILImage.BILINEAR
            )
        )

        # Semantics: PNG uint8 -> downsample to (32, 64), map k80 -> trainID
        sem_k = np.array(PILImage.open(sem_path))
        sem_ds = np.array(
            PILImage.fromarray(sem_k).resize(
                (FEAT_W, FEAT_H), PILImage.NEAREST
            )
        )
        sem_trainid = self.k_to_trainid[sem_ds]

        # Build input vector: [features, depth*w_d, pos*w_p]
        yy, xx = np.mgrid[0:FEAT_H, 0:FEAT_W]
        pos = np.stack([yy / FEAT_H, xx / FEAT_W], axis=-1)  # (32, 64, 2)

        x = np.concatenate([
            feats,
            depth[:, :, None] * self.depth_weight,
            pos * self.pos_weight,
        ], axis=-1).astype(np.float32)  # (32, 64, 771)

        return (
            torch.from_numpy(x),            # (32, 64, 771)
            torch.from_numpy(depth),         # (32, 64)
            torch.from_numpy(sem_trainid),   # (32, 64) uint8
        )


# ─── Pair Mining + InfoNCE ───

def mine_and_compute_loss(
    model: ContrastiveProjectionHead,
    x_batch: torch.Tensor,
    depth_batch: torch.Tensor,
    sem_batch: torch.Tensor,
    tau: float = 0.07,
    delta_depth: float = 0.02,
    tau_edge: float = 0.08,
    max_anchors: int = 256,
    n_negatives: int = 32,
) -> torch.Tensor:
    """Mine pairs from a batch and compute InfoNCE loss.

    Args:
        model: projection head.
        x_batch: (B, H, W, D_in) input vectors.
        depth_batch: (B, H, W) depth at patch resolution.
        sem_batch: (B, H, W) trainID labels.
        tau: InfoNCE temperature.
        delta_depth: max depth difference for positive pairs.
        tau_edge: min depth gradient for negative pairs.
        max_anchors: max anchor patches per image.
        n_negatives: negatives per anchor.

    Returns:
        Scalar loss.
    """
    B, H, W, D_in = x_batch.shape
    device = x_batch.device
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for b in range(B):
        x = x_batch[b]        # (H, W, D_in)
        depth = depth_batch[b]  # (H, W)
        sem = sem_batch[b]      # (H, W)

        # Compute depth gradient magnitude at patch level
        grad_y = torch.zeros_like(depth)
        grad_x = torch.zeros_like(depth)
        grad_y[:-1, :] = (depth[1:, :] - depth[:-1, :]).abs()
        grad_x[:, :-1] = (depth[:, 1:] - depth[:, :-1]).abs()
        depth_grad = torch.max(grad_y, grad_x)  # (H, W)

        # Process each thing class
        for cls in THING_IDS:
            cls_mask = (sem == cls)
            n_cls = cls_mask.sum().item()
            if n_cls < 4:
                continue

            cls_ys, cls_xs = torch.where(cls_mask)
            cls_depth = depth[cls_ys, cls_xs]  # (N,)
            cls_x = x[cls_ys, cls_xs]          # (N, D_in)
            cls_grad = depth_grad[cls_ys, cls_xs]  # (N,)

            # Sample anchors
            n_anchors = min(max_anchors, n_cls)
            anchor_idx = torch.randperm(n_cls, device=device)[:n_anchors]

            a_depth = cls_depth[anchor_idx]  # (A,)
            a_x = cls_x[anchor_idx]          # (A, D_in)

            # Depth distances from anchors to all class pixels
            depth_diff = (a_depth[:, None] - cls_depth[None, :]).abs()  # (A, N)

            # Spatial distances (Manhattan) for proximity constraint
            a_ys = cls_ys[anchor_idx].float()
            a_xs = cls_xs[anchor_idx].float()
            spatial_dist = (
                (a_ys[:, None] - cls_ys[None, :].float()).abs() +
                (a_xs[:, None] - cls_xs[None, :].float()).abs()
            )  # (A, N)

            # Positive mask: close depth + spatially near (within 3 patches)
            pos_mask = (depth_diff < delta_depth) & (spatial_dist <= 3.0)
            # Exclude self
            pos_mask[torch.arange(n_anchors, device=device),
                     anchor_idx] = False

            # Negative mask: high depth gradient region (cross depth edge)
            neg_candidates = cls_grad > tau_edge  # (N,) global mask
            neg_mask = neg_candidates[None, :].expand(n_anchors, -1)
            # Also require depth difference > delta_depth to avoid false negatives
            neg_mask = neg_mask & (depth_diff > delta_depth)

            # Skip if no valid pairs for this class
            has_pos = pos_mask.any(dim=1)  # (A,)
            has_neg = neg_mask.any(dim=1)  # (A,)
            valid = has_pos & has_neg
            if valid.sum() == 0:
                continue

            valid_idx = torch.where(valid)[0]
            # Limit for memory
            if len(valid_idx) > max_anchors // 2:
                valid_idx = valid_idx[
                    torch.randperm(len(valid_idx))[:max_anchors // 2]
                ]

            # For each valid anchor: sample 1 positive + n_negatives
            anchors_list = []
            positives_list = []
            negatives_list = []

            for ai in valid_idx:
                # Sample 1 positive
                p_idx = torch.where(pos_mask[ai])[0]
                p_choice = p_idx[torch.randint(len(p_idx), (1,))].item()

                # Sample negatives
                n_idx = torch.where(neg_mask[ai])[0]
                n_count = min(n_negatives, len(n_idx))
                n_choices = n_idx[
                    torch.randperm(len(n_idx))[:n_count]
                ]

                anchors_list.append(a_x[ai])
                positives_list.append(cls_x[p_choice])
                negatives_list.append(cls_x[n_choices])

            if len(anchors_list) < 2:
                # BatchNorm requires >1 sample
                continue

            # Stack and project
            anchors_t = torch.stack(anchors_list)    # (K, D_in)
            positives_t = torch.stack(positives_list)  # (K, D_in)

            # Pad negatives to same size
            max_neg = max(n.shape[0] for n in negatives_list)
            neg_padded = torch.zeros(
                len(negatives_list), max_neg, D_in, device=device
            )
            neg_valid_mask = torch.zeros(
                len(negatives_list), max_neg, dtype=torch.bool, device=device
            )
            for i, n in enumerate(negatives_list):
                neg_padded[i, :n.shape[0]] = n
                neg_valid_mask[i, :n.shape[0]] = True

            # Project through model
            z_a = model(anchors_t)    # (K, embed_dim)
            z_p = model(positives_t)  # (K, embed_dim)

            K = z_a.shape[0]
            neg_flat = neg_padded.reshape(-1, D_in)
            z_n_flat = model(neg_flat)  # (K*max_neg, embed_dim)
            z_n = z_n_flat.reshape(K, max_neg, -1)  # (K, max_neg, embed_dim)

            # InfoNCE: L = -log(exp(sim_pos/τ) / (exp(sim_pos/τ) + Σ exp(sim_neg/τ)))
            sim_pos = (z_a * z_p).sum(dim=-1) / tau  # (K,)
            sim_neg = torch.bmm(
                z_n, z_a.unsqueeze(-1)
            ).squeeze(-1) / tau  # (K, max_neg)

            # Mask invalid negatives with large negative value
            sim_neg[~neg_valid_mask] = -1e9

            # Log-sum-exp over pos + neg
            logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # (K, 1+max_neg)
            labels = torch.zeros(K, dtype=torch.long, device=device)  # pos is index 0
            loss = F.cross_entropy(logits, labels)

            total_loss = total_loss + loss
            n_valid += 1

    if n_valid > 0:
        return total_loss / n_valid
    return total_loss


# ─── Training Loop ───

# ─── Validation (PQ on val subset) ───

def validate_pq(model, val_files, k_to_trainid, eval_hw,
                depth_weight: float = 2.0, pos_weight: float = 0.5):
    """Run PQ evaluation on val images using the trained model.

    Args:
        model: trained ContrastiveProjectionHead (already on device, eval mode).
        val_files: list of (sem, depth, gt_label, gt_inst, feat) paths.
        k_to_trainid: numpy array mapping k80 clusters to trainIDs.
        eval_hw: (H, W) evaluation resolution.
        depth_weight: depth scaling for input vector.
        pos_weight: position scaling for input vector.

    Returns:
        dict with PQ, PQ_stuff, PQ_things, per_class, etc.
    """
    from PIL import Image as PILImage

    H, W = eval_hw
    method_fn = METHODS["contrastive"]
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    total_instances = 0
    n_images = 0

    for sem_path, depth_path, gt_label_path, gt_inst_path, feat_path in tqdm(
        val_files, desc="Val PQ", leave=False
    ):
        pred_k = np.array(PILImage.open(sem_path))
        pred_sem = k_to_trainid[pred_k]
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = np.array(
                PILImage.fromarray(depth).resize((W, H), PILImage.BILINEAR)
            )

        features = None
        if feat_path is not None:
            features = load_features(str(feat_path))

        kwargs = {
            "thing_ids": THING_IDS,
            "features": features,
            "model": model,
            "depth_weight": depth_weight,
            "pos_weight": pos_weight,
            "hdbscan_min_cluster": 5,
            "hdbscan_min_samples": 3,
            "min_area": 1000,
            "dilation_iters": 3,
        }

        try:
            instances = method_fn(pred_sem, depth, **kwargs)
        except Exception:
            instances = []

        gt_raw = np.array(PILImage.open(gt_label_path))
        gt_sem = remap_gt_to_trainids(gt_raw)
        if gt_sem.shape != (H, W):
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(PILImage.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                PILImage.fromarray(gt_inst_map).resize(
                    (W, H), PILImage.NEAREST
                )
            )

        tp, fp, fn, iou_s, n_inst = evaluate_panoptic_single(
            pred_sem, instances, gt_sem, gt_inst_map, eval_hw
        )
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        iou_acc += iou_s
        total_instances += n_inst
        n_images += 1

    metrics = compute_pq_from_accumulators(tp_acc, fp_acc, fn_acc, iou_acc)
    metrics["avg_instances"] = round(total_instances / max(n_images, 1), 1)
    metrics["n_images"] = n_images
    return metrics


RUN_CONFIGS = {
    1: {"tau": 0.07, "delta_depth": 0.02, "depth_weight": 2.0, "pos_weight": 0.5},
    2: {"tau": 0.07, "delta_depth": 0.05, "depth_weight": 5.0, "pos_weight": 0.5},
    3: {"tau": 0.10, "delta_depth": 0.02, "depth_weight": 2.0, "pos_weight": 0.5},
    4: {"tau": 0.10, "delta_depth": 0.05, "depth_weight": 5.0, "pos_weight": 0.5},
    5: {"tau": 0.07, "delta_depth": 0.02, "depth_weight": 0.0, "pos_weight": 0.0},
}


def train(args: argparse.Namespace) -> None:
    """Train contrastive projection head."""
    # Resolve run config
    run_cfg = RUN_CONFIGS.get(args.run_id, RUN_CONFIGS[1])
    depth_weight = run_cfg["depth_weight"]
    pos_weight = run_cfg["pos_weight"]
    tau = run_cfg["tau"]
    delta_depth = run_cfg["delta_depth"]

    input_dim = FEAT_DIM + (1 if depth_weight > 0 else 0) + (2 if pos_weight > 0 else 0)
    # Always use 771 for consistency with inference code
    input_dim = FEAT_DIM + 1 + 2  # 771

    logger.info(f"Run {args.run_id}: tau={tau}, delta_depth={delta_depth}, "
                f"depth_weight={depth_weight}, pos_weight={pos_weight}")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Dataset
    dataset = ContrastiveInstanceDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        depth_subdir=args.depth_subdir,
        feature_subdir=args.feature_subdir,
        centroids_path=args.centroids_path,
        depth_weight=depth_weight,
        pos_weight=pos_weight,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model
    model = ContrastiveProjectionHead(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
    ).to(device)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Validation setup: discover val files for PQ eval
    val_semantic_subdir = args.semantic_subdir.replace("dinov3_k80", "k80")
    if "dinov3" in args.semantic_subdir:
        val_semantic_subdir = "pseudo_semantic_raw_k80"
    else:
        val_semantic_subdir = args.semantic_subdir
    val_files = discover_files(
        Path(args.cityscapes_root), "val", val_semantic_subdir,
        args.depth_subdir, args.feature_subdir,
    )
    # Use subset for fast validation (50 images ~ 5s)
    val_files = val_files[:args.val_images]
    logger.info(f"Validation: {len(val_files)} images from val split")

    # Load k_to_trainid for val evaluation
    centroids_path = args.centroids_path or str(
        Path(args.cityscapes_root) / "pseudo_semantic_raw_k80"
        / "kmeans_centroids.npz"
    )
    cent = np.load(centroids_path)
    c2c = cent["cluster_to_class"]
    k_to_trainid_val = np.full(256, 255, dtype=np.uint8)
    for k in range(len(c2c)):
        k_to_trainid_val[k] = int(c2c[k])

    eval_hw = (512, 1024)

    # Checkpoint dir
    ckpt_dir = Path(args.output_dir) / f"run{args.run_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_pq_things = -1.0
    config_record = {
        "run_id": args.run_id, "tau": tau, "delta_depth": delta_depth,
        "depth_weight": depth_weight, "pos_weight": pos_weight,
        "input_dim": input_dim, "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for batch_idx, (x, depth, sem) in enumerate(pbar):
            x = x.to(device)
            depth = depth.to(device)
            sem = sem.to(device)

            optimizer.zero_grad()
            loss = mine_and_compute_loss(
                model, x, depth, sem,
                tau=tau, delta_depth=delta_depth,
                tau_edge=args.tau_edge,
                max_anchors=args.max_anchors,
                n_negatives=args.n_negatives,
            )

            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        scheduler.step()
        dt = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)

        # Check embedding health
        model.eval()
        with torch.no_grad():
            sample_x = x[:1].reshape(-1, x.shape[-1])[:100]
            sample_z = model(sample_x)
            embed_std = sample_z.std(dim=0).mean().item()

        # PQ validation
        val_metrics = validate_pq(
            model, val_files, k_to_trainid_val, eval_hw,
            depth_weight=depth_weight, pos_weight=pos_weight,
        )
        pq_th = val_metrics["PQ_things"]
        pq = val_metrics["PQ"]
        pq_st = val_metrics["PQ_stuff"]

        logger.info(
            f"Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}, "
            f"embed_std={embed_std:.4f}, "
            f"PQ={pq:.2f} PQ_st={pq_st:.2f} PQ_th={pq_th:.2f} "
            f"inst/img={val_metrics['avg_instances']:.1f}, "
            f"lr={scheduler.get_last_lr()[0]:.6f}, time={dt:.1f}s"
        )

        # Track best training loss
        if avg_loss < best_loss and n_batches > 0:
            best_loss = avg_loss

        # Save best checkpoint by PQ_things (the metric we care about)
        if pq_th > best_pq_things:
            best_pq_things = pq_th
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "embed_std": embed_std,
                "PQ_things": pq_th,
                "PQ": pq,
                "val_metrics": val_metrics,
                "config": config_record,
            }, ckpt_dir / "best.pth")
            logger.info(
                f"  New best PQ_things={pq_th:.2f} (epoch {epoch})"
            )

        # Also save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "embed_std": embed_std,
            "PQ_things": pq_th,
            "PQ": pq,
            "val_metrics": val_metrics,
            "config": config_record,
        }, ckpt_dir / "latest.pth")

    # Save final config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    logger.info(f"Training complete. Best PQ_things: {best_pq_things:.2f}, "
                f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive projection head for instance decomposition"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--run_id", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Hyperparameter config (1-5)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--tau_edge", type=float, default=0.08,
                        help="Depth gradient threshold for negative pairs")
    parser.add_argument("--max_anchors", type=int, default=256)
    parser.add_argument("--n_negatives", type=int, default=32)
    parser.add_argument("--val_images", type=int, default=50,
                        help="Number of val images for PQ eval each epoch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_raw_dinov3_k80")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--feature_subdir", type=str, default="dinov2_features")
    parser.add_argument("--centroids_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/contrastive_embed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args)


if __name__ == "__main__":
    main()
