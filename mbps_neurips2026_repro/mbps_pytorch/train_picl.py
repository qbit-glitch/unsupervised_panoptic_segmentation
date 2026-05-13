#!/usr/bin/env python3
"""Train PICL: Pseudo-Instance Contrastive Learning for instance decomposition.

Unlike the prior contrastive approach (depth-proximity pairs), PICL mines pairs
from pseudo-instance MASKS:
  - Positives:      patches from the SAME pseudo-instance mask
  - Hard negatives: patches from DIFFERENT instances of the SAME class
  - Easy negatives: patches from stuff regions

This trains instance-discriminative features rather than depth-discriminative
ones, breaking the co-planar pedestrian/car ceiling without stereo or motion.

Usage:
    python mbps_pytorch/train_picl.py \
        --cityscapes_root /path/to/cityscapes \
        --picl_config 1 --epochs 20
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ablate_instance_methods import (
    NUM_CLASSES, discover_files, remap_gt_to_trainids, resize_nearest,
    evaluate_panoptic_single, compute_pq_from_accumulators,
)
from instance_methods.utils import load_features

logger = logging.getLogger(__name__)

THING_IDS = set(range(11, 19))
FEAT_H, FEAT_W, FEAT_DIM = 32, 64, 768
INPUT_DIM = FEAT_DIM + 1 + 2  # 771: feats + depth + pos


# ─── Model ───────────────────────────────────────────────────────────────────

class PICLProjectionHead(nn.Module):
    """MLP projection head with LayerNorm (not BatchNorm) for stable MPS training.

    Input:  (N, 771) = [DINOv2(768) | depth*w_d(1) | pos*w_p(2)]
    Output: (N, 128) L2-normalised embeddings
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 512,
                 embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


# ─── Dataset ─────────────────────────────────────────────────────────────────

def _project_masks_to_feat(
    masks_flat: np.ndarray,
    sem_ds: np.ndarray,
    fh: int = FEAT_H,
    fw: int = FEAT_W,
    min_patches: int = 3,
) -> Tuple[List[np.ndarray], List[int]]:
    """OR-pool pixel-space instance masks to DINOv2 feature-patch resolution.

    Args:
        masks_flat:  (N, H*W) bool array at full pixel resolution.
        sem_ds:      (fh, fw) semantic trainID map at feature resolution.
        fh, fw:      Feature grid height and width (32, 64).
        min_patches: Minimum patches per instance to include.

    Returns:
        feat_masks:  List of (fh, fw) bool arrays, one per valid instance.
        class_ids:   List of int trainIDs (majority class within mask).
    """
    if masks_flat.ndim != 2 or masks_flat.shape[0] == 0:
        return [], []

    N = masks_flat.shape[0]
    total_pixels = masks_flat.shape[1]  # e.g. 512*1024 = 524288
    # Infer full-res dimensions from aspect ratio matching feature grid
    full_h = int(round(fh * (total_pixels / (fh * fw)) ** 0.5 * (fw / fh) ** 0))
    # Safest: assume standard Cityscapes 512×1024
    ph, pw = 512, 1024
    scale_y, scale_x = ph // fh, pw // fw  # 16, 16

    # Reshape and OR-pool: (N,512,1024) -> (N,32,16,64,16).any((2,4)) -> (N,32,64)
    masks_2d = masks_flat.reshape(N, ph, pw)
    feat_masks_all = masks_2d.reshape(N, fh, scale_y, fw, scale_x).any(axis=(2, 4))

    result_masks: List[np.ndarray] = []
    result_class_ids: List[int] = []

    for i in range(N):
        feat_mask = feat_masks_all[i]  # (fh, fw) bool
        n_patches = int(feat_mask.sum())
        if n_patches < min_patches:
            continue

        sem_in_mask = sem_ds[feat_mask]
        if len(sem_in_mask) == 0:
            continue

        counts = np.bincount(sem_in_mask.astype(np.int32), minlength=19)
        cls = int(counts.argmax())
        if cls not in THING_IDS:
            continue

        result_masks.append(feat_mask)
        result_class_ids.append(cls)

    return result_masks, result_class_ids


class PICLDataset(Dataset):
    """Loads DINOv2 features + depth + pseudo-instance masks for PICL training.

    Each sample yields patch-level input vectors + projected instance masks
    for mask-based pair mining.
    """

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        instance_subdir: str = "pseudo_instance_spidepth",
        semantic_subdir: str = "pseudo_semantic_mapped_k80",
        depth_subdir: str = "depth_spidepth",
        feature_subdir: str = "dinov2_features",
        depth_weight: float = 2.0,
        pos_weight: float = 0.5,
        min_patches_per_instance: int = 3,
    ):
        self.depth_weight = depth_weight
        self.pos_weight = pos_weight
        self.min_patches = min_patches_per_instance

        root = Path(cityscapes_root)
        feat_dir = root / feature_subdir / split
        sem_dir = root / semantic_subdir / split
        depth_dir = root / depth_subdir / split
        inst_dir = root / instance_subdir / split

        self.samples: List[Tuple[Path, Path, Path, Path]] = []
        for feat_path in sorted(feat_dir.rglob("*.npy")):
            city = feat_path.parent.name
            stem = feat_path.stem.replace("_leftImg8bit", "")

            sem_path = sem_dir / city / f"{stem}.png"
            depth_path = depth_dir / city / f"{stem}.npy"
            inst_path = inst_dir / city / f"{stem}.npz"

            if sem_path.exists() and depth_path.exists() and inst_path.exists():
                self.samples.append((feat_path, depth_path, sem_path, inst_path))

        logger.info(f"PICLDataset [{split}]: {len(self.samples)} images "
                    f"({instance_subdir})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat_path, depth_path, sem_path, inst_path = self.samples[idx]

        # (1) DINOv2 features: (2048,768) float16 → (32,64,768) float32
        feats = np.load(str(feat_path)).astype(np.float32).reshape(FEAT_H, FEAT_W, FEAT_DIM)

        # (2) Depth: (512,1024) → bilinear downsample to (32,64)
        depth_full = np.load(str(depth_path))
        depth_ds = np.array(
            PILImage.fromarray(depth_full).resize((FEAT_W, FEAT_H), PILImage.BILINEAR)
        )

        # (3) Semantics: (1024,2048) uint8 trainIDs → nearest to (32,64)
        sem_full = np.array(PILImage.open(sem_path))
        sem_ds = np.array(
            PILImage.fromarray(sem_full).resize((FEAT_W, FEAT_H), PILImage.NEAREST)
        )

        # (4) Instance masks from .npz
        inst_data = np.load(str(inst_path))
        masks_flat = inst_data["masks"]  # (N, 524288) bool
        num_valid = int(inst_data.get("num_valid", masks_flat.shape[0]))
        masks_flat = masks_flat[:num_valid]

        # (5) Project masks to feature resolution
        inst_masks, inst_class_ids = _project_masks_to_feat(
            masks_flat, sem_ds, FEAT_H, FEAT_W, self.min_patches
        )

        # (6) Build input vector [feats | depth*w_d | pos*w_p]
        yy, xx = np.mgrid[0:FEAT_H, 0:FEAT_W]
        pos = np.stack([yy / FEAT_H, xx / FEAT_W], axis=-1)  # (32,64,2)
        x = np.concatenate([
            feats,
            depth_ds[:, :, None] * self.depth_weight,
            pos * self.pos_weight,
        ], axis=-1).astype(np.float32)  # (32,64,771)

        return (
            torch.from_numpy(x),           # (32,64,771)
            torch.from_numpy(sem_ds),      # (32,64) uint8
            inst_masks,                    # List[np.ndarray(32,64) bool]
            inst_class_ids,                # List[int]
        )


def picl_collate(batch):
    """Collate variable-length instance lists into a batch."""
    xs = torch.stack([b[0] for b in batch])        # (B,32,64,771)
    sems = torch.stack([b[1] for b in batch])      # (B,32,64)
    inst_masks = [b[2] for b in batch]             # List[B] of List[arr]
    inst_class_ids = [b[3] for b in batch]         # List[B] of List[int]
    return xs, sems, inst_masks, inst_class_ids


# ─── Loss ────────────────────────────────────────────────────────────────────

def mine_and_compute_loss_picl(
    model: PICLProjectionHead,
    x_batch: torch.Tensor,
    sem_batch: torch.Tensor,
    inst_masks_batch: List,
    inst_class_ids_batch: List,
    tau: float = 0.07,
    K_pos: int = 4,
    n_hard_neg: int = 16,
    n_easy_neg: int = 16,
    max_anchors_per_inst: int = 8,
) -> torch.Tensor:
    """Multi-positive InfoNCE with mask-based pair mining.

    Positives:      patches from the SAME pseudo-instance as the anchor.
    Hard negatives: patches from DIFFERENT instances of the SAME class.
    Easy negatives: patches from stuff regions (not in any thing instance).

    Args:
        model:                Projection head.
        x_batch:              (B, H, W, D_in) input vectors.
        sem_batch:            (B, H, W) trainID semantics.
        inst_masks_batch:     List[B] of List[np.ndarray(H,W) bool].
        inst_class_ids_batch: List[B] of List[int].
        tau:                  InfoNCE temperature.
        K_pos:                Positives per anchor.
        n_hard_neg:           Hard negatives (same class, diff instance) per anchor.
        n_easy_neg:           Easy negatives (stuff) per anchor.
        max_anchors_per_inst: Max anchors sampled from each instance.

    Returns:
        Scalar loss tensor.
    """
    B, H, W, D_in = x_batch.shape
    device = x_batch.device
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    all_vectors: List[torch.Tensor] = []
    all_triplets: List[Tuple[int, List[int], List[int]]] = []
    # triplets: (anchor_vec_idx, [pos_vec_indices], [neg_vec_indices])

    for b in range(B):
        x = x_batch[b]                      # (H, W, D_in)
        masks = inst_masks_batch[b]          # List[np.ndarray(H,W)]
        class_ids = inst_class_ids_batch[b]  # List[int]

        if len(masks) < 1:
            continue

        # Build union of all thing-instance pixels for easy-neg mining
        union_mask = np.zeros((H, W), dtype=bool)
        for m in masks:
            union_mask |= m
        stuff_ys, stuff_xs = np.where(~union_mask)
        has_easy_neg = len(stuff_ys) >= n_easy_neg

        # Group instances by class
        class_to_insts: dict = defaultdict(list)
        for inst_idx, cls in enumerate(class_ids):
            class_to_insts[cls].append(inst_idx)

        for cls, inst_indices in class_to_insts.items():
            # Build hard-neg pool: all patches from OTHER instances of same class
            hard_neg_ys_list, hard_neg_xs_list = [], []
            for other_idx in inst_indices:
                oys, oxs = np.where(masks[other_idx])
                hard_neg_ys_list.append(oys)
                hard_neg_xs_list.append(oxs)

            for ai, anchor_inst_idx in enumerate(inst_indices):
                anchor_mask = masks[anchor_inst_idx]  # (H,W) bool
                anchor_ys, anchor_xs = np.where(anchor_mask)
                n_avail = len(anchor_ys)
                if n_avail < 2:
                    continue

                # Hard neg pool = all OTHER instances of same class
                hard_ys = np.concatenate(
                    [hard_neg_ys_list[j] for j in range(len(inst_indices)) if j != ai],
                    axis=0
                ) if len(inst_indices) > 1 else np.array([], dtype=int)
                hard_xs = np.concatenate(
                    [hard_neg_xs_list[j] for j in range(len(inst_indices)) if j != ai],
                    axis=0
                ) if len(inst_indices) > 1 else np.array([], dtype=int)
                has_hard_neg = len(hard_ys) >= 1

                if not has_hard_neg and not has_easy_neg:
                    continue

                # Sample anchors
                n_anchors = min(max_anchors_per_inst, n_avail)
                perm = np.random.permutation(n_avail)[:n_anchors]
                a_ys, a_xs = anchor_ys[perm], anchor_xs[perm]

                for ai2 in range(n_anchors):
                    ay, ax = int(a_ys[ai2]), int(a_xs[ai2])

                    # Positive pool: other patches in same instance
                    pos_pool = [(int(y), int(x)) for y, x in
                                zip(anchor_ys, anchor_xs) if not (y == ay and x == ax)]
                    if len(pos_pool) < 1:
                        continue
                    pos_idx = np.random.choice(len(pos_pool),
                                               min(K_pos, len(pos_pool)),
                                               replace=False)
                    pos_coords = [pos_pool[i] for i in pos_idx]

                    # Negative pool
                    neg_coords = []
                    if has_hard_neg and len(hard_ys) > 0:
                        h_idx = np.random.choice(len(hard_ys),
                                                 min(n_hard_neg, len(hard_ys)),
                                                 replace=False)
                        neg_coords.extend(zip(hard_ys[h_idx].tolist(),
                                              hard_xs[h_idx].tolist()))
                    if has_easy_neg:
                        e_idx = np.random.choice(len(stuff_ys),
                                                 min(n_easy_neg, len(stuff_ys)),
                                                 replace=False)
                        neg_coords.extend(zip(stuff_ys[e_idx].tolist(),
                                              stuff_xs[e_idx].tolist()))
                    if len(neg_coords) == 0:
                        continue

                    # Record vector indices (to batch-project later)
                    base = len(all_vectors)

                    all_vectors.append(x[ay, ax])  # anchor
                    anchor_vi = base

                    pos_vis = []
                    for py, px in pos_coords:
                        all_vectors.append(x[py, px])
                        pos_vis.append(len(all_vectors) - 1)

                    neg_vis = []
                    for ny, nx in neg_coords:
                        all_vectors.append(x[ny, nx])
                        neg_vis.append(len(all_vectors) - 1)

                    all_triplets.append((anchor_vi, pos_vis, neg_vis))

    if len(all_triplets) < 2 or len(all_vectors) < 4:
        return total_loss

    # Project all vectors in one batch-pass
    vecs = torch.stack(all_vectors).to(device)  # (V, D_in)
    z = model(vecs)  # (V, embed_dim)

    # Compute loss per triplet
    losses = []
    for anchor_vi, pos_vis, neg_vis in all_triplets:
        z_a = z[anchor_vi]                           # (E,)
        z_pos = z[pos_vis]                           # (K, E)
        z_neg = z[neg_vis]                           # (M, E)

        sim_pos = (z_a * z_pos).sum(dim=-1) / tau   # (K,)
        sim_neg = (z_a * z_neg).sum(dim=-1) / tau   # (M,)

        # Multi-positive InfoNCE: for each positive, compute its loss
        # denominator = all positives + all negatives (excluding current positive)
        all_sim = torch.cat([sim_pos, sim_neg])      # (K+M,)
        log_denom = torch.logsumexp(all_sim, dim=0)

        loss_k = -(sim_pos - log_denom).mean()       # average over K positives
        losses.append(loss_k)

    if len(losses) < 2:
        return total_loss

    total_loss = torch.stack(losses).mean()
    return total_loss


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_pq(
    model: PICLProjectionHead,
    val_files: List,
    eval_hw: Tuple[int, int],
    depth_weight: float = 2.0,
    pos_weight: float = 0.5,
    hdbscan_min_cluster: int = 5,
    hdbscan_min_samples: int = 3,
    min_area: int = 1000,
) -> dict:
    """Evaluate PQ on a val subset using PICL + HDBSCAN."""
    from instance_methods.picl_embed import picl_instances

    H, W = eval_hw
    tp_acc = np.zeros(NUM_CLASSES)
    fp_acc = np.zeros(NUM_CLASSES)
    fn_acc = np.zeros(NUM_CLASSES)
    iou_acc = np.zeros(NUM_CLASSES)
    total_instances = 0
    n_images = 0

    for sem_path, depth_path, gt_label_path, gt_inst_path, feat_path in tqdm(
        val_files, desc="Val PQ", leave=False
    ):
        pred_sem = np.array(PILImage.open(sem_path))
        if pred_sem.shape != (H, W):
            pred_sem = resize_nearest(pred_sem, eval_hw)

        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = np.array(
                PILImage.fromarray(depth).resize((W, H), PILImage.BILINEAR)
            )

        features = load_features(str(feat_path)) if feat_path else None

        try:
            instances = picl_instances(
                pred_sem, depth,
                thing_ids=THING_IDS,
                features=features,
                model=model,
                depth_weight=depth_weight,
                pos_weight=pos_weight,
                hdbscan_min_cluster=hdbscan_min_cluster,
                hdbscan_min_samples=hdbscan_min_samples,
                min_area=min_area,
                dilation_iters=3,
            )
        except Exception:
            instances = []

        gt_raw = np.array(PILImage.open(gt_label_path))
        gt_sem = remap_gt_to_trainids(gt_raw)
        if gt_sem.shape != (H, W):
            gt_sem = resize_nearest(gt_sem, eval_hw)

        gt_inst_map = np.array(PILImage.open(gt_inst_path), dtype=np.int32)
        if gt_inst_map.shape != (H, W):
            gt_inst_map = np.array(
                PILImage.fromarray(gt_inst_map).resize((W, H), PILImage.NEAREST)
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


# ─── Hyperparameter Configs ───────────────────────────────────────────────────

PICL_CONFIGS = {
    1: {"tau": 0.07, "K_pos": 4, "n_hard_neg": 16, "n_easy_neg": 16,
        "depth_weight": 2.0, "pos_weight": 0.5},   # balanced
    2: {"tau": 0.07, "K_pos": 4, "n_hard_neg": 32, "n_easy_neg": 8,
        "depth_weight": 2.0, "pos_weight": 0.5},   # more hard negatives
    3: {"tau": 0.10, "K_pos": 8, "n_hard_neg": 8,  "n_easy_neg": 32,
        "depth_weight": 2.0, "pos_weight": 0.5},   # more easy negatives, higher tau
    4: {"tau": 0.07, "K_pos": 4, "n_hard_neg": 16, "n_easy_neg": 0,
        "depth_weight": 2.0, "pos_weight": 0.5},   # hard negatives only
}


# ─── Training Loop ───────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Train PICL projection head."""
    cfg = PICL_CONFIGS.get(args.picl_config, PICL_CONFIGS[1])
    tau = cfg["tau"]
    K_pos = cfg["K_pos"]
    n_hard_neg = cfg["n_hard_neg"]
    n_easy_neg = cfg["n_easy_neg"]
    depth_weight = cfg["depth_weight"]
    pos_weight = cfg["pos_weight"]

    logger.info(f"PICL config {args.picl_config}: {cfg}")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Dataset
    dataset = PICLDataset(
        args.cityscapes_root,
        split="train",
        instance_subdir=args.instance_subdir,
        semantic_subdir=args.semantic_subdir,
        depth_subdir=args.depth_subdir,
        feature_subdir=args.feature_subdir,
        depth_weight=depth_weight,
        pos_weight=pos_weight,
        min_patches_per_instance=args.min_patches,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=picl_collate,
        drop_last=False,
    )

    # Model
    model = PICLProjectionHead(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PICLProjectionHead: {n_params:,} params")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Val files (use same semantic as training, no centroids needed)
    val_files = discover_files(
        Path(args.cityscapes_root), "val",
        args.semantic_subdir, args.depth_subdir, args.feature_subdir,
    )[:args.val_images]
    logger.info(f"Validation: {len(val_files)} images")

    eval_hw = (512, 1024)

    # Checkpoint dir
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        "picl_config": args.picl_config,
        "tau": tau, "K_pos": K_pos,
        "n_hard_neg": n_hard_neg, "n_easy_neg": n_easy_neg,
        "depth_weight": depth_weight, "pos_weight": pos_weight,
        "input_dim": INPUT_DIM, "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "instance_subdir": args.instance_subdir,
        "pair_mining": "instance_mask",  # distinguishes from old depth-based approach
    }

    best_pq_things = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for x, sem, inst_masks, inst_class_ids in pbar:
            x = x.to(device)
            sem = sem.to(device)

            optimizer.zero_grad()
            loss = mine_and_compute_loss_picl(
                model, x, sem, inst_masks, inst_class_ids,
                tau=tau, K_pos=K_pos,
                n_hard_neg=n_hard_neg, n_easy_neg=n_easy_neg,
                max_anchors_per_inst=args.max_anchors_per_inst,
            )

            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            pbar.set_postfix(loss=f"{avg:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.1e}")

        scheduler.step()
        dt = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)

        # Embedding health check
        model.eval()
        with torch.no_grad():
            sample_x = x[:1].reshape(-1, x.shape[-1])[:128].to(device)
            sample_z = model(sample_x)
            embed_std = sample_z.std(dim=0).mean().item()

        # PQ validation
        val_metrics = validate_pq(
            model, val_files, eval_hw,
            depth_weight=depth_weight, pos_weight=pos_weight,
        )
        pq_th = val_metrics["PQ_things"]
        pq = val_metrics["PQ"]
        pq_st = val_metrics["PQ_stuff"]

        logger.info(
            f"Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f} "
            f"embed_std={embed_std:.4f} "
            f"PQ={pq:.2f} PQ_st={pq_st:.2f} PQ_th={pq_th:.2f} "
            f"inst/img={val_metrics['avg_instances']:.1f} "
            f"time={dt:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "embed_std": embed_std,
            "PQ_things": pq_th,
            "PQ": pq,
            "val_metrics": val_metrics,
            "config": config_record,
        }

        if pq_th > best_pq_things:
            best_pq_things = pq_th
            torch.save(ckpt, ckpt_dir / "best.pth")
            logger.info(f"  New best PQ_things={pq_th:.2f} (epoch {epoch})")

        torch.save(ckpt, ckpt_dir / "latest.pth")

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    logger.info(f"Done. Best PQ_things={best_pq_things:.2f}")
    logger.info(f"Checkpoints: {ckpt_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PICL: Pseudo-Instance Contrastive Learning"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--picl_config", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Hyperparameter preset (see PICL_CONFIGS)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--max_anchors_per_inst", type=int, default=8)
    parser.add_argument("--min_patches", type=int, default=3,
                        help="Min patches per instance to include in training")
    parser.add_argument("--val_images", type=int, default=50,
                        help="Val images for PQ eval per epoch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--instance_subdir", type=str,
                        default="pseudo_instance_spidepth")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_mapped_k80")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--feature_subdir", type=str, default="dinov2_features")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/picl/round1")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args)


if __name__ == "__main__":
    main()
