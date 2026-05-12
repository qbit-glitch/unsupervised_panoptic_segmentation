#!/usr/bin/env python3
"""Train Depth-Conditioned Slot Attention Decoder for instance segmentation.

3-phase hybrid bootstrap curriculum:
  Phase 1 (recon):     Pure feature reconstruction (MSE on DINOv3 features)
  Phase 2 (bootstrap): + pseudo-label matching loss (slot ↔ instance assignment)
  Phase 3 (self-train): Generate masks → filter → retrain (iterative)

Usage:
    python mbps_pytorch/train_slot_decoder.py \
        --feature_dir /path/to/cityscapes/dinov3_features_vitl16/train \
        --val_feature_dir /path/to/cityscapes/dinov3_features_vitl16/val \
        --depth_dir /path/to/cityscapes/depth_depthpro/train \
        --val_depth_dir /path/to/cityscapes/depth_depthpro/val \
        --instance_dir /path/to/cityscapes/pseudo_instance_depthpro/train \
        --output_dir checkpoints/slot_decoder/ \
        --device mps --epochs 90
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow direct invocation: python mbps_pytorch/train_slot_decoder.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from mbps_pytorch.models.slot_decoder import DepthSlotDecoder, DepthSlotDecoderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GRID_H, GRID_W = 32, 64
N_PATCHES = GRID_H * GRID_W
FEAT_DIM = 1024


class CityscapesSlotDataset(Dataset):
    """Dataset loading DINOv3 features + DepthPro depth + instance pseudo-labels."""

    def __init__(
        self,
        feature_dir: str,
        depth_dir: str,
        instance_dir: str | None = None,
        limit: int | None = None,
    ):
        self.feature_dir = Path(feature_dir)
        self.depth_dir = Path(depth_dir)
        self.instance_dir = Path(instance_dir) if instance_dir else None

        # Collect feature files
        self.feature_files = sorted(self.feature_dir.rglob("*.npy"))
        if limit:
            self.feature_files = self.feature_files[:limit]

        logger.info(f"Dataset: {len(self.feature_files)} samples from {feature_dir}")

    def __len__(self) -> int:
        return len(self.feature_files)

    def _get_depth_path(self, feat_path: Path) -> Path:
        """Map feature path to corresponding depth path."""
        # Feature: .../train/aachen/aachen_000000_000019_leftImg8bit.npy
        # Depth:   .../train/aachen/aachen_000000_000019.npy
        city = feat_path.parent.name
        stem = feat_path.stem.replace("_leftImg8bit", "")
        return self.depth_dir / city / f"{stem}.npy"

    def _get_instance_path(self, feat_path: Path) -> Path | None:
        """Map feature path to corresponding instance pseudo-label path."""
        if self.instance_dir is None:
            return None
        city = feat_path.parent.name
        stem = feat_path.stem.replace("_leftImg8bit", "")
        path = self.instance_dir / city / f"{stem}.npz"
        return path if path.exists() else None

    def __getitem__(self, idx: int) -> dict:
        feat_path = self.feature_files[idx]

        # Load DINOv3 features: (N_PATCHES, FEAT_DIM)
        features = np.load(feat_path).astype(np.float32)
        features = torch.from_numpy(features)

        # Load depth map: (512, 1024)
        depth_path = self._get_depth_path(feat_path)
        depth = np.load(depth_path).astype(np.float32)
        depth = torch.from_numpy(depth)

        sample = {
            "features": features,
            "depth": depth,
            "name": feat_path.stem,
        }

        # Load instance pseudo-labels if available
        inst_path = self._get_instance_path(feat_path)
        if inst_path is not None:
            inst_data = np.load(inst_path, allow_pickle=True)
            masks = inst_data["masks"]  # (num_instances, H*W)
            num_valid = int(inst_data["num_valid"])
            # Downsample masks to patch grid
            patch_masks = self._downsample_masks(masks[:num_valid])
            sample["instance_masks"] = patch_masks  # (num_valid, N_PATCHES)
            sample["num_instances"] = num_valid

        return sample

    def _downsample_masks(self, masks: np.ndarray) -> torch.Tensor:
        """Downsample binary masks from 512×1024 to 32×64 patch grid.

        Args:
            masks: (N_inst, H*W) flattened binary masks at full resolution.

        Returns:
            (N_inst, N_PATCHES) soft masks at patch resolution.
        """
        if masks.shape[0] == 0:
            return torch.zeros(0, N_PATCHES, dtype=torch.float32)

        H, W = 512, 1024
        masks_2d = masks.reshape(-1, H, W).astype(np.float32)
        masks_tensor = torch.from_numpy(masks_2d).unsqueeze(1)  # (N, 1, H, W)

        # Average pooling to patch grid
        patch_masks = F.adaptive_avg_pool2d(masks_tensor, (GRID_H, GRID_W))
        patch_masks = patch_masks.squeeze(1).reshape(-1, N_PATCHES)  # (N, N_PATCHES)

        return patch_masks


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-size instance masks."""
    features = torch.stack([s["features"] for s in batch])
    depth = torch.stack([s["depth"] for s in batch])

    result = {"features": features, "depth": depth}

    # Instance masks: pad to max instances in batch
    if "instance_masks" in batch[0]:
        max_inst = max(s.get("num_instances", 0) for s in batch)
        if max_inst > 0:
            padded_masks = torch.zeros(len(batch), max_inst, N_PATCHES)
            num_instances = []
            for i, s in enumerate(batch):
                n = s.get("num_instances", 0)
                if n > 0:
                    padded_masks[i, :n] = s["instance_masks"]
                num_instances.append(n)
            result["instance_masks"] = padded_masks
            result["num_instances"] = torch.tensor(num_instances)

    return result


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss on DINOv3 features."""
    return F.mse_loss(recon, target)


def pseudo_label_loss(
    masks: torch.Tensor,
    instance_masks: torch.Tensor,
    num_instances: torch.Tensor,
) -> torch.Tensor:
    """Matching loss between predicted slot masks and pseudo-label instance masks.

    Uses Hungarian-free approach: for each pseudo-label instance, find the
    best-matching slot (highest IoU) and encourage that assignment.

    Args:
        masks: (B, K, N) predicted slot masks (soft, sum-to-1 over K).
        instance_masks: (B, max_inst, N) pseudo-label masks (soft, from downsampling).
        num_instances: (B,) number of valid instances per sample.

    Returns:
        Scalar loss.
    """
    B, K, N = masks.shape
    max_inst = instance_masks.shape[1]

    total_loss = 0.0
    count = 0

    for b in range(B):
        n_inst = num_instances[b].item()
        if n_inst == 0:
            continue

        pred = masks[b]  # (K, N)
        gt = instance_masks[b, :n_inst]  # (n_inst, N)

        # Compute IoU-like overlap: (n_inst, K)
        # For soft masks: overlap = sum(min(pred, gt)) / sum(max(pred, gt))
        gt_expanded = gt.unsqueeze(1).expand(-1, K, -1)  # (n_inst, K, N)
        pred_expanded = pred.unsqueeze(0).expand(n_inst, -1, -1)  # (n_inst, K, N)

        intersection = torch.min(gt_expanded, pred_expanded).sum(dim=-1)  # (n_inst, K)
        union = torch.max(gt_expanded, pred_expanded).sum(dim=-1)  # (n_inst, K)
        iou = intersection / (union + 1e-8)  # (n_inst, K)

        # Best slot for each GT instance
        best_slot_idx = iou.argmax(dim=1)  # (n_inst,)

        # Cross-entropy: encourage matched slot to cover GT instance
        for i in range(n_inst):
            slot_mask = pred[best_slot_idx[i]]  # (N,)
            target_mask = gt[i]  # (N,)
            # Binary CE between matched slot mask and GT mask
            loss_i = F.binary_cross_entropy(
                slot_mask.clamp(1e-6, 1 - 1e-6),
                (target_mask > 0.5).float(),
                reduction="mean",
            )
            total_loss = total_loss + loss_i
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=masks.device, requires_grad=True)
    return total_loss / count


def depth_consistency_loss(
    masks: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Encourage each slot to attend to patches at similar depths.

    Penalizes high depth variance within each slot's attention distribution.

    Args:
        masks: (B, K, N) soft slot masks.
        depth: (B, H, W) full-resolution depth map.

    Returns:
        Scalar loss (mean intra-slot depth variance).
    """
    B, K, N = masks.shape

    # Downsample depth to patch grid
    depth_patches = F.adaptive_avg_pool2d(
        depth.unsqueeze(1), (GRID_H, GRID_W)
    ).reshape(B, N)  # (B, N)

    # Weighted depth mean per slot
    depth_mean = torch.einsum("bkn,bn->bk", masks, depth_patches)  # (B, K)
    # Normalize by mask sum
    mask_sum = masks.sum(dim=-1).clamp(min=1e-8)  # (B, K)
    depth_mean = depth_mean / mask_sum  # (B, K)

    # Weighted depth variance per slot
    depth_diff_sq = (depth_patches.unsqueeze(1) - depth_mean.unsqueeze(-1)) ** 2  # (B, K, N)
    depth_var = (masks * depth_diff_sq).sum(dim=-1) / mask_sum  # (B, K)

    return depth_var.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    phase: int,
    lambda_recon: float = 1.0,
    lambda_pl: float = 0.5,
    lambda_depth: float = 0.1,
) -> dict:
    """Train one epoch.

    Args:
        phase: 1=recon only, 2=+pseudo-labels, 3=+depth consistency.
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_pl = 0.0
    total_depth = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [P{phase}]", leave=False)
    for batch in pbar:
        features = batch["features"].to(device)
        depth = batch["depth"].to(device)

        optimizer.zero_grad()

        out = model(features, depth)
        recon = out["recon"]
        masks = out["masks"]

        # Phase 1: reconstruction only
        loss_recon = reconstruction_loss(recon, features)
        loss = lambda_recon * loss_recon

        # Phase 2+: add pseudo-label matching
        loss_pl = torch.tensor(0.0, device=device)
        if phase >= 2 and "instance_masks" in batch:
            inst_masks = batch["instance_masks"].to(device)
            num_inst = batch["num_instances"].to(device)
            loss_pl = pseudo_label_loss(masks, inst_masks, num_inst)
            loss = loss + lambda_pl * loss_pl

        # Phase 3: add depth consistency
        loss_depth = torch.tensor(0.0, device=device)
        if phase >= 3:
            loss_depth = depth_consistency_loss(masks, depth)
            loss = loss + lambda_depth * loss_depth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_pl += loss_pl.item()
        total_depth += loss_depth.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss/n_batches:.4f}",
            recon=f"{total_recon/n_batches:.4f}",
            pl=f"{total_pl/n_batches:.4f}",
        )

    pbar.close()
    return {
        "loss": total_loss / n_batches,
        "loss_recon": total_recon / n_batches,
        "loss_pl": total_pl / n_batches,
        "loss_depth": total_depth / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate reconstruction quality."""
    model.eval()
    total_recon = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch in pbar:
        features = batch["features"].to(device)
        depth = batch["depth"].to(device)

        out = model(features, depth)
        loss_recon = reconstruction_loss(out["recon"], features)

        total_recon += loss_recon.item()
        n_batches += 1
        pbar.set_postfix(val_recon=f"{total_recon/n_batches:.4f}")

    pbar.close()
    return {"val_recon": total_recon / n_batches}


def get_phase(epoch: int, phase1_end: int = 30, phase2_end: int = 60) -> int:
    """Determine training phase from epoch number."""
    if epoch <= phase1_end:
        return 1
    elif epoch <= phase2_end:
        return 2
    else:
        return 3


def main():
    parser = argparse.ArgumentParser(description="Train Depth-Conditioned Slot Decoder")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--val_feature_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--val_depth_dir", type=str, required=True)
    parser.add_argument("--instance_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/slot_decoder")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--slot_iters", type=int, default=5)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_pl", type=float, default=0.5)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument("--phase1_end", type=int, default=30)
    parser.add_argument("--phase2_end", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Model config
    cfg = DepthSlotDecoderConfig(
        feat_dim=FEAT_DIM,
        slot_dim=args.slot_dim,
        num_slots=args.num_slots,
        slot_iters=args.slot_iters,
    )
    model = DepthSlotDecoder(cfg).to(device)
    n_params = model.count_parameters()
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args) | {"model_params": n_params}, f, indent=2)

    # Datasets
    train_dataset = CityscapesSlotDataset(
        feature_dir=args.feature_dir,
        depth_dir=args.depth_dir,
        instance_dir=args.instance_dir,
        limit=args.limit,
    )
    val_dataset = CityscapesSlotDataset(
        feature_dir=args.val_feature_dir,
        depth_dir=args.val_depth_dir,
        instance_dir=None,
        limit=args.limit,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    # Training loop
    logger.info(f"Training {args.epochs} epochs: Phase1=[1,{args.phase1_end}], "
                f"Phase2=[{args.phase1_end+1},{args.phase2_end}], "
                f"Phase3=[{args.phase2_end+1},{args.epochs}]")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    prev_phase = get_phase(start_epoch, args.phase1_end, args.phase2_end)
    best_val_per_phase = {1: float("inf"), 2: float("inf"), 3: float("inf")}

    def _save_ckpt(path: str, epoch: int, extra: dict | None = None) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": vars(args),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    for epoch in range(start_epoch, args.epochs + 1):
        phase = get_phase(epoch, args.phase1_end, args.phase2_end)

        # Save checkpoint at phase transitions
        if phase != prev_phase:
            trans_path = os.path.join(args.output_dir, f"phase{prev_phase}_final.pth")
            _save_ckpt(trans_path, epoch - 1)
            logger.info(f"═══ Phase {prev_phase} → {phase} transition. Saved {trans_path}")
            prev_phase = phase

        t0 = time.time()

        metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, phase,
            lambda_recon=args.lambda_recon,
            lambda_pl=args.lambda_pl,
            lambda_depth=args.lambda_depth,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} [Phase {phase}] "
            f"loss={metrics['loss']:.4f} recon={metrics['loss_recon']:.4f} "
            f"pl={metrics['loss_pl']:.4f} depth={metrics['loss_depth']:.4f} "
            f"lr={lr:.2e} ({elapsed:.1f}s)"
        )

        # Validation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, device)
            logger.info(f"  Val: recon={val_metrics['val_recon']:.4f}")

            # Save best overall
            if val_metrics["val_recon"] < best_val_loss:
                best_val_loss = val_metrics["val_recon"]
                _save_ckpt(os.path.join(args.output_dir, "best.pth"), epoch)
                logger.info(f"  ★ New best val_recon={best_val_loss:.4f} → saved best.pth")

            # Save best per-phase
            if val_metrics["val_recon"] < best_val_per_phase[phase]:
                best_val_per_phase[phase] = val_metrics["val_recon"]
                phase_best_path = os.path.join(args.output_dir, f"best_phase{phase}.pth")
                _save_ckpt(phase_best_path, epoch)
                logger.info(f"  ★ New best Phase {phase} val_recon={val_metrics['val_recon']:.4f}")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            _save_ckpt(os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth"), epoch)

    # Final save
    final_path = os.path.join(args.output_dir, "final.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": vars(args),
    }, final_path)
    logger.info(f"Training complete. Best val_recon={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
