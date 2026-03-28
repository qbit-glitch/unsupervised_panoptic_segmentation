#!/usr/bin/env python3
"""CutS3D CAD Training — PyTorch GPU.

Trains a Cascade Mask R-CNN (class-agnostic detector) on CutS3D pseudo-masks
following Sick et al., ICCV 2025. Supports multi-GPU via DataParallel.

Components:
  - DINO ViT-S/8 backbone (frozen, from torch.hub)
  - Cascade Mask R-CNN (InstanceHead) with 3-stage refinement
  - Spatial Confidence Soft Target Loss (Eq. 6)
  - DropLoss for unmatched proposals
  - Hungarian matching for pred-target assignment
  - Self-training (3 rounds of pseudo-label regeneration)

Usage:
    # Train CAD on pseudo-masks (30 epochs)
    python scripts/train_cuts3d_pytorch.py \
        --image-dir datasets/coco/train2017 \
        --mask-dir data/pseudo_masks_coco \
        --epochs 30 --batch-size 16

    # With self-training (3 rounds × 10 epochs each)
    python scripts/train_cuts3d_pytorch.py \
        --image-dir datasets/coco/train2017 \
        --mask-dir data/pseudo_masks_coco \
        --epochs 30 --batch-size 16 \
        --self-train --self-train-rounds 3 --self-train-epochs 10

    # Resume from checkpoint
    python scripts/train_cuts3d_pytorch.py \
        --image-dir datasets/coco/train2017 \
        --mask-dir data/pseudo_masks_coco \
        --epochs 30 --checkpoint checkpoints/cuts3d/cuts3d_cad_epoch030.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps_pytorch.models.instance.cascade_mask_rcnn import InstanceHead
from mbps_pytorch.models.instance.instance_loss import (
    drop_loss,
    spatial_confidence_soft_target_loss,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class COCOPseudoMaskDataset(Dataset):
    """COCO images paired with CutS3D pseudo-masks.

    On first run, builds an image_id mapping from mask .npz files and caches
    it to a JSON file for instant loading on subsequent runs.

    Args:
        image_dir: Path to COCO train2017 images.
        mask_dir: Path to CutS3D pseudo-mask .npz files.
        image_size: Target image size (square).
    """

    def __init__(self, image_dir: str, mask_dir: str, image_size: int = 480,
                 max_instances: int = 3):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.max_instances = max_instances
        self.num_patches = (image_size // 8) ** 2

        # Build or load cached mapping: mask_index -> image_id
        cache_path = self.mask_dir / "image_id_mapping.json"
        if cache_path.exists():
            print(f"  Loading cached mapping: {cache_path}")
            with open(cache_path) as f:
                mapping = json.load(f)
        else:
            print(f"  Building mask -> image_id mapping (first run, may take a few minutes)...")
            mapping = {}
            mask_files = sorted(self.mask_dir.glob("masks_*.npz"))
            for i, mf in enumerate(mask_files):
                data = np.load(mf, allow_pickle=True)
                mapping[mf.name] = str(data["image_id"])
                if (i + 1) % 10000 == 0:
                    print(f"    {i + 1}/{len(mask_files)}")
            with open(cache_path, "w") as f:
                json.dump(mapping, f)
            print(f"  Saved mapping to {cache_path}")

        # Build item list: (image_path, mask_path) pairs
        self.items = []
        missing = 0
        for mask_name in sorted(mapping.keys()):
            image_id = mapping[mask_name]
            image_path = self.image_dir / f"{image_id}.jpg"
            mask_path = self.mask_dir / mask_name
            if image_path.exists():
                self.items.append((str(image_path), str(mask_path)))
            else:
                missing += 1

        print(f"  Dataset: {len(self.items)} images ({missing} missing)")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, mask_path = self.items[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Load pseudo-masks
        data = np.load(mask_path, allow_pickle=True)
        raw_masks = data["masks"].astype(np.float32)            # (n, K)
        raw_sc = data["spatial_confidence"].astype(np.float32)  # (n, K)
        raw_scores = data["scores"].astype(np.float32)          # (n,)
        num_valid = int(data["num_valid"])

        # Pad to fixed max_instances (some files have fewer masks)
        M = self.max_instances
        K = self.num_patches
        n = raw_masks.shape[0]
        masks = np.zeros((M, K), dtype=np.float32)
        sc = np.ones((M, K), dtype=np.float32)
        scores = np.zeros((M,), dtype=np.float32)
        n_use = min(n, M)
        masks[:n_use] = raw_masks[:n_use]
        sc[:n_use] = raw_sc[:n_use]
        scores[:n_use] = raw_scores[:n_use]

        return {
            "image": image,
            "masks": torch.from_numpy(masks),
            "spatial_confidence": torch.from_numpy(sc),
            "scores": torch.from_numpy(scores),
            "num_valid": min(num_valid, M),
        }


# ---------------------------------------------------------------------------
# Model wrapper (backbone + CAD)
# ---------------------------------------------------------------------------


class CutS3DModel(nn.Module):
    """Combined DINO backbone + Cascade Mask R-CNN CAD.

    Wraps frozen DINO ViT-S/8 and trainable InstanceHead into a single
    module for DataParallel compatibility.
    """

    def __init__(self, backbone: nn.Module, cad: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.cad = cad

    def forward(self, images: torch.Tensor):
        """Extract features and predict instance masks.

        Args:
            images: (B, 3, H, W) normalized images.

        Returns:
            pred_masks: (B, M, N) mask logits.
            pred_scores: (B, M) confidence scores.
        """
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(images, n=1)[0][:, 1:]
        features = features.detach()
        pred_masks, pred_scores = self.cad(features, deterministic=not self.training)
        return pred_masks, pred_scores


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_predictions_to_targets(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    num_valid: torch.Tensor,
    iou_threshold: float = 0.1,
) -> torch.Tensor:
    """Hungarian matching between predicted and target masks.

    Args:
        pred_masks: (B, M, N) predicted mask logits.
        target_masks: (B, M, N) target binary masks.
        num_valid: (B,) number of valid target masks per image.
        iou_threshold: Minimum IoU to count as matched.

    Returns:
        matched: (B, M) boolean tensor — True for matched predictions.
        reordered_targets: (B, M, N) target masks reordered to match predictions.
        reordered_sc: None (caller handles SC reordering separately).
    """
    B, M, N = pred_masks.shape
    device = pred_masks.device
    pred_probs = torch.sigmoid(pred_masks.detach())

    matched = torch.zeros(B, M, dtype=torch.bool, device=device)
    perm = torch.zeros(B, M, dtype=torch.long, device=device)

    for b in range(B):
        nv = int(num_valid[b].item())
        if nv == 0:
            continue

        # IoU cost matrix: (M_pred, nv_target)
        pred_b = pred_probs[b]              # (M, N)
        target_b = target_masks[b, :nv]     # (nv, N)

        intersection = torch.mm(pred_b, target_b.T)       # (M, nv)
        pred_sum = pred_b.sum(dim=-1, keepdim=True)        # (M, 1)
        target_sum = target_b.sum(dim=-1).unsqueeze(0)     # (1, nv)
        union = pred_sum + target_sum - intersection + 1e-8
        iou = intersection / union                         # (M, nv)

        # Hungarian matching (maximize IoU = minimize -IoU)
        cost = -iou.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        for ri, ci in zip(row_ind, col_ind):
            if iou[ri, ci] > iou_threshold:
                matched[b, ri] = True
                perm[b, ri] = ci

    return matched, perm


# ---------------------------------------------------------------------------
# LR Schedule with warmup
# ---------------------------------------------------------------------------


class WarmupCosineSchedule:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            factor = self.step_count / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            factor = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + np.cos(np.pi * progress)
            )
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    max_instances: int = 3,
    lambda_drop: float = 0.5,
    grad_clip: float = 1.0,
    log_every: int = 50,
):
    """Train one epoch.

    Returns:
        Dict of average losses for the epoch.
    """
    model.train()

    epoch_losses = []
    t_start = time.time()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        target_masks = batch["masks"].to(device)         # (B, M, K)
        target_sc = batch["spatial_confidence"].to(device)  # (B, M, K)
        num_valid = batch["num_valid"].to(device)

        # Forward
        pred_masks, pred_scores = model(images)

        # Match predictions to targets (Hungarian)
        matched, perm = match_predictions_to_targets(
            pred_masks, target_masks, num_valid
        )

        # Reorder targets to align with matched predictions
        B, M, K = target_masks.shape
        reordered_masks = torch.zeros_like(target_masks)
        reordered_sc = torch.ones_like(target_sc)
        for b in range(B):
            nv = int(num_valid[b].item())
            for m in range(M):
                if matched[b, m]:
                    ti = perm[b, m].item()
                    reordered_masks[b, m] = target_masks[b, ti]
                    reordered_sc[b, m] = target_sc[b, ti]

        # SC Soft Target Loss (Eq. 6)
        l_mask = spatial_confidence_soft_target_loss(
            pred_masks, reordered_masks, reordered_sc
        )

        # Drop loss for unmatched predictions
        l_drop = drop_loss(pred_masks, pred_scores, matched)

        total_loss = l_mask + lambda_drop * l_drop

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        )
        optimizer.step()
        scheduler.step()

        # Track
        losses = {
            "total": total_loss.item(),
            "sc_mask": l_mask.item(),
            "drop": l_drop.item(),
            "n_matched": matched.float().sum().item() / B,
        }
        epoch_losses.append(losses)

        if (step + 1) % log_every == 0:
            recent = epoch_losses[-log_every:]
            avg = {k: np.mean([l[k] for l in recent]) for k in losses}
            elapsed = time.time() - t_start
            lr = scheduler.get_lr()
            print(
                f"  [{epoch}:{step + 1:>5d}] L={avg['total']:.4f} "
                f"(mask={avg['sc_mask']:.4f} drop={avg['drop']:.4f}) "
                f"matched={avg['n_matched']:.2f} lr={lr:.2e} [{elapsed:.0f}s]"
            )
            sys.stdout.flush()

    # Epoch averages
    avg = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
    return avg


# ---------------------------------------------------------------------------
# Self-Training: generate labels with CAD
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_labels_with_cad(
    model: nn.Module,
    image_dir: str,
    output_dir: str,
    image_size: int,
    max_instances: int,
    device: torch.device,
    batch_size: int = 16,
):
    """Generate new pseudo-labels using the trained CAD.

    Runs inference on all images and saves predictions as .npz pseudo-masks.
    """
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_files = sorted(Path(image_dir).glob("*.jpg"))
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    t_start = time.time()

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]

        images = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            images.append(transform(img))
        images = torch.stack(images).to(device)

        # Forward (handles DataParallel)
        pred_masks, pred_scores = model(images)
        pred_probs = torch.sigmoid(pred_masks).cpu().numpy()
        pred_scores_np = pred_scores.cpu().numpy()

        K = pred_probs.shape[2]
        for b, f in enumerate(batch_files):
            image_id = f.stem
            n_det = max(1, int(np.sum(pred_scores_np[b] > 0.3)))
            n_use = min(n_det, max_instances)

            np.savez_compressed(
                os.path.join(output_dir, f"masks_{total:08d}.npz"),
                masks=pred_probs[b, :n_use],
                spatial_confidence=np.ones((n_use, K), dtype=np.float32),
                scores=pred_scores_np[b, :n_use],
                num_valid=n_use,
                image_id=image_id,
            )
            total += 1

        if total % 5000 < batch_size:
            elapsed = time.time() - t_start
            rate = total / max(elapsed, 1)
            print(f"    [{total}/{len(image_files)}] {rate:.1f} img/s")
            sys.stdout.flush()

    model.train()
    elapsed = time.time() - t_start
    print(f"  Generated {total} labels in {elapsed:.0f}s -> {output_dir}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(model, optimizer, scheduler, epoch, losses, path):
    """Save training checkpoint."""
    # Unwrap DataParallel
    cad_state = (
        model.module.cad.state_dict()
        if hasattr(model, "module")
        else model.cad.state_dict()
    )
    torch.save(
        {
            "epoch": epoch,
            "cad_state_dict": cad_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step": scheduler.step_count,
            "losses": losses,
        },
        path,
    )
    print(f"  Saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load training checkpoint. Returns start epoch."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Load CAD weights (handles DataParallel)
    cad = model.module.cad if hasattr(model, "module") else model.cad
    cad.load_state_dict(ckpt["cad_state_dict"])

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_step" in ckpt and scheduler is not None:
        scheduler.step_count = ckpt["scheduler_step"]

    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"  Loaded checkpoint: {path} (epoch {start_epoch - 1})")
    return start_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CutS3D CAD Training (PyTorch GPU)")
    # Data
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Path to COCO train2017 images")
    parser.add_argument("--mask-dir", type=str, required=True,
                        help="Path to CutS3D pseudo-mask .npz files")
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-drop", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--max-instances", type=int, default=3)
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/cuts3d")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from")
    parser.add_argument("--save-every", type=int, default=5)
    # Self-training
    parser.add_argument("--self-train", action="store_true")
    parser.add_argument("--self-train-rounds", type=int, default=3)
    parser.add_argument("--self-train-epochs", type=int, default=10)
    # Misc
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    print("=" * 80)
    print("  CutS3D CAD Training — PyTorch GPU")
    print("=" * 80)
    print(f"  Device: {device}, GPUs: {num_gpus}")
    if torch.cuda.is_available():
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {name} ({mem:.1f} GB)")
    print(f"  Image dir: {args.image_dir}")
    print(f"  Mask dir: {args.mask_dir}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}, Warmup: {args.warmup_steps} steps")
    print(f"  Image size: {args.image_size}, Max instances: {args.max_instances}")
    print()

    # Dataset
    dataset = COCOPseudoMaskDataset(args.image_dir, args.mask_dir, args.image_size,
                                     max_instances=args.max_instances)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Models
    print("Loading DINO ViT-S/8 backbone (torch.hub, frozen)...")
    backbone = torch.hub.load(
        "facebookresearch/dino:main", "dino_vits8",
        pretrained=True, trust_repo=True,
    )
    backbone.eval()

    K = (args.image_size // 8) ** 2  # 3600 for 480x480
    cad = InstanceHead(
        max_instances=args.max_instances,
        input_dim=384,
        hidden_dim=256,
        num_patches=K,
        num_refinement_stages=3,
        num_classes=1,
    )

    model = CutS3DModel(backbone, cad)
    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in cad.parameters())
    trainable_params = sum(p.numel() for p in cad.parameters() if p.requires_grad)
    print(f"  CAD parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"  Patches: {K} ({args.image_size // 8}x{args.image_size // 8})")

    # Optimizer (only CAD parameters)
    optimizer = torch.optim.AdamW(
        cad.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    # Resume
    start_epoch = 1
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)

    # Training
    print(f"\n{'=' * 80}")
    print(f"  Phase 1: CAD Training ({start_epoch}-{args.epochs})")
    print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"{'=' * 80}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()

        avg = train_one_epoch(
            model, optimizer, scheduler, dataloader, device, epoch,
            max_instances=args.max_instances,
            lambda_drop=args.lambda_drop,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )

        elapsed = time.time() - t_epoch
        lr = scheduler.get_lr()
        print(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"L={avg['total']:.5f} (mask={avg['sc_mask']:.5f} "
            f"drop={avg['drop']:.5f}) | "
            f"matched={avg['n_matched']:.2f} | "
            f"lr={lr:.2e} | {elapsed:.0f}s"
        )
        sys.stdout.flush()

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"cuts3d_cad_epoch{epoch:03d}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, avg, ckpt_path)

    print(f"\nPhase 1 complete!")

    # Self-training
    if args.self_train:
        print(f"\n{'=' * 80}")
        print(f"  Phase 2: Self-Training ({args.self_train_rounds} rounds)")
        print(f"{'=' * 80}\n")

        for r in range(1, args.self_train_rounds + 1):
            print(f"\n--- Self-Training Round {r}/{args.self_train_rounds} ---")

            # Step 1: Generate new pseudo-labels with current CAD
            new_mask_dir = os.path.join(args.mask_dir, f"self_train_round{r}")
            print(f"  Generating pseudo-labels -> {new_mask_dir}")
            generate_labels_with_cad(
                model, args.image_dir, new_mask_dir,
                args.image_size, args.max_instances,
                device, args.batch_size,
            )

            # Step 2: Retrain on new pseudo-labels
            st_dataset = COCOPseudoMaskDataset(
                args.image_dir, new_mask_dir, args.image_size,
                max_instances=args.max_instances,
            )
            st_dataloader = DataLoader(
                st_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True if args.num_workers > 0 else False,
            )

            # Fresh optimizer with lower LR for self-training
            st_optimizer = torch.optim.AdamW(
                cad.parameters(),
                lr=args.lr * 0.5,
                weight_decay=args.weight_decay,
            )
            st_steps = len(st_dataloader) * args.self_train_epochs
            st_scheduler = WarmupCosineSchedule(
                st_optimizer,
                warmup_steps=100,
                total_steps=st_steps,
            )

            for epoch in range(1, args.self_train_epochs + 1):
                avg = train_one_epoch(
                    model, st_optimizer, st_scheduler, st_dataloader,
                    device, epoch,
                    max_instances=args.max_instances,
                    lambda_drop=args.lambda_drop,
                    grad_clip=args.grad_clip,
                    log_every=args.log_every,
                )
                print(
                    f"  [R{r} E{epoch}] L={avg['total']:.5f} "
                    f"matched={avg['n_matched']:.2f}"
                )

            # Save round checkpoint
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"cuts3d_cad_selftrain_round{r}.pt"
            )
            save_checkpoint(model, st_optimizer, st_scheduler, r, avg, ckpt_path)

        print(f"\nSelf-training complete!")

    print("\nDone.")


if __name__ == "__main__":
    main()
