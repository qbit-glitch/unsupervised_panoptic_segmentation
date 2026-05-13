#!/usr/bin/env python3
"""Train the Mamba2 Panoptic Refiner (M2PR).

Trains on precomputed features (DINOv2, CAUSE logits, depth, instances)
with unsupervised losses. No ground truth labels required.

Usage:
    python -m mbps_pytorch.scripts.train_refiner \
        --cityscapes_root /path/to/cityscapes \
        --epochs 30 --batch_size 8 --device mps

    # Quick smoke test:
    python -m mbps_pytorch.scripts.train_refiner \
        --cityscapes_root /path/to/cityscapes \
        --epochs 2 --batch_size 2 --limit 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.models.refiner.mamba2_panoptic_refiner import Mamba2PanopticRefiner
from mbps_pytorch.losses.refiner_loss import RefinerLoss
from mbps_pytorch.data.refiner_dataset import RefinerDataset, refiner_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train M2PR refiner")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/m2pr")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--bridge_dim", type=int, default=256)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--inst_embed_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_soft_logits", action="store_true", default=True)
    parser.add_argument("--no_soft_logits", dest="use_soft_logits", action="store_false")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit dataset size (0 = all, for debugging)")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def get_phase(epoch: int, total_epochs: int) -> str:
    """Determine training phase from epoch number."""
    third = total_epochs // 3
    if epoch < third:
        return "warmup"
    elif epoch < 2 * third:
        return "rampup"
    return "full"


def cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
) -> float:
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(
    model: Mamba2PanopticRefiner,
    criterion: RefinerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_every: int = 50,
) -> dict:
    """Train for one epoch."""
    model.train()
    phase = get_phase(epoch, total_epochs)

    running_losses = {}
    num_batches = 0
    t0 = time.time()

    for step, batch in enumerate(dataloader):
        # Move to device
        dino_feat = batch["dino_features"].to(device)
        cause_logits = batch["cause_logits"].to(device)
        geo_features = batch["geo_features"].to(device)
        inst_desc = batch["inst_descriptor"].to(device)
        depth = batch["depth"].to(device)
        inst_labels = batch["instance_labels"].to(device)

        # Forward
        outputs = model(
            dino_features=dino_feat,
            cause_logits=cause_logits,
            geo_features=geo_features,
            inst_descriptor=inst_desc,
            depth=depth,
            deterministic=False,
        )

        # Loss
        losses = criterion(
            model_outputs=outputs,
            dino_features=dino_feat.detach(),
            depth=depth,
            instance_labels=inst_labels,
            phase=phase,
        )

        total_loss = losses["total"]

        # Check for NaN
        if torch.isnan(total_loss):
            print(f"  WARNING: NaN loss at step {step}, skipping")
            optimizer.zero_grad()
            continue

        # Backward
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            if k not in running_losses:
                running_losses[k] = 0.0
            running_losses[k] += v.item()
        num_batches += 1

        # Log
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg_total = running_losses.get("total", 0) / max(num_batches, 1)
            gate = outputs["gate"].item()
            print(
                f"  [{phase}] step {step+1}/{len(dataloader)} | "
                f"loss={avg_total:.4f} | gate={gate:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

    # Average losses
    avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
    model: Mamba2PanopticRefiner,
    criterion: RefinerLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate on val set."""
    model.eval()
    running_losses = {}
    num_batches = 0

    for batch in dataloader:
        dino_feat = batch["dino_features"].to(device)
        cause_logits = batch["cause_logits"].to(device)
        geo_features = batch["geo_features"].to(device)
        inst_desc = batch["inst_descriptor"].to(device)
        depth = batch["depth"].to(device)
        inst_labels = batch["instance_labels"].to(device)

        outputs = model(
            dino_features=dino_feat,
            cause_logits=cause_logits,
            geo_features=geo_features,
            inst_descriptor=inst_desc,
            depth=depth,
            deterministic=True,
        )

        losses = criterion(
            model_outputs=outputs,
            dino_features=dino_feat,
            depth=depth,
            instance_labels=inst_labels,
            phase="full",
        )

        for k, v in losses.items():
            if k not in running_losses:
                running_losses[k] = 0.0
            running_losses[k] += v.item()
        num_batches += 1

    avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
    return avg_losses


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataset ---
    print("Loading training dataset...")
    train_ds = RefinerDataset(
        cityscapes_root=args.cityscapes_root,
        split="train",
        use_soft_logits=args.use_soft_logits,
    )
    if args.limit > 0:
        train_ds.samples = train_ds.samples[:args.limit]
    print(f"  Train samples: {len(train_ds)}")

    val_ds = RefinerDataset(
        cityscapes_root=args.cityscapes_root,
        split="val",
        use_soft_logits=args.use_soft_logits,
    )
    if args.limit > 0:
        val_ds.samples = val_ds.samples[:args.limit]
    print(f"  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=refiner_collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=refiner_collate_fn,
        pin_memory=False,
    )

    # --- Model ---
    model = Mamba2PanopticRefiner(
        bridge_dim=args.bridge_dim,
        mamba_layers=args.mamba_layers,
        inst_embed_dim=args.inst_embed_dim,
    ).to(device)

    params = model.count_parameters()
    print(f"Model: {params['trainable']/1e6:.1f}M trainable / {params['total']/1e6:.1f}M total")

    # --- Loss ---
    criterion = RefinerLoss()

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Resume ---
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed at epoch {start_epoch}")

    # --- Training Loop ---
    print(f"\n{'='*60}")
    print(f"Training M2PR for {args.epochs} epochs")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        phase = get_phase(epoch, args.epochs)
        lr = cosine_lr_with_warmup(
            optimizer, epoch, args.epochs, args.warmup_epochs, args.lr
        )
        print(f"\nEpoch {epoch+1}/{args.epochs} | phase={phase} | lr={lr:.2e}")

        # Train
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch, args.epochs, args.log_every,
        )
        print(f"  Train loss: {train_losses['total']:.4f}")

        # Validate
        val_losses = validate(model, criterion, val_loader, device)
        print(f"  Val loss:   {val_losses['total']:.4f}")

        # Print key loss components
        for key in ["stego", "depthg", "kl", "inst_discrim", "affinity"]:
            t_val = train_losses.get(key, 0)
            v_val = val_losses.get(key, 0)
            print(f"    {key:20s}: train={t_val:.4f}  val={v_val:.4f}")

        gate = train_losses.get("gate", 0)
        print(f"    {'gate':20s}: {gate:.4f}" if isinstance(gate, float) else "")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Track best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_losses": val_losses,
            }, best_path)
            print(f"  New best val loss: {best_val_loss:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
