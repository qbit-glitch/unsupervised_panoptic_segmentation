"""MSDA training script.

Usage:
    python -m mbps_pytorch.msda.train \
        --arch conv --loss hybrid \
        --feature_dir /path/to/dinov3_features_vitl16 \
        --depth_dir /path/to/depth_depthpro \
        --output_dir checkpoints/msda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .architectures import ARCH_REGISTRY, AdapterConfig, create_adapter
from .dataset import CachedFeatureDataset
from .losses import LOSS_REGISTRY, create_loss

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    if hasattr(loss_fn, "train"):
        loss_fn.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        features = batch["features"].to(device)  # (B, N, D)
        depth = batch["depth"].to(device)  # (B, 1, H, W)

        optimizer.zero_grad()

        adapted = model(features, depth)  # (B, N_out, D_out)

        loss = loss_fn(adapted, features, depth)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(loss_fn.parameters()),
                grad_clip,
            )
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}] loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return {"train_loss": avg_loss}


@torch.no_grad()
def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    if hasattr(loss_fn, "eval"):
        loss_fn.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        depth = batch["depth"].to(device)

        adapted = model(features, depth)
        loss = loss_fn(adapted, features, depth)

        total_loss += loss.item()
        num_batches += 1

    return {"val_loss": total_loss / max(num_batches, 1)}


def save_checkpoint(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    arch_name: str,
    loss_name: str,
    cfg: AdapterConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "arch": arch_name,
        "loss": loss_name,
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
    }
    if hasattr(loss_fn, "state_dict"):
        state["loss_state_dict"] = loss_fn.state_dict()
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MSDA Training")
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=list(ARCH_REGISTRY.keys()),
        help="Architecture variant",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=list(LOSS_REGISTRY.keys()),
        help="Loss function variant",
    )
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/msda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(args.output_dir) / f"{args.arch}_{args.loss}.log",
                mode="a",
            ),
        ],
    )

    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    run_name = f"{args.arch}_{args.loss}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = AdapterConfig(
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    )

    model = create_adapter(args.arch, cfg).to(device)
    loss_fn = create_loss(args.loss, output_dim=args.output_dim).to(device)

    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "loss_state_dict" in ckpt and hasattr(loss_fn, "load_state_dict"):
            loss_fn.load_state_dict(ckpt["loss_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["metrics"].get("val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    train_ds = CachedFeatureDataset(
        args.feature_dir, args.depth_dir, split="train", augment=True
    )
    val_ds = CachedFeatureDataset(
        args.feature_dir, args.depth_dir, split="val", augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    param_info = model.count_parameters()
    logger.info(f"Architecture: {args.arch} ({param_info['trainable']/1e6:.1f}M params)")
    logger.info(f"Loss: {args.loss}")
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, epoch, args.grad_clip
        )
        val_metrics = validate(model, loss_fn, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"lr={lr:.2e} "
            f"time={elapsed:.0f}s"
        )

        all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": lr}

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model, loss_fn, optimizer, epoch, all_metrics,
                run_dir / "best.pt", args.arch, args.loss, cfg,
            )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, loss_fn, optimizer, epoch, all_metrics,
                run_dir / f"epoch_{epoch:03d}.pt", args.arch, args.loss, cfg,
            )

    save_checkpoint(
        model, loss_fn, optimizer, args.epochs - 1,
        {**train_metrics, **val_metrics},
        run_dir / "last.pt", args.arch, args.loss, cfg,
    )
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
