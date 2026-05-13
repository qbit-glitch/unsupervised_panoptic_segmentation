#!/usr/bin/env python3
"""Train Strong Baseline: DINOv3 ViT-L/16 + Mask2Former on Cityscapes Pseudo-Labels.

Usage:
    python scripts/train_baseline.py --config configs/baseline_cityscapes.yaml
    python scripts/train_baseline.py --config configs/baseline_cityscapes.yaml --resume checkpoints/baseline_cityscapes/epoch_25.pt
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps_pytorch.models.mask2former.mask2former_model import (
    DINOv3Backbone,
    Mask2FormerPanoptic,
)
from mbps_pytorch.models.mask2former.panoptic_postprocessor import PanopticPostProcessor
from mbps_pytorch.losses.mask2former_loss import Mask2FormerCriterion
from mbps_pytorch.data.panoptic_dataset import CityscapesPanopticPseudoDataset
from mbps_pytorch.evaluation.panoptic_quality import compute_panoptic_quality
from mbps_pytorch.training.ema import EMAState

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack images, keep targets as list."""
    images = torch.stack([b["image"] for b in batch])
    targets = [b["targets"] for b in batch]
    metadata = [b["metadata"] for b in batch]
    return {"image": images, "targets": targets, "metadata": metadata}


class WarmupCosineSchedule:
    """Linear warmup + cosine decay learning rate schedule."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count <= self.warmup_steps:
                lr = base_lr * self.step_count / max(1, self.warmup_steps)
            else:
                progress = (self.step_count - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    config: dict,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    # Keep backbone in eval mode (frozen BatchNorm/dropout)
    model.backbone.eval()

    grad_accum = config["training"]["grad_accum_steps"]
    clip_norm = config["training"]["gradient_clip"]
    log_every = config["logging"]["log_every_n_steps"]

    running_losses = {}
    num_steps = 0
    t_start = time.time()

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]

        # Forward
        outputs = model(images)

        # Loss
        losses = criterion(outputs, targets)
        loss = losses["total_loss"] / grad_accum

        # Backward
        loss.backward()

        # Accumulate losses for logging
        for k, v in losses.items():
            if k not in running_losses:
                running_losses[k] = 0.0
            running_losses[k] += v.item()

        # Optimizer step after gradient accumulation
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                clip_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            num_steps += 1

            # Log
            if num_steps % log_every == 0:
                avg_losses = {k: v / (step + 1) for k, v in running_losses.items()}
                lr = scheduler.get_lr()
                elapsed = time.time() - t_start
                logger.info(
                    f"Epoch {epoch} | Step {num_steps} | "
                    f"Loss: {avg_losses.get('total_loss', 0):.4f} | "
                    f"CE: {avg_losses.get('loss_ce', 0):.4f} | "
                    f"Mask: {avg_losses.get('loss_mask', 0):.4f} | "
                    f"Dice: {avg_losses.get('loss_dice', 0):.4f} | "
                    f"LR: {lr:.2e} | Time: {elapsed:.0f}s"
                )

    avg_losses = {k: v / max(1, len(dataloader)) for k, v in running_losses.items()}
    return avg_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    postprocessor: PanopticPostProcessor,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    """Evaluate panoptic quality on validation set."""
    model.eval()

    all_pq = []
    all_sq = []
    all_rq = []

    thing_classes = postprocessor.thing_classes
    stuff_classes = postprocessor.stuff_classes

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["targets"]

        outputs = model(images)

        # Post-process predictions
        B = images.shape[0]
        results = postprocessor(
            outputs["pred_logits"],
            outputs["pred_masks"],
            target_size=images.shape[-2:],
        )

        # Compute PQ per image
        for b in range(B):
            pred_result = results[b]

            # Build ground truth panoptic from targets
            tgt_labels = targets[b]["labels"]
            tgt_masks = targets[b]["masks"]

            if len(tgt_labels) == 0:
                continue

            # Upsample target masks to image resolution
            tgt_masks_full = torch.nn.functional.interpolate(
                tgt_masks.unsqueeze(0).to(device),
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (M, H, W)

            # Create GT panoptic map
            H, W = images.shape[-2:]
            gt_panoptic = np.zeros((H, W), dtype=np.int64)
            gt_segments = []
            for m_idx in range(len(tgt_labels)):
                cat_id = tgt_labels[m_idx].item()
                seg_id = cat_id * 1000 + m_idx + 1
                mask = (tgt_masks_full[m_idx] > 0.5).cpu().numpy()
                gt_panoptic[mask] = seg_id
                gt_segments.append({"id": seg_id, "category_id": cat_id})

            pred_panoptic = pred_result.panoptic_map.cpu().numpy()
            pred_segments = pred_result.segments_info

            pq_result = compute_panoptic_quality(
                pred_panoptic, gt_panoptic,
                pred_segments, gt_segments,
                thing_classes=thing_classes,
                stuff_classes=stuff_classes,
            )
            all_pq.append(pq_result.pq)
            all_sq.append(pq_result.sq)
            all_rq.append(pq_result.rq)

    metrics = {
        "PQ": np.mean(all_pq) * 100 if all_pq else 0.0,
        "SQ": np.mean(all_sq) * 100 if all_sq else 0.0,
        "RQ": np.mean(all_rq) * 100 if all_rq else 0.0,
    }
    return metrics


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule, epoch: int, metrics: dict,
    path: str,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Only save trainable state (not frozen backbone)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("backbone.")
    }
    torch.save({
        "epoch": epoch,
        "model_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step": scheduler.step_count,
        "metrics": metrics,
    }, path)
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer | None,
    scheduler: WarmupCosineSchedule | None, path: str,
) -> int:
    """Load training checkpoint. Returns the epoch number."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_step" in ckpt:
        scheduler.step_count = ckpt["scheduler_step"]
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}: {path}")
    return ckpt["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Train Mask2Former Baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Device: {device}")

    # ---- Model ----
    logger.info("Loading DINOv3 ViT-L/16 backbone...")
    backbone = DINOv3Backbone(
        model_name=config["architecture"]["backbone_model_name"],
    )
    backbone.to(device)

    model = Mask2FormerPanoptic(
        backbone=backbone,
        num_classes=config["architecture"]["num_classes"],
        hidden_dim=config["architecture"]["hidden_dim"],
        num_queries=config["architecture"]["num_queries"],
        nheads=config["architecture"]["nheads"],
        dim_feedforward=config["architecture"]["dim_feedforward"],
        dec_layers=config["architecture"]["dec_layers"],
        backbone_dim=config["architecture"]["backbone_dim"],
    ).to(device)

    logger.info(f"Trainable parameters: {model.num_trainable_params():,}")

    # ---- Dataset ----
    train_dataset = CityscapesPanopticPseudoDataset(
        data_dir=config["data"]["data_dir"],
        pseudo_semantic_dir=config["data"]["pseudo_semantic_dir"],
        pseudo_instance_dir=config["data"]["pseudo_instance_dir"],
        stuff_things_path=config["data"].get("stuff_things_path"),
        split="train",
        crop_size=config["data"]["crop_size"],
        min_scale=config["data"]["min_scale"],
        max_scale=config["data"]["max_scale"],
        mask_stride=config["data"]["mask_stride"],
    )

    val_dataset = CityscapesPanopticPseudoDataset(
        data_dir=config["data"]["data_dir"],
        pseudo_semantic_dir=config["data"]["pseudo_semantic_dir"],
        pseudo_instance_dir=config["data"]["pseudo_instance_dir"],
        stuff_things_path=config["data"].get("stuff_things_path"),
        split="val",
        crop_size=config["data"]["crop_size"],
        min_scale=1.0,
        max_scale=1.0,
        mask_stride=config["data"]["mask_stride"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
    )

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # ---- Loss ----
    criterion = Mask2FormerCriterion(
        num_classes=config["architecture"]["num_classes"],
        eos_coef=config["loss"]["eos_coef"],
        cost_class=config["loss"]["cost_class"],
        cost_mask=config["loss"]["cost_mask"],
        cost_dice=config["loss"]["cost_dice"],
        weight_class=config["loss"]["weight_class"],
        weight_mask=config["loss"]["weight_mask"],
        weight_dice=config["loss"]["weight_dice"],
        num_points=config["loss"]["num_points"],
        deep_supervision=config["loss"]["deep_supervision"],
    ).to(device)

    # ---- Optimizer ----
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_epochs = config["training"]["total_epochs"]
    steps_per_epoch = len(train_loader) // config["training"]["grad_accum_steps"]
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = config["training"]["warmup_epochs"] * steps_per_epoch

    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps, total_steps,
        min_lr=config["training"]["min_learning_rate"],
    )

    # ---- Post-processor ----
    postprocessor = PanopticPostProcessor(
        thing_classes=train_dataset.thing_classes,
        stuff_classes=train_dataset.stuff_classes,
        score_threshold=config["evaluation"]["score_threshold"],
        overlap_threshold=config["evaluation"]["overlap_threshold"],
    )

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume) + 1

    # ---- Eval only ----
    if args.eval_only:
        metrics = evaluate(model, val_loader, postprocessor, device, config)
        logger.info(f"Evaluation: PQ={metrics['PQ']:.2f} SQ={metrics['SQ']:.2f} RQ={metrics['RQ']:.2f}")
        return

    # ---- Phase 1: Bootstrap Training ----
    logger.info(f"=== Phase 1: Bootstrap Training ({total_epochs} epochs) ===")

    best_pq = 0.0
    for epoch in range(start_epoch, total_epochs):
        t_epoch = time.time()

        train_losses = train_one_epoch(
            model, criterion, optimizer, scheduler,
            train_loader, device, epoch, config,
        )

        elapsed = time.time() - t_epoch
        logger.info(
            f"Epoch {epoch}/{total_epochs} complete | "
            f"Loss: {train_losses.get('total_loss', 0):.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Evaluate
        eval_every = config["logging"]["eval_every_n_epochs"]
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            metrics = evaluate(model, val_loader, postprocessor, device, config)
            logger.info(
                f"  Val PQ={metrics['PQ']:.2f} SQ={metrics['SQ']:.2f} RQ={metrics['RQ']:.2f}"
            )
            if metrics["PQ"] > best_pq:
                best_pq = metrics["PQ"]
                save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics,
                    os.path.join(config["checkpointing"]["checkpoint_dir"], "best.pt"),
                )

        # Save periodic checkpoint
        save_every = config["checkpointing"]["save_every_n_epochs"]
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_losses,
                os.path.join(config["checkpointing"]["checkpoint_dir"], f"epoch_{epoch:03d}.pt"),
            )

    logger.info(f"Phase 1 complete. Best PQ: {best_pq:.2f}")

    # ---- Phase 2: Self-Training ----
    st_config = config.get("self_training", {})
    if not st_config.get("enabled", False):
        logger.info("Self-training disabled. Done.")
        return

    logger.info("=== Phase 2: Self-Training ===")
    ema = EMAState(model, momentum=st_config["ema_decay"])
    thresholds = st_config.get("confidence_thresholds", [0.7, 0.75, 0.8])

    for round_idx in range(st_config["num_rounds"]):
        threshold = thresholds[min(round_idx, len(thresholds) - 1)]
        logger.info(f"--- Self-training round {round_idx + 1}/{st_config['num_rounds']} (threshold={threshold}) ---")

        # Update postprocessor threshold for pseudo-label generation
        postprocessor.score_threshold = threshold

        # Lower LR for self-training
        st_lr = st_config.get("learning_rate", 1e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = st_lr

        # Train for N epochs
        for ep in range(st_config["epochs_per_round"]):
            global_epoch = total_epochs + round_idx * st_config["epochs_per_round"] + ep
            train_losses = train_one_epoch(
                model, criterion, optimizer, scheduler,
                train_loader, device, global_epoch, config,
            )
            ema.update(model)
            logger.info(
                f"  ST Round {round_idx + 1} Epoch {ep + 1}/{st_config['epochs_per_round']} | "
                f"Loss: {train_losses.get('total_loss', 0):.4f}"
            )

        # Evaluate after each round
        metrics = evaluate(model, val_loader, postprocessor, device, config)
        logger.info(
            f"  ST Round {round_idx + 1} Val PQ={metrics['PQ']:.2f} SQ={metrics['SQ']:.2f} RQ={metrics['RQ']:.2f}"
        )
        if metrics["PQ"] > best_pq:
            best_pq = metrics["PQ"]
            save_checkpoint(
                model, optimizer, scheduler, global_epoch, metrics,
                os.path.join(config["checkpointing"]["checkpoint_dir"], "best.pt"),
            )

        # Save round checkpoint
        save_checkpoint(
            model, optimizer, scheduler, global_epoch, metrics,
            os.path.join(config["checkpointing"]["checkpoint_dir"], f"st_round_{round_idx + 1}.pt"),
        )

    logger.info(f"Training complete. Best PQ: {best_pq:.2f}")


if __name__ == "__main__":
    main()
