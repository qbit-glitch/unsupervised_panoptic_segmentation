#!/usr/bin/env python3
"""Self-supervised training of LoRA/DoRA adapters on depth models (DAv3 / DepthPro).

Trains adapter parameters while keeping pretrained depth model frozen.
Uses self-supervised objectives:
    - Self-distillation from frozen teacher
    - Relative depth ranking loss (pairwise depth ordering)
    - Scale-invariant depth consistency

Usage:
    python mbps_pytorch/train_depth_adapter_lora.py \\
        --model_type dav3 \\
        --data_dir /path/to/cityscapes/leftImg8bit/train \\
        --output_dir results/depth_adapter_dora \\
        --variant dora --rank 4 --epochs 10
"""

import argparse
import copy
import logging
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from mbps_pytorch.models.adapters import (
    inject_lora_into_depth_model,
    inject_lora_into_depthpro,
    freeze_non_adapter_params,
    count_adapter_params,
)
from mbps_pytorch.models.adapters.lora_layers import LoRALinear, DoRALinear, ConvDoRALinear

try:
    import yaml
except ImportError:
    yaml = None

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # NOTE: PYTHONHASHSEED must be set in the shell before invoking Python:
    #   PYTHONHASHSEED=42 python train_depth_adapter_lora.py ...


# --------------------------------------------------------------------------- #
# Depth model loaders
# --------------------------------------------------------------------------- #

def _to_device(model, device):
    """Standardize device handling across all loaders."""
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    return model.to(device_obj)


def _dav3_inference_batch(model, img_tensor, grad=False):
    """Run DA3 inference on a batch of images, preserving gradients for student.

    The official DepthAnything3.forward() is decorated with @torch.inference_mode()
    and torch.no_grad(), which blocks gradient flow. This helper calls the
    underlying model directly (model.model) to allow adapter gradients.

    Automatically resizes inputs to dimensions divisible by the model's patch size
    (typically 14 for DINOv2-based DA3) to avoid assertion errors in patch_embed.

    Args:
        model: DepthAnything3 instance
        img_tensor: (B, 3, H, W) tensor
        grad: If True, run with gradient support (student). If False, no_grad (teacher).

    Returns:
        depth: (B, H, W) tensor
    """
    # Detect patch size from the backbone's patch_embed layer
    # DinoV2 wrapper: backbone.pretrained.patch_embed
    try:
        patch_embed = model.model.backbone.pretrained.patch_embed
        patch_size = getattr(patch_embed, "patch_size", 14)
    except AttributeError:
        patch_size = 14  # Default for all DINOv2-based DA3 models
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size

    # DA3 expects (B, N, 3, H, W) where N=1 for monocular
    if img_tensor.dim() == 4:
        x = img_tensor.unsqueeze(1)  # (B, 1, 3, H, W)
    else:
        x = img_tensor

    B, N, C, H, W = x.shape
    # Ensure dimensions are divisible by patch size
    h_valid = (H // patch_h) * patch_h
    w_valid = (W // patch_w) * patch_w
    if h_valid != H or w_valid != W:
        x = F.interpolate(
            x.view(B * N, C, H, W),
            size=(h_valid, w_valid),
            mode="bilinear",
            align_corners=False,
        ).view(B, N, C, h_valid, w_valid)

    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        output = model.model(x, extrinsics=None, intrinsics=None)
        # output is a Dict-like object; depth shape is (B, S, H, W) with S=1
        depth = output.depth
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth.squeeze(1)  # (B, H, W)
        elif depth.dim() == 3:
            pass  # already (B, H, W)
        else:
            raise ValueError(f"Unexpected DA3 depth shape: {depth.shape}")
    return depth


def load_dav3_model(model_name="depth-anything/DA3MONO-LARGE", device="cpu"):
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_name)
    model = _to_device(model, device)
    # Attach inference_batch helper for unified API
    model.inference_batch = lambda img, grad=False, _model=model: _dav3_inference_batch(_model, img, grad)
    return model


def load_da2_model(model_name="depth-anything/Depth-Anything-V2-Large-hf", device="cpu", cache_dir=None):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # Support local path fallback
    if os.path.isdir(model_name):
        logger.info("Loading DA2 from local path: %s", model_name)

    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, cache_dir=cache_dir)
    model = _to_device(model, device)
    model.processor = processor
    return model


def load_depthpro_model(model_name="apple/DepthPro-hf", device="cpu", cache_dir=None):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    if os.path.isdir(model_name):
        logger.info("Loading DepthPro from local path: %s", model_name)
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, cache_dir=cache_dir)
    model = _to_device(model, device)
    model.processor = processor
    return model


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #

def self_distillation_loss(student_out, teacher_out, mask=None, loss_type="log_l1"):
    """Distillation loss appropriate for metric/relative depth.

    Supports:
        - "mse": standard MSE (biases toward close-range accuracy)
        - "log_l1": log-space L1, more balanced across depth ranges
        - "relative_l1": scale-invariant per-pixel relative error
    """
    teacher_out = teacher_out.detach()
    if loss_type == "mse":
        if mask is not None:
            diff = (student_out - teacher_out) ** 2
            return (diff * mask).sum() / mask.sum().clamp(min=1)
        return F.mse_loss(student_out, teacher_out)
    elif loss_type == "log_l1":
        student_log = torch.log(student_out.clamp(min=1e-3))
        teacher_log = torch.log(teacher_out.clamp(min=1e-3))
        if mask is not None:
            diff = (student_log - teacher_log).abs()
            return (diff * mask).sum() / mask.sum().clamp(min=1)
        return (student_log - teacher_log).abs().mean()
    elif loss_type == "relative_l1":
        diff = (student_out - teacher_out).abs() / (teacher_out + 1e-3)
        if mask is not None:
            return (diff * mask).sum() / mask.sum().clamp(min=1)
        return diff.mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def _extract_depth(output):
    """Extract depth tensor from model output with shape validation."""
    depth = output.predicted_depth if hasattr(output, "predicted_depth") else output
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    if depth.dim() != 3:
        raise ValueError(f"Expected depth shape (B,H,W), got {depth.shape}")
    return depth


def relative_depth_ranking_loss(student_depth, teacher_depth, num_pairs=2048, margin=0.05):
    """Pairwise depth ranking loss.

    Samples pairs of pixels and penalizes when the student's predicted depth order
    contradicts the teacher's order.

    NOTE: num_pairs=2048 covers ~0.4% of pixels in a 512x1024 image.
    For boundary-rich scenes, consider increasing to 4096+.
    The margin is in the same units as the depth values. When using
    log_l1 distillation, depth values are in log-space and margin=0.05
    corresponds to ~5% relative depth difference.
    """
    B, H, W = student_depth.shape
    device = student_depth.device
    student_flat = student_depth.view(B, -1)
    teacher_flat = teacher_depth.view(B, -1)

    loss = torch.tensor(0.0, device=device)
    for b in range(B):
        idx_i = torch.randint(0, H * W, (num_pairs // B,), device=device)
        idx_j = torch.randint(0, H * W, (num_pairs // B,), device=device)
        # Ensure no self-pairs
        mask_same = idx_i == idx_j
        while mask_same.any():
            idx_j[mask_same] = torch.randint(0, H * W, (mask_same.sum(),), device=device)
            mask_same = idx_i == idx_j

        s_i = student_flat[b, idx_i]
        s_j = student_flat[b, idx_j]
        t_i = teacher_flat[b, idx_i]
        t_j = teacher_flat[b, idx_j]
        # Teacher defines the ground-truth ordering
        target = torch.sign(t_i - t_j)
        # Skip pairs where teacher has equal depth (target=0)
        valid = target != 0
        if valid.any():
            l = F.margin_ranking_loss(s_i[valid], s_j[valid], target[valid], margin=margin)
            loss = loss + l
    return loss / B


def scale_invariant_loss(pred, target, lambda_si=0.5, min_depth=1e-3):
    """Scale-invariant log loss (Eigen et al.) with gradient clipping."""
    pred_clamped = torch.clamp(pred, min=min_depth)
    target_clamped = torch.clamp(target.detach(), min=min_depth)
    diff = torch.log(pred_clamped) - torch.log(target_clamped)
    n = pred.numel()
    loss = (diff ** 2).mean() - lambda_si * (diff.sum() ** 2) / (n ** 2)
    return loss


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class DepthAdapterDataset(Dataset):
    def __init__(self, image_dir, image_size=(512, 1024), augment=True, return_pil=False):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment
        self.return_pil = return_pil
        self.image_files = sorted(
            list(self.image_dir.rglob("*.png")) + list(self.image_dir.rglob("*.jpg"))
        )
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        logger.info("Found %d images", len(self.image_files))

        self.base_transform = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)
        self.to_tensor = T.ToTensor()
        if augment:
            self.aug_transform = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.base_transform(img)  # Resize only, keep PIL

        if self.return_pil:
            item = {"img_pil": img, "path": str(img_path)}
        else:
            img_tensor = self.to_tensor(img)
            item = {"img": img_tensor, "path": str(img_path)}

        if self.aug_transform:
            img_aug = self.aug_transform(img)  # Augment PIL directly
            if self.return_pil:
                item["img_aug_pil"] = img_aug
            else:
                item["img_aug"] = self.to_tensor(img_aug)
        return item


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_depth_adapter(model, teacher_model, processor, val_loader, device, model_type):
    """Compute teacher-student divergence metrics on a validation set."""
    model.eval()
    metrics = {"mse": 0.0, "mae": 0.0, "rmse": 0.0}
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if model_type == "dav3":
                img = batch["img"].to(device)
                teacher_out = teacher_model.inference_batch(img, grad=False)
                student_out = model.inference_batch(img, grad=False)
            else:
                img = batch["img_pil"]
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                teacher_out = _extract_depth(teacher_model(**inputs))
                student_out = _extract_depth(model(**inputs))

            mse = F.mse_loss(student_out, teacher_out).item()
            mae = (student_out - teacher_out).abs().mean().item()
            metrics["mse"] += mse
            metrics["mae"] += mae
            metrics["rmse"] += mse ** 0.5
            count += 1

    for k in metrics:
        metrics[k] /= max(count, 1)
    return metrics


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_depth_adapter(
    model, teacher_model, processor, train_loader, device, output_dir,
    losses, loss_weights, lr=1e-4, epochs=10, save_every=5,
    model_type="dav3", grad_accum_steps=1, adapter_config=None,
    val_loader=None, val_every=1, distill_loss_type="log_l1",
    num_pairs=2048, ranking_margin=0.05,
):
    logger.info("=== Depth Adapter Training ===")
    logger.info("Losses: %s", losses)
    logger.info("Weights: %s", loss_weights)
    batch_size = getattr(train_loader, 'batch_size', 1)
    effective_batch = batch_size * grad_accum_steps
    logger.info("Effective batch size: %d (batch=%d * accum=%d)", effective_batch, batch_size, grad_accum_steps)
    logger.info("LR=%.1e, epochs=%d, grad_accum=%d", lr, epochs, grad_accum_steps)
    logger.info("Distill loss type: %s", distill_loss_type)

    # Loss weight recommendations:
    # - distillation (log_l1): weight 1.0 — primary objective
    # - ranking: weight 0.1 — enforces ordering consistency
    # - scale_invariant: weight 0.1 or less — conflicts with log_l1 if weight is high
    #   The SI loss discards global scale, which log_l1 already handles gracefully.
    if "scale_invariant" in losses and "distillation" in losses:
        logger.info("NOTE: scale_invariant and distillation losses may conflict. "
                    "Consider setting scale_invariant weight <= 0.1.")

    adapter_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable adapter params: %d", sum(p.numel() for p in adapter_params))

    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=1e-4)

    # Warmup + cosine schedule
    total_steps = epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    scaler = GradScaler() if device.type == "cuda" else None
    best_loss = float("inf")
    best_val_mae = float("inf")

    for epoch in range(epochs):
        model.eval()  # Frozen base in eval mode
        for m in model.modules():
            if isinstance(m, (LoRALinear, DoRALinear, ConvDoRALinear)):
                m.train()  # Only adapters in train mode
        totals = {k: 0.0 for k in losses + ["total"]}
        count = 0
        step = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()
        for batch in prog:
            if model_type == "dav3":
                img = batch["img"].to(device)
                img_aug = batch.get("img_aug")
                if img_aug is not None:
                    img_aug = img_aug.to(device)
            else:
                img = batch["img_pil"]
                img_aug = batch.get("img_aug_pil")

            # Teacher forward on clean image (no grad)
            with torch.no_grad():
                if model_type == "dav3":
                    teacher_out = teacher_model.inference_batch(img, grad=False)
                else:
                    teacher_inputs = processor(images=img, return_tensors="pt")
                    teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}
                    teacher_out = _extract_depth(teacher_model(**teacher_inputs))

            # Student forward on augmented image if available
            student_input = img_aug if img_aug is not None else img

            amp_context = autocast() if scaler else nullcontext()
            with amp_context:
                if model_type == "dav3":
                    student_out = model.inference_batch(student_input, grad=True)
                else:
                    student_inputs = processor(images=student_input, return_tensors="pt")
                    student_inputs = {k: v.to(device) for k, v in student_inputs.items()}
                    student_out = _extract_depth(model(**student_inputs))

                loss_total = torch.tensor(0.0, device=device)

                if "distillation" in losses:
                    l_dist = self_distillation_loss(student_out, teacher_out, loss_type=distill_loss_type)
                    w = loss_weights.get("distillation", 1.0)
                    loss_total = loss_total + w * l_dist
                    totals["distillation"] += l_dist.item()

                if "ranking" in losses:
                    l_rank = relative_depth_ranking_loss(student_out, teacher_out, num_pairs=num_pairs, margin=ranking_margin)
                    w = loss_weights.get("ranking", 0.1)
                    loss_total = loss_total + w * l_rank
                    totals["ranking"] += l_rank.item()

                if "scale_invariant" in losses:
                    l_si = scale_invariant_loss(student_out, teacher_out)
                    w = loss_weights.get("scale_invariant", 0.5)
                    loss_total = loss_total + w * l_si
                    totals["scale_invariant"] += l_si.item()

            # Gradient accumulation
            # Gradient clipping note:
            # - clip_grad_norm_ is called AFTER loss.backward() and BEFORE optimizer.step()
            # - With grad_accum_steps > 1, gradients accumulate across steps before clipping
            # - With AMP, scaler.unscale_() normalizes gradients before clipping
            # - Effective gradient scale = per-step-gradient / grad_accum_steps
            # - max_norm=1.0 is conservative; for stable adapter training, 5.0 may be acceptable
            loss_total = loss_total / grad_accum_steps
            if scaler:
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()

            totals["total"] += loss_total.item() * grad_accum_steps
            count += 1
            step += 1
            prog.set_postfix(total=f"{loss_total.item() * grad_accum_steps:.4f}")

            if step % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Handle remaining gradients
        if step % grad_accum_steps != 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        avg_total = totals["total"] / max(count, 1)
        logger.info("Epoch %d: total=%.4f", epoch + 1, avg_total)
        for k in losses:
            logger.info("  %s: %.4f", k, totals[k] / max(count, 1))

        # Validation
        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_metrics = validate_depth_adapter(
                model, teacher_model, processor, val_loader, device, model_type
            )
            logger.info(
                "Validation epoch %d: MSE=%.4f MAE=%.4f RMSE=%.4f",
                epoch + 1, val_metrics["mse"], val_metrics["mae"], val_metrics["rmse"],
            )
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                ckpt_path = os.path.join(output_dir, "best_val.pt")
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1, "adapter_config": adapter_config, "val_mae": best_val_mae},
                    ckpt_path,
                )
                logger.info("New best val MAE=%.4f, saved to %s", best_val_mae, ckpt_path)

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "adapter_config": adapter_config}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        if avg_total < best_loss:
            best_loss = avg_total
            ckpt_path = os.path.join(output_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "adapter_config": adapter_config}, ckpt_path)
            logger.info("New best loss=%.4f, saved to %s", best_loss, ckpt_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _build_parser():
    parser = argparse.ArgumentParser(description="Train depth model adapters")
    parser.add_argument("--model_type", type=str, required=True, choices=["dav3", "da2", "depthpro"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--variant", type=str, default="dora", choices=["lora", "dora", "conv_dora"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--late_block_start", type=int, default=None)
    parser.add_argument("--adapt_decoder", action="store_true")
    parser.add_argument("--losses", type=str, default="distillation,ranking,scale_invariant")
    parser.add_argument("--loss_weights", type=str, default="")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--distill_loss_type", type=str, default="log_l1",
                        choices=["mse", "log_l1", "relative_l1"],
                        help="Loss type for self-distillation objective")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of training data to reserve for validation")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run validation every N epochs")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory")
    parser.add_argument("--num_pairs", type=int, default=2048, help="Number of pixel pairs for ranking loss")
    parser.add_argument("--ranking_margin", type=float, default=0.05, help="Margin for ranking loss (in log-space if using log-depth)")
    return parser


def main():
    # Pre-parse to get config path before building full defaults
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    parser = _build_parser()

    if pre_args.config and os.path.isfile(pre_args.config):
        if yaml is None:
            raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")
        with open(pre_args.config, "r") as f:
            config = yaml.safe_load(f)
        # Map nested config to flat argparse names
        # e.g. model: {rank: 4} -> --rank 4
        config_defaults = {}
        for section, values in config.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    if isinstance(v, list):
                        config_defaults[k] = [str(x) for x in v]
                    else:
                        config_defaults[k] = v
        parser.set_defaults(**config_defaults)
        logger.info("Loaded config from %s", pre_args.config)

    args = parser.parse_args(remaining_argv)

    # Architecture-specific defaults for late_block_start
    if args.late_block_start is None:
        if args.model_type == "da2":
            args.late_block_start = 18  # DA2-Large: 24 blocks
        elif args.model_type == "depthpro":
            args.late_block_start = 18  # DepthPro: 24-layer DINOv2-Large encoders
        elif args.model_type == "dav3":
            args.late_block_start = 6   # DA3: typically 12 blocks
        else:
            args.late_block_start = 6
        logger.info("Auto-set late_block_start=%d for %s", args.late_block_start, args.model_type)

    set_seed(args.seed)
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading %s model...", args.model_type)
    if args.model_type == "dav3":
        model_name = args.model_name or "depth-anything/DA3MONO-LARGE"
        model = load_dav3_model(model_name, device=str(device))
        processor = None
    elif args.model_type == "da2":
        model_name = args.model_name or "depth-anything/Depth-Anything-V2-Large-hf"
        model = load_da2_model(model_name, device=str(device), cache_dir=args.cache_dir)
        processor = model.processor
    elif args.model_type == "depthpro":
        model_name = args.model_name or "apple/DepthPro-hf"
        model = load_depthpro_model(model_name, device=str(device), cache_dir=args.cache_dir)
        processor = model.processor
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Create frozen teacher model (original, no adapters)
    logger.info("Creating frozen teacher model...")
    if args.model_type == "dav3":
        teacher_model = load_dav3_model(model_name, device="cpu")
    elif args.model_type == "da2":
        teacher_model = load_da2_model(model_name, device="cpu", cache_dir=args.cache_dir)
    elif args.model_type == "depthpro":
        teacher_model = load_depthpro_model(model_name, device="cpu", cache_dir=args.cache_dir)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model = teacher_model.to(device)

    if args.model_type == "depthpro":
        inject_lora_into_depthpro(
            model, variant=args.variant, rank=args.rank, alpha=args.alpha,
            dropout=args.dropout, late_block_start=args.late_block_start,
            adapt_patch_encoder=True, adapt_image_encoder=True, adapt_fov_encoder=False,
        )
    else:
        inject_lora_into_depth_model(
            model, variant=args.variant, rank=args.rank, alpha=args.alpha,
            dropout=args.dropout, late_block_start=args.late_block_start,
            adapt_decoder=args.adapt_decoder,
        )
    freeze_non_adapter_params(model)
    model = model.to(device)
    logger.info("Total trainable params: %d", count_adapter_params(model))

    # Dataset: always augment for self-supervised training
    return_pil = args.model_type != "dav3"
    full_dataset = DepthAdapterDataset(
        args.data_dir, image_size=tuple(args.image_size), augment=True, return_pil=return_pil,
    )

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    nw = 0 if device.type == "mps" else args.num_workers
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=nw, pin_memory=pin, drop_last=False,
        )
        logger.info("Train: %d samples, Val: %d samples", train_size, val_size)
    else:
        logger.info("Train: %d samples (no val split)", train_size)

    loss_list = [x.strip() for x in args.losses.split(",")]
    loss_weights = {}
    if args.loss_weights:
        import json
        loss_weights = json.loads(args.loss_weights.replace("'", "\""))

    adapter_config = {
        "variant": args.variant,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "late_block_start": args.late_block_start,
        "adapt_decoder": args.adapt_decoder,
        "model_type": args.model_type,
        "distill_loss_type": args.distill_loss_type,
    }

    train_depth_adapter(
        model, teacher_model, processor, train_loader, device, args.output_dir,
        losses=loss_list, loss_weights=loss_weights, lr=args.lr, epochs=args.epochs,
        save_every=args.save_every, model_type=args.model_type,
        grad_accum_steps=args.grad_accum_steps, adapter_config=adapter_config,
        val_loader=val_loader, val_every=args.val_every,
        distill_loss_type=args.distill_loss_type,
        num_pairs=args.num_pairs, ranking_margin=args.ranking_margin,
    )

    logger.info("Training complete! Checkpoints in %s", args.output_dir)


if __name__ == "__main__":
    main()
