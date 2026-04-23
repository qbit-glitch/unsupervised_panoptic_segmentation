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
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from mbps_pytorch.models.adapters import (
    inject_lora_into_depth_model,
    inject_lora_into_depthpro,
    freeze_non_adapter_params,
    count_adapter_params,
)

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
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Depth model loaders
# --------------------------------------------------------------------------- #

def load_dav3_model(model_name="depth-anything/DA3MONO-LARGE", device="cpu"):
    try:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(model_name)
        model = model.to(device=torch.device(device))
        return model
    except ImportError:
        logger.error("depth_anything_3 not installed. Falling back to DA2.")
        return load_da2_model(device=device)


def load_da2_model(model_name="depth-anything/Depth-Anything-V2-Large-hf", device="cpu"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.processor = processor
    return model


def load_depthpro_model(model_name="apple/DepthPro-hf", device="cpu"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.processor = processor
    return model


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #

def self_distillation_loss(student_out, teacher_out, mask=None):
    """MSE between student and frozen teacher outputs."""
    if mask is not None:
        diff = (student_out - teacher_out.detach()) ** 2
        return (diff * mask).sum() / mask.sum().clamp(min=1)
    return F.mse_loss(student_out, teacher_out.detach())


def relative_depth_ranking_loss(depth_pred, num_pairs=1024):
    """Encourage correct pairwise depth ordering.

    Samples pairs of pixels and penalizes when the predicted depth order
    contradicts the teacher's order.
    """
    B, H, W = depth_pred.shape
    device = depth_pred.device
    flat = depth_pred.view(B, -1)

    loss = torch.tensor(0.0, device=device)
    for b in range(B):
        idx_i = torch.randint(0, H * W, (num_pairs // B,), device=device)
        idx_j = torch.randint(0, H * W, (num_pairs // B,), device=device)
        d_i = flat[b, idx_i]
        d_j = flat[b, idx_j]
        # Ranking: if d_i > d_j, we want pred_i > pred_j
        # Use margin ranking loss
        target = torch.sign(d_i.detach() - d_j.detach())
        margin = 0.1
        l = F.margin_ranking_loss(d_i, d_j, target, margin=margin)
        loss = loss + l
    return loss / B


def scale_invariant_loss(pred, target, lambda_si=0.5):
    """Scale-invariant log loss (Eigen et al.)."""
    diff = torch.log(pred + 1e-6) - torch.log(target.detach() + 1e-6)
    n = pred.numel()
    loss = (diff ** 2).mean() - lambda_si * (diff.sum() ** 2) / (n ** 2)
    return loss


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class DepthAdapterDataset(Dataset):
    def __init__(self, image_dir, image_size=(512, 1024), augment=True):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment
        self.image_files = sorted(
            list(self.image_dir.rglob("*.png")) + list(self.image_dir.rglob("*.jpg"))
        )
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        logger.info("Found %d images", len(self.image_files))

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
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
        img_tensor = self.transform(img)  # (3, H, W)
        img_aug = None
        if self.aug_transform:
            img_aug = T.ToTensor()(self.aug_transform(Image.fromarray(
                (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )))
        return {"img": img_tensor, "img_aug": img_aug, "path": str(img_path)}


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_depth_adapter(
    model, teacher_model, processor, train_loader, device, output_dir,
    losses, loss_weights, lr=1e-4, epochs=10, save_every=5,
    model_type="dav3", grad_accum_steps=1, adapter_config=None,
):
    logger.info("=== Depth Adapter Training ===")
    logger.info("Losses: %s", losses)
    logger.info("Weights: %s", loss_weights)
    logger.info("LR=%.1e, epochs=%d, grad_accum=%d", lr, epochs, grad_accum_steps)

    adapter_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable adapter params: %d", sum(p.numel() for p in adapter_params))

    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        totals = {k: 0.0 for k in losses + ["total"]}
        count = 0
        step = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()
        for batch in prog:
            img = batch["img"].to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                if model_type == "dav3":
                    teacher_out = teacher_model.inference_batch(img)
                else:
                    inputs = processor(images=[Image.fromarray((i.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for i in img], return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    teacher_out = teacher_model(**inputs).predicted_depth

            # Student forward
            if model_type == "dav3":
                student_out = model.inference_batch(img)
            else:
                student_out = model(**inputs).predicted_depth

            loss_total = torch.tensor(0.0, device=device)

            if "distillation" in losses:
                l_dist = self_distillation_loss(student_out, teacher_out)
                w = loss_weights.get("distillation", 1.0)
                loss_total = loss_total + w * l_dist
                totals["distillation"] += l_dist.item()

            if "ranking" in losses:
                l_rank = relative_depth_ranking_loss(student_out)
                w = loss_weights.get("ranking", 0.1)
                loss_total = loss_total + w * l_rank
                totals["ranking"] += l_rank.item()

            if "scale_invariant" in losses:
                l_si = scale_invariant_loss(student_out, teacher_out)
                w = loss_weights.get("scale_invariant", 0.5)
                loss_total = loss_total + w * l_si
                totals["scale_invariant"] += l_si.item()

            # Gradient accumulation
            loss_total = loss_total / grad_accum_steps
            loss_total.backward()

            totals["total"] += loss_total.item() * grad_accum_steps
            count += 1
            step += 1
            prog.set_postfix(total=f"{loss_total.item() * grad_accum_steps:.4f}")

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Handle remaining gradients
        if step % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_total = totals["total"] / max(count, 1)
        logger.info("Epoch %d: total=%.4f", epoch + 1, avg_total)
        for k in losses:
            logger.info("  %s: %.4f", k, totals[k] / max(count, 1))
        scheduler.step()

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

def main():
    parser = argparse.ArgumentParser(description="Train depth model adapters")
    parser.add_argument("--model_type", type=str, required=True, choices=["dav3", "da2", "depthpro"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--variant", type=str, default="dora", choices=["lora", "dora", "conv_dora"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--late_block_start", type=int, default=6)
    parser.add_argument("--adapt_decoder", action="store_true")
    parser.add_argument("--losses", type=str, default="distillation,ranking")
    parser.add_argument("--loss_weights", type=str, default="")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    args = parser.parse_args()

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
        model = load_da2_model(model_name, device=str(device))
        processor = model.processor
    elif args.model_type == "depthpro":
        model_name = args.model_name or "apple/DepthPro-hf"
        model = load_depthpro_model(model_name, device=str(device))
        processor = model.processor
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Create frozen teacher model (original, no adapters)
    logger.info("Creating frozen teacher model...")
    teacher_model = copy.deepcopy(model)
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

    dataset = DepthAdapterDataset(
        args.data_dir, image_size=tuple(args.image_size), augment="distillation" in args.losses,
    )
    nw = 0 if device.type == "mps" else args.num_workers
    pin = device.type == "cuda"
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )

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
    }

    train_depth_adapter(
        model, teacher_model, processor, train_loader, device, args.output_dir,
        losses=loss_list, loss_weights=loss_weights, lr=args.lr, epochs=args.epochs,
        save_every=args.save_every, model_type=args.model_type,
        grad_accum_steps=args.grad_accum_steps, adapter_config=adapter_config,
    )

    logger.info("Training complete! Checkpoints in %s", args.output_dir)


if __name__ == "__main__":
    main()
