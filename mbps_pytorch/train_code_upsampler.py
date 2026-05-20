#!/usr/bin/env python3
"""Train residual or dynamic-kernel upsamplers for frozen DCFA 90D codes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.code_upsampler import (
    AttentiveCodeUpsampler,
    DynamicKernelCodeUpsampler,
    ResidualCodeUpsampler,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DCFACodePairDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        limit_images: int = 0,
        random_flip: bool = True,
    ) -> None:
        self.root = Path(cache_dir) / split
        self.paths = sorted(self.root.rglob("*.npz"))
        if limit_images > 0:
            self.paths = self.paths[:limit_images]
        if not self.paths:
            raise FileNotFoundError(f"No cache files found under {self.root}")
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.paths[idx])
        low = torch.from_numpy(data["low_code"].astype(np.float32))
        teacher = torch.from_numpy(data["teacher_code"].astype(np.float32))
        rgb = torch.from_numpy(data["rgb"].astype(np.float32) / 255.0).permute(2, 0, 1)
        depth = torch.from_numpy(data["depth"].astype(np.float32)).unsqueeze(0)

        if self.random_flip and random.random() < 0.5:
            low = low.flip(-1)
            teacher = teacher.flip(-1)
            rgb = rgb.flip(-1)
            depth = depth.flip(-1)

        guidance = torch.cat([rgb, depth], dim=0)
        return {
            "low_code": low,
            "teacher_code": teacher,
            "guidance": guidance,
        }


def cosine_code_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred.float(), dim=1, eps=1e-6)
    target_n = F.normalize(target.float(), dim=1, eps=1e-6)
    return 1.0 - (pred_n * target_n).sum(dim=1).mean()


def crop_teacher_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_h: int,
    crop_w: int,
    is_train: bool,
) -> torch.Tensor:
    if not is_train:
        return cosine_code_loss(pred, target)
    height, width = pred.shape[-2:]
    crop_h = min(max(1, crop_h), height)
    crop_w = min(max(1, crop_w), width)
    if crop_h == height and crop_w == width:
        return cosine_code_loss(pred, target)
    top = random.randint(0, height - crop_h)
    left = random.randint(0, width - crop_w)
    pred_crop = pred[:, :, top : top + crop_h, left : left + crop_w]
    target_crop = target[:, :, top : top + crop_h, left : left + crop_w]
    return cosine_code_loss(pred_crop, target_crop)


def downsample_consistency_loss(pred: torch.Tensor, low_code: torch.Tensor) -> torch.Tensor:
    pred_low = F.interpolate(
        pred,
        size=low_code.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return cosine_code_loss(pred_low, low_code)


def boundary_smoothness_loss(
    pred: torch.Tensor,
    guidance: torch.Tensor,
    edge_alpha: float = 12.0,
) -> torch.Tensor:
    rgb = guidance[:, :3]
    depth = guidance[:, 3:4]

    code_dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean(dim=1)
    code_dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean(dim=1)

    rgb_dx = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean(dim=1)
    rgb_dy = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=1)
    dep_dx = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs().mean(dim=1)
    dep_dy = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs().mean(dim=1)

    wx = torch.exp(-edge_alpha * (rgb_dx + dep_dx)).detach()
    wy = torch.exp(-edge_alpha * (rgb_dy + dep_dy)).detach()
    return (code_dx * wx).mean() + (code_dy * wy).mean()


def variance_preservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_std = pred.flatten(2).std(dim=2)
    target_std = target.flatten(2).std(dim=2).detach()
    floor = 0.5 * target_std
    return F.relu(floor - pred_std).mean()


def _shift_pair(x: torch.Tensor, dy: int, dx: int) -> tuple[torch.Tensor, torch.Tensor]:
    y0_a = max(0, -dy)
    y1_a = x.shape[-2] - max(0, dy)
    x0_a = max(0, -dx)
    x1_a = x.shape[-1] - max(0, dx)
    y0_b = max(0, dy)
    y1_b = x.shape[-2] - max(0, -dy)
    x0_b = max(0, dx)
    x1_b = x.shape[-1] - max(0, -dx)
    return x[:, :, y0_a:y1_a, x0_a:x1_a], x[:, :, y0_b:y1_b, x0_b:x1_b]


def mask_self_distill_loss(
    pred: torch.Tensor,
    teacher: torch.Tensor,
    guidance: torch.Tensor,
    edge_alpha: float = 12.0,
    teacher_temp: float = 10.0,
    neg_margin: float = 0.25,
) -> torch.Tensor:
    """Local class-agnostic mask distillation from teacher/guidance affinities."""
    pred_n = F.normalize(pred.float(), dim=1, eps=1e-6)
    teacher_n = F.normalize(teacher.float(), dim=1, eps=1e-6).detach()
    guide = guidance.float().detach()
    total = pred.new_tensor(0.0)
    n_terms = 0
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        p_a, p_b = _shift_pair(pred_n, dy, dx)
        t_a, t_b = _shift_pair(teacher_n, dy, dx)
        g_a, g_b = _shift_pair(guide, dy, dx)
        pred_sim = (p_a * p_b).sum(dim=1)
        teacher_sim = (t_a * t_b).sum(dim=1)
        guide_dist = (g_a - g_b).abs().mean(dim=1)
        same_weight = torch.exp(-edge_alpha * guide_dist) * torch.sigmoid(teacher_temp * teacher_sim)
        pos = same_weight * (1.0 - pred_sim)
        neg = (1.0 - same_weight) * F.relu(pred_sim - neg_margin)
        total = total + (pos + neg).mean()
        n_terms += 1
    return total / max(n_terms, 1)


def neco_neighbor_order_loss(
    pred: torch.Tensor,
    teacher: torch.Tensor,
    num_samples: int = 256,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Patch-neighbor ordering loss inspired by NeCo.

    It matches the teacher's dense neighbor distribution instead of only the
    per-pixel teacher vector, making the output easier to cluster at fixed K.
    """
    bsz, _, height, width = pred.shape
    n_tokens = height * width
    n = min(max(2, num_samples), n_tokens)
    pred_flat = F.normalize(pred.float().flatten(2).transpose(1, 2), dim=2, eps=1e-6)
    teacher_flat = F.normalize(teacher.float().flatten(2).transpose(1, 2), dim=2, eps=1e-6).detach()
    losses = []
    for b in range(bsz):
        idx = torch.randperm(n_tokens, device=pred.device)[:n]
        p = pred_flat[b, idx]
        t = teacher_flat[b, idx]
        sim_p = p @ p.t() / temperature
        sim_t = t @ t.t() / temperature
        eye = torch.eye(n, device=pred.device, dtype=torch.bool)
        sim_p = sim_p.masked_fill(eye, -1e4)
        sim_t = sim_t.masked_fill(eye, -1e4)
        target_prob = F.softmax(sim_t, dim=1)
        log_prob = F.log_softmax(sim_p, dim=1)
        losses.append(F.kl_div(log_prob, target_prob, reduction="batchmean"))
    return torch.stack(losses).mean()


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.variant == "residual":
        return ResidualCodeUpsampler(
            code_dim=getattr(args, "code_dim", 90),
            hidden_ch=getattr(args, "hidden_ch", 64),
            guidance_hidden=getattr(args, "guidance_hidden", 32),
            num_blocks=getattr(args, "num_blocks", 3),
            residual_scale=getattr(args, "residual_scale", 0.25),
        )
    if args.variant == "dynamic":
        return DynamicKernelCodeUpsampler(
            code_dim=getattr(args, "code_dim", 90),
            hidden_ch=getattr(args, "hidden_ch", 64),
            guidance_hidden=getattr(args, "guidance_hidden", 32),
            kernel_size=getattr(args, "kernel_size", 3),
            num_blocks=getattr(args, "num_blocks", 3),
            residual_scale=getattr(args, "residual_scale", 0.25),
            use_coords=getattr(args, "use_coords", False),
            coord_freqs=getattr(args, "coord_freqs", 4),
        )
    if args.variant == "attentive":
        return AttentiveCodeUpsampler(
            code_dim=getattr(args, "code_dim", 90),
            hidden_ch=getattr(args, "hidden_ch", 64),
            guidance_hidden=getattr(args, "guidance_hidden", 48),
            attn_dim=getattr(args, "attn_dim", 64),
            window_size=getattr(args, "attn_window", 5),
            num_blocks=getattr(args, "num_blocks", 2),
            residual_scale=getattr(args, "residual_scale", 0.15),
            use_coords=getattr(args, "use_coords", True),
            coord_freqs=getattr(args, "coord_freqs", 4),
        )
    raise ValueError(f"Unknown variant: {args.variant}")


def iter_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {
        "loss": 0.0,
        "teacher": 0.0,
        "down": 0.0,
        "edge": 0.0,
        "var": 0.0,
        "mask": 0.0,
        "neco": 0.0,
    }
    count = 0

    for batch in tqdm(loader, desc=desc):
        low = batch["low_code"].to(device, non_blocking=True)
        teacher = batch["teacher_code"].to(device, non_blocking=True)
        guidance = batch["guidance"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            pred = model(low, guidance, output_size=teacher.shape[-2:])
            if args.crop_teacher:
                loss_teacher = crop_teacher_cosine_loss(
                    pred,
                    teacher,
                    args.crop_h,
                    args.crop_w,
                    is_train,
                )
            else:
                loss_teacher = cosine_code_loss(pred, teacher)
            loss_down = downsample_consistency_loss(pred, low)
            loss_edge = boundary_smoothness_loss(pred, guidance, args.edge_alpha)
            loss_var = variance_preservation_loss(pred, teacher)
            loss_mask = pred.new_tensor(0.0)
            if args.lambda_mask > 0:
                loss_mask = mask_self_distill_loss(
                    pred,
                    teacher,
                    guidance,
                    edge_alpha=args.mask_edge_alpha,
                    teacher_temp=args.mask_teacher_temp,
                    neg_margin=args.mask_neg_margin,
                )
            loss_neco = pred.new_tensor(0.0)
            if args.lambda_neco > 0:
                loss_neco = neco_neighbor_order_loss(
                    pred,
                    teacher,
                    num_samples=args.neco_samples,
                    temperature=args.neco_tau,
                )
            loss = (
                loss_teacher
                + args.lambda_down * loss_down
                + args.lambda_edge * loss_edge
                + args.lambda_var * loss_var
                + args.lambda_mask * loss_mask
                + args.lambda_neco * loss_neco
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        bsz = low.shape[0]
        count += bsz
        totals["loss"] += float(loss.detach().cpu()) * bsz
        totals["teacher"] += float(loss_teacher.detach().cpu()) * bsz
        totals["down"] += float(loss_down.detach().cpu()) * bsz
        totals["edge"] += float(loss_edge.detach().cpu()) * bsz
        totals["var"] += float(loss_var.detach().cpu()) * bsz
        totals["mask"] += float(loss_mask.detach().cpu()) * bsz
        totals["neco"] += float(loss_neco.detach().cpu()) * bsz

    return {key: val / max(count, 1) for key, val in totals.items()}


def write_csv_row(path: Path, row: Dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default="outputs/code_upsampler/cache_dcfa_v3_90d_64x128")
    parser.add_argument("--output_dir", default="outputs/code_upsampler/runs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--variant", choices=["residual", "dynamic", "attentive"], required=True)
    parser.add_argument("--code_dim", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_down", type=float, default=0.25)
    parser.add_argument("--lambda_edge", type=float, default=0.02)
    parser.add_argument("--lambda_var", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.0)
    parser.add_argument("--lambda_neco", type=float, default=0.0)
    parser.add_argument("--edge_alpha", type=float, default=12.0)
    parser.add_argument("--mask_edge_alpha", type=float, default=12.0)
    parser.add_argument("--mask_teacher_temp", type=float, default=10.0)
    parser.add_argument("--mask_neg_margin", type=float, default=0.25)
    parser.add_argument("--neco_samples", type=int, default=256)
    parser.add_argument("--neco_tau", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_ch", type=int, default=64)
    parser.add_argument("--guidance_hidden", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--attn_dim", type=int, default=64)
    parser.add_argument("--attn_window", type=int, default=5)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--use_coords", action="store_true")
    parser.add_argument("--coord_freqs", type=int, default=4)
    parser.add_argument("--crop_teacher", action="store_true")
    parser.add_argument("--crop_h", type=int, default=32)
    parser.add_argument("--crop_w", type=int, default=64)
    parser.add_argument("--limit_images", type=int, default=0)
    parser.add_argument("--val_limit_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_flip", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    run_name = args.run_name or f"{args.variant}_90d_64x128_s{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = DCFACodePairDataset(
        args.cache_dir,
        "train",
        limit_images=args.limit_images,
        random_flip=not args.no_flip,
    )
    val_ds = None
    val_root = Path(args.cache_dir) / "val"
    if val_root.exists() and any(val_root.rglob("*.npz")):
        val_ds = DCFACodePairDataset(
            args.cache_dir,
            "val",
            limit_images=args.val_limit_images,
            random_flip=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = math.inf

    print(f"Run: {run_name}")
    print(f"Variant: {args.variant}")
    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val images: {len(val_ds)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = iter_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            desc=f"train {epoch}/{args.epochs}",
        )
        row: Dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})

        monitor_metric = train_metrics["loss"]
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = iter_epoch(
                    model,
                    val_loader,
                    None,
                    device,
                    args,
                    desc=f"val {epoch}/{args.epochs}",
                )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            monitor_metric = val_metrics["loss"]

        write_csv_row(run_dir / "metrics.csv", row)
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, row, args)
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, row, args)

        compact = " ".join(
            f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)
        )
        print(f"epoch={epoch} {compact}", flush=True)

    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
