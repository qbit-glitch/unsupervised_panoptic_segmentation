#!/usr/bin/env python3
"""Train the DepthG x CAUSE-TR cross-model fusion adapters (spec 2026-06-12 rev 2).

Side A: codes = CAUSE z (23x46x90), cond = DepthG g bilinear-aligned to 23x46.
Side B: codes = DepthG g (40x80x100), cond = CAUSE z bilinear-aligned to 40x80.

teacher_mode:
    plain  A1: all 1024 pairs uniform, cross-teacher only (no stratification)
    strat  A2/B1: 512 short + 512 long pairs per spec 4.4/4.5
    dual   A3: strat, short-range weight = mean(cross-teacher cosine, depth kernel)

Loss (strat, side A):
    L = corr(short; teacher=g) + corr(long; teacher=frozen z)
      + lambda_preserve * MSE(z', z)
Side B swaps the teacher roles (short=self g, long=cross z).

Smoke:
    .venv_cups_cpu/bin/python mbps_pytorch/train_fusion_adapter.py \
        --side A --teacher_mode strat --epochs 1 --limit_train_images 3 \
        --batch_size 1 --output_dir results/fusion_adapter/smoke_A
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.models.semantic.cross_model_adapter import (
    CrossModelAdapter,
    sample_stratified_pairs,
    teacher_guided_correlation_loss,
)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CACHE = "/Volumes/code_files/datasets/cityscapes/fusion_feature_cache"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class FusionPairDataset(Dataset):
    """Loads paired (z, g, depth) grids and aligns the conditioning grid."""

    def __init__(self, cache_root: str, split: str, side: str,
                 g_subdir: str = "depthg_g_mono", limit_images: int = 0) -> None:
        if side not in ("A", "B"):
            raise ValueError(f"side must be A or B, got {side}")
        self.side = side
        self.files: List[Dict[str, str]] = []
        z_root = Path(cache_root) / "cause_z" / split
        g_root = Path(cache_root) / g_subdir / split
        for city in sorted(p.name for p in z_root.iterdir() if p.is_dir()):
            for zf in sorted((z_root / city).glob("*_codes.npy")):
                stem = zf.name.replace("_codes.npy", "")
                gf = g_root / city / f"{stem}_g.npy"
                df = z_root / city / f"{stem}_depth.npy"
                if gf.is_file() and df.is_file():
                    self.files.append({"z": str(zf), "g": str(gf), "d": str(df)})
        if limit_images > 0:
            self.files = self.files[:limit_images]

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _align(grid: torch.Tensor, hw: tuple) -> torch.Tensor:
        """(h, w, C) -> bilinear -> (hw[0]*hw[1], C)."""
        x = grid.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, hw, mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0).reshape(-1, x.shape[1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.files[idx]
        z = torch.from_numpy(np.load(e["z"]).astype(np.float32))   # (23, 46, 90)
        g = torch.from_numpy(np.load(e["g"]).astype(np.float32))   # (40, 80, d_g)
        d = torch.from_numpy(np.load(e["d"]).astype(np.float32))   # (23, 46)
        if self.side == "A":
            h, w = z.shape[:2]
            return {"codes": z.reshape(-1, z.shape[-1]),
                    "cond": self._align(g, (h, w)),
                    "depth": d.reshape(-1),
                    "spatial_shape": torch.tensor([h, w], dtype=torch.long)}
        h, w = g.shape[:2]
        d_up = F.interpolate(d[None, None], (h, w), mode="bilinear", align_corners=False)
        return {"codes": g.reshape(-1, g.shape[-1]),
                "cond": self._align(z, (h, w)),
                "depth": d_up.reshape(-1),
                "spatial_shape": torch.tensor([h, w], dtype=torch.long)}


def fusion_loss(adjusted: torch.Tensor, codes: torch.Tensor, cond: torch.Tensor,
                depth: torch.Tensor, hw: tuple, side: str, teacher_mode: str,
                lambda_preserve: float, sigma_d: float = 0.5) -> Dict[str, torch.Tensor]:
    """Per-batch fusion loss. adjusted/codes: (B,N,Dc); cond: (B,N,Dx); depth: (B,N)."""
    b = adjusted.shape[0]
    device = adjusted.device
    l_corr = torch.tensor(0.0, device=device)
    for i in range(b):
        if teacher_mode == "plain":
            n = adjusted.shape[1]
            idx_i = torch.randint(0, n, (1024,), device=device)
            idx_j = torch.randint(0, n, (1024,), device=device)
            l_corr = l_corr + teacher_guided_correlation_loss(adjusted[i], cond[i], idx_i, idx_j)
            continue
        (si, sj), (li, lj) = sample_stratified_pairs(hw[0], hw[1], device=device)
        # side A: short <- cross-teacher (DepthG cond), long <- self-teacher (frozen z codes)
        # side B: short <- self-teacher (frozen g codes), long <- cross-teacher (CAUSE cond)
        if side == "A":
            l_short = teacher_guided_correlation_loss(adjusted[i], cond[i], si, sj)
            l_long = teacher_guided_correlation_loss(adjusted[i], codes[i].detach(), li, lj)
        else:
            l_short = teacher_guided_correlation_loss(adjusted[i], codes[i].detach(), si, sj)
            l_long = teacher_guided_correlation_loss(adjusted[i], cond[i], li, lj)
        if teacher_mode == "dual" and side == "A":
            with torch.no_grad():
                w_d = torch.exp(-(depth[i][si] - depth[i][sj]) ** 2 / (2 * sigma_d ** 2))
                w_t = F.cosine_similarity(cond[i][si], cond[i][sj], dim=-1).clamp_min(0.0)
                w = 0.5 * (w_d + w_t)
            cos = F.cosine_similarity(adjusted[i][si], adjusted[i][sj], dim=-1)
            l_short = (w * (1.0 - cos) ** 2).mean()
        l_corr = l_corr + l_short + l_long
    l_corr = l_corr / b
    l_pres = F.mse_loss(adjusted, codes)
    return {"loss": l_corr + lambda_preserve * l_pres, "corr": l_corr, "preserve": l_pres}


def run_epoch(adapter, loader, device, args, optimizer=None, epoch=0) -> Dict[str, float]:
    is_train = optimizer is not None
    adapter.train() if is_train else adapter.eval()
    totals = {"loss": 0.0, "corr": 0.0, "preserve": 0.0, "drift": 0.0}
    count = 0
    with torch.enable_grad() if is_train else torch.no_grad():
        for step, batch in enumerate(loader):
            codes = batch["codes"].to(device)
            cond = batch["cond"].to(device)
            depth = batch["depth"].to(device)
            hw = tuple(batch["spatial_shape"][0].tolist())
            if is_train:
                optimizer.zero_grad()
            adjusted = adapter(codes, cond)
            parts = fusion_loss(adjusted, codes, cond, depth, hw, args.side,
                                args.teacher_mode, args.lambda_preserve)
            if is_train:
                parts["loss"].backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
            totals["loss"] += parts["loss"].item()
            totals["corr"] += parts["corr"].item()
            totals["preserve"] += parts["preserve"].item()
            totals["drift"] += (adjusted - codes).norm(dim=-1).mean().item()
            count += 1
            if is_train and step % 20 == 0:
                logger.info("ep%d step %d/%d loss=%.4f corr=%.4f pres=%.6f",
                            epoch, step, len(loader), parts["loss"].item(),
                            parts["corr"].item(), parts["preserve"].item())
    return {k: v / max(count, 1) for k, v in totals.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--side", required=True, choices=("A", "B"))
    p.add_argument("--teacher_mode", default="strat", choices=("plain", "strat", "dual"))
    p.add_argument("--g_subdir", default="depthg_g_mono")
    p.add_argument("--proj_width", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--lambda_preserve", type=float, default=20.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--limit_train_images", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.teacher_mode == "dual" and args.side != "A":
        p.error("--teacher_mode dual is defined for side A only (spec 4.4 run A3)")

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    full = FusionPairDataset(args.cache_root, "train", args.side,
                             g_subdir=args.g_subdir, limit_images=args.limit_train_images)
    if len(full) == 0:
        raise RuntimeError(f"no paired cache files under {args.cache_root}")
    n_val = max(1, int(len(full) * args.val_fraction))
    train_ds = torch.utils.data.Subset(full, range(len(full) - n_val))
    val_ds = torch.utils.data.Subset(full, range(len(full) - n_val, len(full)))
    logger.info("side=%s mode=%s train=%d val=%d device=%s", args.side, args.teacher_mode,
                len(train_ds), len(val_ds), device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) >= args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    code_dim, cond_dim = (90, 100) if args.side == "A" else (100, 90)
    adapter = CrossModelAdapter(code_dim, cond_dim, args.proj_width,
                                args.hidden_dim, args.num_layers).to(device)
    adapter_config = {"side": args.side, "code_dim": code_dim, "cond_dim": cond_dim,
                      "proj_width": args.proj_width, "hidden_dim": args.hidden_dim,
                      "num_layers": args.num_layers, "teacher_mode": args.teacher_mode,
                      "g_subdir": args.g_subdir,
                      "pair_offsets": {"r_short": 4, "r_long": 8}}
    logger.info("adapter params=%d", sum(q.numel() for q in adapter.parameters()))

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)

    best_val = float("inf")
    best_ckpt = os.path.join(args.output_dir, "best.pt")
    t0 = time.time()
    for epoch in range(args.epochs):
        tr = run_epoch(adapter, train_loader, device, args, optimizer, epoch)
        va = run_epoch(adapter, val_loader, device, args)
        logger.info("ep %d/%d | train loss=%.4f corr=%.4f pres=%.6f drift=%.4f | "
                    "val loss=%.4f drift=%.4f | %.0fs",
                    epoch, args.epochs - 1, tr["loss"], tr["corr"], tr["preserve"],
                    tr["drift"], va["loss"], va["drift"], time.time() - t0)
        scheduler.step()
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"state_dict": adapter.state_dict(),
                        "adapter_config": adapter_config,
                        "epoch": epoch, "val_loss": best_val,
                        "train_args": vars(args)}, best_ckpt)
            logger.info("  -> new best val=%.4f saved %s", best_val, best_ckpt)
    logger.info("done. best val=%.4f ckpt=%s", best_val, best_ckpt)


if __name__ == "__main__":
    main()
