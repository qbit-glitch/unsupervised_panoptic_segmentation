"""Train AuxThingAdapter on cached Stage-3 P4 features + SAM3 supervision.

Fully GT-free. Uses:
  - Cached P4 features (256-dim per pixel) from `cache_stage3_p4_features.py`.
  - SAM3 masks from `sam_fine_masks_sam3/train/<city>/<id>_fine_masks.npz`,
    each containing (masks, class_labels, iou_scores).

Per-pixel target:
  t(u) = class_labels[k] if u ∈ masks[k] (priority by IoU score), else -1 (ignore).

Loss: cross_entropy(logits, target, ignore_index=-1).

CLI:
    python scripts/train_aux_thing_adapter.py \
        --feat_cache /path/to/p4_cache_stage3/ \
        --sam3_root /path/to/sam_fine_masks_sam3/train/ \
        --out checkpoints/aux_thing_adapter_run1 \
        --epochs 100 \
        --batch_size 8 \
        --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure refs/cups is importable for the AuxThingAdapter module.
_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ / "refs" / "cups"))

from cups.model.aux_thing_adapter import AuxThingAdapter  # noqa: E402


# ───────────────────────────── Dataset ──────────────────────────────────────


class CachedFeaturesSAM3Dataset(Dataset):
    """Pairs cached P4 features with per-pixel SAM3 targets at the same resolution.

    Args:
        feat_cache_dir: directory containing per-image .pt files
            shaped (256, H_p4, W_p4) at fp16.
        sam3_root: directory containing SAM3 .npz files
            (masks, class_labels, iou_scores), recursively.
        min_iou: discard SAM3 masks below this IoU score.
        max_masks_per_img: cap the number of masks per image (sorted by IoU desc).
        target_short: ignored (cache spatial resolution drives target size).
        verbose_missing: log when a feature/mask file pair is missing.
    """

    def __init__(
        self,
        feat_cache_dir: Path,
        sam3_root: Path,
        min_iou: float = 0.10,
        max_masks_per_img: int = 30,
        verbose_missing: bool = False,
    ) -> None:
        self.feat_cache_dir = Path(feat_cache_dir)
        self.sam3_root = Path(sam3_root)
        self.min_iou = min_iou
        self.max_masks_per_img = max_masks_per_img
        self.verbose_missing = verbose_missing

        # Build index by walking feat_cache_dir
        self.entries: List[Tuple[Path, Optional[Path]]] = []
        for feat_path in sorted(self.feat_cache_dir.glob("*.pt")):
            img_id = feat_path.stem  # e.g., "aachen_000000_000019"
            city = img_id.split("_")[0]
            sam3_path = self.sam3_root / city / f"{img_id}_fine_masks.npz"
            if not sam3_path.exists():
                if verbose_missing:
                    print(f"[WARN] missing SAM3 npz for {img_id}", flush=True)
                # Keep the entry so the loss is computed-with-all-ignore (no gradient).
                self.entries.append((feat_path, None))
            else:
                self.entries.append((feat_path, sam3_path))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_path, sam3_path = self.entries[idx]
        feat = torch.load(feat_path, weights_only=True).float()  # (256, H, W)
        c, h, w = feat.shape

        if sam3_path is None:
            target = torch.full((h, w), -1, dtype=torch.long)
            return feat, target

        # Load SAM3 npz
        d = np.load(sam3_path, allow_pickle=False)
        masks = d["masks"]  # (N, H_full, W_full) bool
        class_labels = d["class_labels"]  # (N,) int32
        iou_scores = d["iou_scores"]  # (N,) float32

        # Filter by IoU threshold + cap by IoU desc.
        keep = iou_scores >= self.min_iou
        if keep.sum() == 0:
            target = torch.full((h, w), -1, dtype=torch.long)
            return feat, target

        kept_idx = np.where(keep)[0]
        # Sort by IoU descending; lower-IoU masks are overlaid LAST so higher-IoU wins.
        order = kept_idx[np.argsort(-iou_scores[kept_idx])][: self.max_masks_per_img]

        # Build per-pixel target at full resolution, then downsample to (h, w).
        full_h, full_w = masks.shape[1:]
        target_full = np.full((full_h, full_w), -1, dtype=np.int32)
        # Iterate from LOW IoU to HIGH IoU so higher-IoU masks overwrite.
        for k in reversed(order):
            cls = int(class_labels[k])
            if cls < 0 or cls >= 14:
                continue
            target_full[masks[k]] = cls

        # Downsample to feature resolution (h, w) via nearest-neighbour.
        # Use scipy or torch interpolate — torch is faster + already imported.
        target_t = torch.from_numpy(target_full).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
        target_t = F.interpolate(target_t, size=(h, w), mode="nearest").squeeze(0).squeeze(0).long()
        return feat, target_t


def _collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = torch.stack([b[0] for b in batch], dim=0)  # (B, C, H, W)
    targets = torch.stack([b[1] for b in batch], dim=0)  # (B, H, W)
    return feats, targets


# ───────────────────────────── Trainer ──────────────────────────────────────


def train_one_epoch(
    adapter: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str,
    epoch_idx: int,
    log_every: int = 100,
) -> dict:
    adapter.train()
    total_loss = 0.0
    total_pixels = 0
    sup_pixels = 0
    n_batches = 0
    pbar = tqdm(loader, desc=f"epoch {epoch_idx}", leave=False)
    for batch_idx, (feats, targets) in enumerate(pbar):
        feats = feats.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = adapter(feats)  # (B, 14, H, W)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)

        if torch.isfinite(loss):
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1

        sup_pixels += int((targets != -1).sum().item())
        total_pixels += int(targets.numel())
        if batch_idx % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", sup=f"{sup_pixels/max(1,total_pixels)*100:.1f}%")

    avg_loss = total_loss / max(1, n_batches)
    sup_pct = sup_pixels / max(1, total_pixels) * 100.0
    return {"loss": avg_loss, "sup_pct": sup_pct, "n_batches": n_batches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_cache", required=True, type=Path,
                        help="Directory with cached P4 features (.pt files).")
    parser.add_argument("--sam3_root", required=True, type=Path,
                        help="SAM3 train masks root (e.g., sam_fine_masks_sam3/train).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for checkpoints + log.")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--in_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_sam3_classes", default=14, type=int)
    parser.add_argument("--min_iou", default=0.10, type=float)
    parser.add_argument("--max_masks_per_img", default=30, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--save_every", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[INFO] Building dataset from {args.feat_cache} + {args.sam3_root}", flush=True)
    ds = CachedFeaturesSAM3Dataset(
        feat_cache_dir=args.feat_cache,
        sam3_root=args.sam3_root,
        min_iou=args.min_iou,
        max_masks_per_img=args.max_masks_per_img,
    )
    print(f"[INFO] Dataset has {len(ds)} entries.", flush=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
        pin_memory=False,
        drop_last=False,
    )

    print(f"[INFO] Building AuxThingAdapter (in={args.in_dim}, hidden={args.hidden_dim}, "
          f"out={args.num_sam3_classes})", flush=True)
    adapter = AuxThingAdapter(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_sam3_classes=args.num_sam3_classes,
    ).to(args.device)
    print(f"[INFO] Trainable params: {adapter.num_trainable_params():,}", flush=True)

    optim = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=1e-5,
    )

    history = []
    best_loss = float("inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(adapter, loader, optim, args.device, epoch)
        sched.step()
        elapsed = time.time() - t_start
        log_msg = (f"epoch={epoch:03d}  loss={stats['loss']:.4f}  "
                   f"sup_pct={stats['sup_pct']:.1f}  lr={sched.get_last_lr()[0]:.2e}  "
                   f"elapsed={elapsed/60:.1f}m")
        print(log_msg, flush=True)
        history.append({"epoch": epoch, **stats, "lr": sched.get_last_lr()[0],
                        "elapsed_min": elapsed / 60})

        if stats["loss"] < best_loss:
            best_loss = stats["loss"]
            torch.save({
                "epoch": epoch,
                "adapter_state": adapter.state_dict(),
                "args": vars(args),
                "best_loss": best_loss,
            }, args.out / "best.pt")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "adapter_state": adapter.state_dict(),
                "args": vars(args),
                "loss": stats["loss"],
            }, args.out / f"epoch_{epoch:03d}.pt")

    # Final save
    torch.save({
        "epoch": args.epochs,
        "adapter_state": adapter.state_dict(),
        "args": vars(args),
        "loss": history[-1]["loss"],
    }, args.out / "last.pt")

    with (args.out / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(f"[DONE] Best loss: {best_loss:.4f}; total elapsed: {(time.time()-t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
