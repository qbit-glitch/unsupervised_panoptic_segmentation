#!/usr/bin/env python3
"""Train DepthG projector on pre-extracted DINOv3 features.

Learns a 90-dim semantic code space using STEGO contrastive loss +
depth-guided correlation loss, then clusters the codes via MiniBatchKMeans.

Usage (train):
    python mbps_pytorch/train_depthg_dinov3.py \
        --cityscapes_root /path/to/cityscapes \
        --epochs 20 --batch_size 8

Usage (generate pseudo-labels from best checkpoint):
    python mbps_pytorch/train_depthg_dinov3.py \
        --cityscapes_root /path/to/cityscapes \
        --generate_labels
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mbps_pytorch.generate_dinov3_kmeans import find_feature_files
from mbps_pytorch.models.semantic.depthg_head import DepthGHead
from mbps_pytorch.models.semantic.stego_loss import (
    depth_guided_correlation_loss,
    stego_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
OUT_H, OUT_W = 512, 1024


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Auto-detect best available device (MPS > CUDA > CPU).

    Returns:
        Selected torch device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DINOv3DepthDataset(Dataset):
    """Dataset loading pre-extracted DINOv3 features and DepthPro depth maps."""

    def __init__(self, files: List[Dict], depth_root: Path, split: str) -> None:
        self.files = files
        self.depth_root = depth_root
        self.split = split

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load features (2048, 768) and depth (2048,) for one image."""
        entry = self.files[idx]

        feat = np.load(str(entry["feat"])).astype(np.float32)  # (2048, 768)
        feat = torch.from_numpy(feat)

        depth_path = self.depth_root / self.split / entry["city"] / f"{entry['stem']}.npy"
        depth = np.load(str(depth_path)).astype(np.float32)  # (512, 1024)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
        depth = depth.squeeze().reshape(-1)  # (2048,)

        return feat, depth


def _compute_loss(
    model: DepthGHead,
    features: torch.Tensor,
    depth: torch.Tensor,
    lambda_depthg: float,
    temperature: float,
    knn_k: int,
    sigma_d: float,
) -> Tuple[torch.Tensor, float, float]:
    """Forward pass + combined loss.

    Args:
        model: DepthG projector.
        features: DINOv3 features (B, N, 768).
        depth: Flattened depth (B, N).
        lambda_depthg: Depth loss weight.
        temperature: STEGO temperature.
        knn_k: KNN neighbors for positive pairs.
        sigma_d: Depth bandwidth.

    Returns:
        Tuple of (total_loss, stego_loss_value, depth_loss_value).
    """
    codes = model(features)
    l_stego = stego_loss(codes, features, temperature=temperature, knn_k=knn_k)
    l_depth = depth_guided_correlation_loss(codes, depth, sigma_d=sigma_d)
    return l_stego + lambda_depthg * l_depth, l_stego.item(), l_depth.item()


def run_epoch(
    model: DepthGHead,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: Dict,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
) -> float:
    """Run one training or validation epoch.

    Args:
        model: DepthG projector.
        loader: Data loader.
        device: Torch device.
        loss_kwargs: Dict with lambda_depthg, temperature, knn_k, sigma_d.
        optimizer: If provided, run training (backprop + clip). None = eval.
        epoch: Current epoch (for logging).

    Returns:
        Average loss for the epoch.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    num_steps = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for step, (features, depth) in enumerate(loader):
            features = features.to(device)
            depth = depth.to(device)

            if is_train:
                optimizer.zero_grad()

            loss, l_s, l_d = _compute_loss(model, features, depth, **loss_kwargs)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_steps += 1

            if is_train and step % 10 == 0:
                logger.info(
                    f"Epoch {epoch} step {step}/{len(loader)} | "
                    f"loss={loss.item():.4f} (stego={l_s:.4f}, depth={l_d:.4f})"
                )

    return total_loss / max(num_steps, 1)


@torch.no_grad()
def extract_all_codes(
    model: DepthGHead,
    files: List[Dict],
    depth_root: Path,
    split: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract L2-normalized codes for all images. Returns (N*2048, code_dim)."""
    model.eval()
    dataset = DINOv3DepthDataset(files, depth_root, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_codes = []
    for features, _ in tqdm(loader, desc=f"Extracting {split} codes"):
        features = features.to(device)
        codes = model(features)  # (B, 2048, code_dim)
        codes = F.normalize(codes, dim=-1)  # L2-normalize
        # Flatten batch: (B, 2048, D) -> (B*2048, D)
        codes = codes.reshape(-1, codes.shape[-1])
        all_codes.append(codes.cpu().numpy())

    return np.concatenate(all_codes, axis=0)


def generate_pseudo_labels(
    model: DepthGHead,
    cityscapes_root: Path,
    feat_subdir: str,
    depth_subdir: str,
    output_subdir: str,
    k: int,
    device: torch.device,
) -> None:
    """Cluster trained DepthG codes via MiniBatchKMeans and save as PNG labels."""
    depth_root = cityscapes_root / depth_subdir
    out_dir = cityscapes_root / output_subdir

    # Extract train codes for k-means fitting
    train_files = find_feature_files(cityscapes_root, "train", feat_subdir)
    logger.info(f"Extracting codes from {len(train_files)} train images")
    train_codes = extract_all_codes(model, train_files, depth_root, "train", device)
    logger.info(f"Train codes shape: {train_codes.shape}")

    # Fit MiniBatchKMeans
    logger.info(f"Fitting MiniBatchKMeans with k={k}")
    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=4096, n_init=5, max_iter=150, random_state=42, verbose=1,
    )
    kmeans.fit(train_codes)
    logger.info(f"K-means inertia: {kmeans.inertia_:.3f}")

    # Save centroids
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / "kmeans_centroids.npz"), centers=kmeans.cluster_centers_)
    logger.info(f"Saved centroids to {out_dir / 'kmeans_centroids.npz'}")

    # Assign clusters for train + val
    for split in ["train", "val"]:
        files = find_feature_files(cityscapes_root, split, feat_subdir)
        logger.info(f"Assigning {split}: {len(files)} images")

        for entry in tqdm(files, desc=f"Labeling {split}"):
            feat = np.load(str(entry["feat"])).astype(np.float32)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)  # (1, 2048, 768)
            codes = model(feat_t)
            codes = F.normalize(codes, dim=-1).squeeze(0).cpu().numpy()  # (2048, code_dim)

            cluster_ids = kmeans.predict(codes).astype(np.uint8)  # (2048,)
            cluster_2d = cluster_ids.reshape(FEAT_H, FEAT_W)

            label_full = np.array(
                Image.fromarray(cluster_2d).resize((OUT_W, OUT_H), Image.NEAREST)
            )

            city_dir = out_dir / split / entry["city"]
            city_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(label_full).save(str(city_dir / f"{entry['stem']}.png"))

    logger.info(f"Pseudo-labels saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train DepthG projector on DINOv3 features with STEGO + depth loss",
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--feat_subdir", type=str, default="dinov3_features")
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--output_subdir", type=str, default="pseudo_semantic_depthg_dinov3_k80")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_depthg", type=float, default=0.3)
    parser.add_argument("--code_dim", type=int, default=90)
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/depthg_dinov3")
    parser.add_argument("--generate_labels", action="store_true",
                        help="Load best checkpoint and generate pseudo-labels (skip training)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit images per split (for smoke testing)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Main entry point for training and label generation."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    cityscapes_root = Path(args.cityscapes_root)
    depth_root = cityscapes_root / args.depth_subdir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = DepthGHead(input_dim=768, hidden_dim=384, code_dim=args.code_dim).to(device)
    logger.info(f"DepthGHead: {sum(p.numel() for p in model.parameters()):,} parameters")

    best_ckpt = ckpt_dir / "best.pt"

    if args.generate_labels:
        if not best_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_ckpt}")
        logger.info(f"Loading checkpoint from {best_ckpt}")
        model.load_state_dict(torch.load(str(best_ckpt), map_location=device, weights_only=True))
        generate_pseudo_labels(
            model, cityscapes_root, args.feat_subdir, args.depth_subdir,
            args.output_subdir, args.k, device,
        )
        return

    # --- Training ---
    train_files = find_feature_files(cityscapes_root, "train", args.feat_subdir)
    val_files = find_feature_files(cityscapes_root, "val", args.feat_subdir)
    if args.max_images is not None:
        train_files = train_files[:args.max_images]
        val_files = val_files[:args.max_images]
    logger.info(f"Train: {len(train_files)} images, Val: {len(val_files)} images")

    train_dataset = DINOv3DepthDataset(train_files, depth_root, "train")
    val_dataset = DINOv3DepthDataset(val_files, depth_root, "val")
    nw = 0 if device.type == "mps" else 4
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_kwargs = {
        "lambda_depthg": args.lambda_depthg,
        "temperature": 0.1,
        "knn_k": 7,
        "sigma_d": 0.5,
    }

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, device, loss_kwargs, optimizer=optimizer, epoch=epoch,
        )
        val_loss = run_epoch(model, val_loader, device, loss_kwargs)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch}/{args.epochs-1} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={lr_now:.6f} "
            f"elapsed={time.time()-t0:.0f}s"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(best_ckpt))
            logger.info(f"  -> New best val_loss={val_loss:.4f}, saved to {best_ckpt}")

    # Save final
    final_ckpt = ckpt_dir / "final.pt"
    torch.save(model.state_dict(), str(final_ckpt))
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Checkpoints: best={best_ckpt}, final={final_ckpt}")


if __name__ == "__main__":
    main()
