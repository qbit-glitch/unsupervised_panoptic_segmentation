#!/usr/bin/env python3
"""Train a self-supervised depth edge detector.

Generates edge labels via multi-threshold Sobel consensus on DA3 depth:
  - Pixel is "true edge" if edge at tau_high (0.05)
  - Pixel is "non-edge" if NOT edge at tau_low (0.02)
  - Pixels between thresholds are ignored (ambiguous)

Input: depth (1ch) + Sobel gx,gy (2ch) + PCA-reduced DINOv2 (64ch) = 67ch
Architecture: lightweight 4-layer ConvNet (~150K params)
Output: per-pixel edge probability

Usage:
    python mbps_pytorch/train_learned_edge.py \
        --cityscapes_root /path/to/cityscapes \
        --depth_subdir depth_dav3 \
        --epochs 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from instance_methods.utils import load_features, upsample_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WORK_H, WORK_W = 512, 1024


class EdgePredictor(nn.Module):
    """Lightweight ConvNet for per-pixel edge prediction.

    Args:
        in_channels: number of input channels.
    """

    def __init__(self, in_channels: int = 67):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def discover_files(
    cs_root: Path,
    split: str,
    semantic_subdir: str,
    depth_subdir: str,
    feature_subdir: str,
) -> list:
    """Find matching (semantic, depth, feature) tuples."""
    sem_dir = cs_root / semantic_subdir / split
    depth_dir = cs_root / depth_subdir / split
    feat_dir = cs_root / feature_subdir / split
    files = []

    if not sem_dir.exists():
        logger.error(f"Semantic dir not found: {sem_dir}")
        return files

    for city_dir in sorted(sem_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        for sem_path in sorted(city_dir.glob("*.png")):
            stem = sem_path.stem
            # Try: exact stem, strip _leftImg8bit, add _leftImg8bit
            candidates = [
                stem,
                stem.replace("_leftImg8bit", ""),
                f"{stem}_leftImg8bit",
            ]
            depth_path = None
            for c in candidates:
                p = depth_dir / city / f"{c}.npy"
                if p.exists():
                    depth_path = p
                    break
            feat_path = None
            for c in candidates:
                p = feat_dir / city / f"{c}.npy"
                if p.exists():
                    feat_path = p
                    break
            if depth_path is not None and feat_path is not None:
                files.append((str(sem_path), str(depth_path), str(feat_path)))

    return files


def generate_edge_labels(
    depth: np.ndarray,
    tau_low: float = 0.02,
    tau_high: float = 0.05,
) -> tuple:
    """Generate self-supervised edge labels via multi-threshold consensus.

    Args:
        depth: (H, W) float32 depth map.
        tau_low: lower threshold (below this = non-edge).
        tau_high: upper threshold (above this = edge).

    Returns:
        labels: (H, W) float32 — 1.0 for edge, 0.0 for non-edge.
        mask: (H, W) bool — True for pixels with confident labels.
    """
    depth_f64 = depth.astype(np.float64)
    depth_smooth = gaussian_filter(depth_f64, sigma=1.0)
    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Confident edges: gradient > tau_high
    is_edge = grad_mag > tau_high
    # Confident non-edges: gradient < tau_low
    is_non_edge = grad_mag < tau_low
    # Ambiguous: between tau_low and tau_high -> ignored

    labels = np.zeros_like(depth, dtype=np.float32)
    labels[is_edge] = 1.0

    mask = is_edge | is_non_edge
    return labels, mask


def prepare_input(
    depth: np.ndarray,
    features: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Prepare multi-channel input for edge network.

    Returns:
        channels: (C_in, H, W) float32 array.
    """
    H, W = depth.shape

    # Depth channel
    depth_ch = depth.astype(np.float32)[np.newaxis]

    # Sobel gradient channels
    depth_f64 = depth.astype(np.float64)
    gx = sobel(depth_f64, axis=1).astype(np.float32)
    gy = sobel(depth_f64, axis=0).astype(np.float32)
    grad_ch = np.stack([gx, gy])

    # PCA-reduced features upsampled to (H, W)
    fh, fw, C = features.shape
    flat = features.reshape(-1, C)
    reduced = pca.transform(flat).reshape(fh, fw, -1)
    feat_up = upsample_features(reduced, target_h=H, target_w=W)
    feat_ch = feat_up.transpose(2, 0, 1)

    return np.concatenate([depth_ch, grad_ch, feat_ch], axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Train self-supervised depth edge detector"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--depth_subdir", type=str, default="depth_dav3")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_raw_dinov3_k80")
    parser.add_argument("--feature_subdir", type=str,
                        default="dinov2_features")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--tau_low", type=float, default=0.02)
    parser.add_argument("--tau_high", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/learned_edge")
    parser.add_argument("--max_train_images", type=int, default=200)
    parser.add_argument("--max_val_images", type=int, default=50)
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Discover files
    train_files = discover_files(
        cs_root, args.split, args.semantic_subdir,
        args.depth_subdir, args.feature_subdir,
    )
    val_files = discover_files(
        cs_root, args.val_split, args.semantic_subdir,
        args.depth_subdir, args.feature_subdir,
    )

    if args.max_train_images:
        train_files = train_files[:args.max_train_images]
    if args.max_val_images:
        val_files = val_files[:args.max_val_images]

    logger.info(f"Train images: {len(train_files)}, Val images: {len(val_files)}")

    # Fit PCA on a subset of training features
    logger.info("Fitting PCA on training features...")
    pca_features = []
    for _, _, feat_path in train_files[:50]:
        feats = load_features(feat_path)
        pca_features.append(feats.reshape(-1, feats.shape[-1]))
    all_feats = np.concatenate(pca_features, axis=0)
    pca = PCA(n_components=args.pca_dim)
    pca.fit(all_feats)
    logger.info(f"PCA fitted: {all_feats.shape} -> {args.pca_dim}D, "
                f"variance retained: {pca.explained_variance_ratio_.sum():.3f}")

    # Save PCA
    np.savez(
        output_dir / "pca.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
    )

    # Model
    in_channels = 1 + 2 + args.pca_dim  # depth + sobel_gx,gy + pca features
    model = EdgePredictor(in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"EdgePredictor: {n_params:,} params, {in_channels} input channels")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for sem_path, depth_path, feat_path in tqdm(
            train_files, desc=f"Epoch {epoch}/{args.epochs} [train]",
            leave=False,
        ):
            depth_map = np.load(depth_path).astype(np.float32)
            if depth_map.shape != (WORK_H, WORK_W):
                depth_map = np.array(
                    Image.fromarray(depth_map).resize(
                        (WORK_W, WORK_H), Image.BILINEAR
                    )
                )

            features = load_features(feat_path)

            # Generate labels
            labels, label_mask = generate_edge_labels(
                depth_map, tau_low=args.tau_low, tau_high=args.tau_high
            )

            if label_mask.sum() < 100:
                continue

            # Prepare input
            input_np = prepare_input(depth_map, features, pca)
            input_t = torch.from_numpy(input_np).unsqueeze(0).to(device)
            labels_t = torch.from_numpy(labels).unsqueeze(0).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(label_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            # Forward
            logits = model(input_t)  # (1, 1, H, W)

            # Class-balanced BCE
            n_pos = float(labels[label_mask].sum())
            n_neg = float(label_mask.sum()) - n_pos
            if n_pos > 0 and n_neg > 0:
                pos_weight = torch.tensor([n_neg / n_pos],
                                          dtype=torch.float32, device=device)
            else:
                pos_weight = torch.ones(1, dtype=torch.float32, device=device)

            loss_map = nn.functional.binary_cross_entropy_with_logits(
                logits, labels_t, pos_weight=pos_weight, reduction="none"
            )
            loss = (loss_map * mask_t).sum() / (mask_t.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_train += 1

        scheduler.step()
        avg_train = train_loss_sum / max(n_train, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for sem_path, depth_path, feat_path in tqdm(
                val_files, desc=f"Epoch {epoch}/{args.epochs} [val]",
                leave=False,
            ):
                depth_map = np.load(depth_path).astype(np.float32)
                if depth_map.shape != (WORK_H, WORK_W):
                    depth_map = np.array(
                        Image.fromarray(depth_map).resize(
                            (WORK_W, WORK_H), Image.BILINEAR
                        )
                    )

                features = load_features(feat_path)
                labels, label_mask = generate_edge_labels(
                    depth_map, tau_low=args.tau_low, tau_high=args.tau_high
                )

                if label_mask.sum() < 100:
                    continue

                input_np = prepare_input(depth_map, features, pca)
                input_t = torch.from_numpy(input_np).unsqueeze(0).to(device)
                labels_t = torch.from_numpy(labels).unsqueeze(0).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(label_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

                logits = model(input_t)
                loss_map = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels_t, reduction="none"
                )
                loss = (loss_map * mask_t).sum() / (mask_t.sum() + 1e-8)
                val_loss_sum += loss.item()
                n_val += 1

        avg_val = val_loss_sum / max(n_val, 1)

        logger.info(
            f"Epoch {epoch:3d} | train_loss={avg_train:.4f} "
            f"val_loss={avg_val:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "config": {
                    "in_channels": in_channels,
                    "pca_dim": args.pca_dim,
                    "tau_low": args.tau_low,
                    "tau_high": args.tau_high,
                },
            }, output_dir / "best.pth")
            logger.info(f"  -> Saved best model (epoch {epoch})")

    logger.info(f"\nTraining complete. Best epoch: {best_epoch}, "
                f"val_loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint: {output_dir / 'best.pth'}")
    logger.info(f"PCA: {output_dir / 'pca.npz'}")


if __name__ == "__main__":
    main()
