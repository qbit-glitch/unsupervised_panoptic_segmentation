#!/usr/bin/env python3
"""Train Frozen-Feature Depth Adapter (DCFA).

Loads pre-extracted frozen CAUSE 90D codes + depth patches, trains a tiny
adapter MLP with skip connection to produce depth-aware codes.

Loss = L_depth(adjusted, depth) + lambda_preserve * MSE(adjusted, original)
     + lambda_cluster * L_contrastive(adjusted, centroids)  [optional]

Supports multiple adapter architectures via --adapter_type:
    v3          Original DCFA (MLP, default)
    film        B1: FiLM conditioning
    cross_attn  A1: Cross-attention with DINOv2 768D
    deep        B2: 4-layer bottleneck MLP
    window_attn B3: Local 3x3 window attention
    x           DCFA-X: FiLM + cross-attention + fusion

Usage:
    # V3 baseline:
    python mbps_pytorch/train_depth_adapter.py \
        --cityscapes_root /path/to/cityscapes --epochs 20 --lambda_preserve 20.0

    # A1: DINOv2 cross-attention:
    python mbps_pytorch/train_depth_adapter.py \
        --cityscapes_root /path/to/cityscapes --adapter_type cross_attn --use_dino768

    # DCFA-X combined:
    python mbps_pytorch/train_depth_adapter.py \
        --cityscapes_root /path/to/cityscapes --adapter_type x \
        --use_dino768 --use_normals --lambda_cluster 0.1 \
        --centroids_path /path/to/kmeans_centroids.npz
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.models.semantic.depth_adapter import (
    DepthAdapter,
    sinusoidal_depth_encode,
)
from mbps_pytorch.models.semantic.depth_adapter_v2 import create_adapter
from mbps_pytorch.models.semantic.stego_loss import (
    cluster_assignment_loss,
    contrastive_cluster_loss,
    depth_guided_correlation_loss,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PreextractedCodesDataset(Dataset):
    """Dataset loading pre-extracted CAUSE 90D codes + depth + optional extras.

    Expects files at:
        {codes_dir}/{split}/{city}/{stem}_codes.npy    -> (ph, pw, 90)
        {codes_dir}/{split}/{city}/{stem}_depth.npy    -> (ph, pw)
        {codes_dir}/{split}/{city}/{stem}_dino768.npy  -> (ph, pw, 768) [optional]
        {codes_dir}/{split}/{city}/{stem}_normals.npy  -> (ph, pw, 3) [optional]
    """

    def __init__(
        self,
        codes_dir: str,
        split: str,
        load_dino768: bool = False,
        load_normals: bool = False,
        load_gradients: bool = False,
    ) -> None:
        self.load_dino768 = load_dino768
        self.load_normals = load_normals
        self.load_gradients = load_gradients
        self.files: List[Dict[str, str]] = []
        split_dir = os.path.join(codes_dir, split)
        for city in sorted(os.listdir(split_dir)):
            city_dir = os.path.join(split_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for fname in sorted(os.listdir(city_dir)):
                if fname.endswith("_codes.npy"):
                    stem = fname.replace("_codes.npy", "")
                    codes_path = os.path.join(city_dir, fname)
                    depth_path = os.path.join(city_dir, f"{stem}_depth.npy")
                    if not os.path.isfile(depth_path):
                        continue
                    entry = {"codes": codes_path, "depth": depth_path}
                    if load_dino768:
                        dino_path = os.path.join(city_dir, f"{stem}_dino768.npy")
                        if not os.path.isfile(dino_path):
                            continue
                        entry["dino768"] = dino_path
                    if load_normals:
                        normals_path = os.path.join(city_dir, f"{stem}_normals.npy")
                        if not os.path.isfile(normals_path):
                            continue
                        entry["normals"] = normals_path
                    self.files.append(entry)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load codes, depth, and optional extras for one image.

        Returns:
            Dict with keys: codes (N, 90), depth (N,), and optional
            dino768 (N, 768), normals (N, 3), gradients (N, 3),
            spatial_shape (2,).
        """
        entry = self.files[idx]
        codes = np.load(entry["codes"]).astype(np.float32)  # (ph, pw, 90)
        depth = np.load(entry["depth"]).astype(np.float32)  # (ph, pw)
        ph, pw = depth.shape

        result: Dict[str, torch.Tensor] = {
            "codes": torch.from_numpy(codes.reshape(-1, codes.shape[-1])),
            "depth": torch.from_numpy(depth.reshape(-1)),
            "spatial_shape": torch.tensor([ph, pw], dtype=torch.long),
        }

        if self.load_dino768:
            dino = np.load(entry["dino768"]).astype(np.float32)  # (ph, pw, 768)
            result["dino768"] = torch.from_numpy(dino.reshape(-1, dino.shape[-1]))

        if self.load_normals:
            normals = np.load(entry["normals"]).astype(np.float32)  # (ph, pw, 3)
            result["normals"] = torch.from_numpy(normals.reshape(-1, 3))

        if self.load_gradients:
            from scipy.ndimage import sobel
            gx = sobel(depth, axis=1).astype(np.float32)
            gy = sobel(depth, axis=0).astype(np.float32)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            grads = np.stack([gx.ravel(), gy.ravel(), mag.ravel()], axis=-1)
            result["gradients"] = torch.from_numpy(grads)

        return result


def run_epoch(
    adapter: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_preserve: float,
    use_sinusoidal: bool = False,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
    lambda_cluster: float = 0.0,
    centroids: Optional[torch.Tensor] = None,
    cross_image_mining: bool = False,
    loss_type: str = "depth_corr",
) -> Dict[str, float]:
    """Run one training or validation epoch.

    Args:
        loss_type: "depth_corr" (default) or "cluster_aware".

    Returns:
        Dict with avg loss values.
    """
    is_train = optimizer is not None
    adapter.train() if is_train else adapter.eval()
    totals: Dict[str, float] = {
        "loss": 0.0, "depth": 0.0, "preserve": 0.0,
        "cluster": 0.0, "drift": 0.0,
    }
    count = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for step, batch in enumerate(loader):
            codes = batch["codes"].to(device)      # (B, N, 90)
            depth_raw = batch["depth"].to(device)   # (B, N)

            # Encode depth for adapter input
            if use_sinusoidal:
                depth_input = sinusoidal_depth_encode(depth_raw)  # (B, N, 16)
            else:
                depth_input = depth_raw

            # Prepare optional adapter kwargs
            adapter_kwargs: Dict[str, torch.Tensor] = {}
            if "dino768" in batch:
                adapter_kwargs["dino768"] = batch["dino768"].to(device)
            if "normals" in batch:
                adapter_kwargs["normals"] = batch["normals"].to(device)
            if "spatial_shape" in batch:
                ss = batch["spatial_shape"][0]  # same for all in batch
                adapter_kwargs["spatial_shape"] = (ss[0].item(), ss[1].item())

            # Concatenate normals/gradients to depth_input for adapters that
            # expect them via input concat (v3, film, deep, window_attn).
            # DCFA-X handles normals internally via adapter_kwargs.
            is_x = hasattr(adapter, "geo_dim")  # DepthAdapterX has geo_dim attr
            geo_extras = []
            if "normals" in batch and not is_x:
                geo_extras.append(batch["normals"].to(device))
            if "gradients" in batch:
                geo_extras.append(batch["gradients"].to(device))
            if geo_extras and depth_input.dim() == 3:
                depth_input = torch.cat([depth_input] + geo_extras, dim=-1)

            if is_train:
                optimizer.zero_grad()

            adjusted = adapter(codes, depth_input, **adapter_kwargs)

            l_preserve = F.mse_loss(adjusted, codes)

            if loss_type == "cluster_aware" and centroids is not None:
                # Cluster-aware: cross-entropy toward centroid assignments
                l_depth = cluster_assignment_loss(
                    adjusted, centroids, codes, temperature=0.07,
                )
                loss = l_depth + lambda_preserve * l_preserve
                l_cluster = torch.tensor(0.0, device=device)
            else:
                # Default: depth correlation loss
                if cross_image_mining and codes.shape[0] > 1:
                    l_depth = _cross_image_correlation_loss(adjusted, depth_raw)
                else:
                    l_depth = depth_guided_correlation_loss(
                        adjusted, depth_raw, sigma_d=0.5, num_pairs=1024,
                    )
                loss = l_depth + lambda_preserve * l_preserve

                # Optional contrastive cluster loss
                l_cluster = torch.tensor(0.0, device=device)
                if lambda_cluster > 0 and centroids is not None:
                    l_cluster = contrastive_cluster_loss(
                        adjusted, centroids, temperature=0.1, num_pairs=512,
                    )
                    loss = loss + lambda_cluster * l_cluster

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()

            totals["loss"] += loss.item()
            totals["depth"] += l_depth.item()
            totals["preserve"] += l_preserve.item()
            totals["cluster"] += l_cluster.item()
            totals["drift"] += (adjusted - codes).norm(dim=-1).mean().item()
            count += 1

            if is_train and step % 50 == 0:
                logger.info(
                    "Epoch %d step %d/%d | loss=%.4f (depth=%.4f, preserve=%.6f, "
                    "cluster=%.4f) drift=%.4f",
                    epoch, step, len(loader), loss.item(), l_depth.item(),
                    l_preserve.item(), l_cluster.item(),
                    (adjusted - codes).norm(dim=-1).mean().item(),
                )

    n = max(count, 1)
    return {k: v / n for k, v in totals.items()}


def _cross_image_correlation_loss(
    codes: torch.Tensor, depth: torch.Tensor,
    sigma_d: float = 0.5, num_pairs: int = 1024,
) -> torch.Tensor:
    """Correlation loss combining within-image attraction and cross-image repulsion.

    Within-image: standard depth-guided correlation (pull depth-similar together).
    Cross-image: push features from different scenes apart (uniform weight).
    """
    b, n, d = codes.shape
    device = codes.device
    if b < 2:
        return depth_guided_correlation_loss(codes, depth, sigma_d, num_pairs)

    # Within-image: standard depth-guided attraction
    l_within = depth_guided_correlation_loss(codes, depth, sigma_d, num_pairs // 2)

    # Cross-image: repulsion (push apart features from different scenes)
    l_cross = torch.tensor(0.0, device=device)
    for i in range(b):
        j = (i + 1) % b
        idx_i = torch.randint(0, n, (num_pairs // 2,), device=device)
        idx_j = torch.randint(0, n, (num_pairs // 2,), device=device)

        codes_i = codes[i][idx_i]
        codes_j = codes[j][idx_j]

        # Cross-image pairs should push apart: penalize high similarity
        cos_sim = F.cosine_similarity(codes_i, codes_j, dim=-1)
        l_cross = l_cross + (cos_sim ** 2).mean()

    l_cross = l_cross / b
    return l_within + 0.5 * l_cross


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCFA Depth Adapter")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--codes_subdir", type=str, default="cause_codes_90d")
    parser.add_argument("--output_dir", type=str, default="results/depth_adapter")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_preserve", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--depth_dim", type=int, default=1,
                        help="1 for raw depth, 16 for sinusoidal encoding")
    parser.add_argument("--seed", type=int, default=42)
    # DCFA v2 arguments
    parser.add_argument(
        "--adapter_type", type=str, default="v3",
        choices=["v3", "film", "cross_attn", "deep", "window_attn", "x"],
        help="Adapter architecture variant.",
    )
    parser.add_argument("--use_dino768", action="store_true",
                        help="Load DINOv2 768D features (required for cross_attn, x).")
    parser.add_argument("--use_normals", action="store_true",
                        help="Load surface normals from depth.")
    parser.add_argument("--use_gradients", action="store_true",
                        help="Compute Sobel depth gradients on-the-fly.")
    parser.add_argument("--lambda_cluster", type=float, default=0.0,
                        help="Weight for contrastive cluster loss (0=disabled).")
    parser.add_argument("--centroids_path", type=str, default=None,
                        help="Path to kmeans_centroids.npz for contrastive loss.")
    parser.add_argument("--cross_image_mining", action="store_true",
                        help="Sample correlation pairs across images in batch.")
    parser.add_argument(
        "--loss_type", type=str, default="depth_corr",
        choices=["depth_corr", "cluster_aware"],
        help="Primary loss: depth_corr (default) or cluster_aware (CE toward centroids).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Device: %s", device)
    use_sinusoidal = args.depth_dim == 16
    logger.info(
        "adapter=%s lp=%.1f lr=%.1e epochs=%d hidden=%d layers=%d depth_dim=%d "
        "dino768=%s normals=%s gradients=%s lambda_cluster=%.2f",
        args.adapter_type, args.lambda_preserve, args.lr, args.epochs,
        args.hidden_dim, args.num_layers, args.depth_dim,
        args.use_dino768, args.use_normals, args.use_gradients,
        args.lambda_cluster,
    )

    codes_dir = os.path.join(args.cityscapes_root, args.codes_subdir)

    # Datasets
    train_dataset = PreextractedCodesDataset(
        codes_dir, "train",
        load_dino768=args.use_dino768,
        load_normals=args.use_normals,
        load_gradients=args.use_gradients,
    )
    val_dataset = PreextractedCodesDataset(
        codes_dir, "val",
        load_dino768=args.use_dino768,
        load_normals=args.use_normals,
        load_gradients=args.use_gradients,
    )
    logger.info("Train: %d images, Val: %d images",
                len(train_dataset), len(val_dataset))

    nw = 0 if device.type == "mps" else 4
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    # Compute geo_dim for FiLM/concat variants
    geo_dim = 0
    if args.use_normals:
        geo_dim += 3
    if args.use_gradients:
        geo_dim += 3

    # Create adapter via factory
    # For v3/deep/window_attn/cross_attn: geo extras (normals, gradients) are
    # concatenated to depth_input, so depth_dim must include them.
    # For film: geo_dim passed separately (film's constructor adds it internally).
    # For x: normals passed via kwargs (handled internally).
    concat_geo = args.adapter_type in ("v3", "deep", "window_attn", "cross_attn")
    effective_depth_dim = args.depth_dim + (geo_dim if concat_geo else 0)

    adapter_kwargs: Dict = {"code_dim": 90, "depth_dim": effective_depth_dim}
    if args.adapter_type == "v3":
        adapter_kwargs.update(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.adapter_type == "film":
        adapter_kwargs["depth_dim"] = args.depth_dim  # film handles geo internally
        adapter_kwargs.update(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, geo_dim=geo_dim,
        )
    elif args.adapter_type == "cross_attn":
        adapter_kwargs.update(dino_dim=768, d_attn=64, hidden_dim=256)
    elif args.adapter_type == "deep":
        adapter_kwargs.update(hidden_dim=args.hidden_dim, bottleneck_dim=64)
    elif args.adapter_type == "window_attn":
        adapter_kwargs.update(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.adapter_type == "x":
        adapter_kwargs["depth_dim"] = args.depth_dim  # x handles geo internally
        adapter_kwargs.update(geo_dim=geo_dim, dino_dim=768, d_attn=64, fusion_hidden=256)

    adapter = create_adapter(args.adapter_type, **adapter_kwargs).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    logger.info("%s adapter: %d parameters", args.adapter_type.upper(), n_params)

    # Load centroids for contrastive loss
    centroids = None
    needs_centroids = args.lambda_cluster > 0 or args.loss_type == "cluster_aware"
    if needs_centroids and args.centroids_path:
        data = np.load(args.centroids_path)
        key = "centroids" if "centroids" in data else list(data.keys())[0]
        raw = data[key].astype(np.float32)
        # Centroids may be (K, 106) from k-means on [codes(90)+depth(16)].
        # Slice to code_dim=90 so shapes match adjusted codes.
        code_dim = 90
        if raw.shape[1] > code_dim:
            logger.info("Slicing centroids %s to first %d dims (code space)", raw.shape, code_dim)
            raw = raw[:, :code_dim]
        centroids = torch.from_numpy(raw).to(device)
        logger.info("Loaded centroids: %s from %s", centroids.shape, args.centroids_path)

    if args.loss_type == "cluster_aware" and centroids is None:
        parser.error("--loss_type cluster_aware requires --centroids_path")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_ckpt = os.path.join(args.output_dir, "best.pt")
    t0 = time.time()

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            adapter, train_loader, device, args.lambda_preserve,
            use_sinusoidal=use_sinusoidal, optimizer=optimizer, epoch=epoch,
            lambda_cluster=args.lambda_cluster, centroids=centroids,
            cross_image_mining=args.cross_image_mining,
            loss_type=args.loss_type,
        )
        val_metrics = run_epoch(
            adapter, val_loader, device, args.lambda_preserve,
            use_sinusoidal=use_sinusoidal,
            lambda_cluster=args.lambda_cluster, centroids=centroids,
            loss_type=args.loss_type,
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %d/%d | train: loss=%.4f depth=%.4f pres=%.6f clust=%.4f "
            "drift=%.4f | val: loss=%.4f depth=%.4f pres=%.6f clust=%.4f "
            "drift=%.4f | lr=%.6f | %.0fs",
            epoch, args.epochs - 1,
            train_metrics["loss"], train_metrics["depth"],
            train_metrics["preserve"], train_metrics["cluster"],
            train_metrics["drift"],
            val_metrics["loss"], val_metrics["depth"],
            val_metrics["preserve"], val_metrics["cluster"],
            val_metrics["drift"],
            lr_now, time.time() - t0,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "state_dict": adapter.state_dict(),
                "adapter_type": args.adapter_type,
                "adapter_kwargs": adapter_kwargs,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, best_ckpt)
            logger.info("  -> New best val_loss=%.4f, saved to %s", best_val_loss, best_ckpt)

    # Save final
    final_ckpt = os.path.join(args.output_dir, "final.pt")
    torch.save({
        "state_dict": adapter.state_dict(),
        "adapter_type": args.adapter_type,
        "adapter_kwargs": adapter_kwargs,
        "epoch": args.epochs - 1,
        "val_loss": val_metrics["loss"],
    }, final_ckpt)
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    logger.info("Checkpoints: best=%s, final=%s", best_ckpt, final_ckpt)


if __name__ == "__main__":
    main()
