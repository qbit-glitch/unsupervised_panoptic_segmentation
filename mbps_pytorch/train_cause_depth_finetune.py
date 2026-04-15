#!/usr/bin/env python3
"""Fine-tune CAUSE Segment_TR with DepthG depth correlation loss.

Adds a depth-aware contrastive loss to the existing CAUSE training objective:
  L_total = L_contrastive + L_cluster + λ_depth × L_depth

where L_depth uses DepthG's depth_feature_correlation formulation:
  L_depth = -cd.clamp(0, 0.8) × (dd - shift)
  cd = pairwise cosine similarity of 90D Segment_TR codes
  dd = pairwise cosine similarity of 1D depth maps

Variants:
  B0: λ_depth=0    (baseline, reproduce CAUSE)
  B1: λ_depth>0    (depth correlation loss)
  B2: λ_depth>0 + LHP  (depth + local hidden positive propagation)

Usage:
    # Baseline (reproduce CAUSE fine-tune):
    python mbps_pytorch/train_cause_depth_finetune.py \
        --data_dir /path/to/datasets --lambda_depth 0.0 --device cuda

    # Depth correlation fine-tune:
    python mbps_pytorch/train_cause_depth_finetune.py \
        --data_dir /path/to/datasets --lambda_depth 0.05 --device cuda

    # λ sweep:
    for l in 0.01 0.05 0.1; do
        python mbps_pytorch/train_cause_depth_finetune.py \
            --data_dir /path/to/datasets --lambda_depth $l \
            --output_dir results/cause_depth_ft_lambda${l}
    done
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add CAUSE repo to path
CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── CAUSE Constants ───
TRAIN_RESOLUTION = 322
PATCH_SIZE = 14
DIM = 768
REDUCED_DIM = 90
PROJECTION_DIM = 2048
NUM_CODEBOOK = 2048
N_CLASSES = 27
CONTRASTIVE_TEMP = 0.07
POS_THRESH = 0.3
NEG_THRESH = 0.1
BANK_MAX_SIZE = 100
EMA_MOMENTUM = 0.99
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── DepthG Loss Utilities (ported from modules.py) ───

def norm(t: torch.Tensor) -> torch.Tensor:
    """L2-normalize along channel dim (dim=1)."""
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity. (B,C,H,W) x (B,C,H,W) -> (B,H,W,H',W')."""
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample_coords(batch_size: int, feature_samples: int, device: torch.device) -> torch.Tensor:
    """Generate random sampling coordinates in [-1, 1].

    Returns:
        (B, feature_samples, feature_samples, 2) coords for grid_sample.
    """
    return torch.rand(batch_size, feature_samples, feature_samples, 2,
                      device=device) * 2 - 1


def grid_sample(t: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample from spatial tensor at given coordinates.

    Args:
        t: (B, C, H, W) tensor.
        coords: (B, S, S, 2) normalized coords in [-1, 1].

    Returns:
        (B, C, S, S) sampled values.
    """
    return F.grid_sample(
        t, coords, padding_mode="border", align_corners=True, mode="bilinear",
    )


def depth_correlation_loss(
    code: torch.Tensor,
    depth: torch.Tensor,
    feature_samples: int = 11,
    shift: float = 0.0,
) -> torch.Tensor:
    """Compute DepthG depth-feature correlation loss.

    Loss encourages code correlations to match depth correlations:
    where depth is similar, codes should be similar.

    Args:
        code: (B, 90, 23, 23) Segment_TR features.
        depth: (B, 1, 23, 23) depth maps at patch resolution.
        feature_samples: Grid size for sampling (11 -> 121 pairs).
        shift: Margin parameter.

    Returns:
        Scalar loss.
    """
    B = code.shape[0]
    coords = sample_coords(B, feature_samples, code.device)

    code_sampled = grid_sample(code, coords)    # (B, 90, S, S)
    depth_sampled = grid_sample(depth, coords)  # (B, 1, S, S)

    # Code-code correlation
    cd = tensor_correlation(norm(code_sampled), norm(code_sampled))  # (B, S, S, S, S)

    # Depth-depth correlation
    dd = tensor_correlation(norm(depth_sampled), norm(depth_sampled))  # (B, S, S, S, S)

    # Loss: codes should correlate where depth correlates
    loss = -cd.clamp(0.0, 0.8) * (dd - shift)

    return loss.mean()


# ─── Dataset with Depth ───

class CauseDepthDataset(Dataset):
    """Wraps CAUSE ContrastiveSegDataset, adds depth maps.

    For each training sample, loads the corresponding DepthPro depth map,
    resizes to training resolution, and average-pools to patch grid (23×23).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        cityscapes_root: str,
        depth_subdir: str = "depth_depthpro",
        split: str = "train",
        patch_grid: int = 23,
    ):
        self.base = base_dataset
        self.cityscapes_root = cityscapes_root
        self.depth_subdir = depth_subdir
        self.split = split
        self.patch_grid = patch_grid

        # Build index: map dataset integer index → (city, stem)
        # CroppedDataset uses pre-cropped images stored as {0}.jpg, {1}.jpg...
        # We need to map back to the original Cityscapes image for depth loading.
        # Load the mapping from the cropped dataset's source images.
        self._depth_cache = {}
        self._build_image_index()

    def _build_image_index(self) -> None:
        """Build mapping from crop index to source image for depth loading."""
        img_dir = os.path.join(self.cityscapes_root, "leftImg8bit", self.split)
        self.source_images = []
        for city in sorted(os.listdir(img_dir)):
            city_dir = os.path.join(img_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for fname in sorted(os.listdir(city_dir)):
                if fname.endswith("_leftImg8bit.png"):
                    stem = fname.replace("_leftImg8bit.png", "")
                    self.source_images.append({"city": city, "stem": stem})

        # CroppedDataset with crop_type="five" creates 5 crops per image
        self.crops_per_image = max(1, len(self.base) // max(len(self.source_images), 1))
        logger.info("Depth dataset: %d base samples, %d source images, ~%d crops/img",
                    len(self.base), len(self.source_images), self.crops_per_image)

    def _load_depth_for_index(self, idx: int) -> torch.Tensor:
        """Load and downsample depth for a crop index.

        Returns: (1, patch_grid, patch_grid) tensor.
        """
        # Map crop index to source image
        src_idx = idx // self.crops_per_image
        src_idx = min(src_idx, len(self.source_images) - 1)

        if src_idx in self._depth_cache:
            return self._depth_cache[src_idx]

        entry = self.source_images[src_idx]
        npy_path = os.path.join(
            self.cityscapes_root, self.depth_subdir, self.split,
            entry["city"], f"{entry['stem']}.npy",
        )

        if os.path.isfile(npy_path):
            depth = np.load(npy_path).astype(np.float32)
            depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            depth_patch = F.adaptive_avg_pool2d(
                depth_t, (self.patch_grid, self.patch_grid),
            ).squeeze(0)  # (1, pg, pg)
        else:
            depth_patch = torch.zeros(1, self.patch_grid, self.patch_grid)

        # Cache to avoid repeated disk reads
        if len(self._depth_cache) < 5000:
            self._depth_cache[src_idx] = depth_patch

        return depth_patch

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        depth = self._load_depth_for_index(idx)
        item["depth"] = depth
        return item


# ─── Helper Functions ───

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_cause_args() -> SimpleNamespace:
    return SimpleNamespace(
        dim=DIM, reduced_dim=REDUCED_DIM, projection_dim=PROJECTION_DIM,
        num_codebook=NUM_CODEBOOK, n_classes=N_CLASSES,
        num_queries=23 * 23, crop_size=TRAIN_RESOLUTION, patch_size=PATCH_SIZE,
    )


def ema_init(student: nn.Module, teacher: nn.Module) -> None:
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.copy_(p_s.data)
        p_t.requires_grad = False


def ema_update(student: nn.Module, teacher: nn.Module, lamb: float = 0.99) -> None:
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data = lamb * p_t.data + (1 - lamb) * p_s.data


def patch_cluster_for_device(cluster: Cluster, device: torch.device) -> None:
    """Monkey-patch Cluster bank ops for specified device."""
    def bank_init_device(self):
        self.prime_bank = {}
        start = torch.empty([0, self.projection_dim], device=device)
        for i in range(self.num_codebook):
            self.prime_bank[i] = start

    def bank_compute_device(self):
        bank_vq_feat = torch.empty([0, self.dim], device=device)
        bank_proj_feat_ema = torch.empty([0, self.projection_dim], device=device)
        for key in self.prime_bank.keys():
            num = self.prime_bank[key].shape[0]
            if num == 0:
                continue
            bank_vq_feat = torch.cat(
                [bank_vq_feat, self.codebook[key].unsqueeze(0).repeat(num, 1)], dim=0)
            bank_proj_feat_ema = torch.cat(
                [bank_proj_feat_ema, self.prime_bank[key]], dim=0)
        self.flat_norm_bank_vq_feat = F.normalize(bank_vq_feat, dim=1)
        self.flat_norm_bank_proj_feat_ema = F.normalize(bank_proj_feat_ema, dim=1)

    import types
    cluster.bank_init = types.MethodType(bank_init_device, cluster)
    cluster.bank_compute = types.MethodType(bank_compute_device, cluster)


def save_checkpoint(
    segment: Segment_TR, cluster: Cluster,
    output_dir: str, epoch: int,
) -> None:
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    torch.save(segment.state_dict(), os.path.join(epoch_dir, "segment_tr.pth"))
    torch.save(cluster.state_dict(), os.path.join(epoch_dir, "cluster_tr.pth"))
    logger.info("Saved checkpoint: %s", epoch_dir)


# ─── Training Loop ───

def train_depth_finetune(
    net: nn.Module,
    segment: Segment_TR,
    cluster: Cluster,
    train_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    lambda_depth: float = 0.05,
    lr: float = 1e-5,
    epochs: int = 20,
    val_every: int = 2,
    save_every: int = 5,
    depth_shift: float = 0.0,
    feature_samples: int = 11,
) -> None:
    """Fine-tune Segment_TR with CAUSE losses + depth correlation.

    Args:
        lambda_depth: Weight for depth correlation loss. 0.0 = baseline.
        lr: Learning rate (lower than CAUSE's 5e-5 for fine-tuning).
        depth_shift: Shift parameter in depth correlation loss.
        feature_samples: Grid size for spatial sampling (11 -> 121 pairs).
    """
    logger.info("=== Depth Fine-Tuning ===")
    logger.info("  lambda_depth=%.3f, lr=%.1e, epochs=%d", lambda_depth, lr, epochs)
    logger.info("  depth_shift=%.2f, feature_samples=%d", depth_shift, feature_samples)

    params = (
        list(segment.head.parameters())
        + list(segment.projection_head.parameters())
        + [cluster.cluster_probe]
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        segment.train()
        cluster.train()
        total_nce = 0.0
        total_clust = 0.0
        total_depth = 0.0
        count = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in prog:
            img = batch["img"].to(device)
            depth = batch.get("depth")
            if depth is not None:
                depth = depth.to(device)  # (B, 1, 23, 23)

            # Backbone features (frozen)
            with torch.no_grad():
                feat = net(img)[:, 1:, :]  # (B, 529, 768)

            # Student forward
            seg_feat = segment.head(feat, drop=segment.dropout)
            proj_feat = segment.projection_head(seg_feat)

            # Teacher (EMA) forward
            with torch.no_grad():
                seg_feat_ema = segment.head_ema(feat)
                proj_feat_ema = segment.projection_head_ema(seg_feat_ema)

            cluster.bank_compute()

            # CAUSE contrastive loss
            loss_contrastive = cluster.contrastive_ema_with_codebook_bank(
                feat, proj_feat, proj_feat_ema,
                temp=CONTRASTIVE_TEMP,
                pos_thresh=POS_THRESH,
                neg_thresh=NEG_THRESH,
            )

            # CAUSE cluster loss
            loss_cluster, _ = cluster.forward_centroid(seg_feat_ema)

            # Depth correlation loss
            loss_depth = torch.tensor(0.0, device=device)
            if lambda_depth > 0 and depth is not None:
                # Transform 90D features to spatial: (B, 90, 23, 23)
                code_spatial = transform(seg_feat_ema)
                loss_depth = depth_correlation_loss(
                    code_spatial, depth,
                    feature_samples=feature_samples,
                    shift=depth_shift,
                )

            loss = loss_contrastive + loss_cluster + lambda_depth * loss_depth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            ema_update(segment.head, segment.head_ema, lamb=EMA_MOMENTUM)
            ema_update(segment.projection_head, segment.projection_head_ema,
                       lamb=EMA_MOMENTUM)

            # Bank update
            with torch.no_grad():
                cluster.bank_update(feat, proj_feat_ema, max_num=BANK_MAX_SIZE)

            total_nce += loss_contrastive.item()
            total_clust += loss_cluster.item()
            total_depth += loss_depth.item()
            count += 1
            prog.set_postfix(
                nce=f"{loss_contrastive.item():.3f}",
                cl=f"{loss_cluster.item():.3f}",
                dp=f"{loss_depth.item():.4f}",
            )

        avg_nce = total_nce / max(count, 1)
        avg_clust = total_clust / max(count, 1)
        avg_depth = total_depth / max(count, 1)
        logger.info(
            "Epoch %d: nce=%.4f, cluster=%.4f, depth=%.4f (λ*depth=%.4f)",
            epoch + 1, avg_nce, avg_clust, avg_depth, lambda_depth * avg_depth,
        )

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint(segment, cluster, output_dir, epoch + 1)

        total_loss = avg_nce + avg_clust + lambda_depth * avg_depth
        if total_loss < best_loss:
            best_loss = total_loss
            save_checkpoint(segment, cluster, output_dir, epoch=0)
            logger.info("  New best (loss=%.4f) — saved as epoch_000", best_loss)


# ─── Main ───

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune CAUSE Segment_TR with depth correlation loss",
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root datasets dir (contains cityscapes/)")
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="CAUSE pretrained checkpoint dir")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_depth", type=float, default=0.05)
    parser.add_argument("--depth_shift", type=float, default=0.0)
    parser.add_argument("--feature_samples", type=int, default=11)
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(
            Path(__file__).resolve().parent.parent / "refs" / "cause"
        )

    cityscapes_root = os.path.join(args.data_dir, "cityscapes")

    # Load backbone
    logger.info("Loading DINOv2 backbone...")
    backbone_path = os.path.join(args.checkpoint_dir, "checkpoint", "dinov2_vit_base_14.pth")
    net = dinov2_vit_base_14()
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=False)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False

    # Load pretrained Segment_TR + Cluster
    logger.info("Loading pretrained CAUSE Segment_TR...")
    cause_args = build_cause_args()
    segment = Segment_TR(cause_args).to(device)
    cluster = Cluster(cause_args).to(device)

    seg_path = os.path.join(
        args.checkpoint_dir, "CAUSE", "cityscapes",
        "dinov2_vit_base_14", "2048", "segment_tr.pth",
    )
    segment.load_state_dict(
        torch.load(seg_path, map_location="cpu", weights_only=True), strict=False,
    )

    # Load codebook
    mod_path = os.path.join(
        args.checkpoint_dir, "CAUSE", "cityscapes", "modularity",
        "dinov2_vit_base_14", "2048", "modular.npy",
    )
    cb = torch.from_numpy(np.load(mod_path)).to(device)
    cluster.codebook.data = cb
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    # Init EMA from student
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)

    # Patch cluster for device
    patch_cluster_for_device(cluster, device)
    cluster.bank_init()

    # Build dataset
    logger.info("Building training dataset...")
    from torchvision import transforms as T

    # Mock pydensecrf if not installed — CAUSE utils.py imports it at module level
    # but we don't use CRF during training
    if "pydensecrf" not in sys.modules:
        import types
        mock_crf = types.ModuleType("pydensecrf")
        mock_crf.densecrf = types.ModuleType("pydensecrf.densecrf")
        mock_crf.utils = types.ModuleType("pydensecrf.utils")
        sys.modules["pydensecrf"] = mock_crf
        sys.modules["pydensecrf.densecrf"] = mock_crf.densecrf
        sys.modules["pydensecrf.utils"] = mock_crf.utils

    from loader.dataloader import ContrastiveSegDataset

    img_transform = T.Compose([
        T.Resize(TRAIN_RESOLUTION, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(TRAIN_RESOLUTION),
        T.ToTensor(),
    ])
    label_transform = T.Compose([
        T.Resize(TRAIN_RESOLUTION, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(TRAIN_RESOLUTION),
        T.ToTensor(),
    ])

    base_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir,
        dataset_name="cityscapes",
        crop_type="five",
        image_set="train",
        transform=img_transform,
        target_transform=label_transform,
    )

    train_dataset = CauseDepthDataset(
        base_dataset=base_dataset,
        cityscapes_root=cityscapes_root,
        depth_subdir=args.depth_subdir,
        split="train",
        patch_grid=23,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info("Dataset: %d samples, batch_size=%d", len(train_dataset), args.batch_size)

    # Train
    train_depth_finetune(
        net, segment, cluster, train_loader, device, args.output_dir,
        lambda_depth=args.lambda_depth,
        lr=args.lr,
        epochs=args.epochs,
        val_every=args.val_every,
        save_every=args.save_every,
        depth_shift=args.depth_shift,
        feature_samples=args.feature_samples,
    )

    logger.info("Done! Checkpoints at: %s", args.output_dir)
    logger.info(
        "\nNext: generate pseudo-labels with fine-tuned Segment_TR:\n"
        "  python mbps_pytorch/generate_depth_overclustered_semantics.py \\\n"
        "    --cityscapes_root %s --split val \\\n"
        "    --checkpoint_dir %s/epoch_000 \\\n"
        "    --variant none --k 300",
        cityscapes_root, args.output_dir,
    )


if __name__ == "__main__":
    main()
