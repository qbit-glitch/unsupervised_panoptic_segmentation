#!/usr/bin/env python3
"""
Train CAUSE-TR heads on DINOv3 ViT-L/16 features for Cityscapes.

Shortcut approach (skip STEGO Stage 1):
  Stage 2: Modularity-based codebook learning (~10 epochs)
  Stage 3: TRDecoder segment head + Cluster probe training (~40 epochs)

This retrains CAUSE-TR from scratch using DINOv3 ViT-L/16 (1024-dim, patch 16,
register tokens) instead of the original DINOv2 ViT-B/14 (768-dim, patch 14).

Usage:
    python mbps_pytorch/train_cause_dinov3.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir refs/cause/CAUSE_dinov3 \
        --device mps

    # Stage 2 only (codebook):
    python mbps_pytorch/train_cause_dinov3.py --stage codebook ...

    # Stage 3 only (heads, requires codebook from stage 2):
    python mbps_pytorch/train_cause_dinov3.py --stage heads ...
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Add CAUSE repo to path
CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from modules.segment import Segment_TR
from modules.segment_module import (
    Cluster,
    Decoder,
    TRDecoder,
    HeadSegment,
    ProjectionSegment,
    transform,
    untransform,
    flatten,
    vqt,
    quantize_index,
    cos_distance_matrix,
    cluster_assignment_matrix,
    stochastic_sampling,
    ema_init,
    ema_update,
    reset,
)

# ---- Constants ----
TRAIN_RESOLUTION = 320  # divisible by 16
PATCH_SIZE = 16
NUM_PATCHES_PER_SIDE = TRAIN_RESOLUTION // PATCH_SIZE  # 20
NUM_PATCHES = NUM_PATCHES_PER_SIDE ** 2  # 400
DIM = 1024  # DINOv3 ViT-L/16 feature dim
REDUCED_DIM = 90
PROJECTION_DIM = 2048
NUM_CODEBOOK = 2048
N_CLASSES = 27  # Cityscapes (CAUSE format: labelIDs 7-33)
NUM_REGISTER_TOKENS = 4  # DINOv3 register tokens

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---- Device-aware CAUSE functions (replace .cuda() calls) ----

def get_modularity_matrix_and_edge(x, device):
    """Compute modularity matrix W and edge sum e. Device-aware version."""
    norm = F.normalize(x, dim=2)
    A = (norm @ norm.transpose(2, 1)).clamp(0)
    A = A - A * torch.eye(A.shape[1], device=device)
    d = A.sum(dim=2, keepdims=True)
    e = A.sum(dim=(1, 2), keepdims=True)
    W = A - (d / (e + 1e-10)) @ (d.transpose(2, 1) / (e + 1e-10)) * e
    return W, e


def compute_modularity_loss(codebook, feat, device, temp=0.1, grid=True):
    """Compute modularity-based codebook loss. Device-aware version."""
    feat = feat.detach()
    if grid:
        feat, _ = stochastic_sampling(feat)

    W, e = get_modularity_matrix_and_edge(feat, device)
    C = cluster_assignment_matrix(feat, codebook)

    D = C.transpose(2, 1)
    E = torch.tanh(D.unsqueeze(3) @ D.unsqueeze(2) / temp)
    delta, _ = E.max(dim=1)
    Q = (W / (e + 1e-10)) @ delta

    diag = Q.diagonal(offset=0, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return -trace.mean()


class DeviceAwareCluster(Cluster):
    """Cluster subclass with MPS/CPU compatible bank operations."""

    def __init__(self, args, device):
        super().__init__(args)
        self._device = device

    def bank_init(self):
        self.prime_bank = {}
        start = torch.empty([0, self.projection_dim], device=self._device)
        for i in range(self.num_codebook):
            self.prime_bank[i] = start
        # Initialize empty bank tensors (needed before first bank_compute call)
        self.flat_norm_bank_vq_feat = torch.empty([0, self.dim], device=self._device)
        self.flat_norm_bank_proj_feat_ema = torch.empty([0, self.projection_dim], device=self._device)

    def bank_compute(self):
        bank_vq_feat = torch.empty([0, self.dim], device=self._device)
        bank_proj_feat_ema = torch.empty([0, self.projection_dim], device=self._device)
        for key in self.prime_bank.keys():
            num = self.prime_bank[key].shape[0]
            if num == 0:
                continue
            bank_vq_feat = torch.cat(
                [bank_vq_feat, self.codebook[key].unsqueeze(0).repeat(num, 1)], dim=0
            )
            bank_proj_feat_ema = torch.cat(
                [bank_proj_feat_ema, self.prime_bank[key]], dim=0
            )
        self.flat_norm_bank_vq_feat = F.normalize(bank_vq_feat, dim=1)
        self.flat_norm_bank_proj_feat_ema = F.normalize(bank_proj_feat_ema, dim=1)


# ---- DINOv3 Backbone ----

class DINOv3Backbone(nn.Module):
    """Wraps HuggingFace DINOv3 ViT-L/16 to match CAUSE interface.

    forward(x) returns (B, 1+N, D) where first token is CLS, rest are patches.
    Register tokens are stripped internally.
    """

    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-lvd1689m", device="mps"):
        super().__init__()
        from transformers import AutoModel

        print(f"Loading DINOv3 ViT-L/16 from HuggingFace: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.num_register_tokens = NUM_REGISTER_TOKENS
        self.embed_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        print(f"  embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
              f"registers={self.num_register_tokens}")
        print(f"  Parameters: {sum(p.numel() for p in self.backbone.parameters()) / 1e6:.1f}M (frozen)")

    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x, return_dict=True)
        tokens = outputs.last_hidden_state
        # DINOv3: [CLS, reg1, reg2, reg3, reg4, patch1, patch2, ...]
        cls_token = tokens[:, 0:1, :]
        patch_tokens = tokens[:, 1 + self.num_register_tokens :, :]
        # Return (B, 1+N, D) — compatible with CAUSE's net(img)[:, 1:, :]
        return torch.cat([cls_token, patch_tokens], dim=1)


# ---- Dataset ----

class CityscapesCAUSE(Dataset):
    """Cityscapes dataset for CAUSE training with random crops."""

    def __init__(self, root, split="train", resolution=320, augment=True):
        self.root = Path(root)
        self.resolution = resolution
        self.augment = augment

        img_dir = self.root / "leftImg8bit" / split
        self.image_paths = sorted(img_dir.rglob("*.png"))
        print(f"  {split}: {len(self.image_paths)} images")

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(resolution, scale=(0.5, 1.0),
                                             interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return {"img": img, "ind": idx}


# ---- CAUSE Args namespace ----

def make_cause_args():
    """Create args namespace for CAUSE module initialization."""
    return SimpleNamespace(
        dim=DIM,
        reduced_dim=REDUCED_DIM,
        projection_dim=PROJECTION_DIM,
        num_codebook=NUM_CODEBOOK,
        n_classes=N_CLASSES,
        num_queries=NUM_PATCHES,
    )


# ---- Stage 2: Modularity Codebook ----

def train_codebook(net, cluster, train_loader, device, args):
    """Stage 2: Optimize codebook via modularity maximization."""
    print("\n" + "=" * 60)
    print("  STAGE 2: Modularity-based Codebook Learning")
    print("=" * 60)

    # Only optimize the codebook
    optimizer = torch.optim.Adam([cluster.codebook], lr=args.codebook_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.codebook_epochs * len(train_loader)
    )

    best_loss = float("inf")
    for epoch in range(args.codebook_epochs):
        epoch_loss = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Codebook Epoch {epoch + 1}/{args.codebook_epochs}")

        for batch in pbar:
            img = batch["img"].to(device)

            # Extract features (frozen backbone)
            with torch.no_grad():
                feat = net(img)[:, 1:, :]  # (B, N, 1024)

            # Modularity loss
            loss = compute_modularity_loss(
                cluster.codebook, feat, device, temp=0.1, grid=True
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([cluster.codebook], 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / count
        print(f"  Epoch {epoch + 1}: avg loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save codebook
    codebook_path = os.path.join(args.output_dir, "modular.npy")
    np.save(codebook_path, cluster.codebook.detach().cpu().numpy())
    print(f"\nCodebook saved to {codebook_path}")
    print(f"  Shape: {cluster.codebook.shape}, best loss: {best_loss:.4f}")

    return cluster


# ---- Stage 3: Head + Cluster Training ----

def train_heads(net, segment, cluster, train_loader, device, args):
    """Stage 3: Train TRDecoder segment head + Cluster probe."""
    print("\n" + "=" * 60)
    print("  STAGE 3: TRDecoder + Cluster Probe Training")
    print("=" * 60)

    # Inject frozen codebook into Decoder heads
    cb = cluster.codebook.data.clone()
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    # Initialize EMA from student
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)

    # Parameters to train
    train_params = []
    train_params += list(segment.head.parameters())
    train_params += list(segment.projection_head.parameters())
    train_params += list(segment.linear.parameters())
    train_params.append(cluster.cluster_probe)
    # Note: segment.head_ema and segment.projection_head_ema are EMA-updated, not optimized

    total_trainable = sum(p.numel() for p in train_params if p.requires_grad)
    print(f"  Trainable parameters: {total_trainable / 1e6:.2f}M")

    optimizer = torch.optim.Adam(train_params, lr=args.head_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.head_epochs * len(train_loader)
    )

    best_loss = float("inf")
    for epoch in range(args.head_epochs):
        segment.head.train()
        segment.projection_head.train()
        segment.linear.train()

        # Initialize bank at start of each epoch
        cluster.bank_init()

        epoch_loss_cont = 0.0
        epoch_loss_clust = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Head Epoch {epoch + 1}/{args.head_epochs}")

        for batch_idx, batch in enumerate(pbar):
            img = batch["img"].to(device)

            # Extract features (frozen backbone)
            with torch.no_grad():
                feat = net(img)[:, 1:, :]  # (B, N, 1024)

            # Student forward
            seg_feat = segment.head(feat, segment.dropout)  # (B, N, 90)
            proj_feat = segment.projection_head(seg_feat)  # (B, N, 2048)

            # EMA teacher forward (no grad)
            with torch.no_grad():
                seg_feat_ema = segment.head_ema(feat)  # (B, N, 90)
                proj_feat_ema = segment.projection_head_ema(seg_feat_ema)  # (B, N, 2048)

            # Contrastive loss with codebook bank
            loss_contrastive = cluster.contrastive_ema_with_codebook_bank(
                feat, proj_feat, proj_feat_ema, temp=0.07
            )

            # Cluster loss
            loss_cluster, _ = cluster.forward_centroid(seg_feat_ema)

            # Total loss
            loss = loss_contrastive + loss_cluster

            # Check for NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update teacher
            ema_update(segment.head, segment.head_ema, lamb=0.99)
            ema_update(segment.projection_head, segment.projection_head_ema, lamb=0.99)

            # Update bank (every few batches to reduce overhead)
            if batch_idx % 5 == 0:
                with torch.no_grad():
                    cluster.bank_update(feat, proj_feat_ema)

            epoch_loss_cont += loss_contrastive.item()
            epoch_loss_clust += loss_cluster.item()
            count += 1
            pbar.set_postfix(
                cont=f"{loss_contrastive.item():.4f}",
                clust=f"{loss_cluster.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # Compute bank for next epoch
        cluster.bank_compute()

        avg_cont = epoch_loss_cont / max(count, 1)
        avg_clust = epoch_loss_clust / max(count, 1)
        avg_total = avg_cont + avg_clust
        print(f"  Epoch {epoch + 1}: contrastive={avg_cont:.4f}, cluster={avg_clust:.4f}, total={avg_total:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.head_epochs:
            save_checkpoint(segment, cluster, args.output_dir, epoch + 1)

        if avg_total < best_loss:
            best_loss = avg_total
            save_checkpoint(segment, cluster, args.output_dir, "best")

    print(f"\nHead training complete. Best loss: {best_loss:.4f}")
    return segment, cluster


def save_checkpoint(segment, cluster, output_dir, tag):
    """Save segment and cluster checkpoints."""
    seg_path = os.path.join(output_dir, f"segment_tr_{tag}.pth")
    cluster_path = os.path.join(output_dir, f"cluster_tr_{tag}.pth")

    torch.save(segment.state_dict(), seg_path)
    torch.save(cluster.state_dict(), cluster_path)
    print(f"  Checkpoint saved: {seg_path}, {cluster_path}")


# ---- Main ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CAUSE-TR on DINOv3 ViT-L/16 for Cityscapes"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["codebook", "heads", "all"],
                        help="Which stage to run")
    parser.add_argument("--model_name", type=str,
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="HuggingFace model name for DINOv3 backbone")

    # Codebook (Stage 2)
    parser.add_argument("--codebook_epochs", type=int, default=10)
    parser.add_argument("--codebook_lr", type=float, default=1e-3)
    parser.add_argument("--codebook_path", type=str, default=None,
                        help="Path to pre-computed codebook .npy (for --stage heads)")

    # Head training (Stage 3)
    parser.add_argument("--head_epochs", type=int, default=40)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Data
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def get_device(args):
    if args.device == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{args.gpu}")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif args.device == "cuda":
        return torch.device(f"cuda:{args.gpu}")
    return torch.device(args.device)


def main():
    args = parse_args()
    device = get_device(args)
    print(f"Device: {device}")

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            Path(__file__).resolve().parent.parent, "refs", "cause", "CAUSE_dinov3"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # Save config
    config = vars(args).copy()
    config["device"] = str(device)
    config["dim"] = DIM
    config["patch_size"] = PATCH_SIZE
    config["train_resolution"] = TRAIN_RESOLUTION
    config["num_patches"] = NUM_PATCHES
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load backbone
    net = DINOv3Backbone(model_name=args.model_name, device=device).to(device)
    net.eval()

    # Create CAUSE modules
    cause_args = make_cause_args()
    cluster = DeviceAwareCluster(cause_args, device=device).to(device)
    segment = Segment_TR(cause_args).to(device)

    print(f"\nSegment_TR params: {sum(p.numel() for p in segment.parameters()) / 1e6:.2f}M")
    print(f"Cluster params: {sum(p.numel() for p in cluster.parameters()) / 1e6:.2f}M")

    # Dataset
    train_dataset = CityscapesCAUSE(
        args.cityscapes_root, split="train",
        resolution=TRAIN_RESOLUTION, augment=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    # Stage 2: Codebook
    if args.stage in ("codebook", "all"):
        cluster = train_codebook(net, cluster, train_loader, device, args)

    # Load codebook if skipping stage 2
    if args.stage == "heads":
        codebook_path = args.codebook_path
        if codebook_path is None:
            codebook_path = os.path.join(args.output_dir, "modular.npy")
        print(f"Loading codebook from {codebook_path}")
        cb = torch.from_numpy(np.load(codebook_path)).to(device)
        cluster.codebook.data = cb

    # Stage 3: Heads
    if args.stage in ("heads", "all"):
        segment, cluster = train_heads(net, segment, cluster, train_loader, device, args)

    # Save final checkpoints in CAUSE-compatible format
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(segment.state_dict(), os.path.join(final_dir, "segment_tr.pth"))
    torch.save(cluster.state_dict(), os.path.join(final_dir, "cluster_tr.pth"))
    np.save(
        os.path.join(final_dir, "modular.npy"),
        cluster.codebook.detach().cpu().numpy(),
    )
    print(f"\nFinal checkpoints saved to {final_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
