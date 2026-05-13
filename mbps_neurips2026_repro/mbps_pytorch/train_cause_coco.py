#!/usr/bin/env python3
"""
Train CAUSE-TR heads on DINOv3 ViT-L/16 features for COCO-Stuff-27.

Adapted from train_cause_dinov3.py (Cityscapes) to support COCO images
and a feature-based mode that loads pre-extracted DINOv3 features.

Stages:
  Stage 2: Modularity-based codebook learning (~10 epochs)
  Stage 3: TRDecoder segment head + Cluster probe training (~40 epochs)

Usage:
    # Image mode (runs backbone live):
    python mbps_pytorch/train_cause_coco.py \
        --coco_root /path/to/coco \
        --output_dir refs/cause/CAUSE_coco \
        --device mps

    # Feature mode (32x32 pre-extracted features, no backbone needed):
    python mbps_pytorch/train_cause_coco.py \
        --coco_root /path/to/coco \
        --feature_mode \
        --feature_dir dinov3_features \
        --output_dir refs/cause/CAUSE_coco_feat32 \
        --device mps

    # Feature mode (64x64 pre-extracted features):
    python mbps_pytorch/train_cause_coco.py \
        --coco_root /path/to/coco \
        --feature_mode \
        --feature_dir dinov3_features_64x64 \
        --output_dir refs/cause/CAUSE_coco_feat64 \
        --device mps

    # Stage 2 only (codebook):
    python mbps_pytorch/train_cause_coco.py --stage codebook ...

    # Stage 3 only (heads, requires codebook from stage 2):
    python mbps_pytorch/train_cause_coco.py --stage heads ...
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
from scipy.optimize import linear_sum_assignment
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
N_CLASSES = 27  # COCO-Stuff-27
NUM_REGISTER_TOKENS = 4  # DINOv3 register tokens

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---- COCO-Stuff-27 Constants ----
COCOSTUFF27_CLASSNAMES = [
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]

SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


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
        # Return (B, 1+N, D) -- compatible with CAUSE's net(img)[:, 1:, :]
        return torch.cat([cls_token, patch_tokens], dim=1)


# ---- Datasets ----

class COCOCAUSE(Dataset):
    """COCO dataset for CAUSE training with random crops (image mode)."""

    def __init__(self, coco_root, split="val2017", resolution=320, augment=True):
        self.root = Path(coco_root)
        self.resolution = resolution
        self.augment = augment

        img_dir = self.root / split
        self.image_paths = sorted(img_dir.glob("*.jpg"))
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
        # Extract image_id from filename (e.g., 000000001234.jpg -> 1234)
        image_id = int(self.image_paths[idx].stem)
        return {"img": img, "ind": idx, "image_id": image_id}


class COCOFeatureCAUSE(Dataset):
    """COCO dataset loading pre-extracted DINOv3 features (feature mode).

    Features are stored as .npy files of shape (n_patches, dim), e.g.:
      - 32x32 grid: (1024, 1024)
      - 64x64 grid: (4096, 1024)

    Random crops a 20x20 subgrid to produce 400 patches matching CAUSE's
    expected input (TRAIN_RESOLUTION=320, PATCH_SIZE=16 -> 20x20 patches).
    """

    def __init__(self, feat_dir, patch_grid=32, crop_size=20):
        self.feat_files = sorted(Path(feat_dir).glob("*.npy"))
        self.patch_grid = patch_grid
        self.crop_size = crop_size
        print(f"  Feature dir: {feat_dir}")
        print(f"  Found {len(self.feat_files)} feature files")
        print(f"  Patch grid: {patch_grid}x{patch_grid}, crop: {crop_size}x{crop_size}")

    def __len__(self):
        return len(self.feat_files)

    def __getitem__(self, idx):
        feat = np.load(self.feat_files[idx])  # (n_patches, dim)
        feat = feat.reshape(self.patch_grid, self.patch_grid, -1)
        # Random crop
        top = random.randint(0, self.patch_grid - self.crop_size)
        left = random.randint(0, self.patch_grid - self.crop_size)
        feat_crop = feat[top:top + self.crop_size, left:left + self.crop_size]
        feat_crop = feat_crop.reshape(-1, feat_crop.shape[-1])  # (crop_size^2, dim)
        feat_tensor = torch.from_numpy(feat_crop).float()
        image_id = int(self.feat_files[idx].stem)
        return {"feat": feat_tensor, "ind": idx, "image_id": image_id}


class COCOFeatureCAUSEVal(Dataset):
    """COCO feature dataset for validation -- returns full grid, no cropping."""

    def __init__(self, feat_dir, patch_grid=32):
        self.feat_files = sorted(Path(feat_dir).glob("*.npy"))
        self.patch_grid = patch_grid
        print(f"  Val feature dir: {feat_dir}")
        print(f"  Found {len(self.feat_files)} feature files")

    def __len__(self):
        return len(self.feat_files)

    def __getitem__(self, idx):
        feat = np.load(self.feat_files[idx])  # (n_patches, dim)
        feat_tensor = torch.from_numpy(feat).float()
        image_id = int(self.feat_files[idx].stem)
        return {"feat": feat_tensor, "ind": idx, "image_id": image_id}


# ---- CAUSE Args namespace ----

def make_cause_args(num_patches=NUM_PATCHES):
    """Create args namespace for CAUSE module initialization."""
    return SimpleNamespace(
        dim=DIM,
        reduced_dim=REDUCED_DIM,
        projection_dim=PROJECTION_DIM,
        num_codebook=NUM_CODEBOOK,
        n_classes=N_CLASSES,
        num_queries=num_patches,
    )


# ---- GT Loading for Evaluation ----

def load_coco_panoptic_gt(coco_root):
    """Load COCO panoptic GT for evaluation."""
    panoptic_json = os.path.join(coco_root, "annotations", "panoptic_val2017.json")
    panoptic_dir = os.path.join(coco_root, "annotations", "panoptic_val2017")

    with open(panoptic_json) as f:
        data = json.load(f)

    cat_map = {}
    for cat in data["categories"]:
        cat_map[cat["id"]] = cat["supercategory"]

    ann_map = {}
    for ann in data["annotations"]:
        ann_map[ann["image_id"]] = ann

    return cat_map, ann_map, panoptic_dir


def get_gt_semantic(cat_map, ann_map, panoptic_dir, image_id):
    """Get 27-class semantic GT for one image."""
    if image_id not in ann_map:
        return None
    ann = ann_map[image_id]
    pan_path = os.path.join(panoptic_dir, ann["file_name"])
    pan_img = np.array(Image.open(pan_path))
    pan_id = (pan_img[:, :, 0].astype(np.int32) +
              pan_img[:, :, 1].astype(np.int32) * 256 +
              pan_img[:, :, 2].astype(np.int32) * 256 * 256)

    sem_label = np.full(pan_id.shape, 255, dtype=np.uint8)
    for seg in ann["segments_info"]:
        mask = pan_id == seg["id"]
        supercat = cat_map.get(seg["category_id"])
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            sem_label[mask] = SUPERCATEGORY_TO_COARSE[supercat]
    return sem_label


# ---- Hungarian Matching + mIoU ----

def hungarian_match_and_miou(predictions, gt_labels, image_ids, n_clusters, n_classes=27, n_eval=100):
    """Match cluster IDs to GT classes via Hungarian + compute mIoU.

    Args:
        predictions: list of (H_pred, W_pred) numpy arrays with cluster IDs.
        gt_labels: list of (H_gt, W_gt) numpy arrays with 27-class GT (or None).
        image_ids: list of image IDs (for logging).
        n_clusters: number of clusters to match.
        n_classes: number of GT classes (27).
        n_eval: number of images to use for building cost matrix.

    Returns:
        miou, per_class_iou dict, cluster_to_class mapping.
    """
    cost_matrix = np.zeros((n_clusters, n_classes), dtype=np.float64)

    count_used = 0
    for pred, gt in zip(predictions, gt_labels):
        if gt is None:
            continue
        if count_used >= n_eval:
            break
        # Resize GT to match prediction grid
        gt_resized = np.array(Image.fromarray(gt).resize(
            (pred.shape[1], pred.shape[0]), Image.NEAREST))
        for p, g in zip(pred.flatten(), gt_resized.flatten()):
            if g < n_classes:
                cost_matrix[int(p), g] += 1
        count_used += 1

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c
    for k_id in range(n_clusters):
        if k_id not in cluster_to_class:
            best_class = int(np.argmax(cost_matrix[k_id]))
            cluster_to_class[k_id] = best_class

    # Compute mIoU on ALL images
    iou_per_class = np.zeros(n_classes)
    count_per_class = np.zeros(n_classes)

    for pred, gt in zip(predictions, gt_labels):
        if gt is None:
            continue
        # Map predictions to classes
        pred_mapped = np.vectorize(cluster_to_class.get)(pred).astype(np.uint8)
        pred_full = np.array(Image.fromarray(pred_mapped).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST))

        for c in range(n_classes):
            gt_mask = gt == c
            pred_mask = pred_full == c
            inter = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if union > 0:
                iou_per_class[c] += inter / union
                count_per_class[c] += 1

    valid = count_per_class > 0
    miou = (iou_per_class[valid] / count_per_class[valid]).mean() * 100
    per_class = {}
    for c in range(n_classes):
        if count_per_class[c] > 0:
            per_class[COCOSTUFF27_CLASSNAMES[c]] = round(
                iou_per_class[c] / count_per_class[c] * 100, 1)
    return miou, per_class, cluster_to_class


# ---- Stage 2: Modularity Codebook ----

def train_codebook(net, cluster, train_loader, device, args):
    """Stage 2: Optimize codebook via modularity maximization."""
    print("\n" + "=" * 60)
    print("  STAGE 2: Modularity-based Codebook Learning")
    print("=" * 60)

    feature_mode = args.feature_mode

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
            if feature_mode:
                # Feature mode: features are pre-extracted
                feat = batch["feat"].to(device)  # (B, N, 1024)
            else:
                # Image mode: run backbone
                img = batch["img"].to(device)
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

    feature_mode = args.feature_mode

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
            if feature_mode:
                feat = batch["feat"].to(device)  # (B, N, 1024)
            else:
                img = batch["img"].to(device)
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


# ---- Inference ----

def predict_feature_mode(segment, cluster, val_loader, device, patch_grid):
    """Run inference in feature mode over validation set.

    For each image, feeds the full feature grid through the heads and produces
    a cluster assignment map at the patch grid resolution.

    Since the Decoder's query_pos is sized for crop_size^2 (400) patches,
    but validation features may have a different number of patches (e.g., 1024
    for 32x32 or 4096 for 64x64), we tile the full grid in crop_size x crop_size
    windows and stitch the results.
    """
    segment.head.eval()
    segment.projection_head.eval()
    segment.linear.eval()

    crop_size = NUM_PATCHES_PER_SIDE  # 20
    all_preds = []
    all_image_ids = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference (feature mode)"):
            feat = batch["feat"].to(device)  # (B, N_total, 1024)
            image_ids = batch["image_id"]
            B = feat.shape[0]

            for b in range(B):
                feat_b = feat[b]  # (N_total, 1024)
                feat_grid = feat_b.reshape(patch_grid, patch_grid, -1)

                # Tile with crop_size windows, stepping by crop_size (non-overlapping)
                # If patch_grid is not divisible by crop_size, extend the last tile
                pred_grid = torch.zeros(patch_grid, patch_grid, dtype=torch.long, device=device)

                for top in range(0, patch_grid, crop_size):
                    for left in range(0, patch_grid, crop_size):
                        # Clamp window to fit within grid
                        h_end = min(top + crop_size, patch_grid)
                        w_end = min(left + crop_size, patch_grid)
                        h_start = max(h_end - crop_size, 0)
                        w_start = max(w_end - crop_size, 0)

                        window = feat_grid[h_start:h_end, w_start:w_end]  # (crop, crop, dim)
                        window_flat = window.reshape(1, -1, window.shape[-1])  # (1, crop^2, dim)

                        seg_feat = segment.head_ema(window_flat)  # (1, crop^2, 90)
                        # Use linear head for class logits
                        logits = segment.linear(seg_feat)  # returns transformed output
                        # logits shape depends on ProjectionSegment: (1, crop^2, n_classes) or (1, n_classes, crop, crop)
                        # ProjectionSegment with is_untrans=False does untransform then conv then back
                        # Actually it does transform (reshape to spatial) -> conv -> flatten back
                        # So output is (1, N, n_classes)

                        # Alternatively, use cluster_probe for assignment
                        # seg_feat: (1, crop^2, 90), cluster_probe: (27, 90)
                        seg_flat = seg_feat.squeeze(0)  # (crop^2, 90)
                        sim = F.normalize(seg_flat, dim=1) @ F.normalize(cluster.cluster_probe, dim=1).T
                        pred_flat = sim.argmax(dim=1)  # (crop^2,)
                        pred_window = pred_flat.reshape(crop_size, crop_size)

                        # Place into the grid (only the valid region for non-overlapping tiles)
                        # For overlapping tiles at the boundary, prefer the tile that
                        # starts at the original top/left
                        row_offset = top - h_start
                        col_offset = left - w_start
                        h_valid = min(crop_size - row_offset, patch_grid - top)
                        w_valid = min(crop_size - col_offset, patch_grid - left)
                        pred_grid[top:top + h_valid, left:left + w_valid] = \
                            pred_window[row_offset:row_offset + h_valid, col_offset:col_offset + w_valid]

                all_preds.append(pred_grid.cpu().numpy())
                all_image_ids.append(image_ids[b].item() if torch.is_tensor(image_ids[b]) else int(image_ids[b]))

    return all_preds, all_image_ids


def predict_image_mode(net, segment, cluster, val_loader, device):
    """Run inference in image mode over validation set."""
    segment.head.eval()
    segment.projection_head.eval()
    segment.linear.eval()

    all_preds = []
    all_image_ids = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference (image mode)"):
            img = batch["img"].to(device)
            image_ids = batch["image_id"]

            feat = net(img)[:, 1:, :]  # (B, N, 1024)
            seg_feat = segment.head_ema(feat)  # (B, N, 90)

            B, N, D = seg_feat.shape
            H = W = int(math.sqrt(N))

            for b in range(B):
                seg_flat = seg_feat[b]  # (N, 90)
                sim = F.normalize(seg_flat, dim=1) @ F.normalize(cluster.cluster_probe, dim=1).T
                pred_flat = sim.argmax(dim=1)  # (N,)
                pred_map = pred_flat.reshape(H, W).cpu().numpy()
                all_preds.append(pred_map)
                all_image_ids.append(image_ids[b].item() if torch.is_tensor(image_ids[b]) else int(image_ids[b]))

    return all_preds, all_image_ids


# ---- Evaluation & Pseudo-label Saving ----

def evaluate_and_save(all_preds, all_image_ids, coco_root, output_dir, n_classes=27):
    """Evaluate predictions via Hungarian matching and save pseudo-labels."""
    print("\n" + "=" * 60)
    print("  EVALUATION: Hungarian matching + mIoU")
    print("=" * 60)

    # Load GT
    cat_map, ann_map, panoptic_dir = load_coco_panoptic_gt(coco_root)

    gt_labels = []
    for img_id in all_image_ids:
        gt = get_gt_semantic(cat_map, ann_map, panoptic_dir, img_id)
        gt_labels.append(gt)

    # Find max cluster ID for hungarian matching
    max_cluster = max(p.max() for p in all_preds) + 1
    print(f"  Predictions: {len(all_preds)} images, max cluster ID: {max_cluster - 1}")

    miou, per_class, cluster_to_class = hungarian_match_and_miou(
        all_preds, gt_labels, all_image_ids, n_clusters=max_cluster, n_classes=n_classes
    )

    print(f"\n  mIoU: {miou:.1f}%")
    for name, iou in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}: {iou:.1f}%")

    # Save pseudo-labels as 27-class PNG maps
    pseudo_dir = os.path.join(coco_root, "pseudo_semantic_cause", "val2017")
    os.makedirs(pseudo_dir, exist_ok=True)
    print(f"\n  Saving pseudo-labels to {pseudo_dir}...")

    for idx, img_id in enumerate(tqdm(all_image_ids, desc="Saving pseudo-labels")):
        pred = all_preds[idx]  # (H_pred, W_pred) cluster IDs
        # Map cluster IDs to 27-class labels
        class_map = np.vectorize(cluster_to_class.get)(pred).astype(np.uint8)

        # Upsample to original image size
        img_path = os.path.join(coco_root, "val2017", f"{img_id:012d}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            W_orig, H_orig = img.size
            class_map_full = np.array(
                Image.fromarray(class_map).resize((W_orig, H_orig), Image.NEAREST))
        else:
            class_map_full = class_map

        Image.fromarray(class_map_full).save(
            os.path.join(pseudo_dir, f"{img_id:012d}.png"))

    # Save metadata
    meta = {
        "method": "cause_tr_dinov3_coco",
        "n_classes": n_classes,
        "miou": round(miou, 2),
        "per_class": per_class,
        "cluster_to_class": {str(k): int(v) for k, v in cluster_to_class.items()},
        "num_images": len(all_preds),
    }
    meta_path = os.path.join(pseudo_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    return miou, per_class, cluster_to_class


# ---- Main ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CAUSE-TR on DINOv3 ViT-L/16 for COCO-Stuff-27"
    )
    parser.add_argument("--coco_root", type=str, required=True,
                        help="Path to COCO dataset root (contains val2017/, annotations/, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["codebook", "heads", "all"],
                        help="Which stage to run")
    parser.add_argument("--model_name", type=str,
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="HuggingFace model name for DINOv3 backbone (image mode only)")

    # Feature mode
    parser.add_argument("--feature_mode", action="store_true",
                        help="Use pre-extracted features instead of running backbone")
    parser.add_argument("--feature_dir", type=str, default="dinov3_features",
                        help="Subdirectory under coco_root containing features "
                             "(e.g., 'dinov3_features' for 32x32 or 'dinov3_features_64x64' for 64x64)")
    parser.add_argument("--patch_grid", type=int, default=None,
                        help="Patch grid size (auto-detected from first feature file if not set)")
    parser.add_argument("--train_split", type=str, default="val2017",
                        help="Split to use for training (e.g., 'train2017', 'val2017')")
    parser.add_argument("--val_split", type=str, default="val2017",
                        help="Split to use for evaluation")

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

    # Evaluation
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation and pseudo-label saving after training")

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


def detect_patch_grid(feat_dir):
    """Auto-detect patch grid size from the first .npy file in feat_dir."""
    feat_files = sorted(Path(feat_dir).glob("*.npy"))
    if not feat_files:
        raise FileNotFoundError(f"No .npy files found in {feat_dir}")
    sample = np.load(feat_files[0])
    n_patches = sample.shape[0]
    grid_size = int(math.sqrt(n_patches))
    if grid_size * grid_size != n_patches:
        raise ValueError(
            f"Feature file {feat_files[0]} has {n_patches} patches, "
            f"which is not a perfect square. Cannot auto-detect grid size."
        )
    print(f"  Auto-detected patch_grid={grid_size} from {feat_files[0].name} "
          f"(shape: {sample.shape})")
    return grid_size


def main():
    args = parse_args()
    device = get_device(args)
    print(f"Device: {device}")
    print(f"Mode: {'feature' if args.feature_mode else 'image'}")

    # Output directory
    if args.output_dir is None:
        suffix = "_feat" if args.feature_mode else ""
        args.output_dir = os.path.join(
            Path(__file__).resolve().parent.parent, "refs", "cause", f"CAUSE_coco{suffix}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # Detect patch grid in feature mode
    if args.feature_mode:
        train_feat_dir = os.path.join(args.coco_root, args.feature_dir, args.train_split)
        val_feat_dir = os.path.join(args.coco_root, args.feature_dir, args.val_split)
        if args.patch_grid is None:
            args.patch_grid = detect_patch_grid(train_feat_dir)
        patch_grid = args.patch_grid
        crop_size = NUM_PATCHES_PER_SIDE  # 20
        if patch_grid < crop_size:
            raise ValueError(
                f"patch_grid ({patch_grid}) must be >= crop_size ({crop_size}). "
                f"Cannot crop a {crop_size}x{crop_size} window from a {patch_grid}x{patch_grid} grid."
            )

    # Save config
    config = vars(args).copy()
    config["device"] = str(device)
    config["dim"] = DIM
    config["patch_size"] = PATCH_SIZE
    config["train_resolution"] = TRAIN_RESOLUTION
    config["num_patches"] = NUM_PATCHES
    config["n_classes"] = N_CLASSES
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load backbone (only needed in image mode)
    net = None
    if not args.feature_mode:
        net = DINOv3Backbone(model_name=args.model_name, device=device).to(device)
        net.eval()

    # Create CAUSE modules
    cause_args = make_cause_args(num_patches=NUM_PATCHES)
    cluster = DeviceAwareCluster(cause_args, device=device).to(device)
    segment = Segment_TR(cause_args).to(device)

    print(f"\nSegment_TR params: {sum(p.numel() for p in segment.parameters()) / 1e6:.2f}M")
    print(f"Cluster params: {sum(p.numel() for p in cluster.parameters()) / 1e6:.2f}M")

    # ---- Training dataset ----
    if args.feature_mode:
        train_dataset = COCOFeatureCAUSE(
            train_feat_dir,
            patch_grid=patch_grid,
            crop_size=crop_size,
        )
    else:
        train_dataset = COCOCAUSE(
            args.coco_root, split=args.train_split,
            resolution=TRAIN_RESOLUTION, augment=True,
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

    # ---- Evaluation on val set ----
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("  GENERATING PREDICTIONS ON VALIDATION SET")
        print("=" * 60)

        if args.feature_mode:
            val_dataset = COCOFeatureCAUSEVal(val_feat_dir, patch_grid=patch_grid)
            val_loader = DataLoader(
                val_dataset, batch_size=1,
                shuffle=False, num_workers=0,
                pin_memory=True,
            )
            all_preds, all_image_ids = predict_feature_mode(
                segment, cluster, val_loader, device, patch_grid
            )
        else:
            val_dataset = COCOCAUSE(
                args.coco_root, split=args.val_split,
                resolution=TRAIN_RESOLUTION, augment=False,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                pin_memory=True,
            )
            all_preds, all_image_ids = predict_image_mode(
                net, segment, cluster, val_loader, device
            )

        miou, per_class, cluster_to_class = evaluate_and_save(
            all_preds, all_image_ids, args.coco_root, args.output_dir
        )

        # Append results to config
        config["miou"] = round(miou, 2)
        config["per_class_iou"] = per_class
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
