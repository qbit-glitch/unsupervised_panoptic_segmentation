#!/usr/bin/env python3
"""NeCo post-training of DINOv3 for improved patch-level spatial representations.

NeCo (Neighborhood Consistency, ICLR 2025) fine-tunes a frozen ViT with a
patch neighbor consistency InfoNCE loss. Nearby patches in the image should
have similar representations (positive pairs); distant patches should not
(negative pairs). This directly improves the quality of patch features used
for NCut segmentation without any semantic labels.

Expected gain: +3–5 mIoU on COCO-Stuff-27 (per NeCo paper: +5.5 on ADE20k,
+5.7 on COCO-Stuff for linear probing).

Architecture:
    DINOv3 ViT-B/16 (frozen except last 4 blocks)
    → patch tokens (N, 1024)
    → projection head (1024 → 256, 2-layer MLP, BN, ReLU)
    → L2-normalised embeddings (N, 256)

Loss:
    InfoNCE with spatial positive pairs (|pos_i − pos_j| < R patches)
    and in-image negatives (|pos_i − pos_k| > R_neg patches).
    Temperature τ = 0.07.

Usage:
    # Full training on COCO train2017 (remote GPU recommended)
    python train_neco_dinov3.py \
        --coco_root /media/santosh/data/coco \
        --output_dir checkpoints/neco_dinov3 \
        --epochs 5 --batch_size 32 --lr 1e-5 --device cuda

    # Quick smoke-test (100 images, 1 epoch)
    python train_neco_dinov3.py \
        --coco_root /path/to/coco \
        --output_dir /tmp/neco_test \
        --n_images 100 --epochs 1 --device mps
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════


class COCOPatchDataset(Dataset):
    """COCO train2017 images for NeCo patch-level training.

    Returns a single image as a tensor. Patch pair mining is done in the
    collate/training loop using spatial coordinates derived from the patch grid.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        coco_root: str,
        img_size: int = 512,
        split: str = "train2017",
        n_images: Optional[int] = None,
    ) -> None:
        self.img_dir = Path(coco_root) / split
        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        if n_images:
            self.img_paths = self.img_paths[:n_images]
        logger.info("COCOPatchDataset: %d images from %s", len(self.img_paths), split)

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img)


# ═══════════════════════════════════════════════════════════════════════
# Projection Head
# ═══════════════════════════════════════════════════════════════════════


class ProjectionHead(nn.Module):
    """2-layer MLP with BN + ReLU, as in SimCLR / NeCo.

    Maps (B*N, in_dim) → (B*N, out_dim) with L2 normalisation at output.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 2048,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (N, in_dim). Returns: (N, out_dim) L2-normalised."""
        return F.normalize(self.net(x), p=2, dim=1)


# ═══════════════════════════════════════════════════════════════════════
# NeCo Loss
# ═══════════════════════════════════════════════════════════════════════


def neco_loss(
    embeddings: torch.Tensor,
    h_patches: int,
    w_patches: int,
    r_pos: int = 2,
    r_neg: int = 6,
    temperature: float = 0.07,
    n_pairs_per_image: int = 128,
) -> torch.Tensor:
    """NeCo InfoNCE loss on patch spatial neighbourhood.

    For each anchor patch i, samples:
    - positive j: randomly chosen from patches within L∞ distance r_pos
    - negatives k: randomly chosen patches with L∞ distance > r_neg
                   (within the same image)

    The loss is:
        L = -log( exp(z_i·z_j / τ) / (exp(z_i·z_j / τ) + Σ_k exp(z_i·z_k / τ)) )

    Args:
        embeddings:       (B, N, D) L2-normalised patch embeddings.
        h_patches:        Patch grid height.
        w_patches:        Patch grid width.
        r_pos:            Positive pair neighbourhood radius (L∞).
        r_neg:            Minimum distance for negatives (L∞).
        temperature:      InfoNCE temperature τ.
        n_pairs_per_image: Anchor patches sampled per image.

    Returns:
        Scalar loss tensor.
    """
    B, N, D = embeddings.shape
    assert N == h_patches * w_patches, f"N={N} != {h_patches}×{w_patches}"
    device = embeddings.device

    # Build spatial coordinates (row, col) for each patch
    rows = torch.arange(h_patches, device=device)
    cols = torch.arange(w_patches, device=device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)  # (N, 2)

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for b in range(B):
        z = embeddings[b]  # (N, D)

        # Sample anchors
        anchor_ids = torch.randperm(N, device=device)[:n_pairs_per_image]

        for anchor_idx in anchor_ids:
            r_a, c_a = coords[anchor_idx]

            # L∞ distance from anchor to all other patches
            l_inf = torch.maximum(
                (coords[:, 0] - r_a).abs(),
                (coords[:, 1] - c_a).abs(),
            )

            # Positive candidates: within r_pos (exclude self)
            pos_mask = (l_inf <= r_pos) & (l_inf > 0)
            if not pos_mask.any():
                continue
            pos_ids = torch.where(pos_mask)[0]
            pos_idx = pos_ids[torch.randint(len(pos_ids), (1,), device=device)].item()

            # Negative candidates: further than r_neg
            neg_mask = l_inf > r_neg
            if not neg_mask.any():
                continue
            neg_ids = torch.where(neg_mask)[0]

            # Sample up to 64 negatives for efficiency
            n_neg = min(64, len(neg_ids))
            neg_sample_ids = neg_ids[torch.randperm(len(neg_ids), device=device)[:n_neg]]

            z_a = z[anchor_idx]       # (D,)
            z_p = z[pos_idx]          # (D,)
            z_n = z[neg_sample_ids]   # (n_neg, D)

            # Similarities
            sim_pos = (z_a * z_p).sum() / temperature          # scalar
            sim_neg = (z_a.unsqueeze(0) * z_n).sum(dim=1) / temperature  # (n_neg,)

            # InfoNCE: log( exp(pos) / (exp(pos) + Σ exp(neg)) )
            logits = torch.cat([sim_pos.unsqueeze(0), sim_neg])  # (1 + n_neg,)
            loss_i = -F.log_softmax(logits, dim=0)[0]

            total_loss = total_loss + loss_i
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / n_valid


# ═══════════════════════════════════════════════════════════════════════
# Epoch Validation — K-means mIoU on COCO-Stuff-27
# ═══════════════════════════════════════════════════════════════════════


def _build_val_records(coco_root: str, n_images: int = 200) -> List[Tuple[int, str]]:
    """Return (image_id, filename) tuples for the first N panoptic val2017 images.

    Loads the panoptic JSON once here; subsequent calls to load_coco_panoptic_gt
    reuse its function-level cache so the JSON is never read twice.

    Args:
        coco_root: Path to COCO dataset root.
        n_images:  Number of val images to use for validation.

    Returns:
        List of (image_id, filename) tuples, empty if annotations missing.
    """
    pan_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
    if not pan_json.exists():
        logger.warning("Panoptic annotations not found at %s — validation disabled", pan_json)
        return []
    with open(pan_json) as f:
        data = json.load(f)
    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    records: List[Tuple[int, str]] = []
    for ann in data["annotations"]:
        if len(records) >= n_images:
            break
        fname = id_to_fname.get(ann["image_id"])
        if fname:
            records.append((ann["image_id"], fname))
    logger.info("Val index: %d images for mIoU validation", len(records))
    return records


def validate_epoch_miou(
    backbone: nn.Module,
    coco_root: str,
    device_t: torch.device,
    val_records: List[Tuple[int, str]],
    img_size: int = 448,
    patch_size: int = 14,
    k_clusters: int = 54,
) -> Dict[str, float]:
    """K-means mIoU validation on COCO-Stuff-27.

    Two-pass evaluation:
    Pass 1 — extract patch features for all val images (no gradients).
    Fit MiniBatchKMeans(K=k_clusters) on the concatenated feature matrix.
    Pass 2 — per-image: reshape cluster labels → resize to image resolution
             → Hungarian matching against COCO panoptic GT → compute mIoU.

    Args:
        backbone:    Backbone model (may be DataParallel-wrapped).
        coco_root:   Path to COCO dataset root.
        device_t:    Torch device.
        val_records: List of (image_id, filename) from _build_val_records().
        img_size:    Input image size used during training.
        patch_size:  ViT patch size.
        k_clusters:  K for k-means (54 = best overclustering for COCO-Stuff-27).

    Returns:
        Dict with keys: miou, things_miou, stuff_miou (all in %).
    """
    # Lazy import — diffcut_pseudo_semantics.py is a sibling module in mbps_pytorch/
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from diffcut_pseudo_semantics import (  # type: ignore[import]
        load_coco_panoptic_gt, hungarian_miou,
        NUM_CLASSES, THING_IDS, STUFF_IDS,
    )

    h_patches = img_size // patch_size
    w_patches = img_size // patch_size
    val_img_dir = Path(coco_root) / "val2017"

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Unwrap DataParallel for fallback forward_features call
    raw_backbone = backbone.module if isinstance(backbone, nn.DataParallel) else backbone

    # ── Pass 1: feature extraction ──
    all_features: List[np.ndarray] = []
    used_records: List[Tuple[int, str]] = []

    backbone.eval()
    with torch.no_grad():
        for image_id, filename in tqdm(val_records, desc="Val feat extract", leave=False):
            img_path = val_img_dir / filename
            if not img_path.exists():
                continue
            img_t = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device_t)
            try:
                out = backbone(pixel_values=img_t)
                tokens = out.last_hidden_state
                n_special = tokens.shape[1] - h_patches * w_patches
                patch_feats = tokens[:, n_special:, :].squeeze(0)
            except Exception:
                out = raw_backbone.forward_features(img_t)
                patch_feats = out["x_norm_patchtokens"].squeeze(0)
            all_features.append(patch_feats.cpu().float().numpy())
            used_records.append((image_id, filename))
    backbone.train()

    if not all_features:
        return {"miou": 0.0, "things_miou": 0.0, "stuff_miou": 0.0}

    # ── Fit k-means on all patches ──
    all_patches = np.concatenate(all_features, axis=0)   # (N_total, D)
    kmeans = MiniBatchKMeans(
        n_clusters=k_clusters, random_state=42,
        batch_size=4096, n_init=3, verbose=0,
    )
    kmeans.fit(all_patches)

    # ── Pass 2: per-image mIoU ──
    per_class_iou_acc = np.zeros(NUM_CLASSES)
    per_class_count = np.zeros(NUM_CLASSES)
    offset = 0

    for (image_id, _), feats in zip(used_records, all_features):
        n_patches = feats.shape[0]
        cluster_labels = kmeans.labels_[offset: offset + n_patches]
        offset += n_patches

        # Upsample patch-grid labels to img_size × img_size
        label_grid = cluster_labels.reshape(h_patches, w_patches).astype(np.uint8)
        pred = np.array(
            Image.fromarray(label_grid).resize((img_size, img_size), Image.NEAREST)
        )

        gt = load_coco_panoptic_gt(coco_root, image_id)
        if gt is None:
            continue
        gt_resized = np.array(
            Image.fromarray(gt).resize((img_size, img_size), Image.NEAREST)
        )

        _, iou_per_class, _ = hungarian_miou(pred, gt_resized, n_class=NUM_CLASSES)
        for c in range(NUM_CLASSES):
            if (gt_resized == c).any():
                per_class_iou_acc[c] += iou_per_class[c]
                per_class_count[c] += 1

    valid_cls = per_class_count > 0
    per_class_avg = np.zeros(NUM_CLASSES)
    per_class_avg[valid_cls] = per_class_iou_acc[valid_cls] / per_class_count[valid_cls]
    miou = per_class_avg[valid_cls].mean() * 100

    things_ious = [per_class_avg[c] * 100 for c in range(NUM_CLASSES)
                   if c in THING_IDS and per_class_count[c] > 0]
    stuff_ious = [per_class_avg[c] * 100 for c in range(NUM_CLASSES)
                  if c in STUFF_IDS and per_class_count[c] > 0]

    return {
        "miou": round(float(miou), 2),
        "things_miou": round(float(np.mean(things_ious)), 2) if things_ious else 0.0,
        "stuff_miou": round(float(np.mean(stuff_ious)), 2) if stuff_ious else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════


def build_dinov3_model(device: str) -> Tuple[nn.Module, int]:
    """Load DINOv3 ViT-B/16 and return (model, embed_dim).

    Tries HuggingFace transformers first, falls back to torch.hub DINOv2.
    """
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "facebook/dinov2-base",  # dinov3-vitb16 may not be public on HF
            trust_remote_code=True,
        )
        embed_dim = 768
        logger.info("Loaded DINOv2-base (768-dim) via HuggingFace")
    except Exception:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        embed_dim = 768
        logger.info("Loaded DINOv2-vitb14-reg (768-dim) via torch.hub")

    model = model.to(device)
    return model, embed_dim


def get_patch_features(
    model: nn.Module,
    images: torch.Tensor,
    patch_size: int = 14,
) -> Tuple[torch.Tensor, int, int]:
    """Extract patch tokens from DINOv2/v3 model.

    Args:
        model:      DINOv2/v3 model.
        images:     (B, 3, H, W) input tensor.
        patch_size: Patch size (14 for ViT-B/14, 16 for ViT-B/16).

    Returns:
        (B, N, D) patch tokens, h_patches, w_patches.
    """
    B, _, H, W = images.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    with torch.no_grad():
        # Try HuggingFace API first
        try:
            out = model(pixel_values=images)
            # last_hidden_state: (B, 1 + N_reg + N_patch, D)
            # Skip CLS + register tokens
            tokens = out.last_hidden_state
            # DINOv2 with registers: CLS=1, registers=4, patches=N
            n_special = tokens.shape[1] - h_patches * w_patches
            patch_tokens = tokens[:, n_special:, :]  # (B, N, D)
        except Exception:
            # torch.hub DINOv2 API
            out = model.forward_features(images)
            patch_tokens = out["x_norm_patchtokens"]  # (B, N, D)

    return patch_tokens, h_patches, w_patches


def train_neco(
    coco_root: str,
    output_dir: str,
    img_size: int = 448,
    patch_size: int = 14,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    r_pos: int = 2,
    r_neg: int = 6,
    n_pairs_per_image: int = 64,
    proj_dim: int = 256,
    unfreeze_last_n_blocks: int = 4,
    n_images: Optional[int] = None,
    device: str = "cuda",
    num_workers: int = 4,
    log_interval: int = 50,
    grad_accum: int = 8,
    n_val_images: int = 200,
    val_k_clusters: int = 54,
    val_coco_root: Optional[str] = None,
) -> None:
    """Train NeCo on COCO train2017.

    Args:
        coco_root:              Path to COCO dataset root (for training images).
        output_dir:             Directory to save checkpoints and logs.
        img_size:               Input image size (must be divisible by patch_size).
        patch_size:             ViT patch size.
        epochs:                 Number of training epochs.
        batch_size:             Images per forward pass (keep low for 11GB GPUs; use
                                grad_accum to maintain effective batch size).
        lr:                     Learning rate for unfrozen blocks + projection head.
        weight_decay:           AdamW weight decay.
        temperature:            InfoNCE temperature τ.
        r_pos:                  Positive pair L∞ radius (patches).
        r_neg:                  Minimum L∞ distance for negatives (patches).
        n_pairs_per_image:      Anchor patches sampled per image per batch.
        proj_dim:               Projection head output dimension.
        unfreeze_last_n_blocks: Number of ViT blocks to fine-tune (last N).
        n_images:               Limit dataset size (for smoke tests).
        device:                 Compute device.
        num_workers:            DataLoader workers.
        log_interval:           Log loss every N optimizer steps.
        grad_accum:             Gradient accumulation steps. Effective batch size =
                                batch_size * grad_accum (default: 4 * 8 = 32).
        n_val_images:           Number of COCO val images for per-epoch mIoU validation.
                                Set to 0 to disable validation.
        val_k_clusters:         K for k-means in validation (54 = best for COCO-Stuff-27).
        val_coco_root:          COCO root for validation (panoptic annotations + val2017
                                images). Defaults to coco_root if not specified. Useful
                                when train and val data live on different mounts.
    """
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)

    # ── Dataset & DataLoader ──
    dataset = COCOPatchDataset(coco_root, img_size=img_size, n_images=n_images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "mps"),
        drop_last=True,
    )

    # ── Model ──
    backbone, embed_dim = build_dinov3_model(device)
    head = ProjectionHead(in_dim=embed_dim, hidden_dim=embed_dim * 2, out_dim=proj_dim)
    head = head.to(device_t)

    # Freeze all backbone parameters first
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Unfreeze last N transformer blocks
    unfrozen = 0
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        # HuggingFace ViT
        blocks = backbone.encoder.layer
        for block in blocks[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
            unfrozen += 1
    elif hasattr(backbone, "blocks"):
        # torch.hub DINOv2
        blocks = backbone.blocks
        for block in blocks[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
            unfrozen += 1

    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters())
    logger.info(
        "Trainable: %d backbone params (last %d blocks) + %d head params",
        n_trainable, unfreeze_last_n_blocks, n_head,
    )
    logger.info(
        "Effective batch size: %d (batch_size=%d × grad_accum=%d)",
        batch_size * grad_accum, batch_size, grad_accum,
    )

    # ── Multi-GPU (DataParallel) ──
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    if n_gpus > 1:
        backbone = nn.DataParallel(backbone)
        head = nn.DataParallel(head)
        logger.info("Using %d GPUs via DataParallel", n_gpus)

    # ── Optimiser ──
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, backbone.parameters()))
        + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    n_opt_steps_per_epoch = math.ceil(len(loader) / grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * n_opt_steps_per_epoch, eta_min=lr * 0.01,
    )

    # ── Validation index (built once before training) ──
    _val_coco_root = val_coco_root if val_coco_root else coco_root
    val_records: List[Tuple[int, str]] = []
    if n_val_images > 0:
        val_records = _build_val_records(_val_coco_root, n_images=n_val_images)

    # ── Training Loop ──
    h_patches = img_size // patch_size
    w_patches = img_size // patch_size
    best_loss = float("inf")
    best_miou = -1.0
    history = []

    # Helper: unwrap DataParallel for state_dict access
    def _state_dict(m: nn.Module) -> dict:
        return m.module.state_dict() if isinstance(m, nn.DataParallel) else m.state_dict()

    trainable_params = (
        list(filter(lambda p: p.requires_grad, backbone.parameters()))
        + list(head.parameters())
    )

    for epoch in range(1, epochs + 1):
        backbone.train()
        head.train()
        epoch_loss = 0.0
        opt_steps = 0       # optimizer (effective) steps this epoch
        accum_loss = 0.0    # accumulates scaled loss within one grad_accum window
        t_epoch = time.time()

        optimizer.zero_grad()

        for step, images in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{epochs}")):
            images = images.to(device_t)

            # Extract patch features (with gradients for unfrozen blocks)
            try:
                out = backbone(pixel_values=images)
                tokens = out.last_hidden_state
                n_special = tokens.shape[1] - h_patches * w_patches
                patch_tokens = tokens[:, n_special:, :]  # (B, N, D)
            except Exception:
                feats = backbone.forward_features(images)
                patch_tokens = feats["x_norm_patchtokens"]  # (B, N, D)

            B, N, D = patch_tokens.shape
            # Project: (B*N, D) → (B*N, proj_dim)
            proj = head(patch_tokens.reshape(B * N, D))
            embeddings = proj.reshape(B, N, proj_dim)  # (B, N, proj_dim)

            # Scale loss so gradients match full effective batch
            loss = neco_loss(
                embeddings,
                h_patches=h_patches,
                w_patches=w_patches,
                r_pos=r_pos,
                r_neg=r_neg,
                temperature=temperature,
                n_pairs_per_image=n_pairs_per_image,
            ) / grad_accum
            loss.backward()
            accum_loss += loss.item()

            is_last_step = (step + 1 == len(loader))
            if (step + 1) % grad_accum == 0 or is_last_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += accum_loss
                opt_steps += 1
                accum_loss = 0.0

                if opt_steps % log_interval == 0:
                    avg = epoch_loss / opt_steps
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        "Epoch %d opt_step %d | data_step %d/%d | loss=%.4f | lr=%.2e",
                        epoch, opt_steps, step + 1, len(loader), avg, lr_now,
                    )

        epoch_avg = epoch_loss / max(opt_steps, 1)
        elapsed = time.time() - t_epoch
        logger.info(
            "Epoch %d done | avg_loss=%.4f | opt_steps=%d | time=%.1fs",
            epoch, epoch_avg, opt_steps, elapsed,
        )

        # ── Per-epoch validation ──
        val_metrics: Dict[str, float] = {}
        if val_records:
            logger.info("Epoch %d | running mIoU validation on %d images …", epoch, len(val_records))
            t_val = time.time()
            val_metrics = validate_epoch_miou(
                backbone=backbone,
                coco_root=_val_coco_root,
                device_t=device_t,
                val_records=val_records,
                img_size=img_size,
                patch_size=patch_size,
                k_clusters=val_k_clusters,
            )
            logger.info(
                "Epoch %d | val mIoU=%.2f%% | things=%.2f%% | stuff=%.2f%% | val_time=%.1fs",
                epoch,
                val_metrics["miou"],
                val_metrics["things_miou"],
                val_metrics["stuff_miou"],
                time.time() - t_val,
            )

        history.append({
            "epoch": epoch,
            "loss": epoch_avg,
            "opt_steps": opt_steps,
            **val_metrics,
        })

        # Save per-epoch checkpoint
        ckpt_path = output_p / f"neco_epoch{epoch:02d}.pth"
        torch.save({
            "epoch": epoch,
            "backbone_state": _state_dict(backbone),
            "head_state": _state_dict(head),
            "optimizer_state": optimizer.state_dict(),
            "loss": epoch_avg,
            **val_metrics,
        }, ckpt_path)
        logger.info("Saved checkpoint: %s", ckpt_path)

        # Best checkpoint: prefer mIoU (if validation ran) else fall back to train loss
        is_best = False
        if val_metrics:
            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                is_best = True
        elif epoch_avg < best_loss:
            best_loss = epoch_avg
            is_best = True

        if is_best:
            best_path = output_p / "neco_best.pth"
            torch.save({
                "epoch": epoch,
                "backbone_state": _state_dict(backbone),
                "head_state": _state_dict(head),
                "loss": epoch_avg,
                **val_metrics,
            }, best_path)
            criterion = f"mIoU={best_miou:.2f}%" if val_metrics else f"loss={best_loss:.4f}"
            logger.info("New best checkpoint: %s → %s", criterion, best_path)

    # Save training history
    with open(output_p / "neco_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. Best loss: %.4f", best_loss)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeCo fine-tuning of DINOv3")
    p.add_argument("--coco_root", required=True, help="COCO dataset root")
    p.add_argument("--output_dir", required=True, help="Checkpoint output directory")
    p.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--img_size", type=int, default=448,
                   help="Input size (must be divisible by patch_size; 448=32×14)")
    p.add_argument("--patch_size", type=int, default=14,
                   help="ViT patch size (14 for ViT-B/14, 16 for ViT-B/16)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--r_pos", type=int, default=2,
                   help="Positive pair L∞ radius in patches")
    p.add_argument("--r_neg", type=int, default=6,
                   help="Min L∞ distance for negative pairs in patches")
    p.add_argument("--n_pairs", type=int, default=64,
                   help="Anchor patches sampled per image per batch")
    p.add_argument("--proj_dim", type=int, default=256,
                   help="Projection head output dimension")
    p.add_argument("--unfreeze_blocks", type=int, default=4,
                   help="Number of last ViT blocks to unfreeze for fine-tuning")
    p.add_argument("--n_images", type=int, default=None,
                   help="Limit dataset size (for smoke tests)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective_bs = batch_size * grad_accum)")
    p.add_argument("--n_val_images", type=int, default=200,
                   help="COCO val images for per-epoch mIoU validation (0 = disable)")
    p.add_argument("--val_k_clusters", type=int, default=54,
                   help="K for k-means in validation (54 = best overclustering for COCO-Stuff-27)")
    p.add_argument("--val_coco_root", type=str, default=None,
                   help="COCO root for validation (panoptic annotations + val2017 images). "
                        "Defaults to --coco_root if not set. Useful when train/val data live "
                        "on different mounts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_neco(
        coco_root=args.coco_root,
        output_dir=args.output_dir,
        img_size=args.img_size,
        patch_size=args.patch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        r_pos=args.r_pos,
        r_neg=args.r_neg,
        n_pairs_per_image=args.n_pairs,
        proj_dim=args.proj_dim,
        unfreeze_last_n_blocks=args.unfreeze_blocks,
        n_images=args.n_images,
        device=args.device,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        grad_accum=args.grad_accum,
        n_val_images=args.n_val_images,
        val_k_clusters=args.val_k_clusters,
        val_coco_root=args.val_coco_root,
    )


if __name__ == "__main__":
    main()
