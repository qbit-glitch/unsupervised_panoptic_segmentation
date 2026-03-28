#!/usr/bin/env python3
"""Train a lightweight mobile panoptic segmentation model.

Modular training script supporting multiple ablation configurations:
  - Augmentation levels: minimal, full (CUPS-style photometric + multi-scale)
  - Instance heads: none, embedding, center_offset, boundary, and combinations
  - Self-training: EMA teacher-student after initial convergence
  - CUPS training tricks: DropLoss, copy-paste, resolution jitter, BiFPN,
    norm WD separation, EMA teacher with TTA, per-class confidence thresholding

All ablations are controlled via CLI arguments — no code changes needed.

Usage examples:
  # Vanilla baseline (semantic-only, minimal augmentation)
  python train_mobile_panoptic.py --cityscapes_root ... --instance_head none --augmentation minimal

  # Augmented baseline (semantic-only, full augmentation)
  python train_mobile_panoptic.py --cityscapes_root ... --instance_head none --augmentation full

  # Instance ablation I-B (center/offset head)
  python train_mobile_panoptic.py --cityscapes_root ... --instance_head center_offset --augmentation full \
      --instance_subdir instance_targets --lambda_instance 1.0

  # Pairwise I-BC (center/offset + boundary)
  python train_mobile_panoptic.py --cityscapes_root ... --instance_head center_boundary --augmentation full \
      --instance_subdir instance_targets

  # Full CUPS recipe (semantic-only + all tricks + self-training)
  python train_mobile_panoptic.py --cityscapes_root ... --instance_head none --augmentation cups \
      --fpn_type bifpn --cups_stage2 --cups_stage3

  # CUPS recipe ablation (no DropLoss)
  python train_mobile_panoptic.py --cityscapes_root ... --augmentation cups \
      --fpn_type bifpn --cups_stage2 --no_droploss
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from tqdm import tqdm
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# Cityscapes Constants
# ═══════════════════════════════════════════════════════════════════════════════

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
_STUFF_IDS = set(range(0, 11))   # trainIDs 0-10
_THING_IDS = set(range(11, 19))  # trainIDs 11-18
_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# Discrete resolution levels (capped at 1.25x max)
_RESOLUTION_JITTER_LEVELS = [0.5, 0.75, 1.0, 1.25]


# ═══════════════════════════════════════════════════════════════════════════════
# BiFPN (Bidirectional Feature Pyramid Network)
# ═══════════════════════════════════════════════════════════════════════════════

class BiFPN(nn.Module):
    """BiFPN with learnable weighted fusion (EfficientDet-style)."""

    def __init__(self, in_channels_list, fpn_dim=128, num_repeats=2):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in in_channels_list
        ])
        self.num_repeats = num_repeats
        self.td_weights = nn.ParameterList()
        self.bu_weights = nn.ParameterList()
        self.td_convs = nn.ModuleList()
        self.bu_convs = nn.ModuleList()
        for _ in range(num_repeats):
            td_w = nn.ParameterList([
                nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)
            ])
            bu_w = nn.ParameterList([
                nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)
            ])
            td_c = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim),
                    nn.Conv2d(fpn_dim, fpn_dim, 1),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                ) for _ in range(self.num_levels - 1)
            ])
            bu_c = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim),
                    nn.Conv2d(fpn_dim, fpn_dim, 1),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                ) for _ in range(self.num_levels - 1)
            ])
            self.td_weights.append(td_w)
            self.bu_weights.append(bu_w)
            self.td_convs.append(td_c)
            self.bu_convs.append(bu_c)

    def forward(self, features):
        eps = 1e-4
        feats = [l(f) for l, f in zip(self.laterals, features)]
        for r in range(self.num_repeats):
            td_feats = list(feats)
            for i in range(self.num_levels - 2, -1, -1):
                w = F.relu(self.td_weights[r][i])
                w = w / (w.sum() + eps)
                up = F.interpolate(td_feats[i + 1], size=td_feats[i].shape[2:],
                                   mode='bilinear', align_corners=False)
                td_feats[i] = self.td_convs[r][i](w[0] * td_feats[i] + w[1] * up)
            bu_feats = list(td_feats)
            for i in range(1, self.num_levels):
                w = F.relu(self.bu_weights[r][i - 1])
                w = w / (w.sum() + eps)
                down = F.interpolate(bu_feats[i - 1], size=bu_feats[i].shape[2:],
                                     mode='bilinear', align_corners=False)
                bu_feats[i] = self.bu_convs[r][i - 1](w[0] * bu_feats[i] + w[1] * down)
            feats = bu_feats
        return feats


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS: DropLoss (Pixel-Level)
# ═══════════════════════════════════════════════════════════════════════════════

class DropLoss(nn.Module):
    """Pixel-level DropLoss for thing classes (CUPS adaptation).

    Drops loss on confident thing-class pixels, focusing learning on
    hard/uncertain pixels.
    """

    def __init__(self, num_classes=19, thing_ids=None, ignore_index=255,
                 drop_rate=0.3, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.thing_ids = thing_ids or _THING_IDS
        self.ignore_index = ignore_index
        self.drop_rate = drop_rate
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        ce_loss = F.cross_entropy(
            logits, targets, ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        thing_mask = torch.zeros_like(targets, dtype=torch.bool)
        for tid in self.thing_ids:
            thing_mask |= (targets == tid)
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            max_prob, _ = probs.max(dim=1)
            drop_mask = torch.zeros_like(targets, dtype=torch.bool)
            for b in range(B):
                thing_px = thing_mask[b]
                if thing_px.sum() == 0:
                    continue
                confidences = max_prob[b][thing_px]
                n_drop = int(confidences.numel() * self.drop_rate)
                if n_drop > 0:
                    _, top_idx = confidences.topk(n_drop)
                    thing_locs = thing_px.nonzero(as_tuple=False)
                    drop_locs = thing_locs[top_idx]
                    drop_mask[b, drop_locs[:, 0], drop_locs[:, 1]] = True
        ce_loss[drop_mask] = 0.0
        valid = (targets != self.ignore_index) & (~drop_mask)
        if valid.sum() == 0:
            return ce_loss.sum() * 0.0
        return ce_loss[valid].mean()


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS: Copy-Paste Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

class CopyPasteAugmentation:
    """Instance-level copy-paste augmentation (CUPS adaptation)."""

    def __init__(self, thing_ids=None, paste_prob=0.5, max_instances=3,
                 min_area=200):
        self.thing_ids = thing_ids or _THING_IDS
        self.paste_prob = paste_prob
        self.max_instances = max_instances
        self.min_area = min_area
        self.instance_bank = []
        self.max_bank_size = 200

    def update_bank(self, image, sem_label, inst_ids):
        if inst_ids is None:
            return
        for uid in np.unique(inst_ids):
            if uid <= 0:
                continue
            mask = inst_ids == uid
            if mask.sum() < self.min_area:
                continue
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            cls_vals = sem_label[mask]
            cls = int(np.median(cls_vals))
            if cls not in self.thing_ids:
                continue
            crop_img = image[y0:y1, x0:x1].copy()
            crop_mask = mask[y0:y1, x0:x1].copy()
            self.instance_bank.append((crop_img, crop_mask, cls))
            if len(self.instance_bank) > self.max_bank_size:
                self.instance_bank.pop(0)

    def apply(self, image, sem_label, inst_ids=None):
        if random.random() > self.paste_prob:
            return image, sem_label, inst_ids
        if len(self.instance_bank) == 0:
            return image, sem_label, inst_ids
        H, W = image.shape[:2]
        n_paste = random.randint(1, min(self.max_instances, len(self.instance_bank)))
        instances = random.sample(self.instance_bank, n_paste)
        img_out = image.copy()
        sem_out = sem_label.copy()
        inst_out = inst_ids.copy() if inst_ids is not None else None
        next_id = (inst_ids.max() + 1) if inst_ids is not None else 1000
        for crop_img, crop_mask, cls in instances:
            ch, cw = crop_mask.shape
            if ch >= H or cw >= W:
                continue
            scale = random.uniform(0.8, 1.2)
            new_h, new_w = int(ch * scale), int(cw * scale)
            if new_h < 10 or new_w < 10 or new_h >= H or new_w >= W:
                continue
            crop_img_r = np.array(Image.fromarray(
                (crop_img * 255).astype(np.uint8)
            ).resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0
            crop_mask_r = np.array(Image.fromarray(
                crop_mask.astype(np.uint8) * 255
            ).resize((new_w, new_h), Image.NEAREST)) > 128
            y = random.randint(0, H - new_h)
            x = random.randint(0, W - new_w)
            paste_region = crop_mask_r
            img_out[y:y+new_h, x:x+new_w][paste_region] = crop_img_r[paste_region]
            sem_out[y:y+new_h, x:x+new_w][paste_region] = cls
            if inst_out is not None:
                inst_out[y:y+new_h, x:x+new_w][paste_region] = next_id
                next_id += 1
        return img_out, sem_out, inst_out


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS: EMA Teacher with TTA (Stage-3)
# ═══════════════════════════════════════════════════════════════════════════════

class EMATeacher:
    """EMA teacher with TTA inference for Stage-3 self-training.

    Multi-scale (0.75x, 1.0x, 1.25x) + horizontal flip = 6 forward passes.
    Per-class confidence thresholding with escalation across rounds.
    """

    def __init__(self, model, momentum=0.999, num_classes=19, use_tta=True):
        self.teacher = copy.deepcopy(model)
        self.teacher.requires_grad_(False)
        self.teacher.train(False)
        self.momentum = momentum
        self.num_classes = num_classes
        self.use_tta = use_tta
        self.class_thresholds = torch.ones(num_classes) * 0.7

    @torch.no_grad()
    def update(self, student):
        for ema_p, s_p in zip(self.teacher.parameters(), student.parameters()):
            ema_p.data.mul_(self.momentum).add_(s_p.data, alpha=1.0 - self.momentum)

    def update_thresholds(self, round_idx, base_threshold=0.7, increment=0.05):
        new_thresh = base_threshold + round_idx * increment
        self.class_thresholds.fill_(min(new_thresh, 0.95))

    @torch.no_grad()
    def generate_pseudo_labels(self, images, ignore_index=255):
        device = images.device
        B, _, H, W = images.shape
        if self.use_tta:
            all_probs = []
            scales = [0.75, 1.0, 1.25]
            for s in scales:
                for flip in [False, True]:
                    inp = images
                    if s != 1.0:
                        sh, sw = int(H * s), int(W * s)
                        inp = F.interpolate(inp, size=(sh, sw), mode='bilinear',
                                            align_corners=False)
                    if flip:
                        inp = torch.flip(inp, dims=[3])
                    out = self.teacher(inp)
                    logits = out["logits"]
                    logits = F.interpolate(logits, size=(H, W), mode='bilinear',
                                           align_corners=False)
                    if flip:
                        logits = torch.flip(logits, dims=[3])
                    all_probs.append(F.softmax(logits, dim=1))
            avg_probs = torch.stack(all_probs).mean(dim=0)
        else:
            out = self.teacher(images)
            logits = out["logits"]
            logits = F.interpolate(logits, size=(H, W), mode='bilinear',
                                   align_corners=False)
            avg_probs = F.softmax(logits, dim=1)
        confidence, preds = avg_probs.max(dim=1)
        thresholds = self.class_thresholds.to(device)
        for c in range(self.num_classes):
            cls_mask = preds == c
            low_conf = cls_mask & (confidence < thresholds[c])
            preds[low_conf] = ignore_index
        return preds


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS: Norm Weight Decay Separation
# ═══════════════════════════════════════════════════════════════════════════════

def build_cups_optimizer(model, lr=1e-3, weight_decay=1e-4, backbone_lr_ratio=0.1):
    """Build AdamW with norm/bias weight decay separation (CUPS style)."""
    decay_params, no_decay_params = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        is_norm_or_bias = ("norm" in name or "bn" in name or
                           name.endswith(".bias") or "gamma" in name or "beta" in name)
        if is_backbone:
            (backbone_no_decay if is_norm_or_bias else backbone_decay).append(param)
        else:
            (no_decay_params if is_norm_or_bias else decay_params).append(param)

    param_groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        {"params": backbone_decay, "lr": lr * backbone_lr_ratio, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr * backbone_lr_ratio, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in param_groups if len(g["params"]) > 0])


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS: IGNORE_UNKNOWN_THING_REGIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ignore_unknown_thing_regions(sem_labels, inst_labels, thing_ids=None,
                                        ignore_index=255):
    """Set thing-class pixels without instance assignment to ignore."""
    thing_ids = thing_ids or _THING_IDS
    modified = sem_labels.clone()
    thing_mask = torch.zeros_like(sem_labels, dtype=torch.bool)
    for tid in thing_ids:
        thing_mask |= (sem_labels == tid)
    no_instance = (inst_labels == 0) | (inst_labels < 0)
    modified[thing_mask & no_instance] = ignore_index
    return modified


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class CityscapesPseudoLabelDataset(Dataset):
    """Cityscapes dataset with pseudo-label targets and optional instance targets."""

    def __init__(self, cityscapes_root, split="train", semantic_subdir="pseudo_semantic_mapped_k80",
                 instance_subdir=None, crop_size=(384, 768), is_train=True,
                 augmentation="minimal", copy_paste=None):
        self.root = Path(cityscapes_root)
        self.split = split
        self.is_train = is_train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.instance_subdir = instance_subdir
        self.copy_paste = copy_paste

        # Find all images
        img_dir = self.root / "leftImg8bit" / split
        self.images = sorted(img_dir.glob("*/*_leftImg8bit.png"))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        # Map to pseudo-labels
        self.labels = []
        for img_path in self.images:
            city = img_path.parent.name
            stem = img_path.name.replace("_leftImg8bit.png", "")
            label_path = self.root / semantic_subdir / split / city / f"{stem}_pseudo_semantic.png"
            if not label_path.exists():
                label_path = self.root / semantic_subdir / split / city / f"{stem}.png"
            self.labels.append(label_path)

        existing = sum(1 for p in self.labels if p.exists())
        print(f"[Dataset] {split}: {len(self.images)} images, {existing}/{len(self.labels)} labels found")
        if existing == 0:
            raise RuntimeError(f"No pseudo-labels found in {self.root / semantic_subdir / split}")

        # Check instance targets
        if instance_subdir:
            inst_dir = self.root / instance_subdir / split
            if inst_dir.exists():
                print(f"[Dataset] Instance targets: {inst_dir}")
            else:
                print(f"[Dataset] WARNING: Instance target dir not found: {inst_dir}")
                self.instance_subdir = None

        # Photometric augmentations (only used if augmentation="full")
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))

        # Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("RGB"), dtype=np.float32) / 255.0
        label_path = self.labels[idx]
        if label_path.exists():
            label = np.array(Image.open(label_path), dtype=np.int64)
        else:
            label = np.full(img.shape[:2], 255, dtype=np.int64)

        H, W = img.shape[:2]

        # Load instance targets if available
        center, offset, boundary, inst_ids = None, None, None, None
        if self.instance_subdir and self.is_train:
            city = self.images[idx].parent.name
            stem = self.images[idx].name.replace("_leftImg8bit.png", "")
            base = self.root / self.instance_subdir / self.split / city
            try:
                center = np.load(base / f"{stem}_center.npy")
                offset = np.load(base / f"{stem}_offset.npy")
                boundary = np.load(base / f"{stem}_boundary.npy")
            except FileNotFoundError:
                center, offset, boundary = None, None, None

            # Also load raw instance IDs for embedding loss
            inst_dir = self.root / "cups_pseudo_labels_v3"
            inst_path = inst_dir / f"{stem}_leftImg8bit_instance.png"
            if inst_path.exists():
                inst_ids = np.array(Image.open(inst_path), dtype=np.int32)
                # Resize to match image if needed
                if inst_ids.shape != (H, W):
                    inst_ids = np.array(
                        Image.fromarray(inst_ids.astype(np.uint16)).resize((W, H), Image.NEAREST),
                        dtype=np.int32
                    )

        if self.is_train:
            # ── Copy-paste augmentation (CUPS) ──
            if self.copy_paste is not None and inst_ids is not None:
                self.copy_paste.update_bank(img, label, inst_ids)
                img, label, inst_ids = self.copy_paste.apply(img, label, inst_ids)

            # ── Horizontal flip ──
            flip = np.random.rand() > 0.5
            if flip:
                img = img[:, ::-1].copy()
                label = label[:, ::-1].copy()
                if center is not None:
                    center = center[:, ::-1].copy()
                    offset = offset[:, :, ::-1].copy()
                    offset[1] = -offset[1]  # flip dx
                    boundary = boundary[:, ::-1].copy()
                if inst_ids is not None:
                    inst_ids = inst_ids[:, ::-1].copy()

            # ── Multi-scale resize ──
            if self.augmentation == "cups":
                scale = random.choice(_RESOLUTION_JITTER_LEVELS)
            elif self.augmentation == "full":
                scale = np.random.uniform(0.5, 1.25)
            else:
                scale = None  # minimal: no multi-scale

            if scale is not None:
                new_h, new_w = int(H * scale), int(W * scale)
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
                img = np.array(img_pil, dtype=np.float32) / 255.0

                label_pil = Image.fromarray(label.astype(np.uint8))
                label_pil = label_pil.resize((new_w, new_h), Image.NEAREST)
                label = np.array(label_pil, dtype=np.int64)

                if center is not None:
                    center = np.array(Image.fromarray(center).resize((new_w, new_h), Image.BILINEAR))
                    off_dy = np.array(Image.fromarray(offset[0]).resize((new_w, new_h), Image.BILINEAR)) * scale
                    off_dx = np.array(Image.fromarray(offset[1]).resize((new_w, new_h), Image.BILINEAR)) * scale
                    offset = np.stack([off_dy, off_dx], axis=0)
                    boundary = np.array(
                        Image.fromarray(boundary).resize((new_w, new_h), Image.NEAREST)
                    )
                if inst_ids is not None:
                    inst_ids = np.array(
                        Image.fromarray(inst_ids.astype(np.uint16)).resize((new_w, new_h), Image.NEAREST),
                        dtype=np.int32
                    )

                H, W = img.shape[:2]

            # ── Random crop ──
            ch, cw = self.crop_size
            if H >= ch and W >= cw:
                y = np.random.randint(0, H - ch + 1)
                x = np.random.randint(0, W - cw + 1)
                img = img[y:y+ch, x:x+cw]
                label = label[y:y+ch, x:x+cw]
                if center is not None:
                    center = center[y:y+ch, x:x+cw]
                    offset = offset[:, y:y+ch, x:x+cw]
                    boundary = boundary[y:y+ch, x:x+cw]
                if inst_ids is not None:
                    inst_ids = inst_ids[y:y+ch, x:x+cw]
            else:
                # Pad/resize to crop size
                img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize((cw, ch), Image.BILINEAR)
                img = np.array(img_pil, dtype=np.float32) / 255.0
                label = np.array(
                    Image.fromarray(label.astype(np.uint8)).resize((cw, ch), Image.NEAREST),
                    dtype=np.int64
                )
                if center is not None:
                    center = np.array(Image.fromarray(center).resize((cw, ch), Image.BILINEAR))
                    off_dy = np.array(Image.fromarray(offset[0]).resize((cw, ch), Image.BILINEAR))
                    off_dx = np.array(Image.fromarray(offset[1]).resize((cw, ch), Image.BILINEAR))
                    offset = np.stack([off_dy, off_dx], axis=0)
                    boundary = np.array(
                        Image.fromarray(boundary).resize((cw, ch), Image.NEAREST)
                    )
                if inst_ids is not None:
                    inst_ids = np.array(
                        Image.fromarray(inst_ids.astype(np.uint16)).resize((cw, ch), Image.NEAREST),
                        dtype=np.int32
                    )

            # ── Photometric augmentations ──
            if self.augmentation in ("full", "cups"):
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                if np.random.rand() > 0.5:
                    img_pil = self.color_jitter(img_pil)
                if np.random.rand() > 0.5:
                    img_pil = self.gaussian_blur(img_pil)
                if np.random.rand() > 0.8:
                    img_pil = transforms.functional.rgb_to_grayscale(img_pil, num_output_channels=3)
                img = np.array(img_pil, dtype=np.float32) / 255.0
            elif self.augmentation == "minimal":
                if np.random.rand() > 0.5:
                    img = img * (0.8 + 0.4 * np.random.rand())
                    img = np.clip(img, 0, 1)

        else:
            # Validation: resize to fixed size
            target_h, target_w = 512, 1024
            img = np.array(
                Image.fromarray((img * 255).astype(np.uint8)).resize((target_w, target_h), Image.BILINEAR),
                dtype=np.float32
            ) / 255.0
            label = np.array(
                Image.fromarray(label.astype(np.uint8)).resize((target_w, target_h), Image.NEAREST),
                dtype=np.int64
            )

        # To tensors
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_t = self.normalize(img_t)
        label_t = torch.from_numpy(label).long()

        result = {"image": img_t, "label": label_t}

        if center is not None:
            result["center"] = torch.from_numpy(center.copy()).float()
            result["offset"] = torch.from_numpy(offset.copy()).float()
            result["boundary"] = torch.from_numpy(boundary.copy()).float()
        if inst_ids is not None:
            result["inst_ids"] = torch.from_numpy(inst_ids.copy()).long()

        return result


def collate_fn(batch):
    """Custom collate that handles dict-based samples with optional keys."""
    result = {}
    keys = batch[0].keys()
    for k in keys:
        result[k] = torch.stack([b[k] for b in batch if k in b])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Instance Heads
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingHead(nn.Module):
    """16-dim discriminative embedding head. +0.10M params."""
    def __init__(self, in_dim=128, embed_dim=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),  # depthwise
            nn.Conv2d(in_dim, in_dim, 1),  # pointwise
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, embed_dim, 1),
        )

    def forward(self, x):
        return self.head(x)  # (B, embed_dim, H, W)


class CenterOffsetHead(nn.Module):
    """Center heatmap (1-ch) + offset (2-ch) head. +0.15M params."""
    def __init__(self, in_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.center_conv = nn.Conv2d(in_dim, 1, 1)
        self.offset_conv = nn.Conv2d(in_dim, 2, 1)

    def forward(self, x):
        feat = self.shared(x)
        center = torch.sigmoid(self.center_conv(feat))  # (B, 1, H, W)
        offset = self.offset_conv(feat)                  # (B, 2, H, W)
        return center, offset


class BoundaryHead(nn.Module):
    """Instance boundary prediction (1-ch) head. +0.05M params."""
    def __init__(self, in_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))  # (B, 1, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleFPN(nn.Module):
    """Lightweight Feature Pyramid Network."""
    def __init__(self, in_channels_list, fpn_dim=128):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in in_channels_list
        ])
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim),
                nn.Conv2d(fpn_dim, fpn_dim, 1),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ) for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False
            )
        return [s(l) for s, l in zip(self.smooths, laterals)]


class MobilePanopticModel(nn.Module):
    """Lightweight panoptic model: timm backbone + FPN + semantic head + instance heads."""

    def __init__(self, backbone_name, num_classes=19, fpn_dim=128, pretrained=True,
                 instance_head="none", fpn_type="simple", feature_levels=None):
        super().__init__()
        self.instance_head_type = instance_head

        # Backbone
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        channels = self.backbone.feature_info.channels()
        print(f"[Model] Backbone: {backbone_name}, channels: {channels}")

        # Select feature levels (for 5-level backbones like MobileNetV4,
        # skip level 0 at 1/2 res to save memory — use levels 1-4)
        self.feature_levels = feature_levels
        if feature_levels is not None:
            channels = [channels[i] for i in feature_levels]
            print(f"[Model] Using feature levels {feature_levels}, channels: {channels}")

        # FPN
        if fpn_type == "bifpn":
            self.fpn = BiFPN(channels, fpn_dim, num_repeats=2)
            print(f"[Model] FPN: BiFPN (2 repeats)")
        else:
            self.fpn = SimpleFPN(channels, fpn_dim)
            print(f"[Model] FPN: SimpleFPN")

        # Semantic head
        self.sem_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim),
            nn.Conv2d(fpn_dim, fpn_dim, 1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

        # Instance heads (conditionally created based on --instance_head)
        self.embedding_head = None
        self.center_offset_head = None
        self.boundary_head = None

        heads = _parse_instance_heads(instance_head)
        if "embedding" in heads:
            self.embedding_head = EmbeddingHead(fpn_dim, embed_dim=16)
        if "center_offset" in heads:
            self.center_offset_head = CenterOffsetHead(fpn_dim)
        if "boundary" in heads:
            self.boundary_head = BoundaryHead(fpn_dim)

        # Print param counts
        bb_p = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        dec_p = sum(p.numel() for p in self.fpn.parameters()) / 1e6
        dec_p += sum(p.numel() for p in self.sem_head.parameters()) / 1e6
        inst_p = 0
        for h in [self.embedding_head, self.center_offset_head, self.boundary_head]:
            if h is not None:
                inst_p += sum(p.numel() for p in h.parameters()) / 1e6
        total = bb_p + dec_p + inst_p
        print(f"[Model] Params: backbone={bb_p:.2f}M, decoder={dec_p:.2f}M, "
              f"instance={inst_p:.2f}M, total={total:.2f}M")
        print(f"[Model] Instance heads: {instance_head}")

    def forward(self, x):
        features = self.backbone(x)
        if self.feature_levels is not None:
            features = [features[i] for i in self.feature_levels]
        fpn_feats = self.fpn(features)
        finest = fpn_feats[0]  # finest scale

        result = {"logits": self.sem_head(finest)}

        if self.embedding_head is not None:
            result["embeddings"] = self.embedding_head(finest)
        if self.center_offset_head is not None:
            center, offset = self.center_offset_head(finest)
            result["center"] = center
            result["offset"] = offset
        if self.boundary_head is not None:
            result["boundary"] = self.boundary_head(finest)

        return result


def _parse_instance_heads(head_str):
    """Parse instance head string into set of active heads."""
    mapping = {
        "none": set(),
        "embedding": {"embedding"},
        "center_offset": {"center_offset"},
        "boundary": {"boundary"},
        "embed_center": {"embedding", "center_offset"},
        "embed_boundary": {"embedding", "boundary"},
        "center_boundary": {"center_offset", "boundary"},
        "all": {"embedding", "center_offset", "boundary"},
    }
    return mapping.get(head_str, set())


# ═══════════════════════════════════════════════════════════════════════════════
# Instance Losses
# ═══════════════════════════════════════════════════════════════════════════════

def discriminative_loss(embeddings, instance_ids, delta_v=0.5, delta_d=1.5):
    """Discriminative loss for embedding head [De Brabandere et al., 2017].

    Args:
        embeddings: (B, E, H, W) embedding predictions
        instance_ids: (B, H, W) instance IDs (0=ignore/background)
    Returns:
        Scalar loss
    """
    B = embeddings.shape[0]
    total_loss = 0.0
    valid_batches = 0

    for b in range(B):
        emb = embeddings[b]       # (E, H, W)
        inst = instance_ids[b]    # (H, W)
        ids = inst.unique()
        ids = ids[ids > 0]

        if len(ids) < 2:
            continue

        means = []
        pull_loss = torch.tensor(0.0, device=emb.device)

        for uid in ids:
            mask = inst == uid
            emb_k = emb[:, mask]  # (E, N_k)
            mu_k = emb_k.mean(dim=1, keepdim=True)  # (E, 1)
            means.append(mu_k.squeeze(1))
            # Pull: each pixel toward its cluster mean
            dist = torch.norm(emb_k - mu_k, dim=0)  # (N_k,)
            pull_loss = pull_loss + torch.clamp(dist - delta_v, min=0).pow(2).mean()

        pull_loss = pull_loss / len(ids)

        # Push: cluster means apart
        means_t = torch.stack(means)  # (K, E)
        K = means_t.shape[0]
        push_loss = torch.tensor(0.0, device=emb.device)
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(means_t[i] - means_t[j])
                push_loss = push_loss + torch.clamp(2 * delta_d - dist, min=0).pow(2)
                n_pairs += 1
        push_loss = push_loss / max(n_pairs, 1)

        # Regularization: keep means small
        reg_loss = means_t.norm(dim=1).mean() * 0.001

        total_loss += pull_loss + push_loss + reg_loss
        valid_batches += 1

    return total_loss / max(valid_batches, 1)


def center_offset_loss(pred_center, pred_offset, target_center, target_offset,
                       thing_mask):
    """Center heatmap MSE + offset SmoothL1 loss.

    Args:
        pred_center:   (B, 1, H, W) predicted center heatmap
        pred_offset:   (B, 2, H, W) predicted offsets
        target_center: (B, H, W) target center heatmap
        target_offset: (B, 2, H, W) target offsets
        thing_mask:    (B, H, W) bool mask for thing-class pixels
    """
    # Interpolate predictions to target size if needed
    if pred_center.shape[2:] != target_center.shape[1:]:
        pred_center = F.interpolate(pred_center, size=target_center.shape[1:],
                                     mode='bilinear', align_corners=False)
        pred_offset = F.interpolate(pred_offset, size=target_center.shape[1:],
                                     mode='bilinear', align_corners=False)

    # Center loss (weighted MSE: 10x weight on positive pixels)
    pc = pred_center.squeeze(1)  # (B, H, W)
    weight = torch.where(target_center > 0.01, 10.0, 1.0)
    center_l = (weight * (pc - target_center) ** 2).mean()

    # Offset loss (SmoothL1, only on thing pixels with valid instances)
    valid = thing_mask & (target_center > 0.01)
    if valid.sum() > 100:
        # Expand valid to match offset channels
        valid_exp = valid.unsqueeze(1).expand_as(pred_offset)
        offset_l = F.smooth_l1_loss(pred_offset[valid_exp], target_offset[valid_exp])
    else:
        offset_l = torch.tensor(0.0, device=pred_center.device)

    return center_l + offset_l


def boundary_loss(pred_boundary, target_boundary, thing_mask):
    """Weighted BCE for boundary prediction.

    Args:
        pred_boundary:  (B, 1, H, W) predicted boundary probability
        target_boundary: (B, H, W) target boundary (0 or 1)
        thing_mask:      (B, H, W) bool mask for thing-class pixels
    """
    if pred_boundary.shape[2:] != target_boundary.shape[1:]:
        pred_boundary = F.interpolate(pred_boundary, size=target_boundary.shape[1:],
                                       mode='bilinear', align_corners=False)

    pred = pred_boundary.squeeze(1)  # (B, H, W)
    mask = thing_mask

    if mask.sum() < 100:
        return torch.tensor(0.0, device=pred.device)

    pred_m = pred[mask]
    target_m = target_boundary[mask].float()

    # Class-balanced weight (boundaries are sparse ~2-5% of thing pixels)
    n_pos = (target_m > 0.5).sum().float().clamp(min=1)
    n_neg = (target_m <= 0.5).sum().float().clamp(min=1)
    pos_weight = n_neg / n_pos
    weight = torch.where(target_m > 0.5, pos_weight, torch.ones_like(target_m))

    return F.binary_cross_entropy(pred_m, target_m, weight=weight)


# ═══════════════════════════════════════════════════════════════════════════════
# Instance-Aware Panoptic Inference
# ═══════════════════════════════════════════════════════════════════════════════

def infer_instances_connected_components(pred_sem, cc_min_area=50):
    """Baseline: connected components on semantic predictions."""
    H, W = pred_sem.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}
    nxt = 1

    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pan_map[mask] = nxt
        segments[nxt] = cls
        nxt += 1

    for cls in _THING_IDS:
        cls_mask = pred_sem == cls
        if cls_mask.sum() < cc_min_area:
            continue
        labeled, n_cc = ndimage.label(cls_mask)
        for comp in range(1, n_cc + 1):
            cmask = labeled == comp
            if cmask.sum() < cc_min_area:
                continue
            pan_map[cmask] = nxt
            segments[nxt] = cls
            nxt += 1

    return pan_map, segments


def infer_instances_center_offset(pred_sem, center_map, offset_map,
                                   center_threshold=0.1, cc_min_area=50, nms_kernel=7):
    """Center/offset-based instance grouping (Panoptic-DeepLab style).

    1. Find local maxima in center heatmap (NMS)
    2. For each pixel, add predicted offset → voted center
    3. Assign each pixel to nearest detected center
    """
    H, W = pred_sem.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}
    nxt = 1

    # Stuff classes: same as baseline
    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pan_map[mask] = nxt
        segments[nxt] = cls
        nxt += 1

    # Thing classes: use center/offset
    # NMS on center heatmap
    center_t = torch.from_numpy(center_map).float().unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(center_t, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
    peaks = ((center_t == pooled) & (center_t > center_threshold)).squeeze().numpy()

    peak_ys, peak_xs = np.where(peaks)
    if len(peak_ys) == 0:
        # Fallback to connected components for things
        for cls in _THING_IDS:
            cls_mask = pred_sem == cls
            if cls_mask.sum() < cc_min_area:
                continue
            labeled, n_cc = ndimage.label(cls_mask)
            for comp in range(1, n_cc + 1):
                cmask = labeled == comp
                if cmask.sum() < cc_min_area:
                    continue
                pan_map[cmask] = nxt
                segments[nxt] = cls
                nxt += 1
        return pan_map, segments

    # Create voted center coordinates
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    voted_y = yy + offset_map[0]  # dy offset
    voted_x = xx + offset_map[1]  # dx offset

    # Build thing mask
    thing_mask = np.zeros((H, W), dtype=bool)
    for cls in _THING_IDS:
        thing_mask |= (pred_sem == cls)

    # Assign each thing pixel to nearest peak
    for py, px in zip(peak_ys, peak_xs):
        if not thing_mask[py, px]:
            continue
        # Find pixels whose voted center is close to this peak
        dist = np.sqrt((voted_y - py) ** 2 + (voted_x - px) ** 2)
        close = (dist < 30) & thing_mask & (pan_map == 0)
        if close.sum() < cc_min_area:
            continue
        cls = pred_sem[py, px]
        if cls not in _THING_IDS:
            # Use majority class of close pixels
            vals, counts = np.unique(pred_sem[close], return_counts=True)
            thing_vals = [(v, c) for v, c in zip(vals, counts) if v in _THING_IDS]
            if not thing_vals:
                continue
            cls = max(thing_vals, key=lambda x: x[1])[0]
        pan_map[close] = nxt
        segments[nxt] = int(cls)
        nxt += 1

    # Assign remaining thing pixels via nearest peak
    remaining = thing_mask & (pan_map == 0)
    if remaining.sum() > 0 and len(peak_ys) > 0:
        rem_ys, rem_xs = np.where(remaining)
        rem_voted_y = voted_y[remaining]
        rem_voted_x = voted_x[remaining]

        # Distance to each peak
        peak_coords = np.stack([peak_ys, peak_xs], axis=1)  # (N_peaks, 2)
        rem_voted = np.stack([rem_voted_y, rem_voted_x], axis=1)  # (N_rem, 2)
        dists = np.linalg.norm(rem_voted[:, None, :] - peak_coords[None, :, :], axis=2)
        nearest = dists.argmin(axis=1)

        for i, (ry, rx) in enumerate(zip(rem_ys, rem_xs)):
            pi = nearest[i]
            py, px = peak_ys[pi], peak_xs[pi]
            # Find which segment this peak belongs to
            seg_id = pan_map[py, px]
            if seg_id > 0:
                pan_map[ry, rx] = seg_id

    return pan_map, segments


def infer_instances_boundary(pred_sem, boundary_map, boundary_threshold=0.5,
                              cc_min_area=50):
    """Boundary-based instance separation.

    1. Subtract predicted boundaries from thing-class mask
    2. Apply connected components to boundary-subtracted mask
    """
    H, W = pred_sem.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}
    nxt = 1

    # Stuff classes
    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pan_map[mask] = nxt
        segments[nxt] = cls
        nxt += 1

    # Thing classes: boundary-subtracted CC
    boundary_mask = boundary_map > boundary_threshold

    for cls in _THING_IDS:
        cls_mask = pred_sem == cls
        if cls_mask.sum() < cc_min_area:
            continue
        # Subtract boundaries
        split_mask = cls_mask & ~boundary_mask
        labeled, n_cc = ndimage.label(split_mask)
        for comp in range(1, n_cc + 1):
            cmask = labeled == comp
            if cmask.sum() < cc_min_area:
                continue
            pan_map[cmask] = nxt
            segments[nxt] = cls
            nxt += 1

        # Assign boundary pixels to nearest segment
        boundary_in_cls = cls_mask & boundary_mask & (pan_map == 0)
        if boundary_in_cls.sum() > 0:
            # Dilate each segment to absorb boundary pixels
            dilated = ndimage.binary_dilation(pan_map > 0, iterations=2)
            # Re-label boundary pixels based on nearest segment
            by, bx = np.where(boundary_in_cls)
            for y, x in zip(by, bx):
                # Check 3x3 neighborhood for assigned pixels
                y0, y1 = max(0, y - 2), min(H, y + 3)
                x0, x1 = max(0, x - 2), min(W, x + 3)
                patch = pan_map[y0:y1, x0:x1]
                vals = patch[patch > 0]
                if len(vals) > 0:
                    pan_map[y, x] = np.bincount(vals).argmax()

    return pan_map, segments


def infer_instances_center_boundary(pred_sem, center_map, offset_map, boundary_map,
                                     center_threshold=0.1, boundary_threshold=0.5,
                                     cc_min_area=50, nms_kernel=7):
    """Combined center/offset + boundary inference.

    1. Find centers via NMS on heatmap
    2. Group pixels via offset voting
    3. Split grouped segments along predicted boundaries
    """
    # First do center-offset grouping
    pan_map, segments = infer_instances_center_offset(
        pred_sem, center_map, offset_map,
        center_threshold=center_threshold, cc_min_area=cc_min_area,
        nms_kernel=nms_kernel
    )

    # Then split along boundaries
    boundary_mask = boundary_map > boundary_threshold
    H, W = pred_sem.shape
    new_pan = pan_map.copy()
    new_segments = dict(segments)
    nxt = max(segments.keys()) + 1 if segments else 1

    for seg_id, cls in list(segments.items()):
        if cls not in _THING_IDS:
            continue
        seg_mask = pan_map == seg_id
        # Check if boundary cuts through this segment
        boundary_in_seg = seg_mask & boundary_mask
        if boundary_in_seg.sum() < 5:
            continue
        # Split segment
        split_mask = seg_mask & ~boundary_mask
        labeled, n_cc = ndimage.label(split_mask)
        if n_cc <= 1:
            continue
        # Re-assign to new segment IDs
        for comp in range(1, n_cc + 1):
            cmask = labeled == comp
            if cmask.sum() < cc_min_area:
                continue
            new_pan[cmask] = nxt
            new_segments[nxt] = cls
            nxt += 1
        # Remove old segment
        old_mask = new_pan == seg_id
        if old_mask.sum() > 0:
            new_pan[old_mask] = 0
        if seg_id in new_segments:
            del new_segments[seg_id]

    return new_pan, new_segments


# ═══════════════════════════════════════════════════════════════════════════════
# EMA Self-Training
# ═══════════════════════════════════════════════════════════════════════════════

class EMAModel:
    """Exponential Moving Average model for self-training."""
    def __init__(self, model, momentum=0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.momentum = momentum

    @torch.no_grad()
    def update(self, student_model):
        for ema_p, student_p in zip(self.model.parameters(), student_model.parameters()):
            ema_p.data.mul_(self.momentum).add_(student_p.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def generate_self_labels(self, images, confidence_threshold=0.9):
        """Generate pseudo-labels from EMA teacher predictions.

        Returns semantic labels with low-confidence pixels set to 255 (ignore).
        """
        out = self.model(images)
        logits = out["logits"]
        probs = F.softmax(logits, dim=1)
        confidence, preds = probs.max(dim=1)

        # Mask low-confidence pixels
        preds[confidence < confidence_threshold] = 255
        return preds, out


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=19, cityscapes_root=None,
             instance_head="none", cc_min_area=50):
    """Evaluate semantic (mIoU) + panoptic (PQ) metrics.

    Uses instance head-specific inference when available.
    """
    model.eval()
    H, W = 512, 1024

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    iou_sum = np.zeros(num_classes)

    gt_dir = os.path.join(cityscapes_root, "gtFine", "val") if cityscapes_root else None
    has_gt = gt_dir is not None and os.path.isdir(gt_dir)
    val_dataset = val_loader.dataset
    img_idx = 0

    heads = _parse_instance_heads(instance_head)

    for batch in tqdm(val_loader, desc="Evaluating", ncols=100, leave=False):
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)

        out = model(imgs)
        logits = out["logits"]
        logits = F.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=False)
        preds = logits.argmax(dim=1)

        # Semantic accuracy against pseudo-labels
        valid = labels != 255
        total_correct = (preds[valid] == labels[valid]).sum().item()
        total_pixels = valid.sum().item()

        if has_gt:
            preds_np = preds.cpu().numpy()

            # Get instance head outputs at eval resolution
            center_np = offset_np = boundary_np = None
            if "center_offset" in heads and "center" in out:
                c = F.interpolate(out["center"], size=(H, W), mode='bilinear', align_corners=False)
                o = F.interpolate(out["offset"], size=(H, W), mode='bilinear', align_corners=False)
                center_np = c.squeeze(1).cpu().numpy()
                offset_np = o.cpu().numpy()
            if "boundary" in heads and "boundary" in out:
                b = F.interpolate(out["boundary"], size=(H, W), mode='bilinear', align_corners=False)
                boundary_np = b.squeeze(1).cpu().numpy()

            B = preds_np.shape[0]
            for i in range(B):
                idx = img_idx + i
                if idx >= len(val_dataset.images):
                    break
                img_path = val_dataset.images[idx]
                city = img_path.parent.name
                stem = img_path.name.replace("_leftImg8bit.png", "")

                pred_sem = preds_np[i]

                # Load GT
                gt_path = os.path.join(gt_dir, city, f"{stem}_gtFine_labelIds.png")
                if not os.path.exists(gt_path):
                    continue
                gt_raw = np.array(Image.open(gt_path))
                gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
                for raw_id, tid in _CS_ID_TO_TRAIN.items():
                    gt_sem[gt_raw == raw_id] = tid
                if gt_sem.shape != (H, W):
                    gt_sem = np.array(Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

                # Confusion matrix
                valid_gt = (gt_sem < num_classes) & (pred_sem < num_classes)
                if valid_gt.sum() > 0:
                    np.add.at(confusion, (gt_sem[valid_gt], pred_sem[valid_gt]), 1)

                # Build predicted panoptic map
                if "center_offset" in heads and "boundary" in heads and \
                   center_np is not None and boundary_np is not None:
                    pred_pan, pred_segments = infer_instances_center_boundary(
                        pred_sem, center_np[i], offset_np[i], boundary_np[i],
                        cc_min_area=cc_min_area
                    )
                elif "center_offset" in heads and center_np is not None:
                    pred_pan, pred_segments = infer_instances_center_offset(
                        pred_sem, center_np[i], offset_np[i],
                        cc_min_area=cc_min_area
                    )
                elif "boundary" in heads and boundary_np is not None:
                    pred_pan, pred_segments = infer_instances_boundary(
                        pred_sem, boundary_np[i], cc_min_area=cc_min_area
                    )
                else:
                    pred_pan, pred_segments = infer_instances_connected_components(
                        pred_sem, cc_min_area=cc_min_area
                    )

                # Build GT panoptic map
                gt_pan = np.zeros((H, W), dtype=np.int32)
                gt_segments = {}
                gt_nxt = 1

                for cls in _STUFF_IDS:
                    mask = gt_sem == cls
                    if mask.sum() < 64:
                        continue
                    gt_pan[mask] = gt_nxt
                    gt_segments[gt_nxt] = cls
                    gt_nxt += 1

                gt_inst_path = os.path.join(gt_dir, city, f"{stem}_gtFine_instanceIds.png")
                if os.path.exists(gt_inst_path):
                    gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
                    if gt_inst.shape != (H, W):
                        gt_inst = np.array(
                            Image.fromarray(gt_inst.astype(np.int32)).resize((W, H), Image.NEAREST),
                            dtype=np.int32
                        )
                    for uid in np.unique(gt_inst):
                        if uid < 1000:
                            continue
                        raw_cls = uid // 1000
                        if raw_cls not in _CS_ID_TO_TRAIN:
                            continue
                        tid = _CS_ID_TO_TRAIN[raw_cls]
                        if tid not in _THING_IDS:
                            continue
                        mask = gt_inst == uid
                        if mask.sum() < 10:
                            continue
                        gt_pan[mask] = gt_nxt
                        gt_segments[gt_nxt] = tid
                        gt_nxt += 1

                # Match segments per category
                gt_by_cat = defaultdict(list)
                for sid, cat in gt_segments.items():
                    gt_by_cat[cat].append(sid)
                pred_by_cat = defaultdict(list)
                for sid, cat in pred_segments.items():
                    pred_by_cat[cat].append(sid)

                matched_pred = set()
                for cat in range(num_classes):
                    for gt_id in gt_by_cat.get(cat, []):
                        gt_mask = gt_pan == gt_id
                        best_iou, best_pid = 0.0, None
                        for pid in pred_by_cat.get(cat, []):
                            if pid in matched_pred:
                                continue
                            inter = np.sum(gt_mask & (pred_pan == pid))
                            union = np.sum(gt_mask | (pred_pan == pid))
                            if union == 0:
                                continue
                            iou_val = inter / union
                            if iou_val > best_iou:
                                best_iou, best_pid = iou_val, pid
                        if best_iou > 0.5 and best_pid is not None:
                            tp[cat] += 1
                            iou_sum[cat] += best_iou
                            matched_pred.add(best_pid)
                        else:
                            fn[cat] += 1
                    for pid in pred_by_cat.get(cat, []):
                        if pid not in matched_pred:
                            fp[cat] += 1

            img_idx += B

    # Compute mIoU
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    miou_gt = iou[union > 0].mean() * 100 if has_gt else 0.0

    # Compute PQ
    all_pq, stuff_pq, thing_pq = [], [], []
    per_class_pq = {}
    for c in range(num_classes):
        t, f_p, f_n, s = tp[c], fp[c], fn[c], iou_sum[c]
        if t + f_p + f_n > 0:
            sq = s / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0
        per_class_pq[_CS_CLASS_NAMES[c]] = round(pq * 100, 2)
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            (stuff_pq if c in _STUFF_IDS else thing_pq).append(pq)

    pq_all = float(np.mean(all_pq)) * 100 if all_pq else 0.0
    pq_stuff = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
    pq_things = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0

    model.train()
    return {
        "mIoU": round(miou_gt, 2),
        "PQ": round(pq_all, 2),
        "PQ_stuff": round(pq_stuff, 2),
        "PQ_things": round(pq_things, 2),
        "per_class_pq": per_class_pq,
        "per_class_iou": {_CS_CLASS_NAMES[i]: round(iou[i] * 100, 2) for i in range(num_classes)},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else
                           ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                            else 'cpu')))
    print(f"Device: {device}")
    print(f"Config: augmentation={args.augmentation}, instance_head={args.instance_head}, "
          f"fpn_type={args.fpn_type}, lambda_instance={args.lambda_instance}")
    if args.cups_stage2:
        print(f"[CUPS] Stage-2: DropLoss={not args.no_droploss}, "
              f"CopyPaste={not args.no_copy_paste}, "
              f"ResJitter={args.augmentation == 'cups'}, "
              f"IgnoreUnknown={not args.no_ignore_unknown}")
    if args.cups_stage3:
        print(f"[CUPS] Stage-3: self-training ({args.st_rounds} rounds x "
              f"{args.st_epochs_per_round} epochs, TTA={args.st_tta})")

    heads = _parse_instance_heads(args.instance_head)

    # ── Copy-paste augmentation ──
    copy_paste = None
    if args.cups_stage2 and not args.no_copy_paste:
        copy_paste = CopyPasteAugmentation(thing_ids=_THING_IDS, paste_prob=0.5)
        print("[CUPS] Copy-paste augmentation enabled")

    # Model — auto-detect 5-level backbones and skip 1/2-res level
    feature_levels = None
    _tmp_bb = timm.create_model(args.backbone, pretrained=False, features_only=True)
    _n_levels = len(_tmp_bb.feature_info.channels())
    del _tmp_bb
    if _n_levels == 5:
        feature_levels = [1, 2, 3, 4]  # skip level 0 (1/2 res) to save memory
        print(f"[Model] 5-level backbone detected, using levels 1-4")

    model = MobilePanopticModel(
        backbone_name=args.backbone,
        num_classes=args.num_classes,
        fpn_dim=args.fpn_dim,
        pretrained=True,
        instance_head=args.instance_head,
        fpn_type=args.fpn_type,
        feature_levels=feature_levels,
    ).to(device)

    # Freeze backbone initially
    if args.freeze_backbone_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print(f"[Train] Backbone frozen for first {args.freeze_backbone_epochs} epochs")

    # Dataset
    train_dataset = CityscapesPseudoLabelDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        instance_subdir=args.instance_subdir if (heads or args.cups_stage2) else None,
        crop_size=(args.crop_h, args.crop_w),
        is_train=True,
        augmentation=args.augmentation,
        copy_paste=copy_paste,
    )
    val_dataset = CityscapesPseudoLabelDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        instance_subdir=None,
        is_train=False,
        augmentation="minimal",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    # Optimizer: CUPS norm WD separation or standard
    if args.cups_stage2:
        optimizer = build_cups_optimizer(
            model, lr=args.lr, weight_decay=args.weight_decay,
            backbone_lr_ratio=args.backbone_lr_ratio,
        )
        print("[CUPS] Optimizer: AdamW with norm/bias WD separation")
    else:
        backbone_params = list(model.backbone.parameters())
        decoder_params = list(model.fpn.parameters()) + list(model.sem_head.parameters())
        instance_params = []
        for h in [model.embedding_head, model.center_offset_head, model.boundary_head]:
            if h is not None:
                instance_params += list(h.parameters())
        param_groups = [
            {'params': decoder_params, 'lr': args.lr},
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_ratio},
        ]
        if instance_params:
            param_groups.append({'params': instance_params, 'lr': args.lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

    # Loss function: DropLoss or standard CE
    if args.cups_stage2 and not args.no_droploss:
        criterion = DropLoss(
            num_classes=args.num_classes, thing_ids=_THING_IDS,
            drop_rate=args.drop_rate, label_smoothing=args.label_smoothing,
        )
        print(f"[CUPS] DropLoss enabled (drop_rate={args.drop_rate})")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=args.label_smoothing)

    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # EMA for self-training (old-style or CUPS stage-3)
    ema = None
    ema_teacher = None
    if args.self_training:
        print(f"[Train] Self-training enabled from epoch {args.self_training_start_epoch}")
    if args.cups_stage3:
        print(f"[CUPS] Stage-3 will start after epoch {args.num_epochs}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(args.output_dir, "config.txt")
    with open(config_path, "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    best_score = 0.0
    best_epoch = 0
    start_epoch = 1

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("[Resume] Optimizer state restored")
            except (ValueError, KeyError):
                print("[Resume] Optimizer state mismatch, using fresh optimizer")
        start_epoch = ckpt['epoch'] + 1
        if 'metrics' in ckpt:
            score = ckpt['metrics'].get('PQ', 0) or ckpt['metrics'].get('mIoU', 0)
            best_score = score
            best_epoch = ckpt['epoch']
        # If resuming past freeze point, ensure backbone is unfrozen
        if start_epoch > args.freeze_backbone_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True
        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}, resuming at epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()

        # Unfreeze backbone with LR warmup
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer to include backbone params
            if args.cups_stage2:
                optimizer = build_cups_optimizer(
                    model, lr=args.lr, weight_decay=args.weight_decay,
                    backbone_lr_ratio=args.backbone_lr_ratio,
                )
            else:
                backbone_params = list(model.backbone.parameters())
                decoder_params = list(model.fpn.parameters()) + list(model.sem_head.parameters())
                instance_params = []
                for h in [model.embedding_head, model.center_offset_head, model.boundary_head]:
                    if h is not None:
                        instance_params += list(h.parameters())
                param_groups = [
                    {'params': decoder_params, 'lr': args.lr},
                    {'params': backbone_params, 'lr': args.lr * args.backbone_lr_ratio},
                ]
                if instance_params:
                    param_groups.append({'params': instance_params, 'lr': args.lr})
                optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            # Rebuild scheduler for remaining epochs
            remaining_steps = len(train_loader) * (args.num_epochs - epoch + 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_steps, eta_min=1e-7
            )
            print(f"[Epoch {epoch}] Backbone unfrozen, optimizer rebuilt")

        # Initialize EMA at self-training start
        if args.self_training and epoch == args.self_training_start_epoch and ema is None:
            ema = EMAModel(model, momentum=args.ema_momentum)
            print(f"[Epoch {epoch}] EMA teacher initialized (momentum={args.ema_momentum})")

        epoch_loss = 0.0
        epoch_sem_loss = 0.0
        epoch_inst_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    ncols=120, leave=True)

        for batch in pbar:
            optimizer.zero_grad()
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # CUPS: IGNORE_UNKNOWN_THING_REGIONS
            if args.cups_stage2 and not args.no_ignore_unknown and "inst_ids" in batch:
                inst_ids_batch = batch["inst_ids"].to(device, non_blocking=True)
                labels = apply_ignore_unknown_thing_regions(labels, inst_ids_batch)

            # Self-training: replace labels with EMA teacher predictions
            if ema is not None and epoch >= args.self_training_start_epoch:
                with torch.no_grad():
                    self_labels, _ = ema.generate_self_labels(
                        imgs, confidence_threshold=args.self_training_confidence
                    )
                    # Merge: use self-labels where confident, original labels elsewhere
                    use_self = self_labels != 255
                    labels = torch.where(use_self, self_labels, labels)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(imgs)
                    logits = out["logits"]
                    logits = F.interpolate(logits, size=labels.shape[1:],
                                           mode='bilinear', align_corners=False)
                    sem_loss = criterion(logits, labels)

                    # Instance losses
                    inst_loss = torch.tensor(0.0, device=device)
                    if heads and "center" in batch:
                        target_center = batch["center"].to(device, non_blocking=True)
                        target_offset = batch["offset"].to(device, non_blocking=True)
                        target_boundary = batch["boundary"].to(device, non_blocking=True)

                        # Thing mask from semantic predictions
                        with torch.no_grad():
                            pred_cls = logits.argmax(dim=1)
                            thing_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
                            for tid in _THING_IDS:
                                thing_mask |= (pred_cls == tid)

                        if "center_offset" in heads and "center" in out:
                            inst_loss = inst_loss + center_offset_loss(
                                out["center"], out["offset"],
                                target_center, target_offset, thing_mask
                            )
                        if "boundary" in heads and "boundary" in out:
                            inst_loss = inst_loss + boundary_loss(
                                out["boundary"], target_boundary, thing_mask
                            )

                    if "embedding" in heads and "embeddings" in out and "inst_ids" in batch:
                        inst_ids = batch["inst_ids"].to(device, non_blocking=True)
                        # Resize embeddings to match inst_ids
                        emb = out["embeddings"]
                        if emb.shape[2:] != inst_ids.shape[1:]:
                            emb = F.interpolate(emb, size=inst_ids.shape[1:],
                                                mode='bilinear', align_corners=False)
                        inst_loss = inst_loss + discriminative_loss(emb, inst_ids)

                    loss = sem_loss + args.lambda_instance * inst_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                logits = out["logits"]
                logits = F.interpolate(logits, size=labels.shape[1:],
                                       mode='bilinear', align_corners=False)
                sem_loss = criterion(logits, labels)

                inst_loss = torch.tensor(0.0, device=device)
                if heads and "center" in batch:
                    target_center = batch["center"].to(device, non_blocking=True)
                    target_offset = batch["offset"].to(device, non_blocking=True)
                    target_boundary = batch["boundary"].to(device, non_blocking=True)

                    with torch.no_grad():
                        pred_cls = logits.argmax(dim=1)
                        thing_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
                        for tid in _THING_IDS:
                            thing_mask |= (pred_cls == tid)

                    if "center_offset" in heads and "center" in out:
                        inst_loss = inst_loss + center_offset_loss(
                            out["center"], out["offset"],
                            target_center, target_offset, thing_mask
                        )
                    if "boundary" in heads and "boundary" in out:
                        inst_loss = inst_loss + boundary_loss(
                            out["boundary"], target_boundary, thing_mask
                        )

                if "embedding" in heads and "embeddings" in out and "inst_ids" in batch:
                    inst_ids = batch["inst_ids"].to(device, non_blocking=True)
                    emb = out["embeddings"]
                    if emb.shape[2:] != inst_ids.shape[1:]:
                        emb = F.interpolate(emb, size=inst_ids.shape[1:],
                                            mode='bilinear', align_corners=False)
                    inst_loss = inst_loss + discriminative_loss(emb, inst_ids)

                loss = sem_loss + args.lambda_instance * inst_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()

            # Extract scalar values BEFORE deleting tensors
            loss_val = loss.item()
            sem_loss_val = sem_loss.item()
            inst_loss_val = inst_loss.item() if isinstance(inst_loss, torch.Tensor) else inst_loss

            # Update EMA
            if ema is not None and epoch >= args.self_training_start_epoch:
                ema.update(model)

            # Release computation graph references to prevent memory leak
            del loss, sem_loss, inst_loss, logits, out
            if device.type == "mps":
                torch.mps.synchronize()

            if not math.isfinite(loss_val):
                print(f"[WARNING] NaN/Inf loss at epoch {epoch}, step {epoch_steps}. Skipping batch.")
                optimizer.zero_grad()
                continue
            epoch_loss += loss_val
            epoch_sem_loss += sem_loss_val
            epoch_inst_loss += inst_loss_val
            epoch_steps += 1

            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            if heads:
                postfix["sem"] = f"{sem_loss_val:.3f}"
                postfix["inst"] = f"{inst_loss_val:.3f}"
            pbar.set_postfix(postfix)

        epoch_time = time.time() - t0

        # Free MPS/CUDA cached memory
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_sem = epoch_sem_loss / max(epoch_steps, 1)
        avg_inst = epoch_inst_loss / max(epoch_steps, 1)

        log_msg = f"Epoch {epoch}/{args.num_epochs} | loss={avg_loss:.4f}"
        if heads:
            log_msg += f" (sem={avg_sem:.4f}, inst={avg_inst:.4f})"
        log_msg += f" | lr={optimizer.param_groups[0]['lr']:.2e} | time={epoch_time:.0f}s"
        print(log_msg)

        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            metrics = evaluate(model, val_loader, device, args.num_classes,
                               cityscapes_root=args.cityscapes_root,
                               instance_head=args.instance_head,
                               cc_min_area=args.cc_min_area)
            print(f"  → mIoU={metrics['mIoU']:.2f}% | "
                  f"PQ={metrics['PQ']:.2f} | PQ_st={metrics['PQ_stuff']:.2f} | "
                  f"PQ_th={metrics['PQ_things']:.2f}")

            score = metrics["PQ"] if metrics["PQ"] > 0 else metrics["mIoU"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_path = os.path.join(args.output_dir, "best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'args': vars(args),
                }, save_path)
                print(f"  → New best! Saved to {save_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, save_path)

    print(f"\nStage-2 training complete. Best score={best_score:.2f} at epoch {best_epoch}")

    # ══════════════════════════════════════════════════════════════════════════
    # CUPS Stage-3: Self-Training with EMA Teacher + TTA
    # ══════════════════════════════════════════════════════════════════════════
    if args.cups_stage3:
        print("\n" + "=" * 60)
        print("  CUPS STAGE-3: Self-Training")
        print(f"  Rounds: {args.st_rounds}, Epochs/round: {args.st_epochs_per_round}")
        print(f"  TTA: {args.st_tta}, EMA momentum: {args.ema_momentum}")
        print("=" * 60)

        # Initialize EMA teacher from best model
        best_ckpt = torch.load(os.path.join(args.output_dir, "best.pth"),
                               map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"[Stage-3] Loaded best model from epoch {best_ckpt['epoch']}")

        ema_teacher = EMATeacher(
            model, momentum=args.ema_momentum,
            num_classes=args.num_classes, use_tta=args.st_tta,
        )

        # Freeze backbone for self-training
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("[Stage-3] Backbone frozen for self-training")

        # Smaller LR for self-training
        st_lr = args.lr * 0.1
        st_optimizer = build_cups_optimizer(
            model, lr=st_lr, weight_decay=args.weight_decay,
            backbone_lr_ratio=0.0,  # backbone frozen
        )

        st_epoch = args.num_epochs
        for round_idx in range(args.st_rounds):
            ema_teacher.update_thresholds(round_idx)
            thresh = ema_teacher.class_thresholds[0].item()
            print(f"\n[Stage-3] Round {round_idx+1}/{args.st_rounds} "
                  f"(confidence threshold={thresh:.2f})")

            for ep in range(1, args.st_epochs_per_round + 1):
                st_epoch += 1
                model.train()
                epoch_loss = 0.0
                epoch_steps = 0
                t0 = time.time()

                pbar = tqdm(train_loader,
                            desc=f"ST-R{round_idx+1} Ep{ep}/{args.st_epochs_per_round}",
                            ncols=120, leave=True)

                for batch in pbar:
                    imgs = batch["image"].to(device, non_blocking=True)

                    # Generate pseudo-labels from EMA teacher
                    with torch.no_grad():
                        labels = ema_teacher.generate_pseudo_labels(imgs)

                    st_optimizer.zero_grad()
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            out = model(imgs)
                            logits = out["logits"]
                            logits = F.interpolate(logits, size=labels.shape[1:],
                                                   mode='bilinear', align_corners=False)
                            loss = criterion(logits, labels)
                        scaler.scale(loss).backward()
                        scaler.unscale_(st_optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scaler.step(st_optimizer)
                        scaler.update()
                    else:
                        out = model(imgs)
                        logits = out["logits"]
                        logits = F.interpolate(logits, size=labels.shape[1:],
                                               mode='bilinear', align_corners=False)
                        loss = criterion(logits, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        st_optimizer.step()

                    # Update EMA teacher
                    ema_teacher.update(model)

                    epoch_loss += loss.item()
                    epoch_steps += 1
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                avg_loss = epoch_loss / max(epoch_steps, 1)
                elapsed = time.time() - t0
                print(f"  ST Round {round_idx+1} Ep {ep} | loss={avg_loss:.4f} | "
                      f"time={elapsed:.0f}s")

                # Evaluate every 2 self-training epochs to catch overfitting
                if st_epoch % 2 == 0 or ep == args.st_epochs_per_round:
                    metrics = evaluate(model, val_loader, device, args.num_classes,
                                       cityscapes_root=args.cityscapes_root,
                                       instance_head=args.instance_head,
                                       cc_min_area=args.cc_min_area)
                    print(f"  [ST ep{st_epoch}] mIoU={metrics['mIoU']:.2f}% | "
                          f"PQ={metrics['PQ']:.2f} | PQ_st={metrics['PQ_stuff']:.2f} | "
                          f"PQ_th={metrics['PQ_things']:.2f}")

                    score = metrics["PQ"] if metrics["PQ"] > 0 else metrics["mIoU"]
                    if score > best_score:
                        best_score = score
                        best_epoch = st_epoch
                        save_path = os.path.join(args.output_dir, "best.pth")
                        torch.save({
                            "epoch": st_epoch,
                            "model_state_dict": model.state_dict(),
                            "score": score,
                            "metrics": metrics,
                            "stage": "self_training",
                            "round": round_idx + 1,
                        }, save_path)
                        print(f"  [ST ep{st_epoch}] New best! PQ={score:.2f} saved.")

            print(f"  [ST Round {round_idx+1}] complete")

        print(f"\nStage-3 complete. Best score={best_score:.2f} at epoch {best_epoch}")

    print(f"\nAll training complete. Checkpoints in {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Mobile Panoptic Segmentation — Modular Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation examples:
  Vanilla baseline:     --instance_head none --augmentation minimal
  Augmented baseline:   --instance_head none --augmentation full
  CUPS full recipe:     --augmentation cups --fpn_type bifpn --cups_stage2 --cups_stage3 --instance_subdir instance_targets
  CUPS no DropLoss:     --augmentation cups --fpn_type bifpn --cups_stage2 --no_droploss
  CUPS no copy-paste:   --augmentation cups --fpn_type bifpn --cups_stage2 --no_copy_paste
  BiFPN only (no CUPS): --augmentation full --fpn_type bifpn
  I-B (center/offset):  --instance_head center_offset --augmentation full --instance_subdir instance_targets
        """
    )

    # ── Data ──
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_mapped_k80")
    parser.add_argument("--instance_subdir", type=str, default=None,
                        help="Subdirectory with precomputed instance targets (center/offset/boundary .npy)")
    parser.add_argument("--num_classes", type=int, default=19)

    # ── Model ──
    parser.add_argument("--backbone", type=str, default="repvit_m0_9.dist_450e_in1k")
    parser.add_argument("--fpn_dim", type=int, default=128)

    # ── Instance Head ──
    parser.add_argument("--instance_head", type=str, default="none",
                        choices=["none", "embedding", "center_offset", "boundary",
                                 "embed_center", "embed_boundary", "center_boundary", "all"],
                        help="Instance head configuration for ablation")
    parser.add_argument("--lambda_instance", type=float, default=1.0,
                        help="Weight for instance loss relative to semantic loss")

    # ── Model ──
    parser.add_argument("--fpn_type", type=str, default="simple",
                        choices=["simple", "bifpn"],
                        help="FPN type: simple (top-down) or bifpn (bidirectional)")

    # ── Augmentation ──
    parser.add_argument("--augmentation", type=str, default="minimal",
                        choices=["minimal", "full", "cups"],
                        help="Augmentation: minimal, full (random scale), cups (discrete jitter + copy-paste)")

    # ── Training ──
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5)
    parser.add_argument("--crop_h", type=int, default=384)
    parser.add_argument("--crop_w", type=int, default=768)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping (L2 norm)")

    # ── CUPS Stage-2 ──
    parser.add_argument("--cups_stage2", action="store_true",
                        help="Enable CUPS stage-2 tricks (DropLoss, copy-paste, WD separation)")
    parser.add_argument("--no_droploss", action="store_true",
                        help="Disable DropLoss (ablation)")
    parser.add_argument("--no_copy_paste", action="store_true",
                        help="Disable copy-paste augmentation (ablation)")
    parser.add_argument("--no_ignore_unknown", action="store_true",
                        help="Disable IGNORE_UNKNOWN_THING_REGIONS (ablation)")
    parser.add_argument("--drop_rate", type=float, default=0.3,
                        help="DropLoss drop rate for thing pixels")

    # ── CUPS Stage-3 ──
    parser.add_argument("--cups_stage3", action="store_true",
                        help="Enable CUPS stage-3 self-training with EMA teacher + TTA")
    parser.add_argument("--st_rounds", type=int, default=3,
                        help="Number of self-training rounds")
    parser.add_argument("--st_epochs_per_round", type=int, default=5,
                        help="Epochs per self-training round")
    parser.add_argument("--st_tta", action="store_true", default=True,
                        help="Use TTA for teacher inference")
    parser.add_argument("--no_st_tta", action="store_true",
                        help="Disable TTA for teacher inference")

    # ── Old Self-Training (legacy) ──
    parser.add_argument("--self_training", action="store_true",
                        help="Enable simple EMA self-training (legacy, use --cups_stage3 instead)")
    parser.add_argument("--self_training_start_epoch", type=int, default=30)
    parser.add_argument("--ema_momentum", type=float, default=0.999)
    parser.add_argument("--self_training_confidence", type=float, default=0.9)

    # ── Metrics ──
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--cc_min_area", type=int, default=50)

    # ── Infra ──
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/mobile_repvit")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    # Handle --no_st_tta flag
    if args.no_st_tta:
        args.st_tta = False
    train(args)


if __name__ == "__main__":
    main()
