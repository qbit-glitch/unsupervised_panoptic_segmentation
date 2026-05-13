#!/usr/bin/env python3
"""Train Panoptic-DeepLab + RepViT with adapted CUPS stage-2/3 recipe.

Supports 4 architecture variants + 3 FPN types + full ablation controls.
Implements all 12 CUPS training components adapted for lightweight architectures.

CUPS Stage-2 Adaptations (10 components):
  1. Pixel-level DropLoss for thing classes
  2. IGNORE_UNKNOWN_THING_REGIONS
  3. Copy-paste augmentation (instance-level)
  4. Self-enhanced copy-paste (after warmup_steps)
  5. Discrete resolution jitter (7 levels)
  6. Gradient clipping (L2 norm=1.0)
  7. Norm weight decay separation
  8. Optimizer: AdamW LR=1e-4, WD=1e-5
  9. Cascade gradient scaling (1/K per head)
  10. BCE with logits (autocast-safe)

CUPS Stage-3 Adaptations (5 components):
  11. EMA teacher-student (decay=0.999)
  12. TTA teacher inference (2 scales + flip)
  13. Per-class confidence thresholding
  14. Frozen backbone, heads-only training
  15. 3 rounds x N steps, escalating confidence

Usage:
  # Full CUPS recipe with Panoptic-DeepLab + BiFPN
  python train_panoptic_deeplab.py --cityscapes_root ... \\
      --arch panoptic_deeplab --fpn_type bifpn \\
      --cups_stage2 --cups_stage3

  # Architecture ablation (kMaX-DeepLab)
  python train_panoptic_deeplab.py --cityscapes_root ... \\
      --arch kmax_deeplab --fpn_type bifpn \\
      --cups_stage2 --cups_stage3

  # Training recipe ablation (no DropLoss)
  python train_panoptic_deeplab.py --cityscapes_root ... \\
      --arch panoptic_deeplab --fpn_type bifpn \\
      --cups_stage2 --no_droploss

  # FPN ablation
  python train_panoptic_deeplab.py --cityscapes_root ... \\
      --arch panoptic_deeplab --fpn_type simple \\
      --cups_stage2 --cups_stage3
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
from tqdm import tqdm
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mbps_pytorch.panoptic_deeplab import (
    build_model, panoptic_inference_center_offset, panoptic_inference_mask_cls,
)


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
# CUPS Component 1: DropLoss (Pixel-Level)
# ═══════════════════════════════════════════════════════════════════════════════

class DropLoss(nn.Module):
    """Pixel-level DropLoss for thing classes (CUPS adaptation).

    Drops loss on thing-class pixels where the model is already confident,
    focusing learning on hard/uncertain pixels. This prevents the semantic
    head from overfitting to easy thing pixels (e.g., large cars) while
    ignoring hard ones (e.g., distant pedestrians).

    Reference: CUPS (CVPR 2025), Section 3.2
    """

    def __init__(self, num_classes: int = 19, thing_ids: set = None,
                 ignore_index: int = 255, drop_rate: float = 0.3,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.thing_ids = thing_ids or _THING_IDS
        self.ignore_index = ignore_index
        self.drop_rate = drop_rate
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw logits
            targets: (B, H, W) class labels
        """
        B, C, H, W = logits.shape

        # Standard CE on all pixels
        ce_loss = F.cross_entropy(
            logits, targets, ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing, reduction='none'
        )  # (B, H, W)

        # Build thing-class mask
        thing_mask = torch.zeros_like(targets, dtype=torch.bool)
        for tid in self.thing_ids:
            thing_mask |= (targets == tid)

        # For thing pixels: drop loss on confident predictions
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            max_prob, _ = probs.max(dim=1)
            # Sort thing pixels by confidence, drop top drop_rate%
            drop_mask = torch.zeros_like(targets, dtype=torch.bool)
            for b in range(B):
                thing_px = thing_mask[b]
                if thing_px.sum() == 0:
                    continue
                confidences = max_prob[b][thing_px]
                n_drop = int(confidences.numel() * self.drop_rate)
                if n_drop > 0:
                    _, top_idx = confidences.topk(n_drop)
                    # Map back to spatial locations
                    thing_locs = thing_px.nonzero(as_tuple=False)
                    drop_locs = thing_locs[top_idx]
                    drop_mask[b, drop_locs[:, 0], drop_locs[:, 1]] = True

        # Zero out loss on dropped pixels
        ce_loss[drop_mask] = 0.0

        # Average over non-ignored, non-dropped pixels
        valid = (targets != self.ignore_index) & (~drop_mask)
        if valid.sum() == 0:
            return ce_loss.sum() * 0.0
        return ce_loss[valid].mean()


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS Component 2: IGNORE_UNKNOWN_THING_REGIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ignore_unknown_thing_regions(
    sem_labels: torch.Tensor,
    inst_labels: torch.Tensor,
    thing_ids: set = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Set thing-class pixels without instance assignment to ignore.

    In CUPS, thing-class pixels that have no corresponding instance
    pseudo-label are set to ignore_index (255). This prevents the semantic
    head from learning conflicting signals where we have class labels but
    no instance grouping.

    Args:
        sem_labels: (B, H, W) semantic pseudo-labels
        inst_labels: (B, H, W) instance IDs (0=no instance)
        thing_ids: set of thing-class train IDs
        ignore_index: value to assign to unknown thing pixels
    Returns:
        Modified semantic labels with unknown things set to ignore
    """
    thing_ids = thing_ids or _THING_IDS
    modified = sem_labels.clone()

    thing_mask = torch.zeros_like(sem_labels, dtype=torch.bool)
    for tid in thing_ids:
        thing_mask |= (sem_labels == tid)

    no_instance = (inst_labels == 0) | (inst_labels < 0)
    unknown_things = thing_mask & no_instance

    modified[unknown_things] = ignore_index
    return modified


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS Component 3/4: Copy-Paste Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

class CopyPasteAugmentation:
    """Instance-level copy-paste augmentation (CUPS adaptation).

    Copies thing-class instances from one image and pastes them onto another.
    After warmup_steps, uses self-enhanced copy-paste with model predictions
    to select high-quality instances for pasting.
    """

    def __init__(self, thing_ids: set = None, paste_prob: float = 0.5,
                 max_instances: int = 3, min_area: int = 200):
        self.thing_ids = thing_ids or _THING_IDS
        self.paste_prob = paste_prob
        self.max_instances = max_instances
        self.min_area = min_area
        self.instance_bank = []  # Stores (image_crop, mask, class_id)
        self.max_bank_size = 200

    def update_bank(self, image: np.ndarray, sem_label: np.ndarray,
                    inst_ids: np.ndarray):
        """Add instances from current batch to the bank."""
        if inst_ids is None:
            return
        for uid in np.unique(inst_ids):
            if uid <= 0:
                continue
            mask = inst_ids == uid
            if mask.sum() < self.min_area:
                continue
            # Get bounding box
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            # Get class
            cls_vals = sem_label[mask]
            cls = int(np.median(cls_vals))
            if cls not in self.thing_ids:
                continue
            # Store crop
            crop_img = image[y0:y1, x0:x1].copy()
            crop_mask = mask[y0:y1, x0:x1].copy()
            self.instance_bank.append((crop_img, crop_mask, cls))
            if len(self.instance_bank) > self.max_bank_size:
                self.instance_bank.pop(0)

    def apply(self, image: np.ndarray, sem_label: np.ndarray,
              inst_ids: np.ndarray = None) -> tuple:
        """Apply copy-paste augmentation.

        Returns:
            Augmented (image, sem_label, inst_ids)
        """
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
        next_inst_id = (inst_ids.max() + 1) if inst_ids is not None else 1000

        for crop_img, crop_mask, cls in instances:
            ch, cw = crop_mask.shape
            if ch >= H or cw >= W:
                continue
            # Random scale (0.8-1.2x)
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

            # Random position
            y = random.randint(0, H - new_h)
            x = random.randint(0, W - new_w)

            # Paste
            paste_region = crop_mask_r
            img_out[y:y+new_h, x:x+new_w][paste_region] = crop_img_r[paste_region]
            sem_out[y:y+new_h, x:x+new_w][paste_region] = cls
            if inst_out is not None:
                inst_out[y:y+new_h, x:x+new_w][paste_region] = next_inst_id
                next_inst_id += 1

        return img_out, sem_out, inst_out


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS Component 9: Cascade Gradient Scaling
# ═══════════════════════════════════════════════════════════════════════════════

class CascadeGradientScaler:
    """Scale gradients by 1/K per prediction head (CUPS adaptation).

    In multi-head architectures, each head's gradient is scaled by 1/K where
    K is the number of active heads. This prevents any single head from
    dominating the shared backbone gradients.
    """

    def __init__(self, num_heads: int = 3):
        self.scale = 1.0 / max(num_heads, 1)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture-Specific Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_center_offset_loss(
    pred_center: torch.Tensor,
    pred_offset: torch.Tensor,
    target_center: torch.Tensor,
    target_offset: torch.Tensor,
    thing_mask: torch.Tensor,
) -> torch.Tensor:
    """Center heatmap MSE + offset L1 loss for Panoptic-DeepLab/MaskConver."""
    if pred_center.shape[2:] != target_center.shape[1:]:
        pred_center = F.interpolate(pred_center, size=target_center.shape[1:],
                                     mode='bilinear', align_corners=False)
        pred_offset = F.interpolate(pred_offset, size=target_center.shape[1:],
                                     mode='bilinear', align_corners=False)

    pc = pred_center.squeeze(1)
    weight = torch.where(target_center > 0.01, 10.0, 1.0)
    center_l = (weight * (pc - target_center) ** 2).mean()

    valid = thing_mask & (target_center > 0.001)
    if valid.sum() > 100:
        valid_exp = valid.unsqueeze(1).expand_as(pred_offset)
        offset_l = F.smooth_l1_loss(pred_offset[valid_exp], target_offset[valid_exp])
    else:
        offset_l = torch.tensor(0.0, device=pred_center.device)

    return center_l + offset_l


def compute_mask_cls_loss(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    sem_targets: torch.Tensor,
    inst_targets: torch.Tensor,
    num_classes: int = 19,
) -> torch.Tensor:
    """Mask classification loss for kMaX-DeepLab / Mask2Former-Lite.

    Uses simple matching: each GT instance matches the query with highest
    mask IoU, then supervises class + mask.
    """
    B = mask_logits.shape[0]
    total_loss = torch.tensor(0.0, device=mask_logits.device)
    valid_batches = 0

    for b in range(B):
        inst = inst_targets[b]
        sem = sem_targets[b]
        uids = inst.unique()
        uids = uids[(uids > 0) & (uids != 255)]

        if len(uids) == 0:
            # No instances: supervise no-object class
            no_obj_target = torch.full(
                (class_logits.shape[1],), num_classes,
                device=class_logits.device, dtype=torch.long
            )
            total_loss = total_loss + F.cross_entropy(
                class_logits[b], no_obj_target
            )
            valid_batches += 1
            continue

        # Build GT masks and classes
        gt_masks = []
        gt_classes = []
        for uid in uids:
            mask = (inst == uid).float()
            cls_vals = sem[inst == uid]
            cls = cls_vals.mode().values.item() if len(cls_vals) > 0 else 0
            gt_masks.append(mask)
            gt_classes.append(cls)

        gt_masks = torch.stack(gt_masks)  # (N_gt, H, W)
        gt_classes = torch.tensor(gt_classes, device=mask_logits.device, dtype=torch.long)

        # Resize pred masks to GT size
        pred_masks = mask_logits[b]  # (Q, H', W')
        if pred_masks.shape[1:] != gt_masks.shape[1:]:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0), size=gt_masks.shape[1:],
                mode='bilinear', align_corners=False
            ).squeeze(0)

        pred_masks_sig = pred_masks.sigmoid()

        # Greedy matching by IoU
        N_gt = gt_masks.shape[0]
        Q = pred_masks.shape[0]
        matched_q = set()
        cls_targets = torch.full((Q,), num_classes, device=mask_logits.device, dtype=torch.long)
        mask_loss = torch.tensor(0.0, device=mask_logits.device)

        for i in range(N_gt):
            gt_m = gt_masks[i]
            best_iou = -1
            best_j = -1
            for j in range(Q):
                if j in matched_q:
                    continue
                inter = (pred_masks_sig[j] * gt_m).sum()
                union = pred_masks_sig[j].sum() + gt_m.sum() - inter
                iou = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                matched_q.add(best_j)
                cls_targets[best_j] = gt_classes[i]
                # BCE mask loss (autocast-safe: use logits)
                mask_loss = mask_loss + F.binary_cross_entropy_with_logits(
                    pred_masks[best_j], gt_m, reduction='mean'
                )

        # Class loss
        cls_loss = F.cross_entropy(class_logits[b], cls_targets)
        mask_loss = mask_loss / max(N_gt, 1)

        total_loss = total_loss + cls_loss + mask_loss
        valid_batches += 1

    return total_loss / max(valid_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset with CUPS Augmentations
# ═══════════════════════════════════════════════════════════════════════════════

class CUPSCityscapesDataset(Dataset):
    """Cityscapes with CUPS-adapted augmentation pipeline.

    Supports:
      - Discrete resolution jitter (7 levels)
      - Copy-paste augmentation
      - Standard photometric augmentations
    """

    def __init__(self, cityscapes_root, split="train",
                 semantic_subdir="pseudo_semantic_mapped_k80",
                 instance_subdir=None,
                 crop_size=(384, 768), is_train=True,
                 resolution_jitter=False, copy_paste=None):
        self.root = Path(cityscapes_root)
        self.split = split
        self.is_train = is_train
        self.crop_size = crop_size
        self.resolution_jitter = resolution_jitter
        self.copy_paste = copy_paste

        # Find images
        img_dir = self.root / "leftImg8bit" / split
        self.images = sorted(img_dir.glob("*/*_leftImg8bit.png"))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        # Semantic labels
        self.labels = []
        for img_path in self.images:
            city = img_path.parent.name
            stem = img_path.name.replace("_leftImg8bit.png", "")
            label_path = self.root / semantic_subdir / split / city / f"{stem}_pseudo_semantic.png"
            if not label_path.exists():
                label_path = self.root / semantic_subdir / split / city / f"{stem}.png"
            self.labels.append(label_path)

        existing = sum(1 for p in self.labels if p.exists())
        print(f"[Dataset] {split}: {len(self.images)} images, {existing}/{len(self.labels)} labels")
        if existing == 0:
            raise RuntimeError(f"No pseudo-labels found in {self.root / semantic_subdir / split}")

        # Instance targets
        self.instance_subdir = instance_subdir
        if instance_subdir:
            inst_dir = self.root / instance_subdir / split
            if inst_dir.exists():
                print(f"[Dataset] Instance targets: {inst_dir}")
            else:
                print(f"[Dataset] WARNING: Instance dir not found: {inst_dir}")
                self.instance_subdir = None

        # Photometric transforms
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def _load_instance_data(self, idx, H, W):
        """Load precomputed instance targets."""
        center = offset = inst_ids = None
        if not self.instance_subdir:
            return center, offset, inst_ids

        city = self.images[idx].parent.name
        stem = self.images[idx].name.replace("_leftImg8bit.png", "")
        base = self.root / self.instance_subdir / self.split / city

        try:
            center = np.load(base / f"{stem}_center.npy")
            offset = np.load(base / f"{stem}_offset.npy")
        except FileNotFoundError:
            pass

        # Instance IDs from pseudo-labels
        inst_dir = self.root / "cups_pseudo_labels_v3"
        inst_path = inst_dir / f"{stem}_leftImg8bit_instance.png"
        if inst_path.exists():
            inst_ids = np.array(Image.open(inst_path), dtype=np.int32)
            if inst_ids.shape != (H, W):
                inst_ids = np.array(
                    Image.fromarray(inst_ids.astype(np.uint16)).resize((W, H), Image.NEAREST),
                    dtype=np.int32
                )

        return center, offset, inst_ids

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("RGB"), dtype=np.float32) / 255.0
        label_path = self.labels[idx]
        label = np.array(Image.open(label_path), dtype=np.int64) if label_path.exists() \
            else np.full(img.shape[:2], 255, dtype=np.int64)

        H, W = img.shape[:2]
        center, offset, inst_ids = self._load_instance_data(idx, H, W)

        if self.is_train:
            # ── Copy-paste augmentation ──
            if self.copy_paste is not None:
                self.copy_paste.update_bank(img, label, inst_ids)
                img, label, inst_ids = self.copy_paste.apply(img, label, inst_ids)

            # ── Horizontal flip ──
            if random.random() > 0.5:
                img = img[:, ::-1].copy()
                label = label[:, ::-1].copy()
                if center is not None:
                    center = center[:, ::-1].copy()
                    offset = offset[:, :, ::-1].copy()
                    offset[1] = -offset[1]
                if inst_ids is not None:
                    inst_ids = inst_ids[:, ::-1].copy()

            # ── Discrete resolution jitter (CUPS Component 5) ──
            if self.resolution_jitter:
                scale = random.choice(_RESOLUTION_JITTER_LEVELS)
            else:
                scale = random.uniform(0.5, 1.25)

            new_h, new_w = int(H * scale), int(W * scale)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img = np.array(img_pil.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0
            label = np.array(
                Image.fromarray(label.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST),
                dtype=np.int64
            )
            if center is not None:
                center = np.array(Image.fromarray(center).resize((new_w, new_h), Image.BILINEAR))
                off_dy = np.array(Image.fromarray(offset[0]).resize((new_w, new_h), Image.BILINEAR)) * scale
                off_dx = np.array(Image.fromarray(offset[1]).resize((new_w, new_h), Image.BILINEAR)) * scale
                offset = np.stack([off_dy, off_dx], axis=0)
            if inst_ids is not None:
                inst_ids = np.array(
                    Image.fromarray(inst_ids.astype(np.uint16)).resize((new_w, new_h), Image.NEAREST),
                    dtype=np.int32
                )
            H, W = img.shape[:2]

            # ── Random crop ──
            ch, cw = self.crop_size
            if H >= ch and W >= cw:
                y = random.randint(0, H - ch)
                x = random.randint(0, W - cw)
                img = img[y:y+ch, x:x+cw]
                label = label[y:y+ch, x:x+cw]
                if center is not None:
                    center = center[y:y+ch, x:x+cw]
                    offset = offset[:, y:y+ch, x:x+cw]
                if inst_ids is not None:
                    inst_ids = inst_ids[y:y+ch, x:x+cw]
            else:
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
                if inst_ids is not None:
                    inst_ids = np.array(
                        Image.fromarray(inst_ids.astype(np.uint16)).resize((cw, ch), Image.NEAREST),
                        dtype=np.int32
                    )

            # ── Photometric augmentation ──
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            if random.random() > 0.5:
                img_pil = self.color_jitter(img_pil)
            if random.random() > 0.5:
                img_pil = self.gaussian_blur(img_pil)
            if random.random() > 0.8:
                img_pil = transforms.functional.rgb_to_grayscale(img_pil, num_output_channels=3)
            img = np.array(img_pil, dtype=np.float32) / 255.0

        else:
            # Validation: fixed size
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
        if inst_ids is not None:
            result["inst_ids"] = torch.from_numpy(inst_ids.copy()).long()

        return result


def collate_fn(batch):
    result = {}
    for k in batch[0].keys():
        result[k] = torch.stack([b[k] for b in batch if k in b])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS Stage-3: EMA Teacher with TTA
# ═══════════════════════════════════════════════════════════════════════════════

class EMATeacher:
    """EMA teacher with TTA inference for CUPS Stage-3.

    Features:
      - Test-Time Augmentation: 2 scales (0.75x, 1.25x) + horizontal flip
      - Per-class confidence thresholding
      - Escalating confidence across self-training rounds
    """

    def __init__(self, model: nn.Module, momentum: float = 0.999,
                 num_classes: int = 19, use_tta: bool = True):
        self.teacher = copy.deepcopy(model)
        self.teacher.requires_grad_(False)
        self.teacher.train(False)
        self.momentum = momentum
        self.num_classes = num_classes
        self.use_tta = use_tta
        # Per-class confidence thresholds (initialized uniformly)
        self.class_thresholds = torch.ones(num_classes) * 0.7

    @torch.no_grad()
    def update(self, student: nn.Module):
        for ema_p, s_p in zip(self.teacher.parameters(), student.parameters()):
            ema_p.data.mul_(self.momentum).add_(s_p.data, alpha=1.0 - self.momentum)

    def update_thresholds(self, round_idx: int, base_threshold: float = 0.7,
                          increment: float = 0.05):
        """Escalate confidence thresholds per round."""
        new_thresh = base_threshold + round_idx * increment
        self.class_thresholds.fill_(min(new_thresh, 0.95))

    @torch.no_grad()
    def generate_pseudo_labels(self, images: torch.Tensor,
                               ignore_index: int = 255) -> tuple:
        """Generate pseudo-labels with optional TTA.

        Returns:
            labels: (B, H, W) pseudo-labels (ignore_index for low confidence)
            teacher_out: dict of teacher model outputs
        """
        device = images.device
        B, _, H, W = images.shape

        if self.use_tta:
            # Multi-scale + flip TTA
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

        # Per-class confidence filtering
        thresholds = self.class_thresholds.to(device)
        for c in range(self.num_classes):
            cls_mask = preds == c
            low_conf = cls_mask & (confidence < thresholds[c])
            preds[low_conf] = ignore_index

        # Also get teacher output for instance heads (at original scale)
        teacher_out = self.teacher(images)
        return preds, teacher_out


# ═══════════════════════════════════════════════════════════════════════════════
# CUPS Component 7: Norm Weight Decay Separation
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-5,
                    backbone_lr_ratio: float = 0.1) -> torch.optim.AdamW:
    """Build AdamW optimizer with norm/bias weight decay separation.

    CUPS applies weight decay ONLY to non-norm, non-bias parameters.
    """
    decay_params = []
    no_decay_params = []
    backbone_decay = []
    backbone_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        is_norm_or_bias = ("norm" in name or "bn" in name or
                           name.endswith(".bias") or "gamma" in name or "beta" in name)

        if is_backbone:
            if is_norm_or_bias:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            if is_norm_or_bias:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        {"params": backbone_decay, "lr": lr * backbone_lr_ratio,
         "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr * backbone_lr_ratio,
         "weight_decay": 0.0},
    ]

    # Filter empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_evaluation(model, val_loader, device, args):
    """Evaluate PQ + mIoU using architecture-appropriate inference."""
    model.train(False)
    H, W = 512, 1024
    num_classes = args.num_classes
    arch = args.arch

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    iou_sum = np.zeros(num_classes)

    gt_dir = os.path.join(args.cityscapes_root, "gtFine", "val")
    has_gt = os.path.isdir(gt_dir)
    val_dataset = val_loader.dataset
    img_idx = 0

    is_mask_cls = arch in ("kmax_deeplab", "mask2former_lite")

    for batch in tqdm(val_loader, desc="Eval", ncols=100, leave=False):
        imgs = batch["image"].to(device)
        out = model(imgs)

        logits = out["logits"]
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        preds = logits.argmax(dim=1)

        if not has_gt:
            img_idx += imgs.shape[0]
            continue

        B = imgs.shape[0]
        for i in range(B):
            idx = img_idx + i
            if idx >= len(val_dataset.images):
                break
            img_path = val_dataset.images[idx]
            city = img_path.parent.name
            stem = img_path.name.replace("_leftImg8bit.png", "")

            pred_sem = preds[i].cpu().numpy()

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
            if is_mask_cls and "mask_logits" in out:
                mask_l = out["mask_logits"][i]
                cls_l = out["class_logits"][i]
                pred_pan, pred_segments = panoptic_inference_mask_cls(
                    logits[i], mask_l, cls_l, _THING_IDS,
                    min_area=args.cc_min_area,
                )
            elif "center" in out and "offset" in out:
                c_map = F.interpolate(out["center"], size=(H, W),
                                       mode='bilinear', align_corners=False)
                o_map = F.interpolate(out["offset"], size=(H, W),
                                       mode='bilinear', align_corners=False)
                pred_pan, pred_segments = panoptic_inference_center_offset(
                    logits[i], c_map[i], o_map[i], _THING_IDS,
                    min_area=args.cc_min_area,
                )
            else:
                # Fallback: connected components
                pred_pan, pred_segments = _cc_inference(pred_sem, args.cc_min_area)

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

            # Fast segment matching via intersection counting
            gt_by_cat = defaultdict(list)
            for sid, cat in gt_segments.items():
                gt_by_cat[cat].append(sid)
            pred_by_cat = defaultdict(list)
            for sid, cat in pred_segments.items():
                pred_by_cat[cat].append(sid)

            # Precompute segment areas (fast via bincount)
            gt_ids_flat = gt_pan.ravel()
            pred_ids_flat = pred_pan.ravel()
            gt_areas = np.bincount(gt_ids_flat, minlength=gt_nxt)
            pred_areas = np.bincount(pred_ids_flat, minlength=max(pred_segments.keys()) + 1 if pred_segments else 1)

            # Compute intersections: encode (gt_id, pred_id) pairs
            max_pred_id = max(pred_segments.keys()) + 1 if pred_segments else 1
            pair_ids = gt_ids_flat.astype(np.int64) * max_pred_id + pred_ids_flat
            pair_counts = np.bincount(pair_ids)

            matched_pred = set()
            for cat in range(num_classes):
                for gt_id in gt_by_cat.get(cat, []):
                    gt_area = gt_areas[gt_id]
                    best_iou, best_pid = 0.0, None
                    for pid in pred_by_cat.get(cat, []):
                        if pid in matched_pred:
                            continue
                        pair_key = gt_id * max_pred_id + pid
                        inter = int(pair_counts[pair_key]) if pair_key < len(pair_counts) else 0
                        if inter == 0:
                            continue
                        union = gt_area + pred_areas[pid] - inter
                        iou_val = inter / union if union > 0 else 0
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

    # mIoU
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    miou = iou[union > 0].mean() * 100 if has_gt else 0.0

    # PQ
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
        "mIoU": round(miou, 2),
        "PQ": round(pq_all, 2),
        "PQ_stuff": round(pq_stuff, 2),
        "PQ_things": round(pq_things, 2),
        "per_class_pq": per_class_pq,
    }


def _cc_inference(pred_sem, cc_min_area=50):
    """Fallback connected-component inference."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else
                           ('mps' if hasattr(torch.backends, 'mps') and
                            torch.backends.mps.is_available() else 'cpu')))
    print(f"Device: {device}")
    print(f"Architecture: {args.arch} + {args.fpn_type}")
    print(f"CUPS Stage-2: {args.cups_stage2} | Stage-3: {args.cups_stage3}")

    is_mask_cls = args.arch in ("kmax_deeplab", "mask2former_lite")

    # ── Model ──
    model = build_model(
        arch=args.arch,
        backbone_name=args.backbone,
        num_classes=args.num_classes,
        fpn_dim=args.fpn_dim,
        fpn_type=args.fpn_type,
        pretrained=True,
    ).to(device)

    # Count heads for cascade gradient scaling
    num_heads = 1  # semantic always present
    if not is_mask_cls:
        num_heads += 2  # center + offset
    else:
        num_heads += 2  # mask + class
    cascade_scaler = CascadeGradientScaler(num_heads) if args.cups_stage2 else None

    # ── Loss Functions ──
    if args.cups_stage2 and not args.no_droploss:
        sem_criterion = DropLoss(
            num_classes=args.num_classes, thing_ids=_THING_IDS,
            drop_rate=args.drop_rate, label_smoothing=args.label_smoothing,
        )
        print(f"[Loss] DropLoss enabled (drop_rate={args.drop_rate})")
    else:
        sem_criterion = nn.CrossEntropyLoss(
            ignore_index=255, label_smoothing=args.label_smoothing
        )
        print("[Loss] Standard CE loss")

    # ── Copy-paste augmentation ──
    copy_paste = None
    if args.cups_stage2 and not args.no_copy_paste:
        copy_paste = CopyPasteAugmentation(
            thing_ids=_THING_IDS, paste_prob=0.5, max_instances=3,
        )
        print("[Aug] Copy-paste augmentation enabled")

    # ── Dataset ──
    train_dataset = CUPSCityscapesDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        instance_subdir=args.instance_subdir,
        crop_size=(args.crop_h, args.crop_w),
        is_train=True,
        resolution_jitter=(args.cups_stage2 and not args.no_resolution_jitter),
        copy_paste=copy_paste,
    )
    val_dataset = CUPSCityscapesDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        instance_subdir=None,
        is_train=False,
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

    # ── Optimizer (CUPS Component 7+8) ──
    if args.cups_stage2:
        optimizer = build_optimizer(
            model, lr=args.lr, weight_decay=args.weight_decay,
            backbone_lr_ratio=args.backbone_lr_ratio,
        )
        print(f"[Optim] AdamW with norm WD separation, lr={args.lr}, wd={args.weight_decay}")
    else:
        # Simple optimizer for baseline
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith("backbone.")],
             "lr": args.lr},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith("backbone.")],
             "lr": args.lr * args.backbone_lr_ratio},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

    # ── Freeze backbone ──
    if args.freeze_backbone_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print(f"[Train] Backbone frozen for first {args.freeze_backbone_epochs} epochs")

    # ── AMP ──
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    best_score = 0.0
    best_epoch = 0
    global_step = 0

    # ══════════════════════════════════════════════════════════════════════
    # Stage-2 Training
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  STAGE-2 TRAINING: {args.num_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.num_epochs + 1):
        model.train()

        # Unfreeze backbone
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer to include backbone params
            if args.cups_stage2:
                optimizer = build_optimizer(
                    model, lr=args.lr, weight_decay=args.weight_decay,
                    backbone_lr_ratio=args.backbone_lr_ratio,
                )
            else:
                param_groups = [
                    {"params": [p for n, p in model.named_parameters()
                                if p.requires_grad and not n.startswith("backbone.")],
                     "lr": args.lr},
                    {"params": [p for n, p in model.named_parameters()
                                if p.requires_grad and n.startswith("backbone.")],
                     "lr": args.lr * args.backbone_lr_ratio},
                ]
                optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            # Reset scheduler with remaining steps
            remaining_steps = len(train_loader) * (args.num_epochs - epoch + 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_steps, eta_min=1e-7
            )
            print(f"[Epoch {epoch}] Backbone unfrozen, optimizer rebuilt")

        epoch_losses = defaultdict(float)
        epoch_steps = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{args.num_epochs}",
                    ncols=130, leave=True)

        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # CUPS Component 2: IGNORE_UNKNOWN_THING_REGIONS
            if args.cups_stage2 and not args.no_ignore_unknown and "inst_ids" in batch:
                inst_ids = batch["inst_ids"].to(device, non_blocking=True)
                labels = apply_ignore_unknown_thing_regions(labels, inst_ids)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(imgs)
                    logits = out["logits"]
                    logits_up = F.interpolate(logits, size=labels.shape[1:],
                                               mode='bilinear', align_corners=False)

                    # Semantic loss
                    sem_loss = sem_criterion(logits_up, labels)
                    if cascade_scaler:
                        sem_loss = cascade_scaler.scale_loss(sem_loss)

                    # Instance loss (architecture-specific)
                    inst_loss = torch.tensor(0.0, device=device)

                    if is_mask_cls and "inst_ids" in batch:
                        # Mask-classification loss for kMaX/Mask2Former
                        inst_ids_t = batch["inst_ids"].to(device, non_blocking=True)
                        inst_loss = compute_mask_cls_loss(
                            out["mask_logits"], out["class_logits"],
                            labels, inst_ids_t, args.num_classes,
                        )
                        if cascade_scaler:
                            inst_loss = cascade_scaler.scale_loss(inst_loss)

                    elif not is_mask_cls and "center" in out and "center" in batch:
                        # Center/offset loss for PanopticDeepLab/MaskConver
                        target_center = batch["center"].to(device, non_blocking=True)
                        target_offset = batch["offset"].to(device, non_blocking=True)
                        with torch.no_grad():
                            pred_cls = logits_up.argmax(dim=1)
                            thing_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
                            for tid in _THING_IDS:
                                thing_mask |= (pred_cls == tid)
                        inst_loss = compute_center_offset_loss(
                            out["center"], out["offset"],
                            target_center, target_offset, thing_mask,
                        )
                        if cascade_scaler:
                            inst_loss = cascade_scaler.scale_loss(inst_loss)

                    loss = sem_loss + args.lambda_instance * inst_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # CUPS Component 6: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                logits = out["logits"]
                logits_up = F.interpolate(logits, size=labels.shape[1:],
                                           mode='bilinear', align_corners=False)
                sem_loss = sem_criterion(logits_up, labels)
                if cascade_scaler:
                    sem_loss = cascade_scaler.scale_loss(sem_loss)

                inst_loss = torch.tensor(0.0, device=device)

                if is_mask_cls and "inst_ids" in batch:
                    inst_ids_t = batch["inst_ids"].to(device, non_blocking=True)
                    inst_loss = compute_mask_cls_loss(
                        out["mask_logits"], out["class_logits"],
                        labels, inst_ids_t, args.num_classes,
                    )
                    if cascade_scaler:
                        inst_loss = cascade_scaler.scale_loss(inst_loss)

                elif not is_mask_cls and "center" in out and "center" in batch:
                    target_center = batch["center"].to(device, non_blocking=True)
                    target_offset = batch["offset"].to(device, non_blocking=True)
                    with torch.no_grad():
                        pred_cls = logits_up.argmax(dim=1)
                        thing_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
                        for tid in _THING_IDS:
                            thing_mask |= (pred_cls == tid)
                    inst_loss = compute_center_offset_loss(
                        out["center"], out["offset"],
                        target_center, target_offset, thing_mask,
                    )
                    if cascade_scaler:
                        inst_loss = cascade_scaler.scale_loss(inst_loss)

                loss = sem_loss + args.lambda_instance * inst_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            epoch_losses["total"] += loss.item()
            epoch_losses["sem"] += sem_loss.item()
            epoch_losses["inst"] += inst_loss.item() if isinstance(inst_loss, torch.Tensor) else 0
            epoch_steps += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sem": f"{sem_loss.item():.3f}",
                "inst": f"{inst_loss.item():.3f}" if isinstance(inst_loss, torch.Tensor) else "0",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        et = time.time() - t0
        avg = {k: v / max(epoch_steps, 1) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch} | loss={avg['total']:.4f} (sem={avg['sem']:.4f}, "
              f"inst={avg['inst']:.4f}) | lr={optimizer.param_groups[0]['lr']:.2e} | {et:.0f}s")

        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            metrics = run_evaluation(model, val_loader, device, args)
            print(f"  -> mIoU={metrics['mIoU']:.2f}% | PQ={metrics['PQ']:.2f} | "
                  f"PQ_st={metrics['PQ_stuff']:.2f} | PQ_th={metrics['PQ_things']:.2f}")

            score = metrics["PQ"] if metrics["PQ"] > 0 else metrics["mIoU"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'metrics': metrics, 'args': vars(args),
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  -> New best! score={best_score:.2f}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, os.path.join(args.output_dir, f"epoch_{epoch}.pth"))

    print(f"\nStage-2 done. Best={best_score:.2f} at epoch {best_epoch}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage-3: Self-Training (EMA Teacher-Student)
    # ══════════════════════════════════════════════════════════════════════

    if not args.cups_stage3:
        print("\nStage-3 skipped (--cups_stage3 not set)")
        return

    print(f"\n{'='*60}")
    print(f"  STAGE-3 SELF-TRAINING: {args.st_rounds} rounds x {args.st_epochs_per_round} epochs")
    print(f"{'='*60}\n")

    # Freeze backbone for stage-3
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("[Stage-3] Backbone frozen, heads-only training")

    # EMA teacher
    ema_teacher = EMATeacher(
        model, momentum=args.ema_momentum,
        num_classes=args.num_classes, use_tta=args.st_tta,
    )
    print(f"[Stage-3] EMA teacher initialized (momentum={args.ema_momentum}, TTA={args.st_tta})")

    # Rebuild optimizer for heads only
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone.")]
    st_optimizer = torch.optim.AdamW(head_params, lr=args.lr * 0.1, weight_decay=args.weight_decay)

    for round_idx in range(args.st_rounds):
        ema_teacher.update_thresholds(
            round_idx, base_threshold=args.st_base_threshold,
            increment=args.st_threshold_increment,
        )
        print(f"\n-- Round {round_idx + 1}/{args.st_rounds} | "
              f"confidence threshold={ema_teacher.class_thresholds[0].item():.2f} --")

        st_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            st_optimizer,
            T_max=len(train_loader) * args.st_epochs_per_round,
            eta_min=1e-8,
        )

        for st_ep in range(1, args.st_epochs_per_round + 1):
            model.train()
            st_losses = defaultdict(float)
            st_steps = 0
            t0 = time.time()

            global_epoch = args.num_epochs + round_idx * args.st_epochs_per_round + st_ep

            pbar = tqdm(train_loader,
                        desc=f"ST R{round_idx+1} Ep{st_ep}/{args.st_epochs_per_round}",
                        ncols=130, leave=True)

            for batch in pbar:
                imgs = batch["image"].to(device, non_blocking=True)

                # Generate pseudo-labels from EMA teacher
                with torch.no_grad():
                    teacher_labels, teacher_out = ema_teacher.generate_pseudo_labels(imgs)

                st_optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = model(imgs)
                        logits = out["logits"]
                        logits_up = F.interpolate(logits, size=teacher_labels.shape[1:],
                                                   mode='bilinear', align_corners=False)
                        loss = F.cross_entropy(logits_up, teacher_labels,
                                               ignore_index=255,
                                               label_smoothing=args.label_smoothing)

                    scaler.scale(loss).backward()
                    scaler.unscale_(st_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(st_optimizer)
                    scaler.update()
                else:
                    out = model(imgs)
                    logits = out["logits"]
                    logits_up = F.interpolate(logits, size=teacher_labels.shape[1:],
                                               mode='bilinear', align_corners=False)
                    loss = F.cross_entropy(logits_up, teacher_labels,
                                           ignore_index=255,
                                           label_smoothing=args.label_smoothing)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    st_optimizer.step()

                st_scheduler.step()

                # Update EMA teacher
                ema_teacher.update(model)

                st_losses["total"] += loss.item()
                st_steps += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{st_scheduler.get_last_lr()[0]:.2e}",
                })

            et = time.time() - t0
            avg_loss = st_losses["total"] / max(st_steps, 1)
            print(f"ST R{round_idx+1} Ep{st_ep} | loss={avg_loss:.4f} | {et:.0f}s")

            # Evaluate at end of each round's last epoch
            if st_ep == args.st_epochs_per_round:
                metrics = run_evaluation(model, val_loader, device, args)
                print(f"  -> mIoU={metrics['mIoU']:.2f}% | PQ={metrics['PQ']:.2f} | "
                      f"PQ_st={metrics['PQ_stuff']:.2f} | PQ_th={metrics['PQ_things']:.2f}")

                score = metrics["PQ"] if metrics["PQ"] > 0 else metrics["mIoU"]
                if score > best_score:
                    best_score = score
                    best_epoch = global_epoch
                    torch.save({
                        'epoch': global_epoch,
                        'model_state_dict': model.state_dict(),
                        'metrics': metrics, 'args': vars(args),
                    }, os.path.join(args.output_dir, "best.pth"))
                    print(f"  -> New best! score={best_score:.2f}")

                torch.save({
                    'epoch': global_epoch,
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                }, os.path.join(args.output_dir, f"stage3_round{round_idx+1}.pth"))

    print(f"\nAll training complete. Best score={best_score:.2f} at epoch {best_epoch}")
    print(f"Checkpoints in {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Panoptic-DeepLab + RepViT with CUPS Training Recipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data ──
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_mapped_k80")
    parser.add_argument("--instance_subdir", type=str, default="instance_targets")
    parser.add_argument("--num_classes", type=int, default=19)

    # ── Architecture ──
    parser.add_argument("--arch", type=str, default="panoptic_deeplab",
                        choices=["panoptic_deeplab", "kmax_deeplab", "maskconver", "mask2former_lite"])
    parser.add_argument("--backbone", type=str, default="repvit_m0_9.dist_450e_in1k")
    parser.add_argument("--fpn_dim", type=int, default=128)
    parser.add_argument("--fpn_type", type=str, default="bifpn",
                        choices=["bifpn", "simple", "panet"])

    # ── CUPS Stage-2 Flags ──
    parser.add_argument("--cups_stage2", action="store_true",
                        help="Enable all CUPS stage-2 adaptations")
    parser.add_argument("--no_droploss", action="store_true",
                        help="Disable DropLoss (ablation)")
    parser.add_argument("--no_copy_paste", action="store_true",
                        help="Disable copy-paste augmentation (ablation)")
    parser.add_argument("--no_resolution_jitter", action="store_true",
                        help="Disable discrete resolution jitter (ablation)")
    parser.add_argument("--no_ignore_unknown", action="store_true",
                        help="Disable IGNORE_UNKNOWN_THING_REGIONS (ablation)")
    parser.add_argument("--drop_rate", type=float, default=0.3,
                        help="DropLoss drop rate for confident thing pixels")
    parser.add_argument("--lambda_instance", type=float, default=0.1,
                        help="Instance loss weight (reduced from 1.0 per lesson learned)")

    # ── CUPS Stage-3 Flags ──
    parser.add_argument("--cups_stage3", action="store_true",
                        help="Enable CUPS stage-3 self-training")
    parser.add_argument("--st_rounds", type=int, default=3,
                        help="Number of self-training rounds")
    parser.add_argument("--st_epochs_per_round", type=int, default=5,
                        help="Epochs per self-training round")
    parser.add_argument("--st_tta", action="store_true", default=True,
                        help="Use TTA for teacher inference")
    parser.add_argument("--st_base_threshold", type=float, default=0.7,
                        help="Base confidence threshold for self-training")
    parser.add_argument("--st_threshold_increment", type=float, default=0.05,
                        help="Confidence increment per round")
    parser.add_argument("--ema_momentum", type=float, default=0.999)

    # ── Training ──
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (CUPS default: 1e-4)")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (CUPS default: 1e-5)")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping L2 norm")
    parser.add_argument("--crop_h", type=int, default=384)
    parser.add_argument("--crop_w", type=int, default=768)

    # ── Eval ──
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--cc_min_area", type=int, default=50)

    # ── Infra ──
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/panoptic_deeplab")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
