#!/usr/bin/env python3
"""Train CSCMRefineNet: Coupled-State Cross-Modal Mamba Refinement Network.

Trains the semantic pseudo-label refiner using self-supervised losses:
  1. Cross-view consistency (weak/strong augmentation agreement)
  2. Depth-boundary alignment (semantic edges ↔ depth discontinuities)
  3. Feature-prototype consistency (compact DINOv2 clusters per class)
  4. Entropy minimization (encourage confident predictions)

Usage:
    python mbps_pytorch/train_refine_net.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir checkpoints/refine_net \
        --num_epochs 50 --batch_size 4 --device auto

References:
    - UniMatch V2 (TPAMI 2025): Cross-view consistency
    - DepthG (CVPR 2024): Depth-boundary alignment
    - CAUSE (Pattern Recognition 2024): Feature-prototype concept
    - Goodfellow "Deep Learning" Ch. 19: Entropy minimization
"""

import argparse
import json
import logging
import math
import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mbps_pytorch.refine_net import CSCMRefineNet, HiResRefineNet, DepthGuidedUNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Patch grid dimensions for DINOv2 ViT-B/14 at 448×896 input
PATCH_H, PATCH_W = 32, 64
NUM_CLASSES = 27


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PseudoLabelDataset(Dataset):
    """Load pre-computed CAUSE semantics, DINOv2 features, and SPIdepth depth."""

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        semantic_subdir: str = "pseudo_semantic_cause_crf",
        feature_subdir: str = "dinov2_features",
        depth_subdir: str = "depth_spidepth",
        logits_subdir: str = None,
        num_classes: int = 27,
        target_h: int = None,
        target_w: int = None,
        return_depth_full: bool = False,
        instance_subdir: str = None,
        instance_target_subdir: str = None,
    ):
        self.root = cityscapes_root
        self.split = split
        self.semantic_subdir = semantic_subdir
        self.feature_subdir = feature_subdir
        self.depth_subdir = depth_subdir
        self.logits_subdir = logits_subdir
        self.num_classes = num_classes
        self.target_h = target_h
        self.target_w = target_w
        self.return_depth_full = return_depth_full
        self.instance_subdir = instance_subdir
        self.instance_target_subdir = instance_target_subdir

        # Find all images and extract stems
        img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
        self.entries = []
        for city in sorted(os.listdir(img_dir)):
            city_path = os.path.join(img_dir, city)
            if not os.path.isdir(city_path):
                continue
            for fname in sorted(os.listdir(city_path)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                stem = fname.replace("_leftImg8bit.png", "")
                self.entries.append({"stem": stem, "city": city})

        log.info(f"PseudoLabelDataset: {len(self.entries)} images ({split})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        stem, city = entry["stem"], entry["city"]

        # Load DINOv2 features: (2048, 768) float16 → (768, 32, 64) float32
        feat_path = os.path.join(
            self.root, self.feature_subdir, self.split, city,
            f"{stem}_leftImg8bit.npy",
        )
        features = np.load(feat_path).astype(np.float32)  # (2048, 768)
        features = features.reshape(PATCH_H, PATCH_W, -1)  # (32, 64, 768)
        features = features.transpose(2, 0, 1)  # (768, 32, 64)

        # Load depth: (512, 1024) float32 → downsample to (1, 32, 64)
        # Depth/grads always at patch resolution (model input); labels at target resolution
        depth_path = os.path.join(
            self.root, self.depth_subdir, self.split, city, f"{stem}.npy",
        )
        depth_full = np.load(depth_path)  # (512, 1024)
        depth_patch = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
        depth_patch = F.interpolate(
            depth_patch, size=(PATCH_H, PATCH_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)  # (1, 32, 64)
        depth_np = depth_patch.numpy()

        # Compute Sobel gradients on depth at patch resolution
        depth_grads = self._sobel_gradients(depth_np[0])  # (2, 32, 64)

        # Load semantics: (1024, 2048) uint8 → one-hot (27, 32, 64)
        if self.logits_subdir is not None:
            logits_path = os.path.join(
                self.root, self.logits_subdir, self.split, city,
                f"{stem}_logits.pt",
            )
            if os.path.exists(logits_path):
                cause_logits = torch.load(logits_path, weights_only=True).float()
                # Expected: (27, logits_h, logits_w) → resize to (27, 32, 64)
                if cause_logits.shape[1:] != (PATCH_H, PATCH_W):
                    cause_logits = F.interpolate(
                        cause_logits.unsqueeze(0),
                        size=(PATCH_H, PATCH_W),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                # Stored values are softmax probabilities — convert to
                # log-space so they behave as proper logits for the loss
                # functions (which apply F.softmax internally).
                cause_logits = torch.log(cause_logits.clamp(min=1e-7))
                cause_logits = cause_logits.numpy()
            else:
                cause_logits = self._load_onehot_semantics(city, stem)
        else:
            cause_logits = self._load_onehot_semantics(city, stem)

        result = {
            "cause_logits": torch.from_numpy(cause_logits).float(),
            "dinov2_features": torch.from_numpy(features).float(),
            "depth": torch.from_numpy(depth_np).float(),
            "depth_grads": torch.from_numpy(depth_grads).float(),
            "stem": stem,
            "city": city,
        }
        if self.return_depth_full:
            # Return full-res depth (512x1024) for UNet skip connections
            # Normalize to [0, 1] matching patch-level depth
            d_min, d_max = depth_full.min(), depth_full.max()
            if d_max > d_min:
                depth_full_norm = (depth_full - d_min) / (d_max - d_min)
            else:
                depth_full_norm = np.zeros_like(depth_full)
            result["depth_full"] = torch.from_numpy(
                depth_full_norm[np.newaxis].astype(np.float32)
            )  # (1, 512, 1024)

        if self.instance_subdir is not None:
            # Load pre-computed instance masks (uint16 instance IDs)
            inst_path = os.path.join(
                self.root, self.instance_subdir, self.split, city,
                f"{stem}_instance.png",
            )
            if os.path.exists(inst_path):
                inst_map = np.array(Image.open(inst_path)).astype(np.int32)
                result["instance_full"] = torch.from_numpy(
                    inst_map[np.newaxis].astype(np.float32)
                )  # (1, H, W) as float for interpolation compatibility
            else:
                # Fallback: no instances for this image
                h = depth_full.shape[0] if depth_full is not None else 512
                w = depth_full.shape[1] if depth_full is not None else 1024
                result["instance_full"] = torch.zeros(1, h, w)

        if self.instance_target_subdir is not None:
            # Load pre-computed center heatmap + offset targets for instance heads
            center_path = os.path.join(
                self.root, self.instance_target_subdir, self.split, city,
                f"{stem}_center.npy",
            )
            offset_path = os.path.join(
                self.root, self.instance_target_subdir, self.split, city,
                f"{stem}_offset.npy",
            )
            if os.path.exists(center_path) and os.path.exists(offset_path):
                center = np.load(center_path)  # (target_h, target_w)
                offset = np.load(offset_path)  # (2, target_h, target_w)
                result["center_target"] = torch.from_numpy(center).float()
                result["offset_target"] = torch.from_numpy(offset).float()
                # Compute thing_mask from semantic labels (Cityscapes trainIDs 11-18)
                out_h = self.target_h if self.target_h else PATCH_H
                out_w = self.target_w if self.target_w else PATCH_W
                sem_at_target = result["cause_logits"].argmax(dim=0).numpy()
                if sem_at_target.shape != (out_h, out_w):
                    sem_pil = Image.fromarray(sem_at_target.astype(np.uint8))
                    sem_at_target = np.array(sem_pil.resize((out_w, out_h), Image.NEAREST))
                thing_mask = (sem_at_target >= 11) & (sem_at_target <= 18)
                result["thing_mask"] = torch.from_numpy(thing_mask).bool()
            else:
                out_h = self.target_h if self.target_h else PATCH_H
                out_w = self.target_w if self.target_w else PATCH_W
                result["center_target"] = torch.zeros(out_h, out_w)
                result["offset_target"] = torch.zeros(2, out_h, out_w)
                result["thing_mask"] = torch.zeros(out_h, out_w, dtype=torch.bool)

        return result

    def _load_onehot_semantics(self, city, stem):
        """Load argmax PNG and convert to smoothed one-hot at target resolution."""
        sem_path = os.path.join(
            self.root, self.semantic_subdir, self.split, city, f"{stem}.png",
        )
        sem_full = np.array(Image.open(sem_path))  # (1024, 2048) uint8

        out_h = self.target_h if self.target_h else PATCH_H
        out_w = self.target_w if self.target_w else PATCH_W

        # Downsample to target resolution via nearest neighbor
        sem_pil = Image.fromarray(sem_full)
        sem_patch = np.array(
            sem_pil.resize((out_w, out_h), Image.NEAREST)
        )  # (out_h, out_w)

        # Convert to smoothed one-hot (label smoothing = 0.1)
        nc = self.num_classes
        smooth = 0.1
        onehot = np.zeros((nc, out_h, out_w), dtype=np.float32)
        onehot[:] = smooth / nc
        for c in range(nc):
            mask = sem_patch == c
            onehot[c][mask] = 1.0 - smooth + smooth / nc

        # Pixels with value 255 (ignore) get uniform distribution
        ignore_mask = sem_patch == 255
        if ignore_mask.any():
            onehot[:, ignore_mask] = 1.0 / nc

        # Convert to log-space (consistent with .pt logits path)
        return np.log(np.clip(onehot, 1e-7, None))

    @staticmethod
    def _sobel_gradients(depth_2d):
        """Compute Sobel gradients. depth_2d: (H, W) → (2, H, W)."""
        d = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0).float()
        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).reshape(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).reshape(1, 1, 3, 3)
        grad_x = F.conv2d(d, kx, padding=1).squeeze()  # (H, W)
        grad_y = F.conv2d(d, ky, padding=1).squeeze()  # (H, W)
        return torch.stack([grad_x, grad_y], dim=0).numpy()  # (2, H, W)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def apply_augmentation(batch, mode="weak"):
    """Apply augmentation to a batch dict (in-place friendly).

    Since DINOv2 features are pre-extracted, augmentations are spatial only:
      - weak: random horizontal flip
      - strong: horizontal flip + Gaussian noise on features + spatial dropout
    """
    aug_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    if mode in ("weak", "strong"):
        # Random horizontal flip (per sample)
        B = aug_batch["cause_logits"].shape[0]
        for i in range(B):
            if torch.rand(1).item() > 0.5:
                aug_batch["cause_logits"][i] = torch.flip(
                    aug_batch["cause_logits"][i], [-1])
                aug_batch["dinov2_features"][i] = torch.flip(
                    aug_batch["dinov2_features"][i], [-1])
                aug_batch["depth"][i] = torch.flip(
                    aug_batch["depth"][i], [-1])
                aug_batch["depth_grads"][i] = torch.flip(
                    aug_batch["depth_grads"][i], [-1])
                # Flip sign of horizontal gradient
                aug_batch["depth_grads"][i, 0] *= -1
                if "depth_full" in aug_batch and aug_batch["depth_full"] is not None:
                    aug_batch["depth_full"][i] = torch.flip(
                        aug_batch["depth_full"][i], [-1])

    if mode == "strong":
        # Gaussian noise on features
        noise = torch.randn_like(aug_batch["dinov2_features"]) * 0.05
        aug_batch["dinov2_features"] = aug_batch["dinov2_features"] + noise

        # Spatial dropout: zero out random 10% of spatial positions
        B, C, H, W = aug_batch["dinov2_features"].shape
        mask = (torch.rand(B, 1, H, W, device=aug_batch["dinov2_features"].device)
                > 0.1).float()
        aug_batch["dinov2_features"] = aug_batch["dinov2_features"] * mask

    return aug_batch


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_view_consistency_loss(logits_weak, logits_strong, threshold=0.9):
    """Enforce prediction agreement between weak and strong augmented views.

    Weak predictions (detached) provide pseudo-targets for strong predictions.
    Only high-confidence pixels supervise.

    Reference: UniMatch V2 (TPAMI 2025)
    """
    probs_weak = F.softmax(logits_weak.detach(), dim=1)
    max_probs, pseudo_targets = probs_weak.max(dim=1)  # (B, H, W)
    mask = max_probs > threshold

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_strong.device)

    loss = F.cross_entropy(logits_strong, pseudo_targets, reduction="none")
    loss = (loss * mask.float()).sum() / mask.sum()
    return loss


def depth_boundary_alignment_loss(logits, depth, sigma=0.05):
    """Encourage label consistency between depth-similar neighbors.

    Pixels with similar depth should have similar predictions.
    Pixels across depth discontinuities are allowed to disagree.

    Reference: DepthG (CVPR 2024)
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # Horizontal neighbors
    depth_diff_h = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    weight_h = torch.exp(-depth_diff_h ** 2 / (2 * sigma ** 2))
    prob_diff_h = (probs[:, :, :, 1:] - probs[:, :, :, :-1]) ** 2
    loss_h = (weight_h * prob_diff_h.sum(dim=1, keepdim=True)).mean()

    # Vertical neighbors
    depth_diff_v = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    weight_v = torch.exp(-depth_diff_v ** 2 / (2 * sigma ** 2))
    prob_diff_v = (probs[:, :, 1:, :] - probs[:, :, :-1, :]) ** 2
    loss_v = (weight_v * prob_diff_v.sum(dim=1, keepdim=True)).mean()

    return loss_h + loss_v


def feature_prototype_loss(logits, dinov2_features, temperature=0.1):
    """Encourage compact DINOv2 feature clusters per predicted class.

    Compute per-class prototypes (weighted mean DINOv2 feature), then
    maximize cosine similarity of each pixel to its predicted prototype.
    """
    probs = F.softmax(logits / temperature, dim=1)  # (B, C, H, W)
    B, C, H, W = probs.shape
    D = dinov2_features.shape[1]

    probs_flat = probs.reshape(B, C, H * W)      # (B, C, N)
    feats_flat = dinov2_features.reshape(B, D, H * W)  # (B, D, N)

    # Weighted prototypes per class: (B, D, C)
    prototypes = torch.bmm(feats_flat, probs_flat.permute(0, 2, 1))
    weights = probs_flat.sum(dim=2).unsqueeze(1)  # (B, 1, C)
    prototypes = prototypes / (weights + 1e-6)

    # Cosine similarity
    prototypes_norm = F.normalize(prototypes, dim=1)  # (B, D, C)
    feats_norm = F.normalize(feats_flat, dim=1)       # (B, D, N)
    sim = torch.bmm(prototypes_norm.permute(0, 2, 1), feats_norm)  # (B, C, N)

    # Loss: high-probability pixels should be close to their prototype
    loss = -(probs_flat * sim).sum() / (B * H * W)
    return loss


def entropy_loss(logits):
    """Minimize prediction entropy to encourage confident assignments.

    Reference: Goodfellow "Deep Learning" Ch. 19
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy


def boundary_preservation_loss(logits, cause_logits, label_smoothing=0.1):
    """Penalize prediction changes at pseudo-label semantic boundaries.

    Detects boundaries in the input pseudo-labels (where adjacent pixels
    have different class assignments) and applies stronger cross-entropy
    distillation at those locations to preserve panoptic segment structure.
    """
    cause_labels = cause_logits.argmax(dim=1).detach()  # (B, H, W)

    # Detect boundaries: pixels where any 4-connected neighbor differs
    pad_labels = F.pad(cause_labels.unsqueeze(1).float(), (1, 1, 1, 1),
                       mode='replicate').squeeze(1).long()
    boundary = (
        (cause_labels != pad_labels[:, 1:-1, 2:]) |   # right
        (cause_labels != pad_labels[:, 1:-1, :-2]) |   # left
        (cause_labels != pad_labels[:, 2:, 1:-1]) |    # down
        (cause_labels != pad_labels[:, :-2, 1:-1])     # up
    ).float()  # (B, H, W)

    if boundary.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    ce = F.cross_entropy(logits, cause_labels, reduction='none',
                         label_smoothing=label_smoothing)
    loss = (ce * boundary).sum() / boundary.sum()
    return loss


def instance_uniformity_loss(logits, instance_map, min_pixels=25, max_instances=30):
    """Penalize prediction variance within each instance.

    Encourages consistent semantic predictions for all pixels belonging
    to the same instance mask.

    Args:
        logits: (B, C, H, W) raw logits
        instance_map: (B, 1, H, W) instance IDs (0 = stuff/bg, >0 = instances)
        min_pixels: ignore instances smaller than this
        max_instances: limit per image to avoid slowdown
    Returns:
        scalar loss
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    B, C, H, W = probs.shape
    inst = instance_map[:, 0]  # (B, H, W)

    total_loss = 0.0
    count = 0
    for b in range(B):
        unique_ids = inst[b].unique()
        unique_ids = unique_ids[unique_ids > 0]  # skip background
        if len(unique_ids) == 0:
            continue
        if len(unique_ids) > max_instances:
            # Take largest instances
            sizes = torch.tensor(
                [(inst[b] == uid).sum().item() for uid in unique_ids],
                device=inst.device,
            )
            top_idx = sizes.argsort(descending=True)[:max_instances]
            unique_ids = unique_ids[top_idx]

        for uid in unique_ids:
            mask = inst[b] == uid  # (H, W)
            if mask.sum() < min_pixels:
                continue
            p = probs[b, :, mask]  # (C, N)
            mean_p = p.mean(dim=1, keepdim=True)  # (C, 1)
            var = ((p - mean_p) ** 2).mean()
            total_loss += var
            count += 1

    return total_loss / max(count, 1)


def instance_boundary_loss(logits, instance_map):
    """Encourage semantic boundaries to align with instance boundaries.

    Uses BCE to push predicted class transitions toward instance boundary
    locations and suppress spurious transitions within instances.

    Args:
        logits: (B, C, H, W) raw logits
        instance_map: (B, 1, H, W) instance IDs
    Returns:
        scalar loss
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # Predicted boundary: max class change between adjacent pixels
    diff_h = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().max(dim=1)[0]  # (B, H-1, W)
    diff_w = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().max(dim=1)[0]  # (B, H, W-1)

    # Instance boundary: where adjacent pixels have different IDs
    inst = instance_map[:, 0].float()
    inst_diff_h = (inst[:, 1:, :] != inst[:, :-1, :]).float()
    inst_diff_w = (inst[:, :, 1:] != inst[:, :, :-1]).float()

    # BCE loss: encourage high pred_boundary at instance boundaries
    loss_h = F.binary_cross_entropy(diff_h.clamp(1e-6, 1 - 1e-6), inst_diff_h)
    loss_w = F.binary_cross_entropy(diff_w.clamp(1e-6, 1 - 1e-6), inst_diff_w)

    return (loss_h + loss_w) / 2


def center_offset_loss(pred_center, pred_offset, target_center, target_offset,
                       thing_mask, center_pos_weight=200.0):
    """Center heatmap MSE + offset SmoothL1 loss for instance heads.

    Uses high positive weight (200x) to counteract the extreme class imbalance
    (only ~3% of pixels are positive center targets). Also trains offsets on
    ALL instance pixels (where target offset is nonzero), not just near-center
    pixels, for better offset coverage across entire instances.

    Args:
        pred_center:   (B, 1, H, W) predicted center heatmap (sigmoid applied)
        pred_offset:   (B, 2, H, W) predicted offsets (dy, dx)
        target_center: (B, H, W) target center heatmap
        target_offset: (B, 2, H, W) target offsets
        thing_mask:    (B, H, W) bool mask for thing-class pixels
        center_pos_weight: weight multiplier for positive (center) pixels
    Returns:
        center_l: scalar center loss
        offset_l: scalar offset loss
    """
    # Interpolate predictions to target size if needed
    th, tw = target_center.shape[1:]
    if pred_center.shape[2:] != (th, tw):
        pred_center = F.interpolate(pred_center, size=(th, tw),
                                    mode="bilinear", align_corners=False)
        pred_offset = F.interpolate(pred_offset, size=(th, tw),
                                    mode="bilinear", align_corners=False)

    # Center loss: weighted MSE with focal-style modulation
    # High pos_weight (200x) counteracts 3% positive pixel ratio
    # Safe on MPS (no BCE with probs)
    pc = pred_center.squeeze(1)  # (B, H, W)
    is_positive = target_center > 0.01
    weight = torch.where(is_positive, center_pos_weight, 1.0)
    center_l = (weight * (pc - target_center) ** 2).mean()

    # Offset loss: SmoothL1 on ALL instance pixels (where offset target is nonzero)
    # This gives offset supervision across the full instance, not just near center
    has_offset = target_offset.abs().sum(dim=1) > 0.1  # (B, H, W)
    valid = thing_mask & has_offset
    if valid.sum() > 100:
        valid_exp = valid.unsqueeze(1).expand_as(pred_offset)
        offset_l = F.smooth_l1_loss(pred_offset[valid_exp],
                                    target_offset[valid_exp])
    else:
        offset_l = torch.tensor(0.0, device=pred_center.device)

    return center_l, offset_l


class RefineNetLoss(nn.Module):
    """Combined loss for CSCMRefineNet.

    Uses cross-entropy from CAUSE hard labels (not KL divergence) with
    cosine warmdown: supervision starts strong, drops to near-zero by
    end of training so self-supervised losses dominate.

    Self-supervised losses:
      - depth_boundary_alignment: spatial smoothness weighted by depth
      - feature_prototype: DINOv2 cluster compactness
      - entropy: prediction confidence

    Optional improvements:
      - thing_weight: upweight distillation for thing-class pixels (TAD)
      - lambda_bpl: boundary preservation loss weight (BPL)
      - focal_gamma: focal loss gamma (0=standard CE, >0=focal loss)
    """

    def __init__(
        self,
        lambda_distill: float = 1.0,
        lambda_distill_min: float = 0.0,
        lambda_align: float = 5.0,
        lambda_proto: float = 0.5,
        lambda_ent: float = 0.3,
        label_smoothing: float = 0.2,
        thing_weight: float = 1.0,
        lambda_bpl: float = 0.0,
        focal_gamma: float = 0.0,
        lambda_depth_boundary: float = 0.0,
        lambda_instance_uniform: float = 0.0,
        lambda_instance_boundary: float = 0.0,
    ):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.lambda_distill_min = lambda_distill_min
        self.lambda_align = lambda_align
        self.lambda_proto = lambda_proto
        self.lambda_ent = lambda_ent
        self.label_smoothing = label_smoothing
        self.thing_weight = thing_weight
        self.lambda_bpl = lambda_bpl
        self.focal_gamma = focal_gamma
        self.lambda_depth_boundary = lambda_depth_boundary
        self.lambda_instance_uniform = lambda_instance_uniform
        self.lambda_instance_boundary = lambda_instance_boundary
        self._distill_scale = 1.0

    def set_epoch(self, epoch: int, total_epochs: int):
        """Cosine warmdown: fast decay in middle epochs."""
        progress = (epoch - 1) / max(total_epochs - 1, 1)
        hi, lo = self.lambda_distill, self.lambda_distill_min
        # Cosine decay: stays high initially, drops fast, then flattens near min
        self._distill_scale = lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * progress))

    def forward(self, logits, cause_logits, dinov2_features, depth,
                instance_map=None):
        """
        Args:
            logits: (B, C, H, W) model output logits
            cause_logits: (B, C, H, W) original CAUSE log-probabilities
            dinov2_features: (B, 768, H, W) original features
            depth: (B, 1, H, W) depth map
            instance_map: (B, 1, H, W) instance IDs (optional, for instance losses)
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        eff_distill = self._distill_scale

        # Cross-entropy distillation with focal loss, TAD, and depth-boundary weighting
        if eff_distill > 0:
            cause_labels = cause_logits.argmax(dim=1).detach()  # (B, H, W)
            B, C, H, W = logits.shape

            # Always compute per-pixel CE (needed for focal/TAD/depth-boundary)
            ce = F.cross_entropy(
                logits, cause_labels, reduction='none',
                label_smoothing=self.label_smoothing,
            )  # (B, H, W)
            weights = torch.ones_like(ce)

            # Focal loss: downweight easy pixels
            if self.focal_gamma > 0:
                pt = torch.exp(-ce.detach())  # p(correct class)
                focal_w = (1.0 - pt) ** self.focal_gamma
                weights = weights * focal_w

            # Thing-Aware Distillation (TAD)
            if self.thing_weight != 1.0:
                thing_mask = (cause_labels >= 11) & (cause_labels <= 18)
                weights[thing_mask] = weights[thing_mask] * self.thing_weight

            # Depth-boundary weighting: upweight pixels near depth discontinuities
            if self.lambda_depth_boundary > 0:
                depth_for_bw = depth
                if depth.shape[2:] != logits.shape[2:]:
                    depth_for_bw = F.interpolate(
                        depth, size=(H, W), mode='bilinear', align_corners=False)
                # Compute gradient magnitude
                grad_h = torch.abs(depth_for_bw[:, :, 1:, :] - depth_for_bw[:, :, :-1, :])
                grad_w = torch.abs(depth_for_bw[:, :, :, 1:] - depth_for_bw[:, :, :, :-1])
                grad_h = F.pad(grad_h, (0, 0, 0, 1))  # pad bottom
                grad_w = F.pad(grad_w, (0, 1, 0, 0))  # pad right
                grad_mag = (grad_h + grad_w).squeeze(1)  # (B, H, W)
                # Normalize to [1, 1+lambda] range
                grad_norm = grad_mag / (grad_mag.mean() + 1e-6)
                boundary_w = 1.0 + self.lambda_depth_boundary * grad_norm.clamp(max=3.0)
                weights = weights * boundary_w

            l_distill = (ce * weights).mean()
            losses["distill"] = l_distill
        else:
            l_distill = 0.0

        # Boundary Preservation Loss (BPL)
        if self.lambda_bpl > 0:
            l_bpl = boundary_preservation_loss(
                logits, cause_logits, self.label_smoothing)
            losses["bpl"] = l_bpl
        else:
            l_bpl = 0.0

        if self.lambda_align > 0:
            depth_for_align = depth
            if depth.shape[2:] != logits.shape[2:]:
                depth_for_align = F.interpolate(
                    depth, size=logits.shape[2:],
                    mode='bilinear', align_corners=False)
            l_align = depth_boundary_alignment_loss(logits, depth_for_align)
            losses["align"] = l_align
        else:
            l_align = 0.0

        if self.lambda_proto > 0:
            feat_for_proto = dinov2_features
            if feat_for_proto.shape[2:] != logits.shape[2:]:
                feat_for_proto = F.interpolate(
                    feat_for_proto, size=logits.shape[2:],
                    mode='bilinear', align_corners=False)
            l_proto = feature_prototype_loss(logits, feat_for_proto)
            losses["proto"] = l_proto
        else:
            l_proto = 0.0

        if self.lambda_ent > 0:
            l_ent = entropy_loss(logits)
            losses["entropy"] = l_ent
        else:
            l_ent = 0.0

        # Instance losses (optional, when instance_map provided)
        l_inst_uniform = 0.0
        l_inst_boundary = 0.0
        if instance_map is not None:
            if self.lambda_instance_uniform > 0:
                l_inst_uniform = instance_uniformity_loss(logits, instance_map)
                losses["inst_uniform"] = l_inst_uniform
            if self.lambda_instance_boundary > 0:
                l_inst_boundary = instance_boundary_loss(logits, instance_map)
                losses["inst_boundary"] = l_inst_boundary

        total = (eff_distill * l_distill
                 + self.lambda_bpl * l_bpl
                 + self.lambda_align * l_align
                 + self.lambda_proto * l_proto
                 + self.lambda_ent * l_ent
                 + self.lambda_instance_uniform * l_inst_uniform
                 + self.lambda_instance_boundary * l_inst_boundary)
        losses["total"] = total
        losses["eff_distill_w"] = eff_distill

        return total, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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

# CAUSE 27-class → 19 trainID (validated mapping from evaluate_cascade_pseudolabels.py)
_CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


def evaluate_panoptic(model, val_loader, device, cityscapes_root,
                      eval_hw=(512, 1024), cc_min_area=50, num_classes=27,
                      cluster_to_trainid_lut=None,
                      eval_instance_subdir=None,
                      use_center_offset=False,
                      center_threshold=0.1,
                      nms_kernel=7):
    """Evaluate refined semantics with full panoptic metrics.

    Computes PQ, PQ_stuff, PQ_things, SQ, RQ and mIoU on the val set.
    Thing instances are derived via connected components of the semantic map,
    or from pre-computed instance masks if eval_instance_subdir is set.
    """
    from scipy import ndimage
    from collections import defaultdict

    gt_dir = os.path.join(cityscapes_root, "gtFine", "val")
    H, W = eval_hw
    num_cls = 19

    # Accumulators
    confusion = np.zeros((num_cls, num_cls), dtype=np.int64)
    tp = np.zeros(num_cls)
    fp = np.zeros(num_cls)
    fn = np.zeros(num_cls)
    iou_sum = np.zeros(num_cls)
    changed_pixels = 0
    total_pixels = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=100, leave=False):
            fwd_kwargs = {}
            if "depth_full" in batch:
                fwd_kwargs["depth_full"] = batch["depth_full"].to(device)
            if "instance_full" in batch:
                fwd_kwargs["instance_full"] = batch["instance_full"].to(device)
            output = model(
                batch["dinov2_features"].to(device),
                batch["depth"].to(device),
                batch["depth_grads"].to(device),
                **fwd_kwargs,
            )
            # Handle dict output (instance heads) vs plain tensor
            if isinstance(output, dict):
                logits = output["semantic"]
                pred_center_batch = output.get("center")
                pred_offset_batch = output.get("offset")
            else:
                logits = output
                pred_center_batch = pred_offset_batch = None

            pred_27 = logits.argmax(dim=1).cpu().numpy()  # (B, 32, 64)
            orig_27 = batch["cause_logits"].argmax(dim=1).cpu().numpy()

            for i in range(pred_27.shape[0]):
                city, stem = batch["city"][i], batch["stem"][i]

                # Track prediction changes
                changed_pixels += (pred_27[i] != orig_27[i]).sum()
                total_pixels += pred_27[i].size

                # Map to 19 trainID and upsample to eval resolution
                if cluster_to_trainid_lut is not None:
                    # Overclustered: map k classes → 19 trainIDs via LUT
                    pred_tid_patch = cluster_to_trainid_lut[pred_27[i]]
                elif num_classes == 27:
                    pred_tid_patch = _CAUSE27_TO_TRAINID[pred_27[i]]
                else:
                    # Already in trainID space (0-18 + 255)
                    pred_tid_patch = pred_27[i].astype(np.uint8)
                pred_sem = np.array(
                    Image.fromarray(pred_tid_patch).resize((W, H), Image.NEAREST)
                )

                # Load GT semantic
                gt_path = os.path.join(
                    gt_dir, city, f"{stem}_gtFine_labelIds.png")
                if not os.path.exists(gt_path):
                    continue
                gt_raw = np.array(Image.open(gt_path))
                gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
                for raw_id, tid in _CS_ID_TO_TRAIN.items():
                    gt_sem[gt_raw == raw_id] = tid
                if gt_sem.shape != (H, W):
                    gt_sem = np.array(
                        Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

                # mIoU confusion matrix
                valid = (gt_sem < num_cls) & (pred_sem < num_cls)
                if valid.sum() > 0:
                    np.add.at(confusion, (gt_sem[valid], pred_sem[valid]), 1)

                # --- Panoptic evaluation ---
                if use_center_offset and pred_center_batch is not None:
                    # Run panoptic inference at NATIVE resolution (128x256)
                    # then upsample the panoptic map — 16× faster than full-res
                    from mbps_pytorch.panoptic_deeplab import panoptic_inference_center_offset

                    native_h, native_w = logits.shape[2], logits.shape[3]

                    # Build 19-class semantic at native resolution
                    pred_tid_small = np.array(
                        Image.fromarray(pred_tid_patch).resize(
                            (native_w, native_h), Image.NEAREST))
                    sem_19_small = torch.zeros(19, native_h, native_w)
                    for c in range(19):
                        sem_19_small[c] = torch.from_numpy(
                            (pred_tid_small == c).astype(np.float32))

                    center_i = pred_center_batch[i]  # (1, native_h, native_w)
                    offset_i = pred_offset_batch[i]  # (2, native_h, native_w)

                    pred_pan_small, pred_segments = panoptic_inference_center_offset(
                        sem_19_small, center_i, offset_i, _THING_IDS,
                        center_threshold=center_threshold,
                        nms_kernel=nms_kernel,
                        min_area=max(1, cc_min_area // 16),
                        max_offset_dist=30.0,
                    )
                    if isinstance(pred_pan_small, torch.Tensor):
                        pred_pan_small = pred_pan_small.numpy()
                    # Upsample panoptic map to eval resolution
                    pred_pan = np.array(Image.fromarray(
                        pred_pan_small.astype(np.int32)).resize(
                        (W, H), Image.NEAREST))
                    nxt = max(pred_segments.keys()) + 1 if pred_segments else 1

                else:
                    # Build predicted panoptic map (stuff + things)
                    pred_pan = np.zeros((H, W), dtype=np.int32)
                    pred_segments = {}
                    nxt = 1

                    # Stuff segments
                    for cls in _STUFF_IDS:
                        mask = pred_sem == cls
                        if mask.sum() < 64:
                            continue
                        pred_pan[mask] = nxt
                        pred_segments[nxt] = cls
                        nxt += 1

                    # Thing segments: pre-computed instances or connected components
                    if eval_instance_subdir is not None:
                        # Use pre-computed depth-guided instance masks
                        inst_path = os.path.join(
                            cityscapes_root, eval_instance_subdir, "val",
                            city, f"{stem}_instance.png",
                        )
                        if os.path.exists(inst_path):
                            inst_map = np.array(Image.open(inst_path), dtype=np.int32)
                            if inst_map.shape != (H, W):
                                inst_map = np.array(
                                    Image.fromarray(inst_map.astype(np.uint16)).resize(
                                        (W, H), Image.NEAREST
                                    )
                                ).astype(np.int32)
                            # Assign class to each instance via majority vote
                            for uid in np.unique(inst_map):
                                if uid == 0:
                                    continue
                                mask = inst_map == uid
                                if mask.sum() < cc_min_area:
                                    continue
                                votes = pred_sem[mask]
                                cls = int(np.bincount(votes, minlength=19).argmax())
                                if cls in _THING_IDS:
                                    pred_pan[mask] = nxt
                                    pred_segments[nxt] = cls
                                    nxt += 1
                        else:
                            # Fallback to CC if instance file missing
                            for cls in _THING_IDS:
                                cls_mask = pred_sem == cls
                                if cls_mask.sum() < cc_min_area:
                                    continue
                                labeled, n_cc = ndimage.label(cls_mask)
                                for comp in range(1, n_cc + 1):
                                    cmask = labeled == comp
                                    if cmask.sum() < cc_min_area:
                                        continue
                                    pred_pan[cmask] = nxt
                                    pred_segments[nxt] = cls
                                    nxt += 1
                    else:
                        # Connected components on predicted semantics
                        for cls in _THING_IDS:
                            cls_mask = pred_sem == cls
                            if cls_mask.sum() < cc_min_area:
                                continue
                            labeled, n_cc = ndimage.label(cls_mask)
                            for comp in range(1, n_cc + 1):
                                cmask = labeled == comp
                                if cmask.sum() < cc_min_area:
                                    continue
                                pred_pan[cmask] = nxt
                                pred_segments[nxt] = cls
                                nxt += 1

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

                gt_inst_path = os.path.join(
                    gt_dir, city, f"{stem}_gtFine_instanceIds.png")
                if os.path.exists(gt_inst_path):
                    gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
                    if gt_inst.shape != (H, W):
                        gt_inst = np.array(
                            Image.fromarray(gt_inst).resize((W, H), Image.NEAREST))
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
                for cat in range(num_cls):
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

    # Compute metrics
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    miou = iou[union > 0].mean() * 100

    all_pq, stuff_pq, thing_pq = [], [], []
    per_class = {}
    for c in range(num_cls):
        t, f_p, f_n, s = tp[c], fp[c], fn[c], iou_sum[c]
        if t + f_p + f_n > 0:
            sq = s / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0
        per_class[_CS_CLASS_NAMES[c]] = round(pq * 100, 2)
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            (stuff_pq if c in _STUFF_IDS else thing_pq).append(pq)

    pq_all = float(np.mean(all_pq)) * 100 if all_pq else 0.0
    pq_stuff = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
    pq_things = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0

    change_pct = changed_pixels / max(total_pixels, 1) * 100

    model.train()
    return {
        "PQ": round(pq_all, 2),
        "PQ_stuff": round(pq_stuff, 2),
        "PQ_things": round(pq_things, 2),
        "mIoU": round(miou, 2),
        "changed_pct": round(change_pct, 2),
        "per_class_pq": per_class,
        "per_class_iou": {_CS_CLASS_NAMES[i]: round(iou[i] * 100, 2) for i in range(num_cls)},
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    # Model
    num_classes = args.num_classes if hasattr(args, 'num_classes') else NUM_CLASSES
    model_type = getattr(args, 'model_type', 'cscm')
    if model_type == "unet":
        model = DepthGuidedUNet(
            num_classes=num_classes,
            feature_dim=768,
            bridge_dim=args.bridge_dim,
            num_bottleneck_blocks=getattr(args, 'num_bottleneck_blocks', 2),
            skip_dim=getattr(args, 'skip_dim', 32),
            coupling_strength=args.coupling_strength,
            gradient_checkpointing=args.gradient_checkpointing,
            rich_skip=getattr(args, 'rich_skip', False),
            num_final_blocks=getattr(args, 'num_final_blocks', 0),
            num_decoder_stages=getattr(args, 'num_decoder_stages', 2),
            block_type=args.block_type,
            window_size=getattr(args, 'window_size', 8),
            num_heads=getattr(args, 'num_heads', 4),
            use_instance=getattr(args, 'use_instance', False),
            inst_skip_dim=getattr(args, 'inst_skip_dim', 16),
            use_instance_heads=getattr(args, 'use_instance_heads', False),
        ).to(device)
    elif model_type == "hires":
        model = HiResRefineNet(
            num_classes=num_classes,
            feature_dim=768,
            bridge_dim=args.bridge_dim,
            num_blocks=args.num_blocks,
            block_type=args.block_type,
            upsample_strategy=getattr(args, 'upsample_strategy', 'transposed_conv'),
            coupling_strength=args.coupling_strength,
            window_size=getattr(args, 'window_size', 8),
            num_heads=getattr(args, 'num_heads', 4),
            layer_type=args.layer_type,
            d_state=args.d_state,
            chunk_size=args.chunk_size,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
    else:
        model = CSCMRefineNet(
            num_classes=num_classes,
            feature_dim=768,
            bridge_dim=args.bridge_dim,
            num_blocks=args.num_blocks,
            block_type=args.block_type,
            layer_type=args.layer_type,
            scan_mode=args.scan_mode,
            coupling_strength=args.coupling_strength,
            d_state=args.d_state,
            chunk_size=args.chunk_size,
            gradient_checkpointing=args.gradient_checkpointing,
            use_aspp=getattr(args, 'use_aspp', False),
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {total_params:,} parameters")

    # Load pretrained semantic weights for curriculum training (instance heads)
    pretrained_semantic = getattr(args, 'pretrained_semantic', None)
    if pretrained_semantic:
        ckpt = torch.load(pretrained_semantic, weights_only=False, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded pretrained semantic weights from {pretrained_semantic}")
        if missing:
            log.info(f"  Missing keys (expected for instance heads): {missing}")
        if unexpected:
            log.warning(f"  Unexpected keys: {unexpected}")

    # Build cluster→trainID LUT for overclustered evaluation
    cluster_to_trainid_lut = None
    centroids_path = getattr(args, 'centroids_path', None)
    if centroids_path and os.path.exists(centroids_path):
        c2c = np.load(centroids_path)["cluster_to_class"]
        cluster_to_trainid_lut = np.full(256, 255, dtype=np.uint8)
        for cid in range(len(c2c)):
            cluster_to_trainid_lut[cid] = int(c2c[cid])
        log.info(f"Loaded cluster→trainID LUT from {centroids_path} "
                 f"({len(c2c)} clusters → {len(set(c2c.tolist()))} trainIDs)")

    # Datasets — compute target resolution from decoder stages for UNet
    if model_type == "unet":
        n_stages = getattr(args, 'num_decoder_stages', 2)
        target_h = 32 * (2 ** n_stages)   # 2 stages: 128, 3 stages: 256
        target_w = 64 * (2 ** n_stages)   # 2 stages: 256, 3 stages: 512
        log.info(f"UNet output resolution: {target_h}x{target_w} ({n_stages} decoder stages)")
    elif model_type == "hires":
        target_h = getattr(args, 'target_h', 128)
        target_w = getattr(args, 'target_w', 256)
    else:
        target_h = None
        target_w = None
    return_depth_full = (model_type == "unet")
    instance_subdir = getattr(args, 'instance_subdir', None)
    instance_target_subdir = getattr(args, 'instance_target_subdir', None)
    train_dataset = PseudoLabelDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        logits_subdir=args.logits_subdir,
        num_classes=num_classes,
        target_h=target_h,
        target_w=target_w,
        return_depth_full=return_depth_full,
        instance_subdir=instance_subdir,
        instance_target_subdir=instance_target_subdir,
    )
    val_dataset = PseudoLabelDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        logits_subdir=args.logits_subdir,
        num_classes=num_classes,
        target_h=target_h,
        target_w=target_w,
        return_depth_full=return_depth_full,
        instance_subdir=instance_subdir,
    )

    pin_mem = device.type == "cuda"  # MPS doesn't support pin_memory
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem,
    )

    # Optimizer and scheduler
    use_instance_heads = getattr(args, 'use_instance_heads', False)
    if use_instance_heads:
        instance_lr_mult = getattr(args, 'instance_lr_multiplier', 10.0)
        backbone_params = []
        instance_params = []
        for name, param in model.named_parameters():
            if name.startswith("instance_head."):
                instance_params.append(param)
            else:
                backbone_params.append(param)
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr},
            {"params": instance_params, "lr": args.lr * instance_lr_mult},
        ], weight_decay=args.weight_decay)
        log.info(f"Instance heads: {len(instance_params)} params at LR={args.lr * instance_lr_mult:.6f}, "
                 f"backbone: {len(backbone_params)} params at LR={args.lr:.6f}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    # Loss
    loss_fn = RefineNetLoss(
        lambda_distill=args.lambda_distill,
        lambda_distill_min=args.lambda_distill_min,
        lambda_align=args.lambda_align,
        lambda_proto=args.lambda_proto,
        lambda_ent=args.lambda_ent,
        label_smoothing=args.label_smoothing,
        thing_weight=args.thing_weight,
        lambda_bpl=args.lambda_bpl,
        focal_gamma=args.focal_gamma,
        lambda_depth_boundary=args.lambda_depth_boundary,
        lambda_instance_uniform=getattr(args, 'lambda_instance_uniform', 0.0),
        lambda_instance_boundary=getattr(args, 'lambda_instance_boundary', 0.0),
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["total_params"] = total_params
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Mixed precision setup
    use_amp = args.amp
    if use_amp:
        if device.type == "mps":
            amp_dtype = torch.bfloat16
            amp_device = "mps"
        elif device.type == "cuda":
            amp_dtype = torch.bfloat16
            amp_device = "cuda"
        else:
            use_amp = False
            amp_dtype = torch.float32
            amp_device = "cpu"
    else:
        amp_dtype = torch.float32
        amp_device = device.type
    if use_amp:
        log.info(f"Mixed precision: {amp_dtype} on {amp_device}")

    # Resume from checkpoint
    start_epoch = 1
    best_pq = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_pq = ckpt.get("metrics", {}).get("PQ", 0.0)
        log.info(f"Resumed from {args.resume} (epoch {ckpt.get('epoch')}, PQ={best_pq:.2f})")
        # Advance scheduler to match resumed epoch
        for _ in range(start_epoch - 1):
            scheduler.step()

    # Training
    effective_bs = args.batch_size * args.gradient_accumulation_steps
    log.info(f"Training for epochs {start_epoch}-{args.num_epochs}, "
             f"{len(train_loader)} batches/epoch"
             + (f", gradient accumulation={args.gradient_accumulation_steps} "
                f"(effective batch size={effective_bs})"
                if args.gradient_accumulation_steps > 1 else ""))

    freeze_backbone_epochs = getattr(args, 'freeze_backbone_epochs', 8)

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        loss_fn.set_epoch(epoch, args.num_epochs)
        epoch_losses = {"total": 0, "distill": 0, "align": 0, "proto": 0, "entropy": 0, "bpl": 0,
                        "inst_uniform": 0, "inst_boundary": 0, "center": 0, "offset": 0}
        num_batches = 0
        t0 = time.time()

        # Curriculum: freeze/unfreeze backbone for instance head training
        if use_instance_heads:
            if epoch <= freeze_backbone_epochs:
                # Phase 1: freeze backbone, train only instance heads
                for name, param in model.named_parameters():
                    param.requires_grad = name.startswith("instance_head.")
                if epoch == start_epoch:
                    n_frozen = sum(1 for n, p in model.named_parameters() if not p.requires_grad)
                    n_trainable = sum(1 for n, p in model.named_parameters() if p.requires_grad)
                    log.info(f"Phase 1 (ep 1-{freeze_backbone_epochs}): backbone FROZEN "
                             f"({n_frozen} params frozen, {n_trainable} trainable)")
            elif epoch == freeze_backbone_epochs + 1:
                # Phase 2: unfreeze all, reduce backbone LR
                for param in model.parameters():
                    param.requires_grad = True
                optimizer.param_groups[0]["lr"] = args.lr * 0.1  # backbone LR
                log.info(f"Phase 2 (ep {freeze_backbone_epochs+1}+): all unfrozen, "
                         f"backbone LR = {args.lr * 0.1:.6f}")

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    ncols=140, leave=True)
        for batch_idx, batch in enumerate(pbar):
            # Move tensors to device
            cause_logits = batch["cause_logits"].to(device)
            dinov2_features = batch["dinov2_features"].to(device)
            depth = batch["depth"].to(device)
            depth_grads = batch["depth_grads"].to(device)
            depth_full = batch["depth_full"].to(device) if "depth_full" in batch else None
            instance_full = batch["instance_full"].to(device) if "instance_full" in batch else None

            # Feature augmentation (noise + spatial dropout on DINOv2 features)
            if args.feature_augment:
                noise = torch.randn_like(dinov2_features) * args.feature_noise_sigma
                dinov2_features = dinov2_features + noise
                B_f, C_f, H_f, W_f = dinov2_features.shape
                drop_mask = (torch.rand(B_f, 1, H_f, W_f, device=dinov2_features.device)
                             > args.feature_dropout_rate).float()
                dinov2_features = dinov2_features * drop_mask

            # Single forward pass (no dual augmentation)
            accum_steps = args.gradient_accumulation_steps
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                fwd_kwargs = {}
                if depth_full is not None:
                    fwd_kwargs["depth_full"] = depth_full
                if instance_full is not None:
                    fwd_kwargs["instance_full"] = instance_full
                output = model(dinov2_features, depth, depth_grads, **fwd_kwargs)

                # Handle dict output (instance heads) vs plain tensor
                if isinstance(output, dict):
                    logits = output["semantic"]
                    pred_center = output["center"]
                    pred_offset = output["offset"]
                else:
                    logits = output
                    pred_center = pred_offset = None

                # Downsample instance map to logits resolution for loss
                instance_map_for_loss = None
                if instance_full is not None:
                    _, _, lH, lW = logits.shape
                    instance_map_for_loss = F.interpolate(
                        instance_full.float(), size=(lH, lW), mode="nearest"
                    ).long()
                total_loss, loss_dict = loss_fn(
                    logits, cause_logits, dinov2_features, depth,
                    instance_map=instance_map_for_loss,
                )

                # Instance head losses (center heatmap + offset)
                if pred_center is not None and "center_target" in batch:
                    center_target = batch["center_target"].to(device)
                    offset_target = batch["offset_target"].to(device)
                    thing_mask = batch["thing_mask"].to(device)
                    lambda_center = getattr(args, 'lambda_center', 1.0)
                    lambda_offset = getattr(args, 'lambda_offset', 1.0)
                    l_center, l_offset = center_offset_loss(
                        pred_center, pred_offset, center_target, offset_target, thing_mask)
                    total_loss = total_loss + lambda_center * l_center + lambda_offset * l_offset
                    loss_dict["center"] = l_center
                    loss_dict["offset"] = l_offset

                # Scale loss for gradient accumulation
                if accum_steps > 1:
                    total_loss = total_loss / accum_steps

            # Backward (outside autocast — gradients computed in float32)
            total_loss.backward()

            # Step every accum_steps micro-batches (or at end of epoch)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Sanitize NaN/Inf gradients (GatedDeltaNet backward on MPS
                # produces NaN in the SSD exp-cumsum chain)
                nan_grad_count = 0
                for p in model.parameters():
                    if p.grad is not None and not p.grad.isfinite().all():
                        nan_grad_count += p.grad.isfinite().logical_not().sum().item()
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if nan_grad_count > 0 and (batch_idx + 1) % 50 == 0:
                    log.warning(f"  Replaced {nan_grad_count} NaN/Inf gradient elements")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    continue
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += float(v)
            num_batches += 1

            # Update progress bar
            postfix = {
                "loss": f"{loss_dict['total'].item():.4f}",
                "dist": f"{epoch_losses['distill']/num_batches:.4f}",
                "align": f"{epoch_losses['align']/num_batches:.4f}",
                "proto": f"{epoch_losses['proto']/num_batches:.4f}",
                "ent": f"{epoch_losses['entropy']/num_batches:.4f}",
            }
            if epoch_losses.get("inst_uniform", 0) != 0:
                postfix["iu"] = f"{epoch_losses['inst_uniform']/num_batches:.4f}"
            if epoch_losses.get("inst_boundary", 0) != 0:
                postfix["ib"] = f"{epoch_losses['inst_boundary']/num_batches:.4f}"
            if epoch_losses.get("center", 0) != 0:
                postfix["ctr"] = f"{epoch_losses['center']/num_batches:.4f}"
            if epoch_losses.get("offset", 0) != 0:
                postfix["off"] = f"{epoch_losses['offset']/num_batches:.4f}"
            pbar.set_postfix(**postfix)

        pbar.close()
        scheduler.step()

        # Epoch summary
        dt = time.time() - t0
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        inst_summary = ""
        if avg_losses.get("inst_uniform", 0) > 0:
            inst_summary += f" inst_uniform={avg_losses['inst_uniform']:.4f}"
        if avg_losses.get("inst_boundary", 0) > 0:
            inst_summary += f" inst_boundary={avg_losses['inst_boundary']:.4f}"
        if avg_losses.get("center", 0) > 0:
            inst_summary += f" center={avg_losses['center']:.4f}"
        if avg_losses.get("offset", 0) > 0:
            inst_summary += f" offset={avg_losses['offset']:.4f}"
        log.info(
            f"Epoch {epoch}/{args.num_epochs} ({dt:.0f}s) | "
            f"loss={avg_losses['total']:.4f} "
            f"distill={avg_losses['distill']:.4f} (w={loss_fn._distill_scale:.3f}) "
            f"align={avg_losses['align']:.4f} "
            f"proto={avg_losses['proto']:.4f} "
            f"ent={avg_losses['entropy']:.4f}"
            f"{inst_summary} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Evaluate every eval_interval epochs
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            log.info(f"Evaluating at epoch {epoch}...")
            metrics = evaluate_panoptic(
                model, val_loader, device, args.cityscapes_root,
                num_classes=num_classes,
                cluster_to_trainid_lut=cluster_to_trainid_lut,
                eval_instance_subdir=getattr(args, 'eval_instance_subdir', None),
                use_center_offset=use_instance_heads,
                center_threshold=getattr(args, 'center_threshold', 0.1),
                nms_kernel=getattr(args, 'nms_kernel', 7))
            log.info(
                f"  PQ={metrics['PQ']:.2f} | "
                f"PQ_stuff={metrics['PQ_stuff']:.2f} | "
                f"PQ_things={metrics['PQ_things']:.2f} | "
                f"mIoU={metrics['mIoU']:.2f} | "
                f"changed={metrics['changed_pct']:.1f}%"
            )

            # Log coupling strengths
            if hasattr(model, 'blocks'):
                blocks_to_log = model.blocks
            elif hasattr(model, 'bottleneck'):
                # DepthGuidedUNet: bottleneck + decoder stage blocks + final blocks
                blocks_to_log = list(model.bottleneck) + [
                    s.block for s in model.decoder_stages
                ] + list(getattr(model, 'final_blocks', []))
            else:
                blocks_to_log = []
            for i, block in enumerate(blocks_to_log):
                log.info(
                    f"  Block {i}: alpha={block.alpha.item():.4f}, "
                    f"beta={block.beta.item():.4f}"
                )

            # Save checkpoint
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "config": config,
            }, ckpt_path)

            # Save best (track PQ as primary metric)
            if metrics["PQ"] > best_pq:
                best_pq = metrics["PQ"]
                best_path = os.path.join(args.output_dir, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config": config,
                }, best_path)
                log.info(f"  New best PQ: {best_pq:.2f}% (saved to best.pth)")

            # Save metrics history
            metrics["epoch"] = epoch
            metrics_path = os.path.join(args.output_dir, "metrics_history.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    log.info(f"Training complete. Best PQ: {best_pq:.2f}%")
    log.info(f"Checkpoints saved to: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CSCMRefineNet for semantic pseudo-label refinement")

    # Data
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause_crf",
                        help="Subdirectory for semantic pseudo-labels")
    parser.add_argument("--logits_subdir", type=str, default=None,
                        help="Subdirectory for soft logits (optional)")
    parser.add_argument("--centroids_path", type=str, default=None,
                        help="Path to kmeans_centroids.npz for overclustered eval (k>19)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/refine_net")

    # Model architecture
    parser.add_argument("--num_classes", type=int, default=27,
                        help="Number of semantic classes (27 for CAUSE, 19 for mapped overclusters)")
    parser.add_argument("--bridge_dim", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="cscm",
                        choices=["cscm", "hires", "unet"],
                        help="Model type: cscm (original 32x64), hires (upsampled), or unet (depth-guided UNet)")
    parser.add_argument("--num_bottleneck_blocks", type=int, default=2,
                        help="Number of bottleneck blocks at 32x64 (UNet only)")
    parser.add_argument("--skip_dim", type=int, default=32,
                        help="Channel dim for depth skip features (UNet only)")
    parser.add_argument("--block_type", type=str, default="conv",
                        choices=["conv", "mamba", "attention", "mamba_bidir", "mamba_spatial", "mambaout"])
    parser.add_argument("--layer_type", type=str, default="gated_delta_net",
                        choices=["mamba2", "gated_delta_net"])
    parser.add_argument("--scan_mode", type=str, default="bidirectional",
                        choices=["raster", "bidirectional", "cross_scan"])
    parser.add_argument("--coupling_strength", type=float, default=0.1)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--upsample_strategy", type=str, default="transposed_conv",
                        choices=["bilinear", "transposed_conv", "pixel_shuffle", "none"],
                        help="Upsampling strategy for HiRes model")
    parser.add_argument("--target_h", type=int, default=128,
                        help="Target spatial height for HiRes model")
    parser.add_argument("--target_w", type=int, default=256,
                        help="Target spatial width for HiRes model")
    parser.add_argument("--window_size", type=int, default=8,
                        help="Window size for windowed attention blocks")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        default=True,
                        help="Enable gradient checkpointing to reduce memory")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable bfloat16 mixed precision (off by default — fp32 is faster on MPS)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")

    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over K micro-batches before stepping. "
                             "Effective batch size = batch_size * K.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=1)

    # Loss weights
    parser.add_argument("--lambda_distill", type=float, default=1.0,
                        help="Initial supervision weight (cosine decay to --lambda_distill_min)")
    parser.add_argument("--lambda_distill_min", type=float, default=0.0,
                        help="Final supervision weight after cosine warmdown")
    parser.add_argument("--lambda_align", type=float, default=5.0)
    parser.add_argument("--lambda_proto", type=float, default=0.5)
    parser.add_argument("--lambda_ent", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--thing_weight", type=float, default=1.0,
                        help="Distillation weight multiplier for thing classes (TAD, default=1.0=off)")
    parser.add_argument("--lambda_bpl", type=float, default=0.0,
                        help="Boundary preservation loss weight (BPL, default=0.0=off)")
    parser.add_argument("--use_aspp", action="store_true", default=False,
                        help="Use ASPP-lite multi-scale dilated convolutions in CoupledConvBlock")
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal loss gamma (0=standard CE, >0=focal loss)")
    parser.add_argument("--lambda_depth_boundary", type=float, default=0.0,
                        help="Depth-boundary weighting on distillation loss (0=off)")
    parser.add_argument("--feature_augment", action="store_true", default=False,
                        help="Enable feature noise + spatial dropout augmentation")
    parser.add_argument("--feature_noise_sigma", type=float, default=0.02,
                        help="Gaussian noise std on DINOv2 features (default=0.02)")
    parser.add_argument("--feature_dropout_rate", type=float, default=0.05,
                        help="Spatial dropout rate on DINOv2 features (default=0.05)")
    parser.add_argument("--rich_skip", action="store_true", default=False,
                        help="Use richer depth skip (2nd conv + Laplacian, UNet only)")
    parser.add_argument("--num_final_blocks", type=int, default=0,
                        help="Extra CoupledConvBlocks at output resolution after decoder (UNet only)")
    parser.add_argument("--num_decoder_stages", type=int, default=2,
                        help="Number of 2x upsample stages (2=128x256, 3=256x512, UNet only)")

    # Instance conditioning
    parser.add_argument("--use_instance", action="store_true", default=False,
                        help="Enable instance conditioning in UNet decoder (inject instance boundary skip connections)")
    parser.add_argument("--inst_skip_dim", type=int, default=16,
                        help="Instance skip feature channels (default=16)")
    parser.add_argument("--instance_subdir", type=str, default=None,
                        help="Subdirectory for instance pseudo-labels (e.g. pseudo_instance_spidepth)")
    parser.add_argument("--eval_instance_subdir", type=str, default=None,
                        help="Use pre-computed instances for eval instead of CC (e.g. pseudo_instance_spidepth)")
    parser.add_argument("--lambda_instance_uniform", type=float, default=0.0,
                        help="Instance uniformity loss weight (0=off)")
    parser.add_argument("--lambda_instance_boundary", type=float, default=0.0,
                        help="Instance boundary alignment loss weight (0=off)")

    # Instance heads (center heatmap + offset)
    parser.add_argument("--use_instance_heads", action="store_true", default=False,
                        help="Add center heatmap + offset heads for instance segmentation")
    parser.add_argument("--instance_target_subdir", type=str, default=None,
                        help="Subdirectory with center/offset .npy targets (e.g. instance_targets_128x256)")
    parser.add_argument("--lambda_center", type=float, default=1.0,
                        help="Center heatmap loss weight")
    parser.add_argument("--lambda_offset", type=float, default=1.0,
                        help="Offset regression loss weight")
    parser.add_argument("--pretrained_semantic", type=str, default=None,
                        help="Path to pretrained UNet P2-B checkpoint for curriculum training")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=8,
                        help="Number of epochs to freeze backbone in Phase 1 of curriculum")
    parser.add_argument("--instance_lr_multiplier", type=float, default=10.0,
                        help="LR multiplier for instance heads vs backbone")
    parser.add_argument("--center_threshold", type=float, default=0.1,
                        help="Center heatmap threshold for inference (at eval resolution)")
    parser.add_argument("--nms_kernel", type=int, default=7,
                        help="NMS kernel size at model resolution (auto-scaled to eval res)")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
