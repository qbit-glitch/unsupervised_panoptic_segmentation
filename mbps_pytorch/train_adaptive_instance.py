#!/usr/bin/env python3
"""Train AdaptiveInstanceNet: learned instance boundary predictor.

Replaces the fixed depth-gradient threshold (τ=0.10) with a learned,
spatially-adaptive split predictor trained with self-supervised losses:
  1. Split distillation: soft BCE from depth gradient teacher
  2. Feature-guided boundary: DINOv2 feature discontinuity alignment
  3. Contrastive embedding: same-instance pixels → close embeddings
  4. Embedding regularization: prevent collapse to constant

Usage:
    python mbps_pytorch/train_adaptive_instance.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir checkpoints/adaptive_instance \
        --num_epochs 30 --batch_size 4 --device auto

References:
    - Method 2 from mamba_bridge_pseudo_label_refinement.md
    - MTMamba (ECCV 2024): Multi-task Mamba decomposition
    - CutS3D (ICCV 2025): Spatial Importance Sharpening
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mbps_pytorch.adaptive_instance_net import AdaptiveInstanceNet
from mbps_pytorch.adaptive_instance_semantics import (
    SemanticSpec,
    encode_semantic_onehot,
    infer_semantic_spec,
    map_to_trainid,
    validate_semantic_inputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PATCH_H, PATCH_W = 32, 64
FUSION_H, FUSION_W = 128, 256
FUSION_CHANNELS = 9
EVAL_H, EVAL_W = 512, 1024
SAM3_MAX_MASKS = 20  # max per-image SAM3 masks to load (pad to this)

# SAM3 class labels that are THINGS (not stuff).
# 0=person,1=bicycle,2=motorcycle,3=rider,6=truck,7=bus,8=train,10=caravan,11=trailer,12=car
_SAM3_THING_LABELS: frozenset = frozenset({0, 1, 2, 3, 6, 7, 8, 10, 11, 12})
_SAM3_TO_TRAINID = {
    0: 11,   # person
    1: 18,   # bicycle
    2: 17,   # motorcycle
    3: 12,   # rider
    6: 14,   # truck
    7: 15,   # bus
    8: 16,   # train
    12: 13,  # car
}

# Cityscapes trainID constants
_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
_STUFF_IDS = set(range(0, 11))
_THING_IDS = set(range(11, 19))
_THING_IDS_LIST = list(range(11, 19))
_THING_BACKGROUND_IDX = 0
_TRAINID_TO_THING_HEAD_IDX = {tid: i + 1 for i, tid in enumerate(_THING_IDS_LIST)}
_THING_HEAD_IDX_TO_TRAINID = np.array([-1] + _THING_IDS_LIST, dtype=np.int16)
_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InstanceDataset(Dataset):
    """Load DINOv2 features, depth, semantics, and SAM3 masks for instance training."""

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        semantic_subdir: str = "pseudo_semantic_cause_crf",
        feature_subdir: str = "dinov2_features",
        depth_subdir: str = "depth_spidepth",
        sam3_subdir: str = "sam_fine_masks_sam3",
        load_sam3: bool = True,
        semantic_spec: Optional[SemanticSpec] = None,
        use_highres_fusion: bool = False,
        fusion_h: int = FUSION_H,
        fusion_w: int = FUSION_W,
        sam3_mask_threshold: float = 0.25,
        sam3_boundary_small_weight: float = 4.0,
        sam3_fusion_dropout: float = 0.0,
        use_fused_targets: bool = True,
        fused_sobel_tau: float = 0.10,
        fused_min_area: int = 16,
    ):
        self.root = cityscapes_root
        self.split = split
        self.semantic_subdir = semantic_subdir
        self.feature_subdir = feature_subdir
        self.depth_subdir = depth_subdir
        self.sam3_subdir = sam3_subdir
        self.load_sam3 = load_sam3
        self.semantic_spec = semantic_spec or infer_semantic_spec(
            cityscapes_root, semantic_subdir, split=split)
        self.use_highres_fusion = use_highres_fusion
        self.fusion_h = fusion_h
        self.fusion_w = fusion_w
        self.sam3_mask_threshold = sam3_mask_threshold
        self.sam3_boundary_small_weight = sam3_boundary_small_weight
        self.sam3_fusion_dropout = float(np.clip(sam3_fusion_dropout, 0.0, 1.0))
        self.use_fused_targets = use_fused_targets
        self.fused_sobel_tau = fused_sobel_tau
        self.fused_min_area = fused_min_area

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

        log.info(f"InstanceDataset: {len(self.entries)} images ({split})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        stem, city = entry["stem"], entry["city"]

        # DINOv2 features: (2048, 768) → (768, 32, 64)
        feat_path = os.path.join(
            self.root, self.feature_subdir, self.split, city,
            f"{stem}_leftImg8bit.npy",
        )
        features = np.load(feat_path).astype(np.float32)
        features = features.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)

        # Depth: (512, 1024) → (1, 32, 64)
        depth_path = os.path.join(
            self.root, self.depth_subdir, self.split, city, f"{stem}.npy",
        )
        depth_full = np.load(depth_path)
        depth_patch = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
        depth_patch = F.interpolate(
            depth_patch, size=(PATCH_H, PATCH_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        depth_np = depth_patch.numpy()

        # Sobel gradients
        depth_grads = self._sobel_gradients(depth_np[0])  # (2, 32, 64)

        # Depth gradient magnitude (teacher signal for split head)
        grad_mag = np.sqrt(depth_grads[0] ** 2 + depth_grads[1] ** 2)  # (32, 64)

        # Semantic probabilities: CAUSE-27, trainID-19, or raw k-cluster one-hot.
        sem_path = os.path.join(
            self.root, self.semantic_subdir, self.split, city, f"{stem}.png",
        )
        sem_full = np.array(Image.open(sem_path))
        sem_patch = np.array(
            Image.fromarray(sem_full).resize((PATCH_W, PATCH_H), Image.NEAREST)
        )
        onehot = encode_semantic_onehot(sem_patch, self.semantic_spec)

        # TrainID semantic map at patch res (for thing mask during training)
        sem_trainid_patch = map_to_trainid(sem_patch, self.semantic_spec)

        target_h = self.fusion_h if self.use_highres_fusion else PATCH_H
        target_w = self.fusion_w if self.use_highres_fusion else PATCH_W
        grad_mag_target = grad_mag
        sem_trainid_target = sem_trainid_patch
        depth_fusion = depth_grads_fusion = grad_mag_fusion = None

        if self.use_highres_fusion:
            depth_fusion = self._resize_depth(depth_full, target_h, target_w)
            depth_fusion = self._normalize_depth(depth_fusion)
            depth_grads_fusion = self._sobel_gradients(depth_fusion)
            grad_mag_fusion = np.sqrt(
                depth_grads_fusion[0] ** 2 + depth_grads_fusion[1] ** 2)
            p95 = np.percentile(grad_mag_fusion, 95)
            if p95 > 1e-6:
                grad_mag_fusion = np.clip(grad_mag_fusion / p95, 0.0, 1.0)

            sem_fusion = np.array(
                Image.fromarray(sem_full).resize(
                    (target_w, target_h), Image.NEAREST)
            )
            sem_trainid_target = map_to_trainid(sem_fusion, self.semantic_spec)
            grad_mag_target = grad_mag_fusion

        # SAM3 instance masks (thing classes only). In high-res fusion mode,
        # keep targets at 128x256 so small masks survive supervision.
        (sam3_boundary, sam3_instance_map, sam3_masks_pad, sam3_ious_pad,
         sam3_smallness, sam3_class_ids) = self._load_sam3_masks(
            city, stem, target_h, target_w)
        if self.use_fused_targets:
            (sam3_boundary, sam3_instance_map, sam3_masks_pad, sam3_ious_pad,
             sam3_smallness, thing_target) = self._build_fused_targets(
                sem_trainid_target,
                grad_mag_target,
                sam3_masks_pad.numpy(),
                sam3_ious_pad.numpy(),
                sam3_class_ids.numpy(),
                target_h,
                target_w,
            )
        else:
            thing_target = self._thing_target_from_instance_map(
                sam3_instance_map.numpy(), sam3_class_ids.numpy())
        sam3_boundary_weight = 1.0 + self.sam3_boundary_small_weight * sam3_smallness

        sample = {
            "dinov2_features": torch.from_numpy(features).float(),
            "depth": torch.from_numpy(depth_np).float(),
            "depth_grads": torch.from_numpy(depth_grads).float(),
            "cause_logits": torch.from_numpy(onehot).float(),  # legacy key name
            "grad_mag": torch.from_numpy(grad_mag).float(),
            "sem_trainid": torch.from_numpy(sem_trainid_patch.astype(np.int64)),
            "sam3_boundary": sam3_boundary,
            "sam3_boundary_weight": sam3_boundary_weight,
            "sam3_instance_map": sam3_instance_map,
            "sam3_masks": sam3_masks_pad,
            "sam3_ious": sam3_ious_pad,           # (SAM3_MAX_MASKS,) float, zero-padded
            "thing_target": thing_target,
            "stem": stem,
            "city": city,
        }

        if self.use_highres_fusion:
            fusion_inputs = self._build_fusion_inputs(
                depth_fusion,
                depth_grads_fusion,
                grad_mag_fusion,
                sam3_boundary.numpy(),
                sam3_instance_map.numpy(),
                sam3_ious_pad.numpy(),
                sam3_smallness.numpy(),
            )
            if (self.sam3_fusion_dropout >= 1.0
                    or (self.split == "train"
                        and self.sam3_fusion_dropout > 0.0
                        and np.random.random() < self.sam3_fusion_dropout)):
                # Keep DepthPro/Sobel channels; drop SAM3 prior channels so SAM3
                # remains a teacher instead of a train-time shortcut.
                fusion_inputs[4:] = 0.0
            sample.update({
                "fusion_inputs": torch.from_numpy(fusion_inputs).float(),
                "grad_mag_fusion": torch.from_numpy(grad_mag_fusion).float(),
                "sem_trainid_fusion": torch.from_numpy(
                    sem_trainid_target.astype(np.int64)),
            })

        return sample

    def _load_sam3_masks(self, city: str, stem: str, out_h: int, out_w: int):
        """Load SAM3 NPZ, filter to thing classes, build boundary + instance map at patch res."""
        empty_boundary = torch.zeros(out_h, out_w)
        empty_inst_map = torch.full((out_h, out_w), -1, dtype=torch.long)
        empty_masks = torch.zeros(SAM3_MAX_MASKS, out_h, out_w)
        empty_ious = torch.zeros(SAM3_MAX_MASKS)
        empty_smallness = torch.zeros(out_h, out_w)
        empty_class_ids = torch.full((SAM3_MAX_MASKS,), -1, dtype=torch.long)

        if not self.load_sam3:
            return (
                empty_boundary, empty_inst_map, empty_masks, empty_ious,
                empty_smallness, empty_class_ids,
            )

        npz_path = os.path.join(
            self.root, self.sam3_subdir, self.split, city,
            f"{stem}_fine_masks.npz",
        )
        if not os.path.exists(npz_path):
            return (
                empty_boundary, empty_inst_map, empty_masks, empty_ious,
                empty_smallness, empty_class_ids,
            )

        try:
            data = np.load(npz_path)
            masks_full = data["masks"].astype(bool)     # (N, H, W)
            ious = data["iou_scores"].astype(np.float32)  # (N,)
            cls_labels = data["class_labels"].astype(np.int32)  # (N,)
        except Exception:
            return (
                empty_boundary, empty_inst_map, empty_masks, empty_ious,
                empty_smallness, empty_class_ids,
            )

        # Keep only Cityscapes-evaluable thing-class masks.
        keep = np.array([int(c) in _SAM3_TO_TRAINID for c in cls_labels])
        if not keep.any():
            return (
                empty_boundary, empty_inst_map, empty_masks, empty_ious,
                empty_smallness, empty_class_ids,
            )
        masks_full = masks_full[keep]
        ious = ious[keep]
        cls_labels = cls_labels[keep]

        # Sort by IoU descending, cap at SAM3_MAX_MASKS
        order = np.argsort(ious)[::-1][:SAM3_MAX_MASKS]
        masks_full = masks_full[order]
        ious = ious[order]
        cls_labels = cls_labels[order]
        class_ids = np.array(
            [_SAM3_TO_TRAINID[int(c)] for c in cls_labels],
            dtype=np.int64,
        )
        N = len(ious)

        masks_patch = np.zeros((N, out_h, out_w), dtype=np.float32)
        H_full, W_full = masks_full.shape[1], masks_full.shape[2]
        for n in range(N):
            if H_full == out_h and W_full == out_w:
                masks_patch[n] = masks_full[n].astype(np.float32)
            else:
                masks_patch[n] = self._resize_mask_soft(
                    masks_full[n], out_h, out_w, self.sam3_mask_threshold)

        # Build instance map (int): -1=background, 0..N-1=mask index (later masks overwrite)
        sam3_instance_id = np.full((out_h, out_w), -1, dtype=np.int32)
        smallness = np.zeros((out_h, out_w), dtype=np.float32)
        for n in range(N):
            mask_n = masks_patch[n] > self.sam3_mask_threshold
            claim = mask_n & (sam3_instance_id < 0)
            sam3_instance_id[claim] = n
            area = max(int(mask_n.sum()), 1)
            small_score = min(1.0, np.sqrt(64.0 / area) / 4.0)
            smallness[claim] = small_score

        # Boundary: pixel where any neighbor has a different instance ID
        boundary = np.zeros((out_h, out_w), dtype=bool)
        h_diff = sam3_instance_id[:, 1:] != sam3_instance_id[:, :-1]
        boundary[:, :-1] |= h_diff
        boundary[:, 1:] |= h_diff
        v_diff = sam3_instance_id[1:, :] != sam3_instance_id[:-1, :]
        boundary[:-1, :] |= v_diff
        boundary[1:, :] |= v_diff

        # Pad masks and ious to fixed size
        masks_pad = np.zeros((SAM3_MAX_MASKS, out_h, out_w), dtype=np.float32)
        ious_pad = np.zeros(SAM3_MAX_MASKS, dtype=np.float32)
        class_ids_pad = np.full((SAM3_MAX_MASKS,), -1, dtype=np.int64)
        masks_pad[:N] = masks_patch
        ious_pad[:N] = ious
        class_ids_pad[:N] = class_ids

        return (
            torch.from_numpy(boundary.astype(np.float32)),
            torch.from_numpy(sam3_instance_id).long(),
            torch.from_numpy(masks_pad).float(),
            torch.from_numpy(ious_pad).float(),
            torch.from_numpy(smallness).float(),
            torch.from_numpy(class_ids_pad).long(),
        )

    def _build_fused_targets(
        self,
        sem_trainid: np.ndarray,
        grad_mag: np.ndarray,
        sam3_masks: np.ndarray,
        sam3_ious: np.ndarray,
        sam3_class_ids: np.ndarray,
        out_h: int,
        out_w: int,
    ):
        """Fuse SAM3 masks with Sobel-split cluster components for training.

        SAM3 provides high-quality object extents where available. The Sobel +
        semantic-cluster path fills gaps so this remains a self-supervised
        target rather than a train-time dependency on SAM3 at validation.
        """
        components = []
        claimed = np.zeros((out_h, out_w), dtype=bool)

        # 1) High-confidence SAM3 thing masks are the strongest pseudo-targets.
        for n in range(min(SAM3_MAX_MASKS, sam3_masks.shape[0])):
            train_id = int(sam3_class_ids[n]) if n < len(sam3_class_ids) else -1
            if train_id not in _TRAINID_TO_THING_HEAD_IDX or sam3_ious[n] <= 0:
                continue
            mask = sam3_masks[n] > self.sam3_mask_threshold
            claim = mask & (~claimed)
            area = int(claim.sum())
            if area < self.fused_min_area:
                continue
            components.append({
                "mask": claim,
                "train_id": train_id,
                "score": 2.0 + float(sam3_ious[n]),
                "area": area,
            })
            claimed |= claim

        # 2) Sobel + cluster connected components backfill uncovered things.
        edge = grad_mag > self.fused_sobel_tau
        for train_id in _THING_IDS_LIST:
            cls_mask = (sem_trainid == train_id) & (~claimed)
            if int(cls_mask.sum()) < self.fused_min_area:
                continue
            split_mask = cls_mask & (~edge)
            labeled, n_cc = ndimage.label(split_mask)
            for cc_id in range(1, n_cc + 1):
                cc_mask = labeled == cc_id
                area = int(cc_mask.sum())
                if area < self.fused_min_area:
                    continue
                components.append({
                    "mask": cc_mask,
                    "train_id": train_id,
                    "score": 1.0 + area / float(out_h * out_w),
                    "area": area,
                })

        components.sort(key=lambda c: (c["score"], c["area"]), reverse=True)
        components = components[:SAM3_MAX_MASKS]

        instance_id = np.full((out_h, out_w), -1, dtype=np.int32)
        masks_pad = np.zeros((SAM3_MAX_MASKS, out_h, out_w), dtype=np.float32)
        ious_pad = np.zeros(SAM3_MAX_MASKS, dtype=np.float32)
        smallness = np.zeros((out_h, out_w), dtype=np.float32)
        thing_target = np.zeros((out_h, out_w), dtype=np.int64)

        for n, comp in enumerate(components):
            mask = comp["mask"] & (instance_id < 0)
            area = int(mask.sum())
            if area < self.fused_min_area:
                continue
            train_id = int(comp["train_id"])
            instance_id[mask] = n
            masks_pad[n] = mask.astype(np.float32)
            ious_pad[n] = float(np.clip(comp["score"] / 3.0, 0.1, 1.0))
            thing_target[mask] = _TRAINID_TO_THING_HEAD_IDX[train_id]
            small_score = min(1.0, np.sqrt(64.0 / max(area, 1)) / 4.0)
            smallness[mask] = small_score

        boundary = self._instance_boundary(instance_id)
        return (
            torch.from_numpy(boundary.astype(np.float32)),
            torch.from_numpy(instance_id).long(),
            torch.from_numpy(masks_pad).float(),
            torch.from_numpy(ious_pad).float(),
            torch.from_numpy(smallness).float(),
            torch.from_numpy(thing_target).long(),
        )

    @staticmethod
    def _thing_target_from_instance_map(
        instance_map: np.ndarray,
        class_ids: np.ndarray,
    ) -> torch.Tensor:
        target = np.zeros(instance_map.shape, dtype=np.int64)
        for n, train_id in enumerate(class_ids):
            train_id = int(train_id)
            if train_id not in _TRAINID_TO_THING_HEAD_IDX:
                continue
            target[instance_map == n] = _TRAINID_TO_THING_HEAD_IDX[train_id]
        return torch.from_numpy(target).long()

    @staticmethod
    def _instance_boundary(instance_id: np.ndarray) -> np.ndarray:
        boundary = np.zeros(instance_id.shape, dtype=bool)
        h_diff = instance_id[:, 1:] != instance_id[:, :-1]
        boundary[:, :-1] |= h_diff
        boundary[:, 1:] |= h_diff
        v_diff = instance_id[1:, :] != instance_id[:-1, :]
        boundary[:-1, :] |= v_diff
        boundary[1:, :] |= v_diff
        return boundary

    @staticmethod
    def _resize_mask_soft(mask: np.ndarray, out_h: int, out_w: int,
                          threshold: float) -> np.ndarray:
        m_img = Image.fromarray((mask * 255).astype(np.uint8))
        m_small = np.array(m_img.resize((out_w, out_h), Image.BILINEAR))
        soft = m_small.astype(np.float32) / 255.0
        if soft.max() > 0 and not (soft > threshold).any():
            y, x = np.unravel_index(np.argmax(soft), soft.shape)
            soft[y, x] = 1.0
        return soft

    @staticmethod
    def _resize_depth(depth: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        return np.array(
            Image.fromarray(depth.astype(np.float32)).resize(
                (out_w, out_h), Image.BILINEAR),
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_depth(depth: np.ndarray) -> np.ndarray:
        d_min = float(np.nanmin(depth))
        d_max = float(np.nanmax(depth))
        if d_max <= d_min + 1e-6:
            return np.zeros_like(depth, dtype=np.float32)
        return ((depth - d_min) / (d_max - d_min)).astype(np.float32)

    @staticmethod
    def _build_fusion_inputs(depth, depth_grads, grad_mag, boundary,
                             instance_map, ious, smallness):
        occupancy = (instance_map >= 0).astype(np.float32)
        confidence = np.zeros_like(occupancy, dtype=np.float32)
        for n, iou in enumerate(ious):
            if iou <= 0:
                continue
            confidence[instance_map == n] = float(iou)

        if occupancy.any():
            inside = ndimage.distance_transform_edt(occupancy > 0)
            outside = ndimage.distance_transform_edt(occupancy <= 0)
            inside = inside / max(float(inside.max()), 1.0)
            outside = outside / max(float(outside.max()), 1.0)
            signed_dist = (inside - outside).astype(np.float32)
        else:
            signed_dist = np.zeros_like(occupancy, dtype=np.float32)

        return np.stack([
            depth.astype(np.float32),
            depth_grads[0].astype(np.float32),
            depth_grads[1].astype(np.float32),
            grad_mag.astype(np.float32),
            occupancy,
            boundary.astype(np.float32),
            signed_dist,
            confidence,
            smallness.astype(np.float32),
        ], axis=0)

    @staticmethod
    def _sobel_gradients(depth_2d):
        d = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0).float()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).reshape(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).reshape(1, 1, 3, 3)
        grad_x = F.conv2d(d, kx, padding=1).squeeze()
        grad_y = F.conv2d(d, ky, padding=1).squeeze()
        return torch.stack([grad_x, grad_y], dim=0).numpy()


def _load_sam3_proposals(root: str, sam3_subdir: str, split: str, city: str,
                         stem: str, out_hw=(EVAL_H, EVAL_W),
                         mask_threshold: float = 0.25,
                         min_area: int = 100,
                         max_masks: int = SAM3_MAX_MASKS):
    """Load SAM3 thing proposals with Cityscapes trainIDs at eval resolution."""
    H, W = out_hw
    npz_path = os.path.join(root, sam3_subdir, split, city, f"{stem}_fine_masks.npz")
    if not os.path.exists(npz_path):
        return []

    try:
        data = np.load(npz_path)
        masks = data["masks"].astype(bool)
        ious = data["iou_scores"].astype(np.float32)
        labels = data["class_labels"].astype(np.int32)
    except Exception:
        return []

    keep = np.array([int(c) in _SAM3_TO_TRAINID for c in labels])
    if not keep.any():
        return []

    masks = masks[keep]
    ious = ious[keep]
    labels = labels[keep]
    order = np.argsort(ious)[::-1][:max_masks]

    proposals = []
    for idx in order:
        cls = _SAM3_TO_TRAINID.get(int(labels[idx]))
        if cls is None:
            continue
        mask = masks[idx]
        if mask.shape != (H, W):
            soft = np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (W, H), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            if soft.max() > 0 and not (soft > mask_threshold).any():
                y, x = np.unravel_index(np.argmax(soft), soft.shape)
                soft[y, x] = 1.0
            mask = soft > mask_threshold
        else:
            mask = mask.astype(bool)

        labeled, n_cc = ndimage.label(mask)
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area < min_area:
                continue
            proposals.append((cc_mask, cls, float(ious[idx]), area))

    proposals.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return proposals


def _count_cityscapes_images(root: str, split: str) -> int:
    img_root = os.path.join(root, "leftImg8bit", split)
    count = 0
    for _dirpath, _dirnames, filenames in os.walk(img_root):
        count += sum(1 for f in filenames if f.endswith("_leftImg8bit.png"))
    return count


def _count_sam3_files(root: str, sam3_subdir: str, split: str) -> int:
    sam_root = os.path.join(root, sam3_subdir, split)
    count = 0
    for _dirpath, _dirnames, filenames in os.walk(sam_root):
        count += sum(1 for f in filenames if f.endswith("_fine_masks.npz"))
    return count


def _unpack_model_outputs(outputs):
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2]
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1], None
    raise ValueError("AdaptiveInstanceNet must return 2 or 3 tensors")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def split_distillation_loss(split_logit, grad_mag, tau=0.10, soft_margin=0.03):
    """Soft BCE distillation from depth gradient threshold.

    Creates soft teacher targets: sigmoid((grad_mag - tau) / soft_margin)
    so the transition is smooth rather than hard binary.

    Args:
        split_logit: (B, 1, H, W) raw logits (before sigmoid)
        grad_mag: (B, H, W) depth gradient magnitude
        tau: depth gradient threshold (matches existing heuristic)
        soft_margin: softness of the teacher transition
    """
    if grad_mag.shape[-2:] != split_logit.shape[-2:]:
        grad_mag = F.interpolate(
            grad_mag.unsqueeze(1).float(),
            size=split_logit.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    teacher = torch.sigmoid((grad_mag.unsqueeze(1) - tau) / soft_margin)
    return F.binary_cross_entropy_with_logits(split_logit, teacher)


def feature_boundary_loss(split_logit, dinov2_features, sigma=0.3):
    """Encourage split logits to align with DINOv2 feature discontinuities.

    Where DINOv2 features change sharply between neighbors → split should be high.
    Where features are similar → split should be low.

    Uses binary_cross_entropy_with_logits for numerical stability.

    Args:
        split_logit: (B, 1, H, W) raw logits (before sigmoid)
        dinov2_features: (B, 768, H, W) DINOv2 features
        sigma: scale for feature distance → probability mapping
    """
    if split_logit.shape[-2:] != dinov2_features.shape[-2:]:
        split_logit = F.interpolate(
            split_logit,
            size=dinov2_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    feats = F.normalize(dinov2_features, dim=1)  # L2 normalize

    # Horizontal feature distance
    cos_h = (feats[:, :, :, 1:] * feats[:, :, :, :-1]).sum(dim=1, keepdim=True)
    feat_edge_h = 1.0 - cos_h.clamp(-1, 1)  # distance in [0, 2]
    # Teacher: high feature distance → high split probability
    teacher_h = torch.sigmoid((feat_edge_h - sigma) / 0.1)
    # Average split logit at the boundary (mean of left and right)
    logit_h = (split_logit[:, :, :, 1:] + split_logit[:, :, :, :-1]) / 2
    loss_h = F.binary_cross_entropy_with_logits(logit_h, teacher_h)

    # Vertical feature distance
    cos_v = (feats[:, :, 1:, :] * feats[:, :, :-1, :]).sum(dim=1, keepdim=True)
    feat_edge_v = 1.0 - cos_v.clamp(-1, 1)
    teacher_v = torch.sigmoid((feat_edge_v - sigma) / 0.1)
    logit_v = (split_logit[:, :, 1:, :] + split_logit[:, :, :-1, :]) / 2
    loss_v = F.binary_cross_entropy_with_logits(logit_v, teacher_v)

    return loss_h + loss_v


def contrastive_embedding_loss(
    instance_embed, dinov2_features, depth, sem_trainid,
    num_pairs=2048, pos_depth_thresh=0.05, neg_depth_thresh=0.10,
    pos_feat_thresh=0.70, margin=1.0,
):
    """Contrastive loss on instance embeddings using DINOv2 + depth proxies.

    Within thing classes:
      Positive pairs: same class + similar depth + high DINOv2 cosine sim
      Negative pairs: same class + large depth difference

    Args:
        instance_embed: (B, E, H, W) instance embeddings
        dinov2_features: (B, 768, H, W)
        depth: (B, 1, H, W)
        sem_trainid: (B, H, W) int64 trainID labels
        num_pairs: number of pairs to sample per batch element
    """
    if instance_embed.shape[-2:] != dinov2_features.shape[-2:]:
        target_hw = dinov2_features.shape[-2:]
        instance_embed = F.interpolate(
            instance_embed, size=target_hw, mode="bilinear",
            align_corners=False)
        if depth.shape[-2:] != target_hw:
            depth = F.interpolate(
                depth.float(), size=target_hw, mode="bilinear",
                align_corners=False)
        if sem_trainid.shape[-2:] != target_hw:
            sem_trainid = F.interpolate(
                sem_trainid.unsqueeze(1).float(), size=target_hw,
                mode="nearest").squeeze(1).long()

    B, E, H, W = instance_embed.shape
    device = instance_embed.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    # Flatten spatial dims
    embed_flat = instance_embed.permute(0, 2, 3, 1).reshape(B, H * W, E)
    feat_flat = F.normalize(
        dinov2_features.permute(0, 2, 3, 1).reshape(B, H * W, -1), dim=-1
    )
    depth_flat = depth.reshape(B, H * W)
    sem_flat = sem_trainid.reshape(B, H * W)

    for b in range(B):
        # Find thing pixels
        thing_mask = torch.zeros(H * W, dtype=torch.bool, device=device)
        for tid in range(11, 19):
            thing_mask |= (sem_flat[b] == tid)

        thing_idx = thing_mask.nonzero(as_tuple=True)[0]
        if thing_idx.shape[0] < 4:
            continue

        # Sample random pairs from thing pixels
        n_thing = thing_idx.shape[0]
        n_pairs = min(num_pairs, n_thing * (n_thing - 1) // 2)
        if n_pairs < 2:
            continue

        idx_a = thing_idx[torch.randint(n_thing, (n_pairs,), device=device)]
        idx_b = thing_idx[torch.randint(n_thing, (n_pairs,), device=device)]

        # Compute pair properties
        same_class = sem_flat[b, idx_a] == sem_flat[b, idx_b]
        depth_diff = (depth_flat[b, idx_a] - depth_flat[b, idx_b]).abs()
        feat_sim = (feat_flat[b, idx_a] * feat_flat[b, idx_b]).sum(dim=-1)

        # Positive: same class + close depth + similar features
        pos_mask = same_class & (depth_diff < pos_depth_thresh) & (feat_sim > pos_feat_thresh)
        # Negative: same class + far depth (inter-object within same class)
        neg_mask = same_class & (depth_diff > neg_depth_thresh)

        # Embedding distances (use squared L2 for stability, avoid sqrt)
        embed_a = embed_flat[b, idx_a]
        embed_b = embed_flat[b, idx_b]
        embed_dist_sq = (embed_a - embed_b).pow(2).sum(dim=-1).clamp(min=1e-8)
        embed_dist = embed_dist_sq.sqrt()

        # Contrastive loss: positive → minimize distance, negative → maximize up to margin
        if pos_mask.sum() > 0:
            pos_loss = embed_dist_sq[pos_mask].mean()
            total_loss = total_loss + pos_loss
            count += 1
        if neg_mask.sum() > 0:
            neg_loss = F.relu(margin - embed_dist[neg_mask]).pow(2).mean()
            total_loss = total_loss + neg_loss
            count += 1

    if count > 0:
        total_loss = total_loss / count
    return total_loss


def embedding_regularization_loss(instance_embed):
    """Prevent embedding collapse by encouraging unit variance per dimension.

    Args:
        instance_embed: (B, E, H, W) instance embeddings
    """
    B, E, H, W = instance_embed.shape
    flat = instance_embed.reshape(B, E, -1)  # (B, E, N)
    var = flat.var(dim=-1)  # (B, E)
    # Target variance = 1.0; penalize deviation
    return (var - 1.0).pow(2).mean()


def sam3_boundary_loss(
    split_logit: torch.Tensor,
    sam3_boundary: torch.Tensor,
    boundary_weight: Optional[torch.Tensor] = None,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Supervised boundary loss using SAM3 instance mask edges.

    SAM3 provides clean, model-based instance boundaries — stronger supervision
    than the soft depth gradient teacher.

    Args:
        split_logit: (B, 1, H, W) raw logits (before sigmoid)
        sam3_boundary: (B, H, W) float boundary map {0, 1} from SAM3 instance edges
    """
    has_boundary = (sam3_boundary.sum(dim=(1, 2)) > 0)  # (B,) bool
    if not has_boundary.any():
        return split_logit.new_zeros(1).squeeze()

    logit = split_logit.squeeze(1)[has_boundary]  # (B', H, W)
    target = sam3_boundary[has_boundary]
    if boundary_weight is not None:
        weight = boundary_weight[has_boundary].float()
    else:
        weight = None

    # Positive weight: boundaries are sparse → upweight positive pixels
    pos_weight = ((target == 0).float().sum() / (target.sum().clamp(min=1))) * 0.5
    pos_weight = pos_weight.clamp(max=20.0)
    bce = F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pos_weight, weight=weight)
    if dice_weight <= 0:
        return bce

    prob = torch.sigmoid(logit)
    dice_mul = weight if weight is not None else 1.0
    inter = (prob * target * dice_mul).sum(dim=(1, 2))
    denom = ((prob + target) * dice_mul).sum(dim=(1, 2)).clamp(min=1e-6)
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0))
    return bce + dice_weight * dice.mean()


def sam3_embedding_consistency_loss(
    instance_embed: torch.Tensor,
    sam3_instance_map: torch.Tensor,
    sam3_masks: torch.Tensor,
    sam3_ious: torch.Tensor,
    margin: float = 0.5,
    max_masks: int = SAM3_MAX_MASKS,
) -> torch.Tensor:
    """Pull embeddings within same SAM3 mask together; push adjacent masks apart.

    Uses SAM3 instance map (cleaner than depth proxy) as the grouping signal.
    Replaces the depth-proxy contrastive loss for pixels covered by SAM3 masks.

    Args:
        instance_embed: (B, E, H, W)
        sam3_instance_map: (B, H, W) long, -1=background, 0..N-1=mask index
        sam3_masks: (B, SAM3_MAX_MASKS, H, W) float, zero-padded
        sam3_ious: (B, SAM3_MAX_MASKS) float, mask confidence
        margin: push margin for negative pairs
        max_masks: number of valid mask slots
    """
    B, E, H, W = instance_embed.shape
    device = instance_embed.device
    total_loss = instance_embed.new_zeros(1).squeeze()
    count = 0

    embed_flat = instance_embed.permute(0, 2, 3, 1).reshape(B, H * W, E)  # (B, HW, E)
    inst_flat = sam3_instance_map.reshape(B, H * W)  # (B, HW) long

    for b in range(B):
        # Determine number of valid masks for this image
        valid = (sam3_ious[b] > 0).sum().item()
        if valid == 0:
            continue

        mask_losses = []
        for n in range(int(valid)):
            iou_weight = sam3_ious[b, n].item()
            in_mask = (inst_flat[b] == n)  # (HW,) bool pixels in mask n

            if in_mask.sum() < 4:
                continue

            # Pull: embeddings within this mask should be similar
            mask_embs = embed_flat[b, in_mask]  # (M, E)
            mask_mean = mask_embs.mean(dim=0, keepdim=True)  # (1, E)
            pull = (mask_embs - mask_mean).pow(2).sum(dim=-1).mean()
            mask_losses.append(iou_weight * pull)

            # Push: embeddings in adjacent different-mask pixels should differ
            # Adjacent = pixels bordering mask n that belong to a different mask (not -1)
            mask_2d = sam3_masks[b, n] > 0.5  # (H, W) bool tensor
            mask_2d_np = mask_2d.cpu().numpy()
            dilated_np = ndimage.binary_dilation(mask_2d_np, iterations=1)
            border_np = dilated_np & (~mask_2d_np)
            border = torch.from_numpy(border_np).to(device).reshape(H * W)

            other_in_border = border & (inst_flat[b] >= 0) & (inst_flat[b] != n)
            if other_in_border.sum() < 2:
                continue

            other_embs = embed_flat[b, other_in_border]  # (K, E)
            n_other = min(other_embs.shape[0], 32)
            other_embs = other_embs[:n_other]
            other_mean = other_embs.mean(dim=0)

            dist = (mask_mean.squeeze(0) - other_mean).pow(2).sum().sqrt()
            push = F.relu(margin - dist).pow(2)
            mask_losses.append(iou_weight * push)

        if mask_losses:
            total_loss = total_loss + torch.stack(mask_losses).mean()
            count += 1

    if count > 0:
        total_loss = total_loss / count
    return total_loss


def thing_classification_loss(
    thing_logits: torch.Tensor,
    thing_target: torch.Tensor,
    background_weight: float = 0.05,
) -> torch.Tensor:
    """Background-aware thing CE from fused SAM3/Sobel pseudo-targets."""
    if thing_target.shape[-2:] != thing_logits.shape[-2:]:
        thing_target = F.interpolate(
            thing_target.unsqueeze(1).float(),
            size=thing_logits.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()

    valid_classes = int(thing_logits.shape[1])
    if valid_classes <= _THING_BACKGROUND_IDX:
        return thing_logits.new_zeros(1).squeeze()
    target = thing_target.clamp(min=0, max=valid_classes - 1)

    weights = thing_logits.new_ones(valid_classes)
    weights[_THING_BACKGROUND_IDX] = float(background_weight)
    return F.cross_entropy(thing_logits, target, weight=weights)


class AdaptiveInstanceLoss(nn.Module):
    """Combined loss for AdaptiveInstanceNet training.

    Loss hierarchy (descending priority):
      1. sam3_boundary: SAM3 instance edges → hard BCE on split_logit (strongest signal)
      2. split_distillation: depth gradient threshold → soft teacher for split_logit
      3. sam3_embedding: pull/push within/across SAM3 masks → instance_embed
      4. feat_boundary: DINOv2 feature discontinuity → split_logit alignment
      5. contrastive: depth+feature proxy → instance_embed (fallback when no SAM3)
      6. thing_cls: fused target thing/background classification
      7. embed_reg: variance regularization
    """

    def __init__(
        self,
        lambda_split: float = 1.0,
        lambda_feat_boundary: float = 0.5,
        lambda_contrastive: float = 0.3,
        lambda_embed_reg: float = 0.1,
        lambda_sam3_boundary: float = 2.0,
        lambda_sam3_embed: float = 0.5,
        lambda_thing: float = 1.0,
        thing_background_weight: float = 0.05,
        split_tau: float = 0.10,
        split_soft_margin: float = 0.03,
        feat_sigma: float = 0.3,
        contrastive_pairs: int = 2048,
        contrastive_margin: float = 0.5,
        sam3_embed_margin: float = 0.5,
        sam3_boundary_dice_weight: float = 0.5,
    ):
        super().__init__()
        self.lambda_split = lambda_split
        self.lambda_feat_boundary = lambda_feat_boundary
        self.lambda_contrastive = lambda_contrastive
        self.lambda_embed_reg = lambda_embed_reg
        self.lambda_sam3_boundary = lambda_sam3_boundary
        self.lambda_sam3_embed = lambda_sam3_embed
        self.lambda_thing = lambda_thing
        self.thing_background_weight = thing_background_weight
        self.split_tau = split_tau
        self.split_soft_margin = split_soft_margin
        self.feat_sigma = feat_sigma
        self.contrastive_pairs = contrastive_pairs
        self.contrastive_margin = contrastive_margin
        self.sam3_embed_margin = sam3_embed_margin
        self.sam3_boundary_dice_weight = sam3_boundary_dice_weight

    def forward(
        self,
        split_logit,
        instance_embed,
        dinov2_features,
        depth,
        depth_grads,
        grad_mag,
        sem_trainid,
        sam3_boundary: Optional[torch.Tensor] = None,
        sam3_boundary_weight: Optional[torch.Tensor] = None,
        sam3_instance_map: Optional[torch.Tensor] = None,
        sam3_masks: Optional[torch.Tensor] = None,
        sam3_ious: Optional[torch.Tensor] = None,
        thing_logits: Optional[torch.Tensor] = None,
        thing_target: Optional[torch.Tensor] = None,
    ):
        losses = {}

        # 1. SAM3 boundary supervision (hard BCE — strongest signal)
        if self.lambda_sam3_boundary > 0 and sam3_boundary is not None:
            l_sam3_bnd = sam3_boundary_loss(
                split_logit,
                sam3_boundary,
                boundary_weight=sam3_boundary_weight,
                dice_weight=self.sam3_boundary_dice_weight,
            )
            losses["sam3_boundary"] = l_sam3_bnd
        else:
            l_sam3_bnd = split_logit.new_zeros(1).squeeze()

        # 2. Split distillation from depth gradient (soft teacher)
        l_split = split_distillation_loss(
            split_logit, grad_mag,
            tau=self.split_tau, soft_margin=self.split_soft_margin,
        )
        losses["split"] = l_split

        # 3. SAM3 embedding consistency (pull within mask, push across adjacent masks)
        if (self.lambda_sam3_embed > 0 and sam3_instance_map is not None
                and sam3_masks is not None and sam3_ious is not None):
            l_sam3_emb = sam3_embedding_consistency_loss(
                instance_embed, sam3_instance_map, sam3_masks, sam3_ious,
                margin=self.sam3_embed_margin,
            )
            losses["sam3_embed"] = l_sam3_emb
        else:
            l_sam3_emb = instance_embed.new_zeros(1).squeeze()

        # 4. Feature-guided boundary alignment
        if self.lambda_feat_boundary > 0:
            l_feat = feature_boundary_loss(
                split_logit, dinov2_features, sigma=self.feat_sigma)
            losses["feat_boundary"] = l_feat
        else:
            l_feat = 0.0

        # 5. Contrastive embedding loss (depth+feature proxy — fallback)
        if self.lambda_contrastive > 0:
            l_contrast = contrastive_embedding_loss(
                instance_embed, dinov2_features, depth, sem_trainid,
                num_pairs=self.contrastive_pairs,
                margin=self.contrastive_margin,
            )
            losses["contrastive"] = l_contrast
        else:
            l_contrast = 0.0

        # 6. Embedding regularization
        if self.lambda_embed_reg > 0:
            l_reg = embedding_regularization_loss(instance_embed)
            losses["embed_reg"] = l_reg
        else:
            l_reg = 0.0

        # 7. Thing/background classification from fused pseudo-targets.
        if self.lambda_thing > 0 and thing_logits is not None and thing_target is not None:
            l_thing = thing_classification_loss(
                thing_logits,
                thing_target,
                background_weight=self.thing_background_weight,
            )
            losses["thing"] = l_thing
        else:
            l_thing = split_logit.new_zeros(1).squeeze()

        total = (self.lambda_sam3_boundary * l_sam3_bnd
                 + self.lambda_split * l_split
                 + self.lambda_sam3_embed * l_sam3_emb
                 + self.lambda_feat_boundary * l_feat
                 + self.lambda_contrastive * l_contrast
                 + self.lambda_embed_reg * l_reg
                 + self.lambda_thing * l_thing)
        losses["total"] = total
        return total, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pq_things(model, val_loader, device, cityscapes_root,
                       split_threshold=0.5, min_area=100,
                       base_tau=0.05, depth_subdir="depth_spidepth",
                       edge_mode="hybrid_or", max_batches: Optional[int] = None,
                       sam3_subdir="sam_fine_masks_sam3",
                       sam3_mask_threshold=0.25,
                       use_sam3_proposals=True,
                       thing_conf_threshold=0.35):
    """Evaluate PQ_things using hybrid approach: model-predicted confidence
    modulates native-resolution depth gradient threshold.

    The model outputs split_prob at 32x64 which is upsampled to native
    resolution as a spatially-varying weight. Hybrid edge detection uses
    native-resolution depth gradients with an adaptive threshold:
        adaptive_tau = base_tau * (1.5 - split_prob)
    Where split_prob is high → lower threshold → more splits.
    Where split_prob is low → higher threshold → fewer splits.

    In hybrid_or mode, direct learned edges are unioned with adaptive depth
    edges so SAM3-trained boundaries can split co-planar objects even when the
    depth gradient is weak.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import sobel as scipy_sobel

    gt_dir = os.path.join(cityscapes_root, "gtFine", "val")
    H, W = EVAL_H, EVAL_W

    tp = np.zeros(19)
    fp = np.zeros(19)
    fn = np.zeros(19)
    iou_sum = np.zeros(19)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(
                tqdm(val_loader, desc="Eval PQ_things", ncols=100, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            fusion_inputs = batch.get("fusion_inputs")
            outputs = model(
                batch["dinov2_features"].to(device),
                batch["depth"].to(device),
                batch["depth_grads"].to(device),
                batch["cause_logits"].to(device),
                fusion_inputs=(fusion_inputs.to(device)
                               if fusion_inputs is not None else None),
            )
            split_logit, embed, thing_logits = _unpack_model_outputs(outputs)
            split_np = torch.sigmoid(split_logit).squeeze(1).cpu().numpy()
            thing_conf_np = thing_idx_np = None
            if thing_logits is not None:
                thing_prob = F.softmax(thing_logits, dim=1)
                if thing_prob.shape[-2:] != (H, W):
                    thing_prob = F.interpolate(
                        thing_prob, size=(H, W),
                        mode="bilinear", align_corners=False,
                    )
                thing_prob_np = thing_prob.cpu().numpy()
                thing_conf_np = thing_prob_np.max(axis=1)
                thing_idx_np = thing_prob_np.argmax(axis=1)
            sem_batch = batch.get("sem_trainid_fusion", batch["sem_trainid"])
            sem_tid = sem_batch.cpu().numpy()

            for i in range(split_np.shape[0]):
                city, stem = batch["city"][i], batch["stem"][i]

                # Load native-resolution depth and compute gradients
                depth_path = os.path.join(
                    cityscapes_root, depth_subdir, "val", city, f"{stem}.npy")
                if os.path.exists(depth_path):
                    depth_native = np.load(depth_path)
                    if depth_native.shape != (H, W):
                        depth_native = np.array(
                            Image.fromarray(depth_native.astype(np.float32)).resize(
                                (W, H), Image.BILINEAR))
                    depth_smooth = gaussian_filter(depth_native.astype(np.float64), sigma=1.0)
                    gx = scipy_sobel(depth_smooth, axis=1)
                    gy = scipy_sobel(depth_smooth, axis=0)
                    grad_mag_native = np.sqrt(gx ** 2 + gy ** 2)
                else:
                    # Fallback: use upsampled patch-level gradients
                    grad_mag_native = None

                # Upsample model split_prob to native resolution (bilinear for smooth modulation)
                split_full = np.array(
                    Image.fromarray(split_np[i].astype(np.float32)).resize(
                        (W, H), Image.BILINEAR)
                )
                sem_full = np.array(
                    Image.fromarray(sem_tid[i].astype(np.uint8)).resize(
                        (W, H), Image.NEAREST)
                )
                thing_trainid_full = thing_conf_full = None
                if thing_idx_np is not None:
                    safe_idx = np.clip(
                        thing_idx_np[i],
                        0,
                        len(_THING_HEAD_IDX_TO_TRAINID) - 1,
                    )
                    thing_trainid_full = _THING_HEAD_IDX_TO_TRAINID[safe_idx]
                    thing_conf_full = thing_conf_np[i]

                model_edge = split_full > split_threshold
                if grad_mag_native is not None:
                    # Hybrid: adaptive threshold on native gradients
                    # split_prob high → tau_local low → more edges
                    tau_local = base_tau * (1.5 - split_full)
                    tau_local = np.clip(tau_local, 0.02, 0.25)
                    depth_edge = grad_mag_native > tau_local
                    if edge_mode == "hybrid":
                        edge_map = depth_edge
                    elif edge_mode == "direct":
                        edge_map = model_edge
                    elif edge_mode == "hybrid_or":
                        edge_map = depth_edge | model_edge
                    else:
                        raise ValueError(f"Unknown edge_mode: {edge_mode}")
                else:
                    # Fallback: direct threshold on model output
                    edge_map = model_edge
                pred_pan = np.zeros((H, W), dtype=np.int32)
                pred_segments = {}
                nxt = 1

                # Stuff segments
                for cls in _STUFF_IDS:
                    mask = sem_full == cls
                    if mask.sum() < 64:
                        continue
                    pred_pan[mask] = nxt
                    pred_segments[nxt] = cls
                    nxt += 1

                thing_assigned = np.zeros((H, W), dtype=bool)

                if use_sam3_proposals:
                    proposals = _load_sam3_proposals(
                        cityscapes_root, sam3_subdir, "val", city, stem,
                        out_hw=(H, W),
                        mask_threshold=sam3_mask_threshold,
                        min_area=min_area,
                    )
                    for prop_mask, cls, _score, _area in proposals:
                        final = prop_mask & (~thing_assigned)
                        if final.sum() < min_area:
                            continue
                        pred_pan[final] = nxt
                        pred_segments[nxt] = cls
                        thing_assigned |= final
                        nxt += 1

                # Thing fallback: CC with adaptive edge removal on uncovered semantic pixels
                for cls in _THING_IDS:
                    if thing_trainid_full is not None:
                        cls_mask = (
                            (thing_trainid_full == cls)
                            & (thing_conf_full >= thing_conf_threshold)
                            & (~thing_assigned)
                        )
                        if cls_mask.sum() < min_area:
                            cls_mask = (sem_full == cls) & (~thing_assigned)
                    else:
                        cls_mask = (sem_full == cls) & (~thing_assigned)
                    if cls_mask.sum() < min_area:
                        continue
                    split_mask = cls_mask & (~edge_map)
                    labeled, n_cc = ndimage.label(split_mask)
                    for cc_id in range(1, n_cc + 1):
                        cc_mask = labeled == cc_id
                        if cc_mask.sum() < min_area:
                            continue
                        # Dilation to reclaim boundary pixels
                        dilated = ndimage.binary_dilation(cc_mask, iterations=3)
                        final = (dilated & cls_mask & (pred_pan == 0)) | cc_mask
                        if final.sum() < min_area:
                            continue
                        pred_pan[final] = nxt
                        pred_segments[nxt] = cls
                        nxt += 1

                # Load GT (resize to eval resolution)
                gt_label_path = os.path.join(
                    gt_dir, city, f"{stem}_gtFine_labelIds.png")
                gt_inst_path = os.path.join(
                    gt_dir, city, f"{stem}_gtFine_instanceIds.png")
                if not os.path.exists(gt_label_path):
                    continue

                gt_raw = np.array(Image.open(gt_label_path))
                gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
                for raw_id, tid in _CS_ID_TO_TRAIN.items():
                    gt_sem[gt_raw == raw_id] = tid
                if gt_sem.shape != (H, W):
                    gt_sem = np.array(
                        Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

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

                if os.path.exists(gt_inst_path):
                    gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
                    if gt_inst.shape != (H, W):
                        gt_inst = np.array(
                            Image.fromarray(gt_inst).resize(
                                (W, H), Image.NEAREST))
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
                for cat in range(19):
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

    # Compute PQ
    all_pq, stuff_pq, thing_pq = [], [], []
    per_class = {}
    for c in range(19):
        t, f_p, f_n, s = tp[c], fp[c], fn[c], iou_sum[c]
        if t + f_p + f_n > 0:
            sq = s / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0
        per_class[_CS_CLASS_NAMES[c]] = {
            "PQ": round(pq * 100, 2), "TP": int(t),
            "FP": int(f_p), "FN": int(f_n),
        }
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            (stuff_pq if c in _STUFF_IDS else thing_pq).append(pq)

    pq_all = float(np.mean(all_pq)) * 100 if all_pq else 0.0
    pq_stuff = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
    pq_things = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0

    model.train()
    return {
        "PQ": round(pq_all, 2),
        "PQ_stuff": round(pq_stuff, 2),
        "PQ_things": round(pq_things, 2),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if args.thing_classes not in (0, len(_THING_HEAD_IDX_TO_TRAINID)):
        raise ValueError(
            f"--thing_classes must be 0 or {len(_THING_HEAD_IDX_TO_TRAINID)} "
            "(background + 8 Cityscapes thing classes)"
        )

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

    semantic_spec = infer_semantic_spec(
        args.cityscapes_root,
        args.semantic_subdir,
        semantic_mode=args.semantic_mode,
        centroids_path=args.centroids_path,
        num_semantic_classes=args.num_semantic_classes,
        split="train",
    )
    log.info(
        "Semantic mode: %s | semantic_dim=%d | centroids=%s",
        semantic_spec.mode, semantic_spec.num_classes,
        semantic_spec.centroids_path or "none",
    )
    log.info("Depth subdir: %s", args.depth_subdir)
    log.info(
        "High-res fusion: %s | fusion_hw=%dx%d | fusion_channels=%d",
        args.use_highres_fusion, args.fusion_h, args.fusion_w, FUSION_CHANNELS,
    )
    log.info("SAM3 fusion dropout: %.2f", args.sam3_fusion_dropout)
    sam3_coverage = {}
    for split in ("train", "val"):
        n_img = _count_cityscapes_images(args.cityscapes_root, split)
        n_sam = _count_sam3_files(args.cityscapes_root, args.sam3_subdir, split)
        coverage = 100.0 * n_sam / max(n_img, 1)
        sam3_coverage[split] = coverage
        log.info(
            "SAM3 coverage %s: %d/%d images (%.1f%%)",
            split, n_sam, n_img, coverage,
        )
    if args.use_highres_fusion and args.sam3_fusion_dropout < 1.0 and sam3_coverage["val"] < 50.0:
        log.warning(
            "Validation SAM3 coverage is low; set --sam3_fusion_dropout 1.0 "
            "to avoid training on SAM3 input channels that are absent at eval time."
        )
    if args.use_sam3_proposals and not args.no_sam3_proposals and sam3_coverage["val"] < 50.0:
        log.warning(
            "SAM3 proposal eval will be sparse because val SAM3 coverage is %.1f%%.",
            sam3_coverage["val"],
        )
    if not args.use_sam3_proposals:
        log.info("Validation uses checkpoint predictions only; SAM3 proposals are disabled.")
    for split in ("train", "val"):
        stats = validate_semantic_inputs(
            args.cityscapes_root,
            args.semantic_subdir,
            args.depth_subdir,
            semantic_spec,
            split=split,
            max_samples=args.diagnostic_samples,
        )
        log.info(
            "Semantic diagnostics %s: labels=%d..%d valid_trainID=%.1f%% "
            "thing_trainID=%.1f%% samples=%d",
            split, stats["label_min"], stats["label_max"],
            100.0 * stats["valid_trainid_frac"],
            100.0 * stats["thing_trainid_frac"],
            stats["samples"],
        )

    # Model
    model = AdaptiveInstanceNet(
        feature_dim=768,
        depth_channels=3,
        semantic_dim=semantic_spec.num_classes,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        use_highres_fusion=args.use_highres_fusion,
        fusion_channels=FUSION_CHANNELS,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_blocks=args.fusion_blocks,
        thing_classes=args.thing_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"AdaptiveInstanceNet: {total_params:,} params")

    # Datasets
    train_dataset = InstanceDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        depth_subdir=args.depth_subdir,
        sam3_subdir=args.sam3_subdir,
        load_sam3=args.load_sam3,
        semantic_spec=semantic_spec,
        use_highres_fusion=args.use_highres_fusion,
        fusion_h=args.fusion_h,
        fusion_w=args.fusion_w,
        sam3_mask_threshold=args.sam3_mask_threshold,
        sam3_boundary_small_weight=args.sam3_boundary_small_weight,
        sam3_fusion_dropout=args.sam3_fusion_dropout,
        use_fused_targets=not args.no_fused_targets,
        fused_sobel_tau=args.fused_sobel_tau,
        fused_min_area=args.fused_min_area,
    )
    val_dataset = InstanceDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        depth_subdir=args.depth_subdir,
        sam3_subdir=args.sam3_subdir,
        load_sam3=False,
        semantic_spec=semantic_spec,
        use_highres_fusion=args.use_highres_fusion,
        fusion_h=args.fusion_h,
        fusion_w=args.fusion_w,
        sam3_mask_threshold=args.sam3_mask_threshold,
        sam3_boundary_small_weight=args.sam3_boundary_small_weight,
        sam3_fusion_dropout=args.sam3_fusion_dropout,
        use_fused_targets=not args.no_fused_targets,
        fused_sobel_tau=args.fused_sobel_tau,
        fused_min_area=args.fused_min_area,
    )

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    # Loss
    loss_fn = AdaptiveInstanceLoss(
        lambda_split=args.lambda_split,
        lambda_feat_boundary=args.lambda_feat_boundary,
        lambda_contrastive=args.lambda_contrastive,
        lambda_embed_reg=args.lambda_embed_reg,
        lambda_sam3_boundary=args.lambda_sam3_boundary,
        lambda_sam3_embed=args.lambda_sam3_embed,
        lambda_thing=args.lambda_thing,
        thing_background_weight=args.thing_background_weight,
        split_tau=args.split_tau,
        contrastive_pairs=args.contrastive_pairs,
        sam3_boundary_dice_weight=args.sam3_boundary_dice_weight,
    )

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args).copy()
    config.update(semantic_spec.to_config())
    config["total_params"] = total_params
    config["fusion_channels"] = FUSION_CHANNELS
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    use_amp = False
    amp_dtype = torch.float32
    amp_device = device.type

    # Training
    best_pq_things = 0.0
    best_loss = float("inf")
    log.info(f"Training for {args.num_epochs} epochs, "
             f"{len(train_loader)} batches/epoch")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "split": 0, "feat_boundary": 0,
                        "contrastive": 0, "embed_reg": 0,
                        "sam3_boundary": 0, "sam3_embed": 0,
                        "thing": 0}
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    ncols=130, leave=True)
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            dinov2 = batch["dinov2_features"].to(device)
            depth = batch["depth"].to(device)
            depth_grads = batch["depth_grads"].to(device)
            cause = batch["cause_logits"].to(device)
            grad_mag = batch["grad_mag"].to(device)
            sem_tid = batch["sem_trainid"].to(device)
            sam3_bnd = batch["sam3_boundary"].to(device)
            sam3_bnd_weight = batch["sam3_boundary_weight"].to(device)
            sam3_imap = batch["sam3_instance_map"].to(device)
            sam3_masks_t = batch["sam3_masks"].to(device)
            sam3_ious_t = batch["sam3_ious"].to(device)
            thing_target = batch["thing_target"].to(device)
            fusion_inputs = (batch["fusion_inputs"].to(device)
                             if args.use_highres_fusion else None)
            if args.use_highres_fusion:
                grad_mag = batch["grad_mag_fusion"].to(device)
                sem_tid = batch["sem_trainid_fusion"].to(device)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype,
                                enabled=use_amp):
                outputs = model(
                    dinov2, depth, depth_grads, cause,
                    fusion_inputs=fusion_inputs)
                split_logit, embed, thing_logits = _unpack_model_outputs(outputs)
                total_loss, loss_dict = loss_fn(
                    split_logit, embed, dinov2, depth, depth_grads, grad_mag,
                    sem_tid,
                    sam3_boundary=sam3_bnd,
                    sam3_boundary_weight=sam3_bnd_weight,
                    sam3_instance_map=sam3_imap,
                    sam3_masks=sam3_masks_t,
                    sam3_ious=sam3_ious_t,
                    thing_logits=thing_logits,
                    thing_target=thing_target,
                )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += (v.item() if isinstance(v, torch.Tensor)
                                        else float(v))
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                s3bnd=f"{epoch_losses['sam3_boundary']/num_batches:.4f}",
                split=f"{epoch_losses['split']/num_batches:.4f}",
                s3emb=f"{epoch_losses['sam3_embed']/num_batches:.4f}",
                thing=f"{epoch_losses['thing']/num_batches:.4f}",
                contr=f"{epoch_losses['contrastive']/num_batches:.4f}",
            )

        pbar.close()
        scheduler.step()

        dt = time.time() - t0
        avg = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch}/{args.num_epochs} ({dt:.0f}s) | "
            f"loss={avg['total']:.4f} "
            f"s3bnd={avg['sam3_boundary']:.4f} split={avg['split']:.4f} "
            f"s3emb={avg['sam3_embed']:.4f} "
            f"thing={avg['thing']:.4f} "
            f"feat={avg['feat_boundary']:.4f} contr={avg['contrastive']:.4f} "
            f"reg={avg['embed_reg']:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            best_loss_path = os.path.join(args.output_dir, "best_loss.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg,
                "config": config,
            }, best_loss_path)
            log.info(
                f"  New best train loss: {best_loss:.4f} "
                f"(saved to best_loss.pth)"
            )

        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            log.info(f"Evaluating at epoch {epoch}...")
            metrics = evaluate_pq_things(
                model, val_loader, device, args.cityscapes_root,
                split_threshold=args.split_threshold,
                min_area=args.min_area,
                base_tau=args.base_tau,
                depth_subdir=args.depth_subdir,
                edge_mode=args.edge_mode,
                max_batches=args.max_eval_batches,
                sam3_subdir=args.sam3_subdir,
                sam3_mask_threshold=args.sam3_mask_threshold,
                use_sam3_proposals=(
                    args.load_sam3
                    and args.use_sam3_proposals
                    and not args.no_sam3_proposals
                ),
                thing_conf_threshold=args.thing_conf_threshold,
            )
            log.info(
                f"  PQ={metrics['PQ']:.2f} | "
                f"PQ_stuff={metrics['PQ_stuff']:.2f} | "
                f"PQ_things={metrics['PQ_things']:.2f}"
            )

            # Per-class thing details
            for cls_name in ["person", "rider", "car", "truck",
                             "bus", "train", "motorcycle", "bicycle"]:
                d = metrics["per_class"].get(cls_name, {})
                log.info(
                    f"    {cls_name:12s}: PQ={d.get('PQ',0):5.1f} "
                    f"TP={d.get('TP',0):4d} FP={d.get('FP',0):4d} "
                    f"FN={d.get('FN',0):4d}"
                )

            # Save checkpoint
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config,
            }, ckpt_path)

            if metrics["PQ_things"] > best_pq_things:
                best_pq_things = metrics["PQ_things"]
                best_path = os.path.join(args.output_dir, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config": config,
                }, best_path)
                log.info(f"  New best PQ_things: {best_pq_things:.2f}% "
                         f"(saved to best.pth)")

            # Metrics history
            metrics["epoch"] = epoch
            metrics_path = os.path.join(args.output_dir, "metrics_history.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    log.info(f"Training complete. Best PQ_things: {best_pq_things:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AdaptiveInstanceNet for instance boundary prediction")

    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause_crf")
    parser.add_argument("--semantic_mode", type=str, default="auto",
                        choices=["auto", "cluster", "cause27", "trainid"],
                        help="How to interpret semantic PNG values")
    parser.add_argument("--centroids_path", type=str, default=None,
                        help="kmeans_centroids.npz for raw cluster semantics")
    parser.add_argument("--num_semantic_classes", type=int, default=None,
                        help="Override semantic input channels; inferred by default")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/adaptive_instance")

    # SAM3 fusion
    parser.add_argument("--sam3_subdir", type=str, default="sam_fine_masks_sam3",
                        help="Subdir under cityscapes_root containing SAM3 NPZ masks")
    parser.add_argument("--no_sam3", action="store_true",
                        help="Disable SAM3 mask loading (ablation)")
    parser.add_argument("--lambda_sam3_boundary", type=float, default=2.0,
                        help="Weight for SAM3-supervised boundary BCE loss")
    parser.add_argument("--lambda_sam3_embed", type=float, default=0.5,
                        help="Weight for SAM3 mask embedding consistency loss")
    parser.add_argument("--sam3_boundary_dice_weight", type=float, default=0.5,
                        help="Dice term mixed into SAM3 boundary loss")
    parser.add_argument("--sam3_mask_threshold", type=float, default=0.25,
                        help="Threshold for resized SAM3 masks")
    parser.add_argument("--sam3_boundary_small_weight", type=float, default=4.0,
                        help="Extra boundary weight for small SAM3 masks")
    parser.add_argument("--no_sam3_proposals", action="store_true",
                        help="Do not use SAM3 masks as eval-time thing proposals")
    parser.add_argument("--use_sam3_proposals", action="store_true",
                        help="Use SAM3 val masks as eval-time thing proposals (diagnostic only)")
    parser.add_argument("--sam3_fusion_dropout", type=float, default=0.0,
                        help="Probability of zeroing SAM3 fusion-input channels")
    parser.add_argument("--no_fused_targets", action="store_true",
                        help="Train directly on SAM3 targets instead of SAM3+Sobel/cluster fusion")
    parser.add_argument("--fused_sobel_tau", type=float, default=0.10,
                        help="Sobel gradient threshold for fused pseudo-target components")
    parser.add_argument("--fused_min_area", type=int, default=16,
                        help="Minimum target component area at training resolution")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--use_highres_fusion", action="store_true",
                        help="Fuse high-resolution DepthPro Sobel and SAM3 priors")
    parser.add_argument("--fusion_h", type=int, default=FUSION_H)
    parser.add_argument("--fusion_w", type=int, default=FUSION_W)
    parser.add_argument("--fusion_hidden_dim", type=int, default=64)
    parser.add_argument("--fusion_blocks", type=int, default=2)
    parser.add_argument("--thing_classes", type=int,
                        default=len(_THING_HEAD_IDX_TO_TRAINID),
                        help="0 disables thing head; 9 means background + 8 Cityscapes things")

    # Training
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--diagnostic_samples", type=int, default=32,
                        help="Samples per split for semantic/depth validation")
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Optional smoke-test cap on train batches per epoch")
    parser.add_argument("--max_eval_batches", type=int, default=None,
                        help="Optional smoke-test cap on eval batches")

    # Loss weights
    parser.add_argument("--lambda_split", type=float, default=1.0)
    parser.add_argument("--lambda_feat_boundary", type=float, default=0.5)
    parser.add_argument("--lambda_contrastive", type=float, default=0.3)
    parser.add_argument("--lambda_embed_reg", type=float, default=0.1)
    parser.add_argument("--lambda_thing", type=float, default=1.0)
    parser.add_argument("--thing_background_weight", type=float, default=0.05,
                        help="CE class weight for background in thing head")
    parser.add_argument("--split_tau", type=float, default=0.10,
                        help="Depth gradient threshold for teacher signal")
    parser.add_argument("--contrastive_pairs", type=int, default=2048)

    # Evaluation
    parser.add_argument("--split_threshold", type=float, default=0.5,
                        help="Threshold on split_prob for instance boundaries")
    parser.add_argument("--min_area", type=int, default=100,
                        help="Minimum thing proposal/component area for eval")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth",
                        help="Subdirectory for native-resolution depth maps")
    parser.add_argument("--base_tau", type=float, default=0.05,
                        help="Base depth gradient threshold for hybrid eval")
    parser.add_argument("--edge_mode", type=str, default="hybrid_or",
                        choices=["hybrid_or", "hybrid", "direct"],
                        help="How learned boundaries combine with depth edges")
    parser.add_argument("--thing_conf_threshold", type=float, default=0.35,
                        help="Confidence threshold for checkpoint-predicted thing masks")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.load_sam3 = not args.no_sam3
    train(args)
