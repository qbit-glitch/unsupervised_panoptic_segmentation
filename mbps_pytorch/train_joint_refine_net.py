#!/usr/bin/env python3
"""Train JointRefineNet: Unified Semantic + Instance Pseudo-Label Refinement.

Trains the joint refiner using:
  Semantic losses (reused from CSCMRefineNet):
    1. Cross-entropy distillation from CAUSE pseudo-labels
    2. Depth-boundary alignment
    3. Feature-prototype consistency
    4. Entropy minimization
  Boundary losses:
    5. BCE from depth-gradient teacher
    6. BCE from instance boundary map
  Embedding loss:
    7. Discriminative embedding loss from pseudo-instance IDs

Usage:
    python mbps_pytorch/train_joint_refine_net.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir checkpoints/joint_refine_net \
        --logits_subdir pseudo_semantic_cause_logits \
        --instance_subdir pseudo_instance_spidepth \
        --num_epochs 30 --batch_size 4 --device auto
"""

import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbps_pytorch.joint_refine_net import JointRefineNet
from mbps_pytorch.train_refine_net import (
    PseudoLabelDataset,
    apply_augmentation,
    depth_boundary_alignment_loss,
    entropy_loss,
    feature_prototype_loss,
)
from mbps_pytorch.losses.instance_embedding_loss import discriminative_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PATCH_H, PATCH_W = 32, 64
NUM_CLASSES = 27


# ---------------------------------------------------------------------------
# Dataset (extends PseudoLabelDataset with instance data)
# ---------------------------------------------------------------------------

class JointPseudoLabelDataset(PseudoLabelDataset):
    """Extends PseudoLabelDataset with instance boundary maps and ID maps."""

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        semantic_subdir: str = "pseudo_semantic_cause_crf",
        feature_subdir: str = "dinov2_features",
        depth_subdir: str = "depth_spidepth",
        logits_subdir: str = None,
        instance_subdir: str = "pseudo_instance_spidepth",
    ):
        super().__init__(
            cityscapes_root, split, semantic_subdir,
            feature_subdir, depth_subdir, logits_subdir,
        )
        self.instance_subdir = instance_subdir

    def __getitem__(self, idx):
        # Get base data from parent
        data = super().__getitem__(idx)
        stem, city = data["stem"], data["city"]

        # Load instance data
        inst_boundary, inst_ids = self._load_instance_data(city, stem)
        data["instance_boundary"] = inst_boundary
        data["instance_ids"] = inst_ids
        return data

    def _load_instance_data(self, city, stem):
        """Load SPIdepth instances → boundary map + ID map at 32×64."""
        inst_path = os.path.join(
            self.root, self.instance_subdir, self.split, city,
            f"{stem}.npz",
        )

        if not os.path.exists(inst_path):
            return (
                torch.zeros(1, PATCH_H, PATCH_W, dtype=torch.float32),
                torch.zeros(PATCH_H, PATCH_W, dtype=torch.long),
            )

        data = np.load(inst_path)
        num_valid = int(data["num_valid"])
        if num_valid == 0:
            return (
                torch.zeros(1, PATCH_H, PATCH_W, dtype=torch.float32),
                torch.zeros(PATCH_H, PATCH_W, dtype=torch.long),
            )

        masks = data["masks"][:num_valid]  # (M, H*W) bool
        hp, wp = int(data["h_patches"]), int(data["w_patches"])

        # Build full-res instance ID map
        id_map_full = np.zeros((hp, wp), dtype=np.int32)
        for i in range(num_valid):
            mask_2d = masks[i].reshape(hp, wp)
            id_map_full[mask_2d] = i + 1  # 1-indexed

        # Downsample to patch resolution via nearest neighbor
        id_map_patch = np.array(
            Image.fromarray(id_map_full.astype(np.int32)).resize(
                (PATCH_W, PATCH_H), Image.NEAREST
            )
        )  # (32, 64)

        # Compute boundary: pixels where any neighbor has a different instance ID
        boundary = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
        # Vertical neighbors
        diff_v = id_map_patch[:-1, :] != id_map_patch[1:, :]
        boundary[:-1, :] = np.maximum(boundary[:-1, :], diff_v.astype(np.float32))
        boundary[1:, :] = np.maximum(boundary[1:, :], diff_v.astype(np.float32))
        # Horizontal neighbors
        diff_h = id_map_patch[:, :-1] != id_map_patch[:, 1:]
        boundary[:, :-1] = np.maximum(boundary[:, :-1], diff_h.astype(np.float32))
        boundary[:, 1:] = np.maximum(boundary[:, 1:], diff_h.astype(np.float32))

        # Don't count transitions to/from background (id=0) as instance boundaries
        bg_mask = id_map_patch == 0
        boundary[bg_mask] = 0.0

        return (
            torch.from_numpy(boundary).unsqueeze(0).float(),  # (1, 32, 64)
            torch.from_numpy(id_map_patch.astype(np.int64)),  # (32, 64)
        )


# ---------------------------------------------------------------------------
# Augmentation (extends base to handle instance fields)
# ---------------------------------------------------------------------------

def apply_joint_augmentation(batch, mode="weak"):
    """Apply augmentation to a joint batch, including instance fields."""
    aug = apply_augmentation(batch, mode)

    # Also flip instance fields
    if mode in ("weak", "strong"):
        B = aug["cause_logits"].shape[0]
        for i in range(B):
            # Check if this sample was flipped by examining if features differ
            # Instead, we just apply the same random flip logic
            pass

    # Note: base apply_augmentation uses per-sample random flip that we can't
    # retroactively apply to instance fields. For simplicity, we'll handle
    # augmentation in the training loop with explicit flip tracking.
    return aug


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def boundary_distillation_loss(boundary_logits, depth_grads, sigma=0.15):
    """BCE loss: learn to predict depth-gradient edges.

    Teacher: soft boundary probability from depth gradient magnitude.
    Args:
        boundary_logits: (B, 1, H, W) raw logits
        depth_grads: (B, 2, H, W) Sobel_x, Sobel_y
        sigma: sharpness of the soft target (smaller = harder edges)
    """
    grad_mag = torch.sqrt(
        depth_grads[:, 0:1] ** 2 + depth_grads[:, 1:2] ** 2 + 1e-8
    )
    # Normalize per-image so threshold is relative
    grad_mean = grad_mag.mean(dim=(-2, -1), keepdim=True)
    grad_norm = grad_mag / (grad_mean + 1e-6)
    # Soft target: ~1 where gradient is above average, ~0 below
    boundary_target = torch.sigmoid((grad_norm - 1.5) / sigma)
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)


def instance_boundary_loss(boundary_logits, instance_boundary_map):
    """BCE loss against binary instance boundary map.

    Args:
        boundary_logits: (B, 1, H, W) raw logits
        instance_boundary_map: (B, 1, H, W) float, 1 at boundaries, 0 elsewhere
    """
    # Handle class imbalance: boundaries are rare (~5% of pixels)
    pos_weight = (1.0 - instance_boundary_map).sum() / (
        instance_boundary_map.sum() + 1.0
    )
    pos_weight = pos_weight.clamp(max=20.0)  # cap to prevent explosion
    return F.binary_cross_entropy_with_logits(
        boundary_logits, instance_boundary_map,
        pos_weight=pos_weight,
    )


def embedding_distillation_loss(embeddings, pseudo_instance_ids,
                                delta_v=0.5, delta_d=1.5):
    """Discriminative embedding loss using pseudo-instance IDs.

    Args:
        embeddings: (B, D, H, W) L2-normalized
        pseudo_instance_ids: (B, H, W) long, 0=background
    """
    B, D, H, W = embeddings.shape
    emb_flat = embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
    ids_flat = pseudo_instance_ids.reshape(B, H * W)
    result = discriminative_loss(
        emb_flat, ids_flat, delta_v=delta_v, delta_d=delta_d,
    )
    return result["total"]


class JointRefineNetLoss(nn.Module):
    """Combined loss for JointRefineNet.

    Semantic losses (proven weights from CSCMRefineNet experiments):
      - distillation: CE from CAUSE labels (floor=0.5 via cosine warmdown)
      - align: depth-boundary alignment (0.5)
      - proto: feature-prototype consistency (0.05)
      - ent: entropy minimization (0.05)

    Boundary losses:
      - bnd_depth: BCE from depth gradient teacher (0.3)
      - bnd_inst: BCE from instance boundary map (0.3)

    Embedding losses:
      - emb_disc: discriminative loss from pseudo-instances (0.3)
    """

    def __init__(
        self,
        # Semantic
        lambda_distill: float = 1.0,
        lambda_distill_min: float = 0.5,
        lambda_align: float = 0.5,
        lambda_proto: float = 0.05,
        lambda_ent: float = 0.05,
        label_smoothing: float = 0.1,
        # Boundary
        lambda_bnd_depth: float = 0.3,
        lambda_bnd_inst: float = 0.3,
        # Embedding
        lambda_emb_disc: float = 0.3,
        # Instance loss warmup: let semantic losses train the backbone first
        instance_warmup_epochs: int = 5,
    ):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.lambda_distill_min = lambda_distill_min
        self.lambda_align = lambda_align
        self.lambda_proto = lambda_proto
        self.lambda_ent = lambda_ent
        self.label_smoothing = label_smoothing
        self.lambda_bnd_depth = lambda_bnd_depth
        self.lambda_bnd_inst = lambda_bnd_inst
        self.lambda_emb_disc = lambda_emb_disc
        self.instance_warmup_epochs = instance_warmup_epochs
        self._distill_scale = 1.0
        self._inst_scale = 0.0  # instance loss warmup scale

    def set_epoch(self, epoch: int, total_epochs: int):
        """Cosine warmdown for distillation (with floor).
        Linear warmup for instance losses (boundary + embedding).
        """
        progress = (epoch - 1) / max(total_epochs - 1, 1)
        hi, lo = self.lambda_distill, self.lambda_distill_min
        self._distill_scale = lo + 0.5 * (hi - lo) * (
            1 + math.cos(math.pi * progress)
        )
        # Instance loss warmup: 0 for first warmup_epochs, then linear to 1.0
        if epoch <= self.instance_warmup_epochs:
            self._inst_scale = 0.0
        else:
            self._inst_scale = min(1.0, (epoch - self.instance_warmup_epochs) /
                                   max(self.instance_warmup_epochs, 1))

    def forward(self, outputs, cause_logits, dinov2_features, depth,
                depth_grads, instance_boundary, instance_ids):
        """
        Args:
            outputs: dict from JointRefineNet.forward()
            cause_logits: (B, 27, H, W) CAUSE log-probabilities
            dinov2_features: (B, 768, H, W)
            depth: (B, 1, H, W)
            depth_grads: (B, 2, H, W)
            instance_boundary: (B, 1, H, W) float
            instance_ids: (B, H, W) long
        Returns:
            total_loss, loss_dict
        """
        sem_logits = outputs["semantic_logits"]
        bnd_logits = outputs["boundary_logits"]
        embeddings = outputs["embeddings"]
        losses = {}
        eff_distill = self._distill_scale

        # === Semantic losses ===
        if eff_distill > 0:
            cause_labels = cause_logits.argmax(dim=1).detach()
            l_distill = F.cross_entropy(
                sem_logits, cause_labels,
                label_smoothing=self.label_smoothing,
            )
            losses["distill"] = l_distill
        else:
            l_distill = 0.0

        l_align = depth_boundary_alignment_loss(sem_logits, depth)
        losses["align"] = l_align

        l_proto = feature_prototype_loss(sem_logits, dinov2_features)
        losses["proto"] = l_proto

        l_ent = entropy_loss(sem_logits)
        losses["entropy"] = l_ent

        # === Boundary losses (with warmup) ===
        inst_w = self._inst_scale
        zero = torch.tensor(0.0, device=sem_logits.device)

        if inst_w > 0:
            l_bnd_depth = boundary_distillation_loss(bnd_logits, depth_grads)
            losses["bnd_depth"] = l_bnd_depth

            if instance_boundary.sum() > 0:
                l_bnd_inst = instance_boundary_loss(bnd_logits, instance_boundary)
            else:
                l_bnd_inst = zero
            losses["bnd_inst"] = l_bnd_inst
        else:
            l_bnd_depth = zero
            l_bnd_inst = zero
            losses["bnd_depth"] = l_bnd_depth
            losses["bnd_inst"] = l_bnd_inst

        # === Embedding losses (with warmup) ===
        if inst_w > 0:
            has_instances = (instance_ids > 0).any(dim=-1).any(dim=-1)
            if has_instances.any():
                l_emb = embedding_distillation_loss(embeddings, instance_ids)
            else:
                l_emb = zero
            losses["emb_disc"] = l_emb
        else:
            l_emb = zero
            losses["emb_disc"] = l_emb

        # === Total ===
        total = (
            eff_distill * l_distill
            + self.lambda_align * l_align
            + self.lambda_proto * l_proto
            + self.lambda_ent * l_ent
            + inst_w * self.lambda_bnd_depth * l_bnd_depth
            + inst_w * self.lambda_bnd_inst * l_bnd_inst
            + inst_w * self.lambda_emb_disc * l_emb
        )
        losses["total"] = total
        losses["eff_distill_w"] = eff_distill
        losses["inst_w"] = inst_w

        return total, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
_STUFF_IDS = set(range(0, 11))
_THING_IDS = set(range(11, 19))
_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

_CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


def boundary_embedding_instances(
    semantic_tid, boundary_prob, embeddings_np,
    thing_ids=_THING_IDS, boundary_threshold=0.5,
    merge_sim=0.85, min_area=50,
):
    """Generate instance masks from boundary + embedding predictions.

    Args:
        semantic_tid: (H, W) uint8 trainID labels
        boundary_prob: (H, W) float [0,1] boundary probability
        embeddings_np: (D, H, W) float L2-normalized embeddings
        thing_ids: set of thing class trainIDs
        boundary_threshold: hard threshold for boundary edges
        merge_sim: cosine similarity threshold for merging fragments
        min_area: minimum instance area in pixels

    Returns:
        instance_id_map: (H, W) int32, 0=background
    """
    H, W = semantic_tid.shape
    boundary_hard = boundary_prob > boundary_threshold
    instance_id_map = np.zeros((H, W), dtype=np.int32)
    assigned = np.zeros((H, W), dtype=bool)
    next_id = 1

    for cls in sorted(thing_ids):
        cls_mask = semantic_tid == cls
        if cls_mask.sum() < min_area:
            continue

        # Split by boundaries
        split_mask = cls_mask & ~boundary_hard
        labeled, n_cc = ndimage.label(split_mask)

        # Collect fragments with their mean embeddings
        fragments = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area < min_area // 2:  # relaxed for merging
                continue
            mean_emb = embeddings_np[:, cc_mask].mean(axis=1)  # (D,)
            norm = np.linalg.norm(mean_emb) + 1e-8
            mean_emb = mean_emb / norm
            fragments.append({
                "mask": cc_mask,
                "emb": mean_emb,
                "area": area,
            })

        # Greedy merge: if two fragments have similar embeddings, merge them
        merged = []
        used = [False] * len(fragments)
        for i in range(len(fragments)):
            if used[i]:
                continue
            group_mask = fragments[i]["mask"].copy()
            group_emb = fragments[i]["emb"]
            used[i] = True

            for j in range(i + 1, len(fragments)):
                if used[j]:
                    continue
                sim = np.dot(group_emb, fragments[j]["emb"])
                if sim > merge_sim:
                    group_mask |= fragments[j]["mask"]
                    # Recompute mean embedding
                    group_emb = embeddings_np[:, group_mask].mean(axis=1)
                    norm = np.linalg.norm(group_emb) + 1e-8
                    group_emb = group_emb / norm
                    used[j] = True

            merged.append(group_mask)

        # Sort by area descending for priority in reclamation
        merged.sort(key=lambda m: -m.sum())

        # Reclaim boundary pixels via dilation
        for frag_mask in merged:
            dilated = ndimage.binary_dilation(frag_mask, iterations=2)
            reclaimed = dilated & cls_mask & ~assigned
            final_mask = frag_mask | reclaimed
            if final_mask.sum() >= min_area:
                instance_id_map[final_mask] = next_id
                assigned |= final_mask
                next_id += 1

    return instance_id_map


def _match_panoptic(pred_pan, pred_segments, gt_pan, gt_segments, num_cls=19):
    """Score one image: match pred vs GT segments, return per-class TP/FP/FN/IoU."""
    tp = np.zeros(num_cls)
    fp = np.zeros(num_cls)
    fn = np.zeros(num_cls)
    iou_sum = np.zeros(num_cls)

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

    return tp, fp, fn, iou_sum


def evaluate_joint_panoptic(model, val_loader, device, cityscapes_root,
                            eval_hw=(512, 1024), cc_min_area=50,
                            eval_boundary=False):
    """Evaluate JointRefineNet with BOTH CC and optionally boundary+embedding.

    Always reports CC-based PQ (consistent baseline). When eval_boundary=True,
    additionally reports boundary+embedding PQ_things for comparison.
    """
    gt_dir = os.path.join(cityscapes_root, "gtFine", "val")
    H, W = eval_hw
    num_cls = 19

    confusion = np.zeros((num_cls, num_cls), dtype=np.int64)
    # CC-based counters (always computed)
    tp_cc = np.zeros(num_cls)
    fp_cc = np.zeros(num_cls)
    fn_cc = np.zeros(num_cls)
    iou_sum_cc = np.zeros(num_cls)
    # Boundary+embedding counters (only when eval_boundary=True)
    tp_bnd = np.zeros(num_cls)
    fp_bnd = np.zeros(num_cls)
    fn_bnd = np.zeros(num_cls)
    iou_sum_bnd = np.zeros(num_cls)
    changed_pixels = 0
    total_pixels = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=100, leave=False):
            outputs = model(
                batch["dinov2_features"].to(device),
                batch["depth"].to(device),
                batch["depth_grads"].to(device),
            )

            pred_27 = outputs["semantic_logits"].argmax(dim=1).cpu().numpy()
            boundary_prob = torch.sigmoid(
                outputs["boundary_logits"]
            ).squeeze(1).cpu().numpy()
            embeddings = outputs["embeddings"].cpu().numpy()
            orig_27 = batch["cause_logits"].argmax(dim=1).cpu().numpy()

            for i in range(pred_27.shape[0]):
                city, stem = batch["city"][i], batch["stem"][i]

                # Track prediction changes
                changed_pixels += (pred_27[i] != orig_27[i]).sum()
                total_pixels += pred_27[i].size

                # Map 27→19 trainID at patch resolution
                pred_tid_patch = _CAUSE27_TO_TRAINID[pred_27[i]]

                # Upsample semantic to eval resolution
                pred_sem = np.array(
                    Image.fromarray(pred_tid_patch).resize(
                        (W, H), Image.NEAREST
                    )
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
                        Image.fromarray(gt_sem).resize(
                            (W, H), Image.NEAREST
                        )
                    )

                # mIoU confusion matrix
                valid = (gt_sem < num_cls) & (pred_sem < num_cls)
                if valid.sum() > 0:
                    np.add.at(confusion, (gt_sem[valid], pred_sem[valid]), 1)

                # --- Build GT panoptic map (shared by both eval methods) ---
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
                    gt_inst = np.array(
                        Image.open(gt_inst_path), dtype=np.int32
                    )
                    if gt_inst.shape != (H, W):
                        gt_inst = np.array(
                            Image.fromarray(gt_inst).resize(
                                (W, H), Image.NEAREST
                            )
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

                # --- Build stuff segments (shared prefix for pred_pan) ---
                stuff_pan = np.zeros((H, W), dtype=np.int32)
                stuff_segments = {}
                nxt = 1
                for cls in _STUFF_IDS:
                    mask = pred_sem == cls
                    if mask.sum() < 64:
                        continue
                    stuff_pan[mask] = nxt
                    stuff_segments[nxt] = cls
                    nxt += 1

                # --- Method 1: CC instances (always) ---
                pred_pan_cc = stuff_pan.copy()
                pred_seg_cc = dict(stuff_segments)
                nxt_cc = nxt
                for cls in _THING_IDS:
                    cls_mask = pred_sem == cls
                    if cls_mask.sum() < cc_min_area:
                        continue
                    labeled, n_cc = ndimage.label(cls_mask)
                    for comp in range(1, n_cc + 1):
                        cmask = labeled == comp
                        if cmask.sum() < cc_min_area:
                            continue
                        pred_pan_cc[cmask] = nxt_cc
                        pred_seg_cc[nxt_cc] = cls
                        nxt_cc += 1

                t, f, n_, s = _match_panoptic(
                    pred_pan_cc, pred_seg_cc, gt_pan, gt_segments, num_cls,
                )
                tp_cc += t; fp_cc += f; fn_cc += n_; iou_sum_cc += s

                # --- Method 2: Boundary+embedding instances (optional) ---
                if eval_boundary:
                    bnd_patch = boundary_prob[i]
                    bnd_full = np.array(
                        Image.fromarray(
                            (bnd_patch * 255).astype(np.uint8)
                        ).resize((W, H), Image.BILINEAR)
                    ).astype(np.float32) / 255.0

                    emb_patch = embeddings[i]
                    emb_tensor = torch.from_numpy(emb_patch).unsqueeze(0)
                    emb_full = F.interpolate(
                        emb_tensor, size=(H, W), mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).numpy()
                    emb_norm = np.linalg.norm(
                        emb_full, axis=0, keepdims=True
                    ) + 1e-8
                    emb_full = emb_full / emb_norm

                    inst_map = boundary_embedding_instances(
                        pred_sem, bnd_full, emb_full,
                        thing_ids=_THING_IDS,
                        min_area=cc_min_area,
                    )

                    pred_pan_bnd = stuff_pan.copy()
                    pred_seg_bnd = dict(stuff_segments)
                    nxt_bnd = nxt
                    for uid in np.unique(inst_map):
                        if uid == 0:
                            continue
                        mask = inst_map == uid
                        if mask.sum() < cc_min_area:
                            continue
                        cls_votes = pred_sem[mask]
                        valid_votes = cls_votes[cls_votes < num_cls]
                        if len(valid_votes) == 0:
                            continue
                        cls = int(np.bincount(valid_votes).argmax())
                        if cls not in _THING_IDS:
                            continue
                        pred_pan_bnd[mask] = nxt_bnd
                        pred_seg_bnd[nxt_bnd] = cls
                        nxt_bnd += 1

                    t, f, n_, s = _match_panoptic(
                        pred_pan_bnd, pred_seg_bnd, gt_pan, gt_segments,
                        num_cls,
                    )
                    tp_bnd += t; fp_bnd += f; fn_bnd += n_; iou_sum_bnd += s

    # --- Compute metrics ---
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    miou = iou[union > 0].mean() * 100

    def _pq_from_counters(tp_arr, fp_arr, fn_arr, iou_arr):
        all_pq, stuff_pq, thing_pq = [], [], []
        per_cls = {}
        for c in range(num_cls):
            t, f_p, f_n, s = tp_arr[c], fp_arr[c], fn_arr[c], iou_arr[c]
            if t + f_p + f_n > 0:
                sq = s / (t + 1e-8)
                rq = t / (t + 0.5 * f_p + 0.5 * f_n)
                pq = sq * rq
            else:
                pq = 0.0
            per_cls[_CS_CLASS_NAMES[c]] = round(pq * 100, 2)
            if t + f_p + f_n > 0:
                all_pq.append(pq)
                (stuff_pq if c in _STUFF_IDS else thing_pq).append(pq)
        pq_all = float(np.mean(all_pq)) * 100 if all_pq else 0.0
        pq_stuff = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
        pq_things = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0
        return pq_all, pq_stuff, pq_things, per_cls

    pq_all, pq_stuff, pq_things, per_class = _pq_from_counters(
        tp_cc, fp_cc, fn_cc, iou_sum_cc,
    )
    change_pct = changed_pixels / max(total_pixels, 1) * 100

    result = {
        "PQ": round(pq_all, 2),
        "PQ_stuff": round(pq_stuff, 2),
        "PQ_things": round(pq_things, 2),
        "mIoU": round(miou, 2),
        "changed_pct": round(change_pct, 2),
        "per_class_pq": per_class,
        "per_class_iou": {
            _CS_CLASS_NAMES[i]: round(iou[i] * 100, 2)
            for i in range(num_cls)
        },
    }

    if eval_boundary:
        pq_all_b, pq_stuff_b, pq_things_b, per_class_b = _pq_from_counters(
            tp_bnd, fp_bnd, fn_bnd, iou_sum_bnd,
        )
        result["PQ_bnd"] = round(pq_all_b, 2)
        result["PQ_stuff_bnd"] = round(pq_stuff_b, 2)
        result["PQ_things_bnd"] = round(pq_things_b, 2)
        result["per_class_pq_bnd"] = per_class_b

    model.train()
    return result


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
    model = JointRefineNet(
        num_classes=NUM_CLASSES,
        feature_dim=768,
        bridge_dim=args.bridge_dim,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        coupling_strength=args.coupling_strength,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"JointRefineNet: {total_params:,} params ({trainable_params:,} trainable)")

    # Datasets
    train_dataset = JointPseudoLabelDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        logits_subdir=args.logits_subdir,
        instance_subdir=args.instance_subdir,
    )
    val_dataset = JointPseudoLabelDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        logits_subdir=args.logits_subdir,
        instance_subdir=args.instance_subdir,
    )

    pin_mem = device.type == "cuda"
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01,
    )

    # Loss
    loss_fn = JointRefineNetLoss(
        lambda_distill=args.lambda_distill,
        lambda_distill_min=args.lambda_distill_min,
        lambda_align=args.lambda_align,
        lambda_proto=args.lambda_proto,
        lambda_ent=args.lambda_ent,
        label_smoothing=args.label_smoothing,
        lambda_bnd_depth=args.lambda_bnd_depth,
        lambda_bnd_inst=args.lambda_bnd_inst,
        lambda_emb_disc=args.lambda_emb_disc,
        instance_warmup_epochs=args.instance_warmup_epochs,
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["total_params"] = total_params
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training
    best_pq = 0.0
    log.info(
        f"Training for {args.num_epochs} epochs, "
        f"{len(train_loader)} batches/epoch"
    )

    loss_keys = [
        "total", "distill", "align", "proto", "entropy",
        "bnd_depth", "bnd_inst", "emb_disc",
    ]

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        loss_fn.set_epoch(epoch, args.num_epochs)
        epoch_losses = {k: 0.0 for k in loss_keys}
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            ncols=140, leave=True,
        )
        for batch_idx, batch in enumerate(pbar):
            cause_logits = batch["cause_logits"].to(device)
            dinov2_features = batch["dinov2_features"].to(device)
            depth = batch["depth"].to(device)
            depth_grads = batch["depth_grads"].to(device)
            inst_boundary = batch["instance_boundary"].to(device)
            inst_ids = batch["instance_ids"].to(device)

            # Forward
            outputs = model(dinov2_features, depth, depth_grads)

            # Loss
            total_loss, loss_dict = loss_fn(
                outputs, cause_logits, dinov2_features, depth,
                depth_grads, inst_boundary, inst_ids,
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            # Sanitize NaN/Inf gradients
            nan_grad_count = 0
            for p in model.parameters():
                if p.grad is not None and not p.grad.isfinite().all():
                    nan_grad_count += p.grad.isfinite().logical_not().sum().item()
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if nan_grad_count > 0 and (batch_idx + 1) % 50 == 0:
                log.warning(
                    f"  Replaced {nan_grad_count} NaN/Inf gradient elements"
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate
            for k in loss_keys:
                v = loss_dict.get(k, 0.0)
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += float(v)
            num_batches += 1

            # Progress bar
            pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                sem=f"{epoch_losses['distill'] / num_batches:.3f}",
                bnd=f"{(epoch_losses['bnd_depth'] + epoch_losses['bnd_inst']) / num_batches:.3f}",
                emb=f"{epoch_losses['emb_disc'] / num_batches:.3f}",
            )

        pbar.close()
        scheduler.step()

        # Epoch summary
        dt = time.time() - t0
        avg = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch}/{args.num_epochs} ({dt:.0f}s) | "
            f"loss={avg['total']:.4f} "
            f"distill={avg['distill']:.4f}(w={loss_fn._distill_scale:.2f}) "
            f"align={avg['align']:.4f} proto={avg['proto']:.4f} "
            f"ent={avg['entropy']:.4f} "
            f"bnd_d={avg['bnd_depth']:.4f} bnd_i={avg['bnd_inst']:.4f} "
            f"emb={avg['emb_disc']:.4f} "
            f"inst_w={loss_fn._inst_scale:.2f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            do_bnd = epoch > args.instance_warmup_epochs
            log.info(f"Evaluating at epoch {epoch} (CC always"
                     f"{' + boundary+embedding' if do_bnd else ''})...")
            metrics = evaluate_joint_panoptic(
                model, val_loader, device, args.cityscapes_root,
                eval_boundary=do_bnd,
            )
            log.info(
                f"  CC:  PQ={metrics['PQ']:.2f} | "
                f"PQ_stuff={metrics['PQ_stuff']:.2f} | "
                f"PQ_things={metrics['PQ_things']:.2f} | "
                f"mIoU={metrics['mIoU']:.2f} | "
                f"changed={metrics['changed_pct']:.1f}%"
            )
            if do_bnd:
                log.info(
                    f"  BND: PQ={metrics['PQ_bnd']:.2f} | "
                    f"PQ_stuff={metrics['PQ_stuff_bnd']:.2f} | "
                    f"PQ_things={metrics['PQ_things_bnd']:.2f}"
                )

            # Log coupling strengths
            for i, block in enumerate(model.blocks):
                log.info(
                    f"  Block {i}: alpha={block.alpha.item():.4f}, "
                    f"beta={block.beta.item():.4f}"
                )

            # Save checkpoint
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch:04d}.pth",
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "config": config,
            }, ckpt_path)

            # Save best
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

            # Append metrics history
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
        description="Train JointRefineNet for unified semantic + instance refinement"
    )

    # Data
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause_crf")
    parser.add_argument("--logits_subdir", type=str, default=None)
    parser.add_argument("--instance_subdir", type=str,
                        default="pseudo_instance_spidepth")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/joint_refine_net")

    # Model architecture
    parser.add_argument("--bridge_dim", type=int, default=192)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--coupling_strength", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        default=True)
    parser.add_argument("--no_gradient_checkpointing",
                        dest="gradient_checkpointing", action="store_false")

    # Training
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=2)

    # Semantic loss weights
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--lambda_distill_min", type=float, default=0.5)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_proto", type=float, default=0.05)
    parser.add_argument("--lambda_ent", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Boundary loss weights
    parser.add_argument("--lambda_bnd_depth", type=float, default=0.3)
    parser.add_argument("--lambda_bnd_inst", type=float, default=0.3)

    # Embedding loss weights
    parser.add_argument("--lambda_emb_disc", type=float, default=0.3)

    # Instance loss warmup
    parser.add_argument("--instance_warmup_epochs", type=int, default=5,
                        help="Epochs of semantic-only training before adding "
                             "boundary/embedding losses (default: 5)")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
