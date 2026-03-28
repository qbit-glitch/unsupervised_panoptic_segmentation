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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mbps_pytorch.adaptive_instance_net import AdaptiveInstanceNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PATCH_H, PATCH_W = 32, 64
NUM_CLASSES = 27
EVAL_H, EVAL_W = 512, 1024

# Cityscapes trainID constants
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

# CAUSE 27-class → 19 trainID mapping
_CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InstanceDataset(Dataset):
    """Load DINOv2 features, depth, semantics for instance training."""

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        semantic_subdir: str = "pseudo_semantic_cause_crf",
        feature_subdir: str = "dinov2_features",
        depth_subdir: str = "depth_spidepth",
    ):
        self.root = cityscapes_root
        self.split = split
        self.semantic_subdir = semantic_subdir
        self.feature_subdir = feature_subdir
        self.depth_subdir = depth_subdir

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

        # CAUSE semantics: one-hot (27, 32, 64) — softmax probabilities
        sem_path = os.path.join(
            self.root, self.semantic_subdir, self.split, city, f"{stem}.png",
        )
        sem_full = np.array(Image.open(sem_path))
        sem_patch = np.array(
            Image.fromarray(sem_full).resize((PATCH_W, PATCH_H), Image.NEAREST)
        )
        # Smoothed one-hot
        smooth = 0.1
        onehot = np.full((NUM_CLASSES, PATCH_H, PATCH_W), smooth / NUM_CLASSES,
                         dtype=np.float32)
        for c in range(NUM_CLASSES):
            mask = sem_patch == c
            onehot[c][mask] = 1.0 - smooth + smooth / NUM_CLASSES

        # TrainID semantic map at patch res (for thing mask during training)
        sem_27_patch = sem_patch  # (32, 64) uint8 CAUSE-27
        sem_trainid_patch = _CAUSE27_TO_TRAINID[sem_27_patch]  # (32, 64) uint8

        return {
            "dinov2_features": torch.from_numpy(features).float(),
            "depth": torch.from_numpy(depth_np).float(),
            "depth_grads": torch.from_numpy(depth_grads).float(),
            "cause_logits": torch.from_numpy(onehot).float(),
            "grad_mag": torch.from_numpy(grad_mag).float(),
            "sem_trainid": torch.from_numpy(sem_trainid_patch.astype(np.int64)),
            "stem": stem,
            "city": city,
        }

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


class AdaptiveInstanceLoss(nn.Module):
    """Combined loss for AdaptiveInstanceNet training."""

    def __init__(
        self,
        lambda_split: float = 1.0,
        lambda_feat_boundary: float = 0.5,
        lambda_contrastive: float = 0.3,
        lambda_embed_reg: float = 0.1,
        split_tau: float = 0.10,
        split_soft_margin: float = 0.03,
        feat_sigma: float = 0.3,
        contrastive_pairs: int = 2048,
        contrastive_margin: float = 0.5,
    ):
        super().__init__()
        self.lambda_split = lambda_split
        self.lambda_feat_boundary = lambda_feat_boundary
        self.lambda_contrastive = lambda_contrastive
        self.lambda_embed_reg = lambda_embed_reg
        self.split_tau = split_tau
        self.split_soft_margin = split_soft_margin
        self.feat_sigma = feat_sigma
        self.contrastive_pairs = contrastive_pairs
        self.contrastive_margin = contrastive_margin

    def forward(self, split_logit, instance_embed, dinov2_features,
                depth, depth_grads, grad_mag, sem_trainid):
        losses = {}

        # 1. Split distillation from depth gradient
        l_split = split_distillation_loss(
            split_logit, grad_mag,
            tau=self.split_tau, soft_margin=self.split_soft_margin,
        )
        losses["split"] = l_split

        # 2. Feature-guided boundary alignment
        if self.lambda_feat_boundary > 0:
            l_feat = feature_boundary_loss(
                split_logit, dinov2_features, sigma=self.feat_sigma)
            losses["feat_boundary"] = l_feat
        else:
            l_feat = 0.0

        # 3. Contrastive embedding loss
        if self.lambda_contrastive > 0:
            l_contrast = contrastive_embedding_loss(
                instance_embed, dinov2_features, depth, sem_trainid,
                num_pairs=self.contrastive_pairs,
                margin=self.contrastive_margin,
            )
            losses["contrastive"] = l_contrast
        else:
            l_contrast = 0.0

        # 4. Embedding regularization
        if self.lambda_embed_reg > 0:
            l_reg = embedding_regularization_loss(instance_embed)
            losses["embed_reg"] = l_reg
        else:
            l_reg = 0.0

        total = (self.lambda_split * l_split
                 + self.lambda_feat_boundary * l_feat
                 + self.lambda_contrastive * l_contrast
                 + self.lambda_embed_reg * l_reg)
        losses["total"] = total
        return total, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pq_things(model, val_loader, device, cityscapes_root,
                       split_threshold=0.5, min_area=100,
                       base_tau=0.05, depth_subdir="depth_spidepth"):
    """Evaluate PQ_things using hybrid approach: model-predicted confidence
    modulates native-resolution depth gradient threshold.

    The model outputs split_prob at 32×64 which is upsampled to native
    resolution as a spatially-varying weight. The actual edge detection
    uses native-resolution depth gradients with an adaptive threshold:
        adaptive_tau = base_tau * (1.5 - split_prob)
    Where split_prob is high → lower threshold → more splits.
    Where split_prob is low → higher threshold → fewer splits.
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
        for batch in tqdm(val_loader, desc="Eval PQ_things", ncols=100, leave=False):
            split_logit, embed = model(
                batch["dinov2_features"].to(device),
                batch["depth"].to(device),
                batch["depth_grads"].to(device),
                batch["cause_logits"].to(device),
            )
            split_np = torch.sigmoid(split_logit).squeeze(1).cpu().numpy()  # (B, 32, 64)
            sem_tid = batch["sem_trainid"].cpu().numpy()  # (B, 32, 64)

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

                if grad_mag_native is not None:
                    # Hybrid: adaptive threshold on native gradients
                    # split_prob high → tau_local low → more edges
                    tau_local = base_tau * (1.5 - split_full)
                    tau_local = np.clip(tau_local, 0.02, 0.25)
                    edge_map = grad_mag_native > tau_local
                else:
                    # Fallback: direct threshold on model output
                    edge_map = split_full > split_threshold
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

                # Thing segments: CC with adaptive edge removal
                for cls in _THING_IDS:
                    cls_mask = sem_full == cls
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
    model = AdaptiveInstanceNet(
        feature_dim=768,
        depth_channels=3,
        semantic_dim=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"AdaptiveInstanceNet: {total_params:,} params")

    # Datasets
    train_dataset = InstanceDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
    )
    val_dataset = InstanceDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
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
        split_tau=args.split_tau,
        contrastive_pairs=args.contrastive_pairs,
    )

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args)
    config["total_params"] = total_params
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    use_amp = False
    amp_dtype = torch.float32
    amp_device = device.type

    # Training
    best_pq_things = 0.0
    log.info(f"Training for {args.num_epochs} epochs, "
             f"{len(train_loader)} batches/epoch")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "split": 0, "feat_boundary": 0,
                        "contrastive": 0, "embed_reg": 0}
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    ncols=130, leave=True)
        for batch_idx, batch in enumerate(pbar):
            dinov2 = batch["dinov2_features"].to(device)
            depth = batch["depth"].to(device)
            depth_grads = batch["depth_grads"].to(device)
            cause = batch["cause_logits"].to(device)
            grad_mag = batch["grad_mag"].to(device)
            sem_tid = batch["sem_trainid"].to(device)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype,
                                enabled=use_amp):
                split_logit, embed = model(dinov2, depth, depth_grads, cause)
                total_loss, loss_dict = loss_fn(
                    split_logit, embed, dinov2, depth, depth_grads, grad_mag,
                    sem_tid)

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
                split=f"{epoch_losses['split']/num_batches:.4f}",
                feat=f"{epoch_losses['feat_boundary']/num_batches:.4f}",
                contr=f"{epoch_losses['contrastive']/num_batches:.4f}",
            )

        pbar.close()
        scheduler.step()

        dt = time.time() - t0
        avg = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch}/{args.num_epochs} ({dt:.0f}s) | "
            f"loss={avg['total']:.4f} split={avg['split']:.4f} "
            f"feat={avg['feat_boundary']:.4f} contr={avg['contrastive']:.4f} "
            f"reg={avg['embed_reg']:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            log.info(f"Evaluating at epoch {epoch}...")
            metrics = evaluate_pq_things(
                model, val_loader, device, args.cityscapes_root,
                split_threshold=args.split_threshold,
                base_tau=args.base_tau,
                depth_subdir=args.depth_subdir,
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
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/adaptive_instance")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=6)

    # Training
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=2)

    # Loss weights
    parser.add_argument("--lambda_split", type=float, default=1.0)
    parser.add_argument("--lambda_feat_boundary", type=float, default=0.5)
    parser.add_argument("--lambda_contrastive", type=float, default=0.3)
    parser.add_argument("--lambda_embed_reg", type=float, default=0.1)
    parser.add_argument("--split_tau", type=float, default=0.10,
                        help="Depth gradient threshold for teacher signal")
    parser.add_argument("--contrastive_pairs", type=int, default=2048)

    # Evaluation
    parser.add_argument("--split_threshold", type=float, default=0.5,
                        help="Threshold on split_prob for instance boundaries")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth",
                        help="Subdirectory for native-resolution depth maps")
    parser.add_argument("--base_tau", type=float, default=0.05,
                        help="Base depth gradient threshold for hybrid eval")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
