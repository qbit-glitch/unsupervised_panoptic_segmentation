#!/usr/bin/env python3
"""Train CRF-TTT-Mamba2 pseudo-label refinement module.

Trains the combined CRF + Test-Time Training Mamba2 refiner on pre-computed
DINOv2 features, depth maps, and CAUSE-TR semantic pseudo-labels.

Training losses:
  1. Distillation: CE(refined, CAUSE labels) with label smoothing
  2. CRF pairwise consistency: penalize pred disagreement for similar features
  3. Depth-boundary alignment: prediction edges ↔ depth edges
  4. Entropy minimization: encourage confident predictions

Evaluation modes:
  - Standard forward (implicit TTT via Mamba2 state)
  - Explicit TTT adaptation (K gradient steps per image, self-supervised)

Usage:
    # Train
    python mbps_pytorch/train_ttt_mamba2.py \
        --cityscapes_root /path/to/cityscapes \
        --num_epochs 20 --batch_size 4 --device auto

    # Evaluate only (compare standard vs TTT)
    python mbps_pytorch/train_ttt_mamba2.py \
        --cityscapes_root /path/to/cityscapes \
        --eval_only --checkpoint checkpoints/ttt_mamba2/best.pth

    # Generate refined pseudo-labels
    python mbps_pytorch/train_ttt_mamba2.py \
        --cityscapes_root /path/to/cityscapes \
        --generate --checkpoint checkpoints/ttt_mamba2/best.pth \
        --output_subdir pseudo_semantic_ttt_mamba2
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from mbps_pytorch.ttt_mamba2_refiner import TTTMamba2Refiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PATCH_H, PATCH_W = 32, 64

# CAUSE 27-class → 19 trainID mapping (for 27-class evaluation only)
CAUSE27_TO_TRAINID = {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}

# Cityscapes raw ID → trainID
CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

# Stuff/thing split for panoptic evaluation
STUFF_IDS = set(range(0, 11))   # trainIDs 0-10
THING_IDS = set(range(11, 19))  # trainIDs 11-18
CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TTTDataset(Dataset):
    """Load pre-computed DINOv2 features, depth, and semantic pseudo-labels."""

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        semantic_subdir: str = "pseudo_semantic_overclustered_k300_nocrf",
        feature_subdir: str = "dinov2_features",
        depth_subdir: str = "depth_spidepth",
        num_classes: int = 19,
        limit: int = 0,
        mask_threshold: int = 6,
    ):
        self.root = cityscapes_root
        self.split = split
        self.semantic_subdir = semantic_subdir
        self.feature_subdir = feature_subdir
        self.depth_subdir = depth_subdir
        self.num_classes = num_classes
        self.mask_threshold = mask_threshold

        # Find all images
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

        if limit > 0:
            self.entries = self.entries[:limit]
        log.info(f"TTTDataset: {len(self.entries)} images ({split})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        stem, city = entry["stem"], entry["city"]

        # DINOv2 features: (2048, 768) float16 → (768, 32, 64) float32
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
        depth_tensor = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
        depth_patch = F.interpolate(
            depth_tensor.float(), size=(PATCH_H, PATCH_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # Normalize depth to [0, 1]
        dmin, dmax = depth_patch.min(), depth_patch.max()
        if dmax > dmin:
            depth_patch = (depth_patch - dmin) / (dmax - dmin)

        # Sobel gradients on depth
        depth_grads = self._sobel_gradients(depth_patch[0].numpy())

        # Semantic label: (1024, 2048) uint8 → (32, 64) at patch resolution
        sem_path = os.path.join(
            self.root, self.semantic_subdir, self.split, city, f"{stem}.png",
        )
        sem_full = np.array(Image.open(sem_path))
        sem_patch = np.array(
            Image.fromarray(sem_full).resize((PATCH_W, PATCH_H), Image.NEAREST)
        )

        # Create soft probabilities from semantic label (label smoothing)
        nc = self.num_classes
        cause_probs = np.full(
            (nc, PATCH_H, PATCH_W), 0.1 / nc, dtype=np.float32
        )
        for c in range(nc):
            mask = sem_patch == c
            cause_probs[c][mask] = 1.0 - 0.1 + 0.1 / nc

        # Target: hard class labels for CE loss
        target = sem_patch.astype(np.int64)

        # Confidence mask via 8-neighbor agreement
        conf_mask = self._neighbor_agreement_mask(sem_patch, self.mask_threshold)

        return {
            "dinov2_features": torch.from_numpy(features),
            "depth": depth_patch,
            "depth_grads": torch.from_numpy(depth_grads),
            "cause_probs": torch.from_numpy(cause_probs),
            "conf_mask": torch.from_numpy(conf_mask),
            "target": torch.from_numpy(target),
            "stem": stem,
            "city": city,
        }

    @staticmethod
    def _sobel_gradients(depth_2d):
        """Compute Sobel gradients. depth_2d: (H, W) → (2, H, W)."""
        d = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0).float()
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        grad_x = F.conv2d(d, kx, padding=1).squeeze()
        grad_y = F.conv2d(d, ky, padding=1).squeeze()
        return torch.stack([grad_x, grad_y], dim=0).numpy()

    @staticmethod
    def _neighbor_agreement_mask(sem_patch, threshold=6):
        """Confidence mask: 1.0 if >= threshold of 8 neighbors share same class.

        Removes spatially inconsistent pseudo-label noise and boundary artifacts.
        """
        H, W = sem_patch.shape
        padded = np.pad(sem_patch, 1, mode='edge')
        agree = np.zeros((H, W), dtype=np.int32)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                neighbor = padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
                agree += (neighbor == sem_patch).astype(np.int32)
        return (agree >= threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class NoiseRobustLoss(nn.Module):
    """Noise-robust loss for learning from noisy pseudo-labels.

    Three mechanisms to handle pseudo-label noise:

    1. Symmetric Cross-Entropy (Wang et al., ICCV 2019):
       L_SCE = α·CE(pred, label) + β·RCE(pred, label)
       Forward CE provides standard learning signal; reverse CE is naturally
       noise-tolerant — its loss landscape preserves the correct minimum even
       under label noise. Combined, SCE converges fast (from CE) while resisting
       memorization of noisy labels (from RCE).

    2. Local-consistency confidence masking:
       Pixels where < k/8 neighbors share the same pseudo-label are masked out
       (target set to ignore=255). Removes spatially inconsistent noise and
       boundary artifacts without requiring any ground truth.

    3. EMA Teacher-Student consistency (Mean Teacher, Tarvainen & Valpola 2017):
       An exponential moving average (EMA) copy of the student provides soft
       pseudo-targets. The teacher's sharpened predictions are more stable and
       denoised than the raw pseudo-labels. Cross-entropy from student to
       sharpened teacher gives self-supervised signal INDEPENDENT of original
       pseudo-label quality — the key mechanism for the student to exceed
       the noisy teacher.

    Also retains Information Maximization (SHOT, ICML 2020) for class diversity.

    References:
      - SCE (Wang et al., ICCV 2019): Symmetric CE for robust noisy-label learning
      - SHOT (Liang et al., ICML 2020): Information maximization prevents collapse
      - Mean Teacher (Tarvainen & Valpola, NeurIPS 2017): EMA teacher-student
      - Class-Balanced Loss (Cui et al., CVPR 2019): inverse-freq weighting
      - Student Denoiser (Jiang et al., 2023): Students exceed noisy teachers
    """

    # Inverse-sqrt-frequency weights for Cityscapes overclustered k=300 (19 trainIDs)
    CLASS_WEIGHTS_19 = [
        0.1193, 0.3035, 0.1573, 0.9296, 0.8115,  # road, sidewalk, building, wall, fence
        0.8888, 1.5517, 1.3069, 0.1947, 0.7355,  # pole, tlight, tsign, veg, terrain
        0.3870, 0.6865, 2.2990, 0.2913, 1.5022,  # sky, person, rider, car, truck
        1.6955, 1.6764, 2.2948, 1.1682,           # bus, train, motorcycle, bicycle
    ]

    def __init__(
        self,
        sce_alpha: float = 1.0,
        sce_beta: float = 1.0,
        im_weight: float = 0.01,
        consist_weight: float = 1.0,
        teacher_temp: float = 0.5,
        teacher_conf: float = 0.7,
        num_classes: int = 19,
        rce_clip: float = 1e-4,
    ):
        super().__init__()
        self.sce_alpha = sce_alpha
        self.sce_beta = sce_beta
        self.w_im = im_weight
        self.w_consist = consist_weight
        self.teacher_temp = teacher_temp
        self.teacher_conf = teacher_conf
        self.rce_clip = rce_clip

        if num_classes == 19:
            w = torch.tensor(self.CLASS_WEIGHTS_19, dtype=torch.float32)
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", w)

    def forward(self, logits, target, conf_mask, teacher_logits=None):
        """
        Args:
            logits: (B, C, H, W) student model output
            target: (B, H, W) long — pseudo-label class indices
            conf_mask: (B, H, W) float — 1.0 for confident, 0.0 for masked
            teacher_logits: (B, C, H, W) EMA teacher output (detached, no grad)
        """
        B, C, H, W = logits.shape

        # ---- 1. Symmetric Cross-Entropy (SCE) ----
        # Apply confidence mask: set uncertain pixels to ignore
        target_masked = target.clone()
        target_masked[conf_mask < 0.5] = 255

        valid = target_masked < C
        if valid.sum() > 0:
            logits_flat = logits.permute(0, 2, 3, 1)[valid].reshape(-1, C)
            target_flat = target_masked[valid].reshape(-1)

            log_p = F.log_softmax(logits_flat, dim=1)
            p = log_p.exp()
            cb_weight = self.class_weights[target_flat]

            # Forward CE: -Σ q_k · log(p_k) with q = one-hot
            fwd_ce_per_pixel = F.nll_loss(log_p, target_flat, reduction='none')
            fwd_ce = (cb_weight * fwd_ce_per_pixel).mean()

            # Reverse CE: -Σ p_k · log(q_k) with q = one-hot clipped at rce_clip
            one_hot = torch.zeros_like(p).scatter_(1, target_flat.unsqueeze(1), 1.0)
            q_clipped = one_hot.clamp(min=self.rce_clip)
            rce_per_pixel = -(p * q_clipped.log()).sum(dim=1)
            rev_ce = (cb_weight * rce_per_pixel).mean()

            sce_loss = self.sce_alpha * fwd_ce + self.sce_beta * rev_ce
        else:
            fwd_ce = rev_ce = torch.tensor(0.0, device=logits.device)
            sce_loss = torch.tensor(0.0, device=logits.device)

        # ---- 2. Information Maximization (SHOT) ----
        probs = logits.softmax(dim=1)
        log_probs = logits.log_softmax(dim=1)
        pixel_entropy = -(probs * log_probs).sum(dim=1).mean()
        p_bar = probs.mean(dim=(0, 2, 3))
        marginal_entropy = -(p_bar * (p_bar + 1e-8).log()).sum()
        im_loss = pixel_entropy - marginal_entropy

        # ---- 3. EMA Teacher-Student Consistency (Mean Teacher) ----
        if teacher_logits is not None:
            # Sharpened teacher soft targets (lower temp = more confident)
            teacher_probs = (teacher_logits / self.teacher_temp).softmax(dim=1)

            # Only enforce on pixels where teacher is confident
            teacher_max_conf = teacher_probs.max(dim=1)[0]  # (B, H, W)
            conf_pixels = (teacher_max_conf > self.teacher_conf).float()

            # Student CE against sharpened teacher (soft cross-entropy)
            student_log_probs = F.log_softmax(logits, dim=1)
            # CE = -Σ_c teacher_probs_c * log(student_probs_c)
            ce_per_pixel = -(teacher_probs * student_log_probs).sum(dim=1)  # (B, H, W)

            denom = conf_pixels.sum().clamp(min=1)
            consist_loss = (ce_per_pixel * conf_pixels).sum() / denom
        else:
            consist_loss = torch.tensor(0.0, device=logits.device)

        # ---- Total ----
        total = sce_loss + self.w_im * im_loss + self.w_consist * consist_loss

        mask_pct = (1.0 - conf_mask.float().mean().item()) * 100
        teacher_conf_pct = 0.0
        if teacher_logits is not None:
            teacher_conf_pct = (conf_pixels.mean().item()) * 100

        return {
            "fwd_ce": fwd_ce.item(),
            "rev_ce": rev_ce.item(),
            "sce": sce_loss.item(),
            "im": im_loss.item(),
            "consist": consist_loss.item(),
            "pixel_ent": pixel_entropy.item(),
            "marginal_ent": marginal_entropy.item(),
            "mask_pct": mask_pct,
            "teacher_conf_pct": teacher_conf_pct,
            "total_val": total.item(),
            "total": total,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_miou(
    model,
    dataset,
    device,
    mode="standard",
    ttt_steps=3,
    ttt_lr=0.01,
    num_classes_eval=19,
):
    """Evaluate mIoU on val set at both patch and full resolution.

    Args:
        mode: "standard" (forward only) or "ttt" (explicit TTT adaptation)

    Returns:
        dict with mIoU (patch-res), mIoU_full (1024x2048), per_class_iou, etc.
    """
    model.eval()
    model_classes = dataset.num_classes

    # Build pred→trainID LUT (identity if already 19-class, remap if 27-class)
    lut = np.full(256, 255, dtype=np.uint8)
    if model_classes == 19:
        for i in range(19):
            lut[i] = i
    else:
        for c27, c19 in CAUSE27_TO_TRAINID.items():
            lut[c27] = c19

    confusion = np.zeros((num_classes_eval, num_classes_eval), dtype=np.int64)
    confusion_full = np.zeros((num_classes_eval, num_classes_eval), dtype=np.int64)

    gt_dir = os.path.join(dataset.root, "gtFine", "val")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in tqdm(loader, desc=f"Eval ({mode})", leave=False):
        stem = batch["stem"][0]
        city = batch["city"][0]

        feats = batch["dinov2_features"].to(device)
        depth = batch["depth"].to(device)
        depth_grads = batch["depth_grads"].to(device)

        if mode == "ttt":
            logits = model.ttt_adapt(
                feats, depth, depth_grads,
                ttt_steps=ttt_steps, ttt_lr=ttt_lr,
            )
        else:
            with torch.no_grad():
                logits = model(feats, depth, depth_grads)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (32, 64)
        pred_19 = lut[pred.astype(np.uint8)]

        gt_path = os.path.join(
            gt_dir, city, f"{stem}_gtFine_labelIds.png"
        )
        if not os.path.exists(gt_path):
            continue
        gt_full = np.array(Image.open(gt_path))

        # --- Patch-res evaluation (32×64) ---
        gt_patch = np.array(
            Image.fromarray(gt_full).resize((PATCH_W, PATCH_H), Image.NEAREST)
        )
        gt_19_patch = np.full_like(gt_patch, 255, dtype=np.uint8)
        for raw_id, train_id in CS_ID_TO_TRAIN.items():
            gt_19_patch[gt_patch == raw_id] = train_id

        valid = (gt_19_patch < num_classes_eval) & (pred_19 < num_classes_eval)
        if valid.sum() > 0:
            np.add.at(
                confusion,
                (gt_19_patch[valid].astype(int), pred_19[valid].astype(int)),
                1,
            )

        # --- Full-res evaluation (1024×2048) ---
        pred_full = np.array(
            Image.fromarray(pred_19).resize((2048, 1024), Image.NEAREST)
        )
        gt_19_full = np.full_like(gt_full, 255, dtype=np.uint8)
        for raw_id, train_id in CS_ID_TO_TRAIN.items():
            gt_19_full[gt_full == raw_id] = train_id

        valid_f = (gt_19_full < num_classes_eval) & (pred_full < num_classes_eval)
        if valid_f.sum() > 0:
            np.add.at(
                confusion_full,
                (gt_19_full[valid_f].astype(int), pred_full[valid_f].astype(int)),
                1,
            )

    def _miou(conf):
        inter = np.diag(conf)
        union = conf.sum(axis=1) + conf.sum(axis=0) - inter
        iou = np.where(union > 0, inter / union, 0.0)
        return iou[union > 0].mean() * 100, iou * 100

    miou_patch, iou_patch = _miou(confusion)
    miou_full, iou_full = _miou(confusion_full)

    return {
        "mIoU": miou_patch,
        "mIoU_full": miou_full,
        "per_class_iou": iou_patch,
        "per_class_iou_full": iou_full,
        "confusion": confusion,
        "confusion_full": confusion_full,
    }


def evaluate_changed_pct(model, dataset, device):
    """Measure what fraction of pixels change vs input pseudo-labels."""
    model.eval()
    nc = dataset.num_classes
    total_pixels = 0
    changed_pixels = 0

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    for batch in loader:
        feats = batch["dinov2_features"].to(device)
        depth = batch["depth"].to(device)
        depth_grads = batch["depth_grads"].to(device)
        target = batch["target"]  # (B, H, W)

        with torch.no_grad():
            logits = model(feats, depth, depth_grads)
            pred = logits.argmax(dim=1).cpu()

        valid = target < nc
        changed = (pred != target) & valid
        changed_pixels += changed.sum().item()
        total_pixels += valid.sum().item()

    return changed_pixels / max(total_pixels, 1) * 100


def evaluate_panoptic(
    model,
    dataset,
    device,
    mode="standard",
    ttt_steps=3,
    ttt_lr=0.01,
    eval_hw=(512, 1024),
    cc_min_area=50,
):
    """Full panoptic evaluation: PQ, SQ, RQ, PQ_stuff, PQ_things, mIoU.

    Thing instances derived via connected components of the semantic map.
    """
    from scipy import ndimage
    from collections import defaultdict

    model.eval()
    model_classes = dataset.num_classes
    H, W = eval_hw
    num_cls = 19

    # Build pred→trainID LUT
    lut = np.full(256, 255, dtype=np.uint8)
    if model_classes == 19:
        for i in range(19):
            lut[i] = i
    else:
        for c27, c19 in CAUSE27_TO_TRAINID.items():
            lut[c27] = c19

    # Accumulators
    confusion = np.zeros((num_cls, num_cls), dtype=np.int64)
    tp = np.zeros(num_cls)
    fp = np.zeros(num_cls)
    fn = np.zeros(num_cls)
    iou_sum = np.zeros(num_cls)
    changed_pixels = 0
    total_pixels = 0

    gt_dir = os.path.join(dataset.root, "gtFine", "val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(loader, desc=f"PanEval ({mode})", leave=False):
        stem = batch["stem"][0]
        city = batch["city"][0]

        feats = batch["dinov2_features"].to(device)
        depth = batch["depth"].to(device)
        depth_grads = batch["depth_grads"].to(device)
        target = batch["target"]

        if mode == "ttt":
            logits = model.ttt_adapt(
                feats, depth, depth_grads,
                ttt_steps=ttt_steps, ttt_lr=ttt_lr,
            )
        else:
            with torch.no_grad():
                logits = model(feats, depth, depth_grads)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        pred_19 = lut[pred.astype(np.uint8)]

        # Track changes vs input pseudo-labels
        tgt = target.squeeze(0).numpy()
        tgt_19 = lut[tgt.astype(np.uint8)]
        valid_change = (tgt_19 < num_cls) & (pred_19 < num_cls)
        changed_pixels += ((pred_19 != tgt_19) & valid_change).sum()
        total_pixels += valid_change.sum()

        # Upsample to eval resolution
        pred_sem = np.array(
            Image.fromarray(pred_19).resize((W, H), Image.NEAREST)
        )

        # Load GT semantic
        gt_path = os.path.join(gt_dir, city, f"{stem}_gtFine_labelIds.png")
        if not os.path.exists(gt_path):
            continue
        gt_raw = np.array(Image.open(gt_path))
        gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
        for raw_id, tid in CS_ID_TO_TRAIN.items():
            gt_sem[gt_raw == raw_id] = tid
        if gt_sem.shape != (H, W):
            gt_sem = np.array(
                Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

        # mIoU confusion
        valid = (gt_sem < num_cls) & (pred_sem < num_cls)
        if valid.sum() > 0:
            np.add.at(confusion, (gt_sem[valid], pred_sem[valid]), 1)

        # --- Panoptic: build predicted panoptic map ---
        pred_pan = np.zeros((H, W), dtype=np.int32)
        pred_segments = {}
        nxt = 1

        for cls in STUFF_IDS:
            mask = pred_sem == cls
            if mask.sum() < 64:
                continue
            pred_pan[mask] = nxt
            pred_segments[nxt] = cls
            nxt += 1

        for cls in THING_IDS:
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

        for cls in STUFF_IDS:
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
                if raw_cls not in CS_ID_TO_TRAIN:
                    continue
                tid = CS_ID_TO_TRAIN[raw_cls]
                if tid not in THING_IDS:
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
    inter_diag = np.diag(confusion)
    union_arr = confusion.sum(1) + confusion.sum(0) - inter_diag
    iou = np.where(union_arr > 0, inter_diag / union_arr, 0.0)
    miou = iou[union_arr > 0].mean() * 100

    all_pq, stuff_pq, thing_pq = [], [], []
    all_sq, all_rq = [], []
    per_class = {}
    for c in range(num_cls):
        t, f_p, f_n, s = tp[c], fp[c], fn[c], iou_sum[c]
        if t + f_p + f_n > 0:
            sq = s / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0
        per_class[CS_CLASS_NAMES[c]] = {"PQ": round(pq * 100, 2), "IoU": round(iou[c] * 100, 2)}
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            all_sq.append(sq)
            all_rq.append(rq)
            (stuff_pq if c in STUFF_IDS else thing_pq).append(pq)

    pq_all = float(np.mean(all_pq)) * 100 if all_pq else 0.0
    pq_stuff = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
    pq_things = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0
    sq_all = float(np.mean(all_sq)) * 100 if all_sq else 0.0
    rq_all = float(np.mean(all_rq)) * 100 if all_rq else 0.0

    return {
        "PQ": round(pq_all, 2),
        "PQ_stuff": round(pq_stuff, 2),
        "PQ_things": round(pq_things, 2),
        "SQ": round(sq_all, 2),
        "RQ": round(rq_all, 2),
        "mIoU": round(miou, 2),
        "changed_pct": round(changed_pixels / max(total_pixels, 1) * 100, 2),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    """Main training loop with optional DDP support."""
    # DDP setup
    rank, world_size, ddp_device = _setup_ddp()
    use_ddp = world_size > 1

    if use_ddp:
        device = ddp_device
        if _is_main(rank):
            log.info(f"DDP: {world_size} GPUs, rank {rank}, device {device}")
    else:
        device = _get_device(args.device)
        log.info(f"Device: {device}")

    # Datasets
    train_ds = TTTDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        feature_subdir=args.feature_subdir,
        depth_subdir=args.depth_subdir,
        num_classes=args.num_classes,
        limit=args.limit,
        mask_threshold=args.mask_threshold,
    )
    val_ds = TTTDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        feature_subdir=args.feature_subdir,
        depth_subdir=args.depth_subdir,
        num_classes=args.num_classes,
        mask_threshold=args.mask_threshold,
    )

    # DataLoader with DistributedSampler for DDP
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if use_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    if _is_main(rank):
        log.info(
            f"Train: {len(train_ds)} images, {len(train_loader)} batches/GPU, "
            f"effective batch={args.batch_size * world_size * args.accum_steps} "
            f"(batch={args.batch_size} x {world_size} GPUs x {args.accum_steps} accum)"
        )

    # Model
    model = TTTMamba2Refiner(
        num_classes=args.num_classes,
        feature_dim=768,
        bridge_dim=args.bridge_dim,
        num_blocks=args.num_blocks,
        scan_mode=args.scan_mode,
        layer_type=args.layer_type,
        context_dropout=args.context_dropout,
        d_state=args.d_state,
        gradient_checkpointing=True,
    ).to(device)

    if _is_main(rank):
        params = model.count_parameters()
        log.info(
            f"Model: {params['total']:,} total, {params['mamba_blocks']:,} "
            f"in Mamba2 blocks, {params['projections']:,} in projections"
        )

    # Wrap with DDP
    if use_ddp:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)

    # EMA Teacher (Mean Teacher — Tarvainen & Valpola, NeurIPS 2017)
    raw_model_for_ema = model.module if use_ddp else model
    teacher = copy.deepcopy(raw_model_for_ema)
    teacher.requires_grad_(False)
    teacher.eval()
    ema_alpha = args.ema_alpha
    if _is_main(rank):
        log.info(f"EMA Teacher: alpha={ema_alpha}, temp={args.teacher_temp}, "
                 f"conf={args.teacher_conf}, warmup={args.ema_warmup_epochs} epochs, "
                 f"feat_drop={args.feat_drop_rate}")

    # Loss and optimizer
    criterion = NoiseRobustLoss(
        sce_alpha=args.sce_alpha,
        sce_beta=args.sce_beta,
        im_weight=args.im_weight,
        consist_weight=args.consist_weight,
        teacher_temp=args.teacher_temp,
        teacher_conf=args.teacher_conf,
        num_classes=args.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6,
    )

    # Output directory
    out_dir = Path(args.output_dir)
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    best_pq = 0.0
    best_epoch = 0

    # AMP scaler for FP16 training
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    if _is_main(rank):
        log.info(f"AMP (FP16): {'enabled' if use_amp else 'disabled'}")

    for epoch in range(1, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses = {
            "fwd_ce": 0, "rev_ce": 0, "sce": 0, "im": 0, "consist": 0,
            "pixel_ent": 0, "marginal_ent": 0, "mask_pct": 0,
            "teacher_conf_pct": 0, "total_val": 0,
        }
        num_batches = 0
        t0 = time.time()

        # EMA teacher consistency starts after warmup
        use_teacher = epoch > args.ema_warmup_epochs

        accum_steps = args.accum_steps
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                     disable=not _is_main(rank))
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            feats = batch["dinov2_features"].to(device)
            depth = batch["depth"].to(device)
            depth_grads = batch["depth_grads"].to(device)
            target = batch["target"].to(device)
            conf_mask = batch["conf_mask"].to(device)

            with autocast(enabled=use_amp):
                # Student forward pass with feature dropout (strong aug)
                # Teacher sees clean features, student sees corrupted —
                # forces robust representations (DINO-style asymmetric aug)
                if use_teacher and args.feat_drop_rate > 0:
                    drop_mask = torch.bernoulli(torch.full(
                        (feats.shape[0], feats.shape[1], 1, 1),
                        1.0 - args.feat_drop_rate,
                        device=feats.device, dtype=feats.dtype,
                    )).expand_as(feats)
                    scale = 1.0 / (1.0 - args.feat_drop_rate)
                    feats_student = feats * drop_mask * scale
                else:
                    feats_student = feats

                logits = model(feats_student, depth, depth_grads)

                # EMA teacher forward pass on clean features (no grad)
                teacher_logits = None
                if use_teacher:
                    with torch.no_grad():
                        teacher_logits = teacher(feats, depth, depth_grads)

                losses = criterion(logits, target, conf_mask, teacher_logits)

                # Scale loss for gradient accumulation
                scaled_loss = losses["total"] / accum_steps

            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA update: teacher ← α·teacher + (1-α)·student
                if use_teacher:
                    raw_student = model.module if use_ddp else model
                    with torch.no_grad():
                        for p_s, p_t in zip(raw_student.parameters(),
                                            teacher.parameters()):
                            p_t.data.mul_(ema_alpha).add_(
                                p_s.data, alpha=1.0 - ema_alpha)

            for k in epoch_losses:
                epoch_losses[k] += losses[k]
            num_batches += 1

            if _is_main(rank):
                pbar.set_postfix(
                    loss=f"{losses['total_val']:.3f}",
                    sce=f"{losses['sce']:.3f}",
                    ema=f"{losses['consist']:.3f}",
                )

        scheduler.step()
        elapsed = time.time() - t0

        if _is_main(rank):
            avg = {k: v / num_batches for k, v in epoch_losses.items()}
            teacher_str = (f" t_conf={avg['teacher_conf_pct']:.1f}%"
                           if use_teacher else " (no teacher)")
            log.info(
                f"Epoch {epoch}: loss={avg['total_val']:.4f} "
                f"sce={avg['sce']:.4f} (fwd={avg['fwd_ce']:.4f} rev={avg['rev_ce']:.4f}) "
                f"im={avg['im']:.4f} ema={avg['consist']:.4f}{teacher_str} "
                f"masked={avg['mask_pct']:.1f}% "
                f"lr={scheduler.get_last_lr()[0]:.6f} ({elapsed:.1f}s)"
            )

        # Evaluate every N epochs — rank 0 only
        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            if _is_main(rank):
                # Use the unwrapped model for eval
                raw_model = model.module if use_ddp else model
                pan = evaluate_panoptic(
                    raw_model, val_ds, device, mode="standard"
                )
                log.info(
                    f"  PQ={pan['PQ']:.2f} PQ_st={pan['PQ_stuff']:.2f} "
                    f"PQ_th={pan['PQ_things']:.2f} SQ={pan['SQ']:.2f} "
                    f"RQ={pan['RQ']:.2f} mIoU={pan['mIoU']:.2f} "
                    f"changed={pan['changed_pct']:.2f}%"
                )

                # Save best (use PQ as primary metric)
                if pan["PQ"] > best_pq:
                    best_pq = pan["PQ"]
                    best_epoch = epoch
                    ckpt_path = out_dir / "best.pth"
                    raw_sd = raw_model.state_dict()
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": raw_sd,
                        "panoptic": pan,
                        "args": vars(args),
                    }, str(ckpt_path))
                    log.info(
                        f"  Saved best model: {ckpt_path} (PQ={pan['PQ']:.2f}%)"
                    )

            # Sync all ranks before next epoch
            if use_ddp:
                dist.barrier()

    if _is_main(rank):
        log.info(
            f"Training complete. Best PQ: {best_pq:.2f}% at epoch {best_epoch}"
        )

        # Final evaluation with TTT (rank 0 only)
        if best_pq > 0:
            log.info("--- Final evaluation with explicit TTT ---")
            raw_model = model.module if use_ddp else model
            ckpt = torch.load(str(out_dir / "best.pth"), map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])

            for ttt_steps in [1, 3, 5]:
                pan = evaluate_panoptic(
                    raw_model, val_ds, device, mode="ttt",
                    ttt_steps=ttt_steps, ttt_lr=args.ttt_lr,
                )
                log.info(
                    f"  TTT steps={ttt_steps}: PQ={pan['PQ']:.2f} "
                    f"PQ_st={pan['PQ_stuff']:.2f} PQ_th={pan['PQ_things']:.2f} "
                    f"mIoU={pan['mIoU']:.2f}"
                )

    _cleanup_ddp()


def evaluate_only(args):
    """Evaluate a trained model."""
    device = _get_device(args.device)
    log.info(f"Device: {device}")

    val_ds = TTTDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        feature_subdir=args.feature_subdir,
        depth_subdir=args.depth_subdir,
        num_classes=args.num_classes,
    )

    model = TTTMamba2Refiner(
        num_classes=args.num_classes,
        bridge_dim=args.bridge_dim,
        num_blocks=args.num_blocks,
        scan_mode=args.scan_mode,
        layer_type=args.layer_type,
        d_state=args.d_state,
        gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    # Standard forward — full panoptic metrics
    pan = evaluate_panoptic(model, val_ds, device, mode="standard")
    log.info(
        f"Standard: PQ={pan['PQ']:.2f} PQ_st={pan['PQ_stuff']:.2f} "
        f"PQ_th={pan['PQ_things']:.2f} SQ={pan['SQ']:.2f} RQ={pan['RQ']:.2f} "
        f"mIoU={pan['mIoU']:.2f} changed={pan['changed_pct']:.2f}%"
    )
    for name, vals in pan["per_class"].items():
        log.info(f"  {name:15s}: PQ={vals['PQ']:5.1f}  IoU={vals['IoU']:5.1f}")

    # TTT adaptation
    for steps in [1, 3, 5]:
        pan = evaluate_panoptic(
            model, val_ds, device, mode="ttt",
            ttt_steps=steps, ttt_lr=args.ttt_lr,
        )
        log.info(
            f"TTT steps={steps}: PQ={pan['PQ']:.2f} PQ_st={pan['PQ_stuff']:.2f} "
            f"PQ_th={pan['PQ_things']:.2f} mIoU={pan['mIoU']:.2f}"
        )


def generate_labels(args):
    """Generate refined pseudo-labels for the full dataset."""
    device = _get_device(args.device)

    ds = TTTDataset(
        args.cityscapes_root, split=args.gen_split,
        semantic_subdir=args.semantic_subdir,
        feature_subdir=args.feature_subdir,
        depth_subdir=args.depth_subdir,
        num_classes=args.num_classes,
    )

    model = TTTMamba2Refiner(
        num_classes=args.num_classes,
        bridge_dim=args.bridge_dim,
        num_blocks=args.num_blocks,
        scan_mode=args.scan_mode,
        layer_type=args.layer_type,
        d_state=args.d_state,
        gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    out_dir = Path(args.cityscapes_root) / args.output_subdir / args.gen_split
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(loader, desc="Generating"):
        stem = batch["stem"][0]
        city = batch["city"][0]

        feats = batch["dinov2_features"].to(device)
        depth = batch["depth"].to(device)
        depth_grads = batch["depth_grads"].to(device)

        if args.ttt_steps > 0:
            logits = model.ttt_adapt(
                feats, depth, depth_grads,
                ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
            )
        else:
            with torch.no_grad():
                logits = model(feats, depth, depth_grads)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Upsample to full resolution (1024, 2048)
        pred_full = np.array(
            Image.fromarray(pred).resize((2048, 1024), Image.NEAREST)
        )

        # Save
        city_dir = out_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred_full).save(str(city_dir / f"{stem}.png"))

    log.info(f"Generated {len(ds)} refined labels in {out_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (rank, world_size, device)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1, None  # not DDP
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank):
    return rank == 0


def _get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(
        description="Train CRF-TTT-Mamba2 pseudo-label refiner"
    )

    # Data
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_overclustered_k300_nocrf")
    parser.add_argument("--feature_subdir", type=str,
                        default="dinov2_features")
    parser.add_argument("--depth_subdir", type=str,
                        default="depth_spidepth")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of semantic classes (19=trainID, 27=CAUSE)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit training images (0=all)")

    # Model
    parser.add_argument("--bridge_dim", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--scan_mode", type=str, default="cross_scan",
                        choices=["raster", "bidirectional", "cross_scan"])
    parser.add_argument("--layer_type", type=str, default="mamba2",
                        choices=["mamba2", "gated_delta_net"])
    parser.add_argument("--context_dropout", type=float, default=0.2)
    parser.add_argument("--d_state", type=int, default=64)

    # Training
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective_batch = batch_size * accum * world_size)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=5)

    # Loss weights (NoiseRobustLoss)
    parser.add_argument("--sce_alpha", type=float, default=1.0,
                        help="Forward CE weight in SCE (Wang et al., ICCV 2019)")
    parser.add_argument("--sce_beta", type=float, default=1.0,
                        help="Reverse CE weight in SCE (noise tolerance)")
    parser.add_argument("--im_weight", type=float, default=0.01,
                        help="Information Maximization weight (SHOT, ICML 2020)")
    parser.add_argument("--consist_weight", type=float, default=1.0,
                        help="EMA teacher-student consistency weight")
    parser.add_argument("--mask_threshold", type=int, default=6,
                        help="Min neighbor agreement for confidence mask (0-8)")

    # EMA Teacher (Mean Teacher)
    parser.add_argument("--ema_alpha", type=float, default=0.999,
                        help="EMA decay rate (0.999 = slow update, stable teacher)")
    parser.add_argument("--teacher_temp", type=float, default=0.5,
                        help="Teacher sharpening temperature (<1 = sharper)")
    parser.add_argument("--teacher_conf", type=float, default=0.7,
                        help="Min teacher confidence to enforce consistency")
    parser.add_argument("--ema_warmup_epochs", type=int, default=2,
                        help="Epochs of pure SCE before enabling EMA teacher")
    parser.add_argument("--feat_drop_rate", type=float, default=0.5,
                        help="Feature dropout rate for student (0=disabled, 0.5=drop 50%% channels)")

    # TTT
    parser.add_argument("--ttt_lr", type=float, default=0.01)
    parser.add_argument("--ttt_steps", type=int, default=3)

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/ttt_mamba2")
    parser.add_argument("--device", type=str, default="auto")

    # Modes
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--gen_split", type=str, default="val")
    parser.add_argument("--output_subdir", type=str,
                        default="pseudo_semantic_ttt_mamba2")
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    # Suppress logging on non-main DDP ranks
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank > 0:
        logging.getLogger().setLevel(logging.WARNING)

    if args.eval_only:
        assert args.checkpoint, "--checkpoint required for --eval_only"
        evaluate_only(args)
    elif args.generate:
        assert args.checkpoint, "--checkpoint required for --generate"
        generate_labels(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
