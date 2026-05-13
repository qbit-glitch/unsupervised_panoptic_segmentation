#!/usr/bin/env python3
"""CAUSE-TR unsupervised semantic segmentation for COCO-Stuff-27.

Trains a Transformer segmentation head (TRDecoder) on frozen DINOv3 ViT-L/16
features using CAUSE's modularity codebook + contrastive learning.

Key differences from failed feature-mode (4.4% mIoU):
  - Live backbone inference with REAL image augmentation
  - Larger batch size (16 vs 4)
  - Full COCO train2017 (118K images vs 5K val)
  - Overclustering (K=80 vs K=27)
  - Entropy regularization to prevent dead centroids

Usage:
    # Stage 2: Train modularity codebook
    python cause_tr_coco.py --stage codebook \
        --coco_root /path/to/coco --device cuda:0

    # Stage 3: Train TRDecoder + cluster probe
    python cause_tr_coco.py --stage heads \
        --coco_root /path/to/coco --device cuda:0

    # All stages
    python cause_tr_coco.py --stage all \
        --coco_root /path/to/coco --device cuda:0

    # Evaluate only
    python cause_tr_coco.py --stage eval \
        --coco_root /path/to/coco --checkpoint_dir checkpoints/cause_tr_k80

References:
    CAUSE (Pattern Recognition 2024): arXiv 2310.07379
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── COCO-Stuff-27 constants ───

COCOSTUFF27_CLASSNAMES = [
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]
NUM_CLASSES = 27
THING_IDS = set(range(12))
STUFF_IDS = set(range(12, 27))

SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


# ═══════════════════════════════════════════════════════════════════════
# DINOv3 Backbone
# ═══════════════════════════════════════════════════════════════════════

class DINOv3Backbone(nn.Module):
    """Frozen DINOv3 ViT-L/16 backbone for feature extraction.

    Returns patch tokens only (CLS + register tokens stripped).
    """

    MODEL_ID = "facebook/dinov3-large"
    NUM_REGISTER_TOKENS = 4

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        from transformers import AutoModel

        self.device = torch.device(device)
        logger.info("Loading DINOv3 ViT-L/16 from %s...", self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.dim = self.model.config.hidden_size  # 1024
        self.patch_size = self.model.config.patch_size  # 16
        logger.info(
            "DINOv3 loaded: dim=%d, patch_size=%d", self.dim, self.patch_size
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from images.

        Args:
            images: (B, 3, H, W) tensor, normalized for DINOv3.

        Returns:
            (B, N, D) patch tokens where N = (H/16)*(W/16).
        """
        out = self.model(pixel_values=images)
        tokens = out.last_hidden_state  # (B, 1+reg+N, D)
        # Strip CLS token and register tokens
        patch_tokens = tokens[:, 1 + self.NUM_REGISTER_TOKENS :, :]
        return patch_tokens


# ═══════════════════════════════════════════════════════════════════════
# CAUSE-TR Model Components
# ═══════════════════════════════════════════════════════════════════════

def _transform(x: torch.Tensor) -> torch.Tensor:
    """(B, P, D) → (B, D, sqrt(P), sqrt(P))"""
    B, P, D = x.shape
    s = int(math.sqrt(P))
    return x.permute(0, 2, 1).view(B, D, s, s)


def _untransform(x: torch.Tensor) -> torch.Tensor:
    """(B, D, H, W) → (B, H*W, D)"""
    B, D, H, W = x.shape
    return x.view(B, D, -1).permute(0, 2, 1)


class TRDecoder(nn.Module):
    """Transformer Decoder head from CAUSE-TR."""

    def __init__(
        self,
        dim: int = 1024,
        reduced_dim: int = 90,
        hidden_dim: int = 2048,
        nhead: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.f1 = nn.Conv2d(dim, reduced_dim, (1, 1))
        self.f2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(dim, reduced_dim, (1, 1)),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor,
        drop: nn.Module = nn.Identity(),
    ) -> torch.Tensor:
        q = k = tgt + pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + pos, key=memory, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = memory + self.norm3(tgt)
        tgt = _transform(tgt.transpose(0, 1))
        tgt = self.f1(drop(tgt)) + self.f2(drop(tgt))
        tgt = _untransform(tgt)
        return tgt


class CAUSETRModel(nn.Module):
    """Full CAUSE-TR model with TRDecoder + codebook + cluster probe."""

    def __init__(
        self,
        dim: int = 1024,
        reduced_dim: int = 90,
        projection_dim: int = 2048,
        num_codebook: int = 2048,
        n_classes: int = 80,
        num_queries: int = 784,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduced_dim = reduced_dim
        self.projection_dim = projection_dim
        self.num_codebook = num_codebook
        self.n_classes = n_classes

        # Codebook
        self.codebook = nn.Parameter(torch.empty(num_codebook, dim))
        nn.init.uniform_(self.codebook, -1.0 / num_codebook, 1.0 / num_codebook)

        # Cluster probe
        self.cluster_probe = nn.Parameter(torch.randn(n_classes, reduced_dim))
        nn.init.uniform_(
            self.cluster_probe, -1.0 / n_classes, 1.0 / n_classes
        )

        # Query positional embedding
        self.query_pos = nn.Parameter(torch.randn(num_queries, dim))

        # Student head
        self.head = TRDecoder(dim, reduced_dim)
        self.projection_head = nn.Conv2d(reduced_dim, projection_dim, 1)
        self.linear = nn.Conv2d(reduced_dim, n_classes, 1)

        # EMA teacher head
        self.head_ema = TRDecoder(dim, reduced_dim)
        self.projection_head_ema = nn.Conv2d(reduced_dim, projection_dim, 1)

        self.dropout = nn.Dropout(0.1)

        # Bank for contrastive learning
        self.prime_bank: Dict[int, torch.Tensor] = {}

    def init_ema(self) -> None:
        """Copy student → teacher (EMA init)."""
        for sp, tp in zip(self.head.parameters(), self.head_ema.parameters()):
            tp.data.copy_(sp.data)
            tp.requires_grad = False
        for sp, tp in zip(
            self.projection_head.parameters(),
            self.projection_head_ema.parameters(),
        ):
            tp.data.copy_(sp.data)
            tp.requires_grad = False

    def update_ema(self, lamb: float = 0.99) -> None:
        """EMA update: teacher ← λ*teacher + (1-λ)*student."""
        for sp, tp in zip(self.head.parameters(), self.head_ema.parameters()):
            tp.data = lamb * tp.data + (1 - lamb) * sp.data
        for sp, tp in zip(
            self.projection_head.parameters(),
            self.projection_head_ema.parameters(),
        ):
            tp.data = lamb * tp.data + (1 - lamb) * sp.data

    def vqt(self, z: torch.Tensor) -> torch.Tensor:
        """Vector quantize tensor z using codebook."""
        z_flat = z.contiguous().view(-1, z.shape[-1])
        norm_z = F.normalize(z_flat, dim=1)
        norm_c = F.normalize(self.codebook, dim=1)
        dist = norm_z @ norm_c.T
        idx = dist.argmax(dim=1)
        return self.codebook[idx].view(*z.shape[:-1], self.codebook.shape[1])

    def forward_student(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Student forward: feat → reduced features + projection."""
        discrete_query = self.vqt(feat)
        tr_feat = self.head(
            discrete_query.transpose(0, 1),
            feat.transpose(0, 1),
            self.query_pos[: feat.shape[1]].unsqueeze(1),
            self.dropout,
        )
        proj_feat = _untransform(self.projection_head(_transform(tr_feat)))
        return tr_feat, proj_feat

    def forward_teacher(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher forward (no gradients)."""
        with torch.no_grad():
            discrete_query = self.vqt(feat)
            tr_feat = self.head_ema(
                discrete_query.transpose(0, 1),
                feat.transpose(0, 1),
                self.query_pos[: feat.shape[1]].unsqueeze(1),
            )
            proj_feat = _untransform(
                self.projection_head_ema(_transform(tr_feat))
            )
        return tr_feat, proj_feat

    def forward_cluster(
        self, feat: torch.Tensor, alpha: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster assignment via cosine similarity with probe centroids.

        Returns:
            (cluster_loss, predictions): loss scalar and (B, H*W) predictions.
        """
        normed_feat = F.normalize(_transform(feat.detach()), dim=1)
        normed_clusters = F.normalize(self.cluster_probe, dim=1)
        inner = torch.einsum("bchw,nc->bnhw", normed_feat, normed_clusters)

        preds = inner.argmax(dim=1)  # (B, H, W)
        probs = F.one_hot(preds, self.n_classes).permute(0, 3, 1, 2).float()
        cluster_loss = -(probs * inner).sum(1).mean()

        return cluster_loss, preds.reshape(preds.shape[0], -1)

    def forward_cluster_inference(self, feat: torch.Tensor) -> torch.Tensor:
        """Cluster prediction at inference time."""
        normed_feat = F.normalize(_transform(feat), dim=1)
        normed_clusters = F.normalize(self.cluster_probe, dim=1)
        inner = torch.einsum("bchw,nc->bnhw", normed_feat, normed_clusters)
        return inner.argmax(dim=1)  # (B, H, W)


# ═══════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════

def modularity_loss(
    codebook: torch.Tensor, feat: torch.Tensor, temp: float = 0.1
) -> torch.Tensor:
    """CAUSE modularity-based codebook loss."""
    feat = feat.detach()
    norm = F.normalize(feat, dim=2)
    A = (norm @ norm.transpose(2, 1)).clamp(0)
    A = A - A * torch.eye(A.shape[1], device=A.device)
    d = A.sum(dim=2, keepdims=True)
    e = A.sum(dim=(1, 2), keepdims=True)
    W = A - (d / (e + 1e-8)) @ (d.transpose(2, 1) / (e + 1e-8)) * e

    # Cluster assignment
    z_flat = feat.contiguous().view(-1, feat.shape[-1])
    norm_z = F.normalize(z_flat, dim=1)
    norm_c = F.normalize(codebook, dim=1)
    C = (norm_z @ norm_c.T).clamp(0).view(*feat.shape[:-1], -1)

    D = C.transpose(2, 1)
    E = torch.tanh(D.unsqueeze(3) @ D.unsqueeze(2) / temp)
    delta, _ = E.max(dim=1)
    Q = (W / (e + 1e-8)) @ delta

    trace = Q.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    return -trace.mean()


def contrastive_loss(
    model: CAUSETRModel,
    feat: torch.Tensor,
    proj_feat: torch.Tensor,
    proj_feat_ema: torch.Tensor,
    temp: float = 0.07,
    pos_thresh: float = 0.3,
    neg_thresh: float = 0.1,
) -> torch.Tensor:
    """CAUSE contrastive NCE loss with codebook bank."""
    vq_feat = model.vqt(feat)
    norm_vq = F.normalize(vq_feat, dim=2)
    flat_norm_vq = norm_vq.contiguous().view(-1, norm_vq.shape[-1])

    norm_proj = F.normalize(proj_feat, dim=2)
    norm_proj_ema = F.normalize(proj_feat_ema, dim=2)
    flat_norm_proj_ema = norm_proj_ema.contiguous().view(-1, norm_proj_ema.shape[-1])

    losses = []
    for b in range(proj_feat.shape[0]):
        anchor_vq = norm_vq[b]
        anchor_proj = norm_proj[b]

        cs_st = anchor_proj @ flat_norm_proj_ema.T
        codebook_dist = anchor_vq @ flat_norm_vq.T

        pos_mask = codebook_dist > pos_thresh
        neg_mask = codebook_dist < neg_thresh

        # Auto-mask: exclude self-pairs
        auto_mask = torch.ones_like(pos_mask)
        n = pos_mask.shape[0]
        auto_mask[:, b * n : (b + 1) * n].fill_diagonal_(0)
        pos_mask = pos_mask * auto_mask

        cs = cs_st / temp
        cs_shifted = cs - cs.max(dim=1, keepdim=True)[0].detach()
        denom = (cs_shifted.exp() * (pos_mask + neg_mask)).sum(dim=1, keepdim=True)
        nce = -cs_shifted + torch.log(denom + 1e-10)

        pos_indices = pos_mask.nonzero(as_tuple=False)
        if len(pos_indices) > 0:
            losses.append(nce[pos_indices[:, 0], pos_indices[:, 1]].mean())

    if not losses:
        return torch.tensor(0.0, device=feat.device, requires_grad=True)
    return sum(losses) / len(losses)


def entropy_regularization(cluster_probs: torch.Tensor) -> torch.Tensor:
    """Maximize entropy of average cluster assignment to prevent collapse."""
    avg_probs = cluster_probs.float().mean(dim=(0, 2, 3))  # (K,)
    avg_probs = avg_probs / (avg_probs.sum() + 1e-10)
    log_probs = torch.log(avg_probs + 1e-10)
    return (avg_probs * log_probs).sum()  # Negative = minimize


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class COCOImageDataset(Dataset):
    """COCO dataset with real image augmentation for CAUSE training."""

    DINOV3_MEAN = [0.485, 0.456, 0.406]
    DINOV3_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        coco_root: str,
        split: str = "train2017",
        resolution: int = 448,
        augment: bool = True,
    ) -> None:
        img_dir = Path(coco_root) / split
        self.img_files = sorted(img_dir.glob("*.jpg"))
        logger.info("COCOImageDataset: %d images from %s", len(self.img_files), img_dir)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(resolution, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.DINOV3_MEAN, self.DINOV3_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(self.DINOV3_MEAN, self.DINOV3_STD),
            ])

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.img_files[idx]).convert("RGB")
        img_id = int(self.img_files[idx].stem)
        return self.transform(img), img_id


# ═══════════════════════════════════════════════════════════════════════
# GT Loading and Evaluation
# ═══════════════════════════════════════════════════════════════════════

def load_coco_panoptic_gt(coco_root: str, image_id: int) -> Optional[np.ndarray]:
    """Load COCO panoptic GT → 27-class semantic label map."""
    panoptic_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
    panoptic_dir = Path(coco_root) / "annotations" / "panoptic_val2017"

    if not hasattr(load_coco_panoptic_gt, "_cache"):
        with open(panoptic_json) as f:
            data = json.load(f)
        cat_map = {c["id"]: c["supercategory"] for c in data["categories"]}
        ann_map = {a["image_id"]: a for a in data["annotations"]}
        load_coco_panoptic_gt._cache = (cat_map, ann_map, str(panoptic_dir))

    cat_map, ann_map, pdir = load_coco_panoptic_gt._cache
    if image_id not in ann_map:
        return None

    ann = ann_map[image_id]
    pan_img = np.array(Image.open(Path(pdir) / ann["file_name"]))
    pan_id = (
        pan_img[:, :, 0].astype(np.int32)
        + pan_img[:, :, 1].astype(np.int32) * 256
        + pan_img[:, :, 2].astype(np.int32) * 256 * 256
    )
    sem_label = np.full(pan_id.shape, 255, dtype=np.uint8)
    for seg in ann["segments_info"]:
        supercat = cat_map.get(seg["category_id"])
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            sem_label[pan_id == seg["id"]] = SUPERCATEGORY_TO_COARSE[supercat]
    return sem_label


def evaluate_model(
    backbone: DINOv3Backbone,
    model: CAUSETRModel,
    coco_root: str,
    device: str = "cuda",
    resolution: int = 448,
    n_images: Optional[int] = None,
) -> Dict:
    """Evaluate CAUSE-TR with per-image Hungarian matching."""
    model.eval()
    val_dataset = COCOImageDataset(
        coco_root, split="val2017", resolution=resolution, augment=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    all_preds = []
    all_gts = []
    all_ids = []

    with torch.no_grad():
        for img_t, img_id in tqdm(val_loader, desc="Evaluating"):
            img_id = img_id.item()
            gt = load_coco_panoptic_gt(coco_root, img_id)
            if gt is None:
                continue

            img_t = img_t.to(device)
            feat = backbone(img_t)
            pred = model.forward_cluster_inference(feat)  # (1, H, W)
            pred_np = pred[0].cpu().numpy()

            # Resize to GT resolution
            pred_resized = np.array(
                Image.fromarray(pred_np.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                )
            )
            all_preds.append(pred_resized)
            all_gts.append(gt)
            all_ids.append(img_id)

            if n_images and len(all_preds) >= n_images:
                break

    # Global Hungarian matching
    n_clusters = model.n_classes
    cost_matrix = np.zeros((n_clusters, NUM_CLASSES))
    for pred, gt in zip(all_preds, all_gts):
        valid = gt != 255
        for cluster_id in range(n_clusters):
            for gt_class in range(NUM_CLASSES):
                cost_matrix[cluster_id, gt_class] += (
                    (pred[valid] == cluster_id) & (gt[valid] == gt_class)
                ).sum()

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c
    for k in range(n_clusters):
        if k not in cluster_to_class and cost_matrix[k].sum() > 0:
            cluster_to_class[k] = cost_matrix[k].argmax()

    # Compute mIoU
    iou_per_class = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for pred, gt in zip(all_preds, all_gts):
        mapped = np.full_like(pred, 255)
        for p_id, c_id in cluster_to_class.items():
            mapped[pred == p_id] = c_id

        valid = gt != 255
        for c in range(NUM_CLASSES):
            gt_mask = (gt == c) & valid
            pred_mask = (mapped == c) & valid
            inter = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if union > 0:
                iou_per_class[c] += inter / union
                count_per_class[c] += 1

    valid_cls = count_per_class > 0
    per_class_avg = np.zeros(NUM_CLASSES)
    per_class_avg[valid_cls] = iou_per_class[valid_cls] / count_per_class[valid_cls]
    miou = per_class_avg[valid_cls].mean() * 100

    things = [per_class_avg[c] * 100 for c in THING_IDS if count_per_class[c] > 0]
    stuff = [per_class_avg[c] * 100 for c in STUFF_IDS if count_per_class[c] > 0]

    result = {
        "miou": round(miou, 2),
        "things_miou": round(np.mean(things), 2) if things else 0.0,
        "stuff_miou": round(np.mean(stuff), 2) if stuff else 0.0,
        "n_images": len(all_preds),
        "n_clusters": n_clusters,
        "active_clusters": len(cluster_to_class),
        "per_class_iou": {
            COCOSTUFF27_CLASSNAMES[c]: round(per_class_avg[c] * 100, 1)
            for c in range(NUM_CLASSES) if count_per_class[c] > 0
        },
    }

    logger.info(
        "Eval: mIoU=%.2f%%, Things=%.2f%%, Stuff=%.2f%%, "
        "active_clusters=%d/%d",
        miou, result["things_miou"], result["stuff_miou"],
        result["active_clusters"], n_clusters,
    )
    model.train()
    return result


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_codebook(
    backbone: DINOv3Backbone,
    model: CAUSETRModel,
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 1e-3,
) -> None:
    """Stage 2: Train modularity-based codebook."""
    logger.info("=== Stage 2: Codebook Training (%d epochs) ===", epochs)

    optimizer = torch.optim.Adam([model.codebook], lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for images, _ in tqdm(train_loader, desc=f"Codebook ep{epoch+1}"):
            images = images.to(device)
            with torch.no_grad():
                feat = backbone(images)  # (B, N, 1024)

            loss = modularity_loss(model.codebook, feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Codebook ep%d: loss=%.4f", epoch + 1, avg_loss)


def train_heads(
    backbone: DINOv3Backbone,
    model: CAUSETRModel,
    train_loader: DataLoader,
    coco_root: str,
    device: str = "cuda",
    epochs: int = 40,
    lr: float = 5e-5,
    lambda_entropy: float = 2.0,
    eval_every: int = 5,
    checkpoint_dir: Optional[str] = None,
    n_eval_images: Optional[int] = 200,
) -> None:
    """Stage 3: Train TRDecoder + projection + cluster probe."""
    logger.info("=== Stage 3: Head Training (%d epochs) ===", epochs)

    # Initialize EMA
    model.init_ema()

    # Trainable parameters: head, projection_head, linear, cluster_probe
    trainable = list(model.head.parameters()) + \
                list(model.projection_head.parameters()) + \
                list(model.linear.parameters()) + \
                [model.cluster_probe]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    best_miou = 0.0

    for epoch in range(epochs):
        total_contrastive = 0.0
        total_cluster = 0.0
        total_entropy = 0.0
        n_batches = 0

        for images, _ in tqdm(train_loader, desc=f"Heads ep{epoch+1}"):
            images = images.to(device)

            with torch.no_grad():
                feat = backbone(images)

            # Student forward
            tr_feat, proj_feat = model.forward_student(feat)
            # Teacher forward
            tr_feat_ema, proj_feat_ema = model.forward_teacher(feat)

            # Contrastive loss
            loss_nce = contrastive_loss(
                model, feat, proj_feat, proj_feat_ema
            )

            # Cluster loss
            loss_cluster, preds = model.forward_cluster(tr_feat_ema)

            # Entropy regularization
            pred_onehot = F.one_hot(
                preds.reshape(-1, int(math.sqrt(preds.shape[1])),
                              int(math.sqrt(preds.shape[1]))),
                model.n_classes
            ).permute(0, 3, 1, 2).float()
            loss_entropy = entropy_regularization(pred_onehot)

            loss = loss_nce + loss_cluster + lambda_entropy * loss_entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # EMA update
            model.update_ema(lamb=0.99)

            total_contrastive += loss_nce.item()
            total_cluster += loss_cluster.item()
            total_entropy += loss_entropy.item()
            n_batches += 1

        avg_nce = total_contrastive / max(n_batches, 1)
        avg_cls = total_cluster / max(n_batches, 1)
        avg_ent = total_entropy / max(n_batches, 1)
        logger.info(
            "Heads ep%d: nce=%.4f, cluster=%.4f, entropy=%.4f",
            epoch + 1, avg_nce, avg_cls, avg_ent,
        )

        # Evaluate periodically
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            result = evaluate_model(
                backbone, model, coco_root, device,
                n_images=n_eval_images,
            )

            if checkpoint_dir and result["miou"] > best_miou:
                best_miou = result["miou"]
                ckpt_path = Path(checkpoint_dir) / "best.pth"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "miou": best_miou,
                }, str(ckpt_path))
                logger.info("Saved best checkpoint: mIoU=%.2f%%", best_miou)

    # Save final checkpoint
    if checkpoint_dir:
        final_path = Path(checkpoint_dir) / "final.pth"
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "miou": best_miou,
        }, str(final_path))
        logger.info("Saved final checkpoint: %s", final_path)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAUSE-TR for COCO-Stuff-27")
    p.add_argument("--stage", choices=["codebook", "heads", "all", "eval"], default="all")
    p.add_argument("--coco_root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_classes", type=int, default=80, help="Cluster probe size (overclustering)")
    p.add_argument("--num_codebook", type=int, default=2048)
    p.add_argument("--reduced_dim", type=int, default=90)
    p.add_argument("--codebook_epochs", type=int, default=10)
    p.add_argument("--codebook_lr", type=float, default=1e-3)
    p.add_argument("--head_epochs", type=int, default=40)
    p.add_argument("--head_lr", type=float, default=5e-5)
    p.add_argument("--lambda_entropy", type=float, default=2.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--n_eval_images", type=int, default=200)
    p.add_argument("--train_split", default="train2017")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    # Compute number of patches
    num_patches = (args.resolution // 16) ** 2  # 448/16 = 28 → 784 patches

    # Initialize backbone
    backbone = DINOv3Backbone(device=device)

    # Initialize model
    model = CAUSETRModel(
        dim=backbone.dim,
        reduced_dim=args.reduced_dim,
        num_codebook=args.num_codebook,
        n_classes=args.n_classes,
        num_queries=num_patches,
    ).to(device)

    # Resume from checkpoint
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Resumed at epoch %d, mIoU=%.2f%%", ckpt["epoch"], ckpt.get("miou", 0))

    # Default checkpoint dir
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(
            Path(args.coco_root) / f"checkpoints/cause_tr_k{args.n_classes}"
        )

    if args.stage == "eval":
        result = evaluate_model(
            backbone, model, args.coco_root, device,
            resolution=args.resolution,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        return

    # Training dataset
    train_dataset = COCOImageDataset(
        args.coco_root,
        split=args.train_split,
        resolution=args.resolution,
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.stage in ("codebook", "all"):
        train_codebook(
            backbone, model, train_loader, device,
            epochs=args.codebook_epochs, lr=args.codebook_lr,
        )

    if args.stage in ("heads", "all"):
        train_heads(
            backbone, model, train_loader,
            coco_root=args.coco_root, device=device,
            epochs=args.head_epochs, lr=args.head_lr,
            lambda_entropy=args.lambda_entropy,
            eval_every=args.eval_every,
            checkpoint_dir=args.checkpoint_dir,
            n_eval_images=args.n_eval_images,
        )

    # Final full evaluation
    logger.info("=== Final Evaluation ===")
    result = evaluate_model(
        backbone, model, args.coco_root, device,
        resolution=args.resolution,
    )

    # Save results
    out_path = args.output or str(
        Path(args.coco_root) / f"cause_tr_k{args.n_classes}_results.json"
    )
    result["config"] = {
        "n_classes": args.n_classes,
        "num_codebook": args.num_codebook,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "codebook_epochs": args.codebook_epochs,
        "head_epochs": args.head_epochs,
        "head_lr": args.head_lr,
        "lambda_entropy": args.lambda_entropy,
        "train_split": args.train_split,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
