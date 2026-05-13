#!/usr/bin/env python3
"""
Train CAUSE-TR heads on DINOv3 features for Cityscapes.

Full 4-stage pipeline:
  Stage 0: HP backbone fine-tuning — full params, 30 epochs (exact CVPR 2023 protocol)
  Stage 1: HP-style reference pool warm-up (~5 epochs) — prevents cluster collapse
  Stage 2: Modularity-based codebook learning (~10 epochs)
  Stage 3: TRDecoder segment head + Cluster probe training (~60 epochs)

Supports both DINOv3 ViT-B/16 (768-dim) and DINOv3 ViT-L/16 (1024-dim).
Default: ViT-B/16 — architecturally matches original CAUSE-TR (DINOv2 ViT-B/14, 768-dim)
and trains 3.5x faster. Use --model_name to switch to ViT-L.

Original CAUSE-TR used DINOv2 ViT-B/14 (768-dim, patch 14).
DINOv3 ViT-B/16 uses same embed_dim=768, patch_size=16 (minor difference).

Usage:
    python mbps_pytorch/train_cause_dinov3.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir refs/cause/CAUSE_dinov3 \
        --device mps

    # Stage 2 only (codebook):
    python mbps_pytorch/train_cause_dinov3.py --stage codebook ...

    # Stage 3 only (heads, requires codebook from stage 2):
    python mbps_pytorch/train_cause_dinov3.py --stage heads ...
"""

import argparse
import copy
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
TRAIN_RESOLUTION = 448  # 448px — divisible by 16; 28×28=784 patches (v7, was 320/400 in v5)
PATCH_SIZE = 16
NUM_PATCHES_PER_SIDE = TRAIN_RESOLUTION // PATCH_SIZE  # 28
NUM_PATCHES = NUM_PATCHES_PER_SIDE ** 2  # 784
DIM = 768   # Set dynamically in main() from backbone.embed_dim; default matches ViT-B/16
REDUCED_DIM = 90
PROJECTION_DIM = 1536  # DIM * 2; updated in main() alongside DIM
NUM_CODEBOOK = 512
N_CLASSES = 27  # Cityscapes (CAUSE format: labelIDs 7-33)
N_CLASSES_CLUSTER = 54  # Overclustering for Stage 3 cluster_probe (K=54 → Hungarian → 27)
NUM_REGISTER_TOKENS = 4  # DINOv3 register tokens

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# HP config — exact refs/hp/json/server/cityscapes.json
HP_POOL_SIZE        = 2048               # reference pool entries (1 random patch per image)
HP_TEMPERATURE      = 0.6                # tau from official Cityscapes config (NOT 0.07)
HP_BASE_TEMPERATURE = 0.07               # base temperature (official SupConLoss default)
HP_LAMBDA_MAX       = 1.0                # final weight for LHP mask
HP_WARMUP_ITERS     = 50                 # iters before lmbd starts increasing from 0
HP_EMA_DECAY        = 0.999              # EMA decay for head_ema + projection_head_ema
HP_RENEW_EVERY      = 20                 # Pool_sp circular-buffer update every N iters
HP_RHO              = 0.163              # 512 target negatives / (bs4 × 784 patches) = 512/3136 ≈ 0.163
                                          # Original HP: rho=0.02 × bs64 × 400 patches = 512 negs/anchor
HP_ALPHA            = 0.05               # consistency loss weight (opt["alpha"])
HP_GRAD_NORM        = 10.0               # gradient clip norm for backbone (opt["train"]["grad_norm"])
HP_SPATIAL_SIZE     = NUM_PATCHES_PER_SIDE   # 28 (v7: 448px) — split = 28×28 = 784 patches/image
HP_SPLIT            = HP_SPATIAL_SIZE ** 2   # 784


class HPHiddenPositiveLoss(nn.Module):
    """Exact HP (Hidden Positives, CVPR 2023) SupConLoss protocol.

    Faithfully reproduces refs/hp/loss.py SupConLoss with loss_version=1,
    reweighting=1, rho=HP_RHO.  Adapted for MPS:
      - .cuda() / float16 / autocast replaced by .to(device) / float32
      - log(0) in denominator guarded with .clamp(min=1e-9) instead of
        producing -inf, which causes NaN on MPS log_softmax.

    Inputs (all L2-normed, float32, on device):
      modeloutput_z:     (B*N, proj_dim) — student projection head output
      modeloutput_z_mix: (B*N, proj_dim) — teacher (EMA) projection head output
      modeloutput_f:     (B*N, feat_dim) — frozen backbone features
      modeloutput_s_pr:  (B*N, code_dim) — EMA TRDecoder head output
      Pool_ag:  (HP_POOL_SIZE, feat_dim) — backbone reference pool
      Pool_sp:  (HP_POOL_SIZE, code_dim) — EMA-code reference pool
      lmbd:     float  — LHP weight (0 → HP_LAMBDA_MAX during warmup)
    """

    def __init__(self, temperature: float = HP_TEMPERATURE,
                 base_temperature: float = HP_BASE_TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        modeloutput_z: torch.Tensor,
        modeloutput_z_mix: torch.Tensor,
        modeloutput_f: torch.Tensor,
        modeloutput_s_pr: torch.Tensor,
        Pool_ag: torch.Tensor,
        Pool_sp: torch.Tensor,
        lmbd: float,
    ) -> torch.Tensor:
        device = modeloutput_z.device
        batch_size = modeloutput_z.shape[0]          # B * HP_SPLIT
        split = HP_SPLIT                              # 400 patches per image
        mini_iters = batch_size // split              # B (number of images)

        # --- Negative mask: (split, batch_size), diagonal = 0 (exclude self-pairs) ---
        negative_mask_one = torch.scatter(
            torch.ones(split, batch_size, device=device),
            1,
            torch.arange(split, device=device).view(-1, 1),
            0.0,
        )  # (split, batch_size)

        # --- Neglect mask: keep rho fraction of negatives in denominator ---
        mask_neglect_base = (
            torch.FloatTensor(split, batch_size).uniform_().to(device) < HP_RHO
        )  # (split, batch_size)

        loss = torch.tensor(0.0, device=device)

        # --- Rpoint: each patch's max similarity to pool (threshold for positives) ---
        Rpoint     = (modeloutput_f    @ Pool_ag.T).max(dim=1).values   # (batch_size,)
        Rpoint_ema = (modeloutput_s_pr @ Pool_sp.T).max(dim=1).values   # (batch_size,)
        # shape (batch_size, split) — needed for symmetric mask (T-side comparison)
        Rpoint_T     = Rpoint.unsqueeze(-1).expand(-1, split)
        Rpoint_ema_T = Rpoint_ema.unsqueeze(-1).expand(-1, split)

        for mi in range(mini_iters):
            s = mi * split
            e = (mi + 1) * split

            # ---- GHP mask (backbone similarity) ----
            modeloutput_f_one     = modeloutput_f[s:e]              # (split, feat_dim)
            output_cossim_one     = modeloutput_f_one @ modeloutput_f.T  # (split, batch_size)
            output_cossim_one_T   = output_cossim_one.T             # (batch_size, split)

            # T-side: is sim(f_j, f_i) > Rp_j  (shape batch_size×split → transpose → split×batch_size)
            mask_one_T = (Rpoint_T < output_cossim_one_T).float().T   # (split, batch_size)
            Rpoint_one = Rpoint[s:e].unsqueeze(-1).expand(-1, batch_size)  # (split, batch_size)
            mask_one   = (Rpoint_one < output_cossim_one).float()    # (split, batch_size)
            # Symmetric OR: positive if EITHER direction exceeds its pool threshold
            mask_one   = torch.logical_or(mask_one.bool(), mask_one_T.bool()).float()

            # ---- LHP mask (EMA code similarity) ----
            modeloutput_s_pr_one   = modeloutput_s_pr[s:e]
            output_cossim_ema_one  = modeloutput_s_pr_one @ modeloutput_s_pr.T
            output_cossim_ema_one_T = output_cossim_ema_one.T
            mask_ema_one_T = (Rpoint_ema_T < output_cossim_ema_one_T).float().T
            Rpoint_ema_one = Rpoint_ema[s:e].unsqueeze(-1).expand(-1, batch_size)
            mask_ema_one   = (Rpoint_ema_one < output_cossim_ema_one).float()
            mask_ema_one   = torch.logical_or(mask_ema_one.bool(), mask_ema_one_T.bool()).float()
            mask_ema_one   = mask_ema_one * negative_mask_one        # exclude self

            # ---- Neglect mask for NCE denominator ----
            neglect_mask = torch.logical_or(
                mask_one.bool(), mask_neglect_base.bool()
            ).float()
            neglect_negative_mask_one = negative_mask_one * neglect_mask  # (split, batch_size)
            mask_one = mask_one * negative_mask_one   # exclude self-pairs from positive mask

            # ---- NCE on student projection (modeloutput_z) ----
            modeloutput_z_one = modeloutput_z[s:e]
            anchor_dot_one    = (modeloutput_z_one @ modeloutput_z.T) / self.temperature
            logits_max_one, _ = torch.max(anchor_dot_one, dim=1, keepdim=True)
            logits_one        = anchor_dot_one - logits_max_one.detach()
            exp_logits_one    = torch.exp(logits_one) * neglect_negative_mask_one
            # MPS fix: clamp denominator before log to avoid log(0) → NaN
            log_prob_one = logits_one - torch.log(
                exp_logits_one.sum(1, keepdim=True).clamp(min=1e-9)
            )

            # loss_version=1: combined GHP + lmbd*LHP, reweighting=1
            nonzero_idx = torch.where(mask_one.sum(1) != 0.0)[0]
            if len(nonzero_idx) > 0:
                mask_one_nz      = mask_one[nonzero_idx]
                log_prob_one_nz  = log_prob_one[nonzero_idx]
                mask_ema_one_nz  = mask_ema_one[nonzero_idx]
                weighted_mask    = mask_one_nz.detach() + mask_ema_one_nz.detach() * lmbd
                # reweighting=1: normalize by per-row positive count
                pnm = weighted_mask.sum(1).float()
                pnm = pnm / pnm.sum().clamp(min=1e-9)
                pnm = pnm / pnm.mean().clamp(min=1e-9)
                mean_log_prob_pos = (
                    (weighted_mask * log_prob_one_nz).sum(1)
                    / weighted_mask.sum(1).clamp(min=1.0)
                )
                loss = loss - torch.mean(
                    (self.temperature / self.base_temperature) * mean_log_prob_pos * pnm
                )

            # ---- NCE on teacher projection (modeloutput_z_mix) ----
            modeloutput_z_mix_one  = modeloutput_z_mix[s:e]
            anchor_dot_mix_one     = (modeloutput_z_mix_one @ modeloutput_z_mix.T) / self.temperature
            logits_max_mix_one, _  = torch.max(anchor_dot_mix_one, dim=1, keepdim=True)
            logits_mix_one         = anchor_dot_mix_one - logits_max_mix_one.detach()
            exp_logits_mix_one     = torch.exp(logits_mix_one) * neglect_negative_mask_one
            log_prob_mix_one       = logits_mix_one - torch.log(
                exp_logits_mix_one.sum(1, keepdim=True).clamp(min=1e-9)
            )

            if len(nonzero_idx) > 0:
                log_prob_mix_one_nz = log_prob_mix_one[nonzero_idx]
                mean_log_prob_mix   = (
                    (weighted_mask * log_prob_mix_one_nz).sum(1)
                    / weighted_mask.sum(1).clamp(min=1.0)
                )
                loss = loss - torch.mean(
                    (self.temperature / self.base_temperature) * mean_log_prob_mix * pnm
                )

            # Shift the diagonal for the next mini-iter
            negative_mask_one = torch.roll(negative_mask_one, split, dims=1)

        loss = loss / max(mini_iters, 1) / 2  # average over mini-iters and both views
        return loss


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
        # Initialize once — must NOT be reset by bank_init() or the contrastive
        # bank built by bank_compute() at epoch-end will be wiped at epoch-start.
        self.flat_norm_bank_vq_feat = torch.empty([0, self.dim], device=self._device)
        self.flat_norm_bank_proj_feat_ema = torch.empty([0, self.projection_dim], device=self._device)

    def bank_init(self):
        # Resets only prime_bank (per-epoch accumulation buffer).
        # flat_norm_bank_* persist across epochs — populated by bank_compute().
        self.prime_bank = {}
        start = torch.empty([0, self.projection_dim], device=self._device)
        for i in range(self.num_codebook):
            self.prime_bank[i] = start

    def bank_compute(self):
        # Pre-collect on CPU to avoid MPS memory fragmentation from 512 torch.cat ops
        vq_parts = []
        proj_parts = []
        for key in self.prime_bank.keys():
            num = self.prime_bank[key].shape[0]
            if num == 0:
                continue
            vq_parts.append(self.codebook[key].detach().cpu().unsqueeze(0).expand(num, -1))
            proj_parts.append(self.prime_bank[key].detach().cpu())
        if vq_parts:
            bank_vq_feat = torch.cat(vq_parts, dim=0).to(self._device)
            bank_proj_feat_ema = torch.cat(proj_parts, dim=0).to(self._device)
        else:
            bank_vq_feat = torch.empty([0, self.dim], device=self._device)
            bank_proj_feat_ema = torch.empty([0, self.projection_dim], device=self._device)
        del vq_parts, proj_parts
        self.flat_norm_bank_vq_feat = F.normalize(bank_vq_feat, dim=1)
        self.flat_norm_bank_proj_feat_ema = F.normalize(bank_proj_feat_ema, dim=1)


# ---- DINOv3 Backbone ----

class DINOv3Backbone(nn.Module):
    """Wraps HuggingFace DINOv3 backbone (ViT-B/16 or ViT-L/16) to match CAUSE interface.

    forward(x) returns (B, 1+N, D) where first token is CLS, rest are patches.
    Register tokens are stripped internally. embed_dim is read from model config.
    """

    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", device="mps",
                 hp_weights=None):
        super().__init__()
        from transformers import AutoModel

        print(f"Loading DINOv3 backbone from HuggingFace: {model_name}")
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

        if hp_weights is not None:
            self.backbone.load_state_dict(torch.load(hp_weights, map_location="cpu"))
            print(f"  Loaded HP-finetuned weights: {hp_weights}")

    def unfreeze(self):
        """Unfreeze backbone for HP fine-tuning (Stage 0)."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print(f"  DINOv3 backbone UNFROZEN: "
              f"{sum(p.numel() for p in self.backbone.parameters()) / 1e6:.1f}M trainable")

    def refreeze(self):
        """Re-freeze backbone after HP fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        print("  DINOv3 backbone re-frozen.")

    def forward(self, x):
        # No forced no_grad here — callers guard with no_grad when backbone is frozen.
        # Stage 0 (hp_backbone) calls without no_grad to allow gradient flow.
        outputs = self.backbone(x, return_dict=True)
        tokens = outputs.last_hidden_state
        # DINOv3: [CLS, reg1, reg2, reg3, reg4, patch1, patch2, ...]
        cls_token = tokens[:, 0:1, :]
        patch_tokens = tokens[:, 1 + self.num_register_tokens :, :]
        # Return (B, 1+N, D) — compatible with CAUSE's net(img)[:, 1:, :]
        return torch.cat([cls_token, patch_tokens], dim=1)


# ---- Dataset ----

class CityscapesCAUSE(Dataset):
    """Cityscapes dataset for CAUSE training with configurable crop strategy."""

    def __init__(self, root, split="train", resolution=320,
                 augment=True, crop_strategy="rect_then_crop"):
        self.root = Path(root)
        self.resolution = resolution
        self.augment = augment

        img_dir = self.root / "leftImg8bit" / split
        self.image_paths = sorted(img_dir.rglob("*.png"))
        print(f"  {split}: {len(self.image_paths)} images")

        if augment:
            if crop_strategy == "rect_then_crop":
                # Resize to half Cityscapes resolution (640×1280), preserving
                # the 2:1 aspect ratio. RandomCrop(resolution) then cuts a
                # geometrically-correct 320×320 tile — identical to inference.
                self.transform = transforms.Compose([
                    transforms.Resize(
                        (resolution * 2, resolution * 4),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
            else:  # "random_resized_crop" — original behaviour (for ablation)
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        resolution, scale=(0.5, 1.0),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
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
        return {"img": img, "ind": idx}


class CityscapesCAUSEPair(Dataset):
    """Cityscapes dataset returning two independently-augmented crops of the same image.

    Used only for Stage 0 (HP backbone fine-tuning) consistency loss:
        loss_consistency = alpha * mean(PairwiseDistance(z, z_aug))
    Each call to __getitem__ applies the stochastic transform twice independently,
    so img and img_aug are different random crops of the same Cityscapes image.
    """

    def __init__(self, root, split="train", resolution=320):
        self.root = Path(root)
        img_dir = self.root / "leftImg8bit" / split
        self.image_paths = sorted(img_dir.rglob("*.png"))
        print(f"  {split} (pair): {len(self.image_paths)} images")
        self.transform = transforms.Compose([
            transforms.Resize(
                (resolution * 2, resolution * 4),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"img": self.transform(img), "img_aug": self.transform(img), "ind": idx}


class CityscapesCAUSEVal(Dataset):
    """Cityscapes val set: image + GT label (27-class, CAUSE labelID 7–33 → 0–26).

    LabelIDs 0–6 (void, ego vehicle, etc.) and any other IDs → -1 (ignored
    in Hungarian matching and mIoU computation).
    """

    # Index table: labelID (uint8, 0–255) → CAUSE class (0–26) or -1
    _REMAP = np.full(256, -1, dtype=np.int64)
    for _lid in range(27):
        _REMAP[7 + _lid] = _lid

    def __init__(self, root, resolution=320):
        self.resolution = resolution
        img_dir = Path(root) / "leftImg8bit" / "val"
        self.image_paths = sorted(img_dir.rglob("*_leftImg8bit.png"))
        print(f"  val: {len(self.image_paths)} images")

        self.img_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _label_path(self, img_path: Path) -> Path:
        p = str(img_path)
        p = p.replace("/leftImg8bit/", "/gtFine/")
        p = p.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        return Path(p)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # Load label, remap labelIDs to 0-26 / -1
        lbl_raw = np.clip(np.array(Image.open(self._label_path(img_path))), 0, 255).astype(np.uint8)
        lbl_mapped = self._REMAP[lbl_raw]  # (1024, 2048) int64

        # Apply the same spatial transform as the image:
        #   Resize(shorter_edge=resolution) using NEAREST, then CenterCrop(resolution).
        # The image transform does Resize(320) → 320×640, then CenterCrop(320) → 320×320.
        # We must do the same on the label or every pixel will be misaligned.
        lbl_pil = Image.fromarray(lbl_mapped.astype(np.int32), mode='I')
        lbl_pil = transforms.functional.resize(
            lbl_pil, self.resolution,
            interpolation=InterpolationMode.NEAREST,
        )  # shorter edge → resolution (preserves aspect ratio)
        lbl_pil = transforms.functional.center_crop(lbl_pil, self.resolution)
        lbl_t = torch.from_numpy(np.array(lbl_pil)).long()  # (H, W)

        return {"img": img, "label": lbl_t}


# ---- Validation: CAUSE-standard mIoU via cluster_probe + Hungarian ----

def validate(net, segment, cluster, val_loader, device):
    """Compute CAUSE-standard mIoU on Cityscapes val.

    Pipeline (v7):
      1. Extract DINOv3 patch features — with flip TTA (matches official test_tr.py)
      2. head_ema → 90-dim spatial features, averaged over orig + hflip
      3. cluster_probe cosine similarity → K cluster predictions (K=N_CLASSES_CLUSTER=54)
      4. Hungarian matching on (K × N_CLASSES) confusion matrix → mIoU, pAcc
    """
    segment.eval()
    n_gt = N_CLASSES
    n_cluster = cluster.cluster_probe.shape[0]  # 54 (overclustering) or 27

    # Confusion matrix: rows = predicted cluster idx, cols = GT class idx
    histogram = torch.zeros(n_cluster, n_gt, dtype=torch.long)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Val", leave=False):
            imgs = batch["img"].to(device)    # (B, 3, H, W)
            labels = batch["label"]           # (B, H, W), values in [-1..26]

            # --- Flip TTA (matches official test_tr.py) ---
            feat      = net(imgs)[:, 1:, :]               # (B, N, DIM)
            feat_flip = net(imgs.flip(dims=[3]))[:, 1:, :] # (B, N, DIM)

            seg_feat      = transform(segment.head_ema(feat))       # (B, 90, sqrt(N), sqrt(N))
            seg_feat_flip = transform(segment.head_ema(feat_flip))   # (B, 90, sqrt(N), sqrt(N))
            # Flip back spatially and average
            seg_feat_avg = (seg_feat + seg_feat_flip.flip(dims=[3])) / 2  # (B, 90, H', W')

            # Upsample to label resolution
            spatial_up = F.interpolate(
                seg_feat_avg, size=labels.shape[-2:], mode='bilinear', align_corners=False
            )  # (B, 90, H_lbl, W_lbl)

            # Cluster assignment
            normed_feats  = F.normalize(spatial_up, dim=1)
            normed_probes = F.normalize(cluster.cluster_probe.detach(), dim=1)  # (K, 90)
            inner = torch.einsum("bchw,nc->bnhw", normed_feats, normed_probes)  # (B, K, H, W)
            preds = inner.argmax(dim=1).cpu()   # (B, H, W) — values in [0, K-1]

            # Accumulate confusion matrix (pred rows, gt cols)
            lt = labels.reshape(-1)
            lp = preds.reshape(-1)
            mask = (lt >= 0) & (lt < n_gt) & (lp >= 0) & (lp < n_cluster)
            lt_valid = lt[mask]
            lp_valid = lp[mask]
            # bincount index: n_gt * pred + gt → shape (n_cluster, n_gt)
            hist = torch.bincount(
                n_gt * lp_valid + lt_valid,
                minlength=n_gt * n_cluster,
            ).reshape(n_cluster, n_gt)
            histogram += hist

    # Hungarian matching on (n_cluster × n_gt) matrix
    # linear_sum_assignment assigns min(n_cluster, n_gt) pairs
    conf_np = histogram.numpy()
    row_ind, col_ind = linear_sum_assignment(conf_np, maximize=True)
    # row_ind: which pred clusters are matched (min(54,27)=27 entries)
    # col_ind: which GT class each matched pred cluster covers

    # For each GT class, record tp, fp, fn using its assigned pred cluster
    tp = np.array([conf_np[row_ind[i], col_ind[i]] for i in range(len(col_ind))], dtype=np.float64)
    fp = np.array([conf_np[row_ind[i], :].sum() - tp[i]     for i in range(len(col_ind))], dtype=np.float64)
    fn = np.array([conf_np[:, col_ind[i]].sum() - tp[i]     for i in range(len(col_ind))], dtype=np.float64)
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, np.nan)

    valid = ~np.isnan(iou)
    miou = float(iou[valid].mean()) * 100
    pacc = float(tp.sum() / max(conf_np.sum(), 1)) * 100

    segment.train()
    return miou, pacc


# ---- CAUSE Args namespace ----

def make_cause_args():
    """Create args namespace for CAUSE module initialization."""
    return SimpleNamespace(
        dim=DIM,
        reduced_dim=REDUCED_DIM,
        projection_dim=PROJECTION_DIM,
        num_codebook=NUM_CODEBOOK,
        n_classes=N_CLASSES,
        num_queries=NUM_PATCHES,
    )


# ---- Stage 0: HP Backbone Fine-tuning ----

def train_hp_backbone(net, segment, cluster, train_pair_loader, device, args):
    """Stage 0: Full HP backbone fine-tuning (exact CVPR 2023 protocol).

    Trains DINOv3 backbone (307M) + simple linear code_head together using:
      - HP SupCon loss (pool-based positive mining, tau=0.6, rho=0.02)
      - Consistency loss: HP_ALPHA * mean(PairwiseDistance(z, z_aug))

    NOTE: Uses a temporary nn.Linear(DIM, REDUCED_DIM) as the code head (bypasses
    TRDecoder VQ). The VQ in segment.head collapses immediately under backbone
    fine-tuning because the v4 codebook is misaligned with the changing features,
    blocking all gradient flow. The simple linear head avoids this.

    Optimizers:
      AdamW(backbone + code_head, lr=hp_backbone_lr, weight_decay=0.1)
      Adam(project_head,         lr=hp_backbone_lr)

    After training: backbone weights saved to {output_dir}/dinov3_hp_finetuned.pth
    Backbone is re-frozen after this stage. Stage 2 (codebook) must re-run.
    """
    print("\n" + "=" * 60)
    print("  STAGE 0: HP Backbone Fine-tuning (full 307M params)")
    print("=" * 60)

    # Unfreeze backbone — gradients flow through the full ViT-L/16
    net.unfreeze()
    net.train()

    # Temporary code head: Linear(DIM, REDUCED_DIM) — simple, no VQ, full gradient flow.
    # Discarded after Stage 0; Stage 2 retrains the real TRDecoder codebook.
    code_head = nn.Linear(DIM, REDUCED_DIM).to(device)
    nn.init.xavier_uniform_(code_head.weight)
    nn.init.zeros_(code_head.bias)
    code_head_ema = copy.deepcopy(code_head)
    for p in code_head_ema.parameters():
        p.requires_grad = False

    # project_head: 2-layer MLP with BatchNorm1d — prevents representation collapse.
    # A single nn.Linear collapses (all outputs → same direction) because there is no
    # barrier against mode collapse. BatchNorm1d forces zero-mean unit-variance per
    # feature, which breaks the symmetry collapse. ReLU adds non-linearity.
    project_head = nn.Sequential(
        nn.Linear(REDUCED_DIM, REDUCED_DIM),
        nn.BatchNorm1d(REDUCED_DIM),
        nn.ReLU(inplace=True),
        nn.Linear(REDUCED_DIM, REDUCED_DIM),
    ).to(device)

    # Optimizers: backbone + code_head with AdamW; project_head with Adam
    backbone_params = list(net.backbone.parameters()) + list(code_head.parameters())
    net_optimizer = torch.optim.AdamW(
        backbone_params, lr=args.hp_backbone_lr, weight_decay=0.1
    )
    head_optimizer = torch.optim.Adam(project_head.parameters(), lr=args.hp_backbone_lr)

    hp_loss_fn = HPHiddenPositiveLoss(
        temperature=HP_TEMPERATURE,
        base_temperature=HP_BASE_TEMPERATURE,
    ).to(device)
    pd = nn.PairwiseDistance()

    # Build initial Pool_ag (backbone features) and Pool_sp (code_head_ema features)
    print("  Building initial reference pools with linear code_head_ema...")
    Pool_ag = torch.zeros(HP_POOL_SIZE, DIM,         device=device)
    Pool_sp = torch.zeros(HP_POOL_SIZE, REDUCED_DIM, device=device)
    filled  = 0
    net.eval()
    code_head_ema.eval()
    with torch.no_grad():
        for batch in train_pair_loader:
            if filled >= HP_POOL_SIZE:
                break
            img  = batch["img"].to(device)
            feat = net(img)[:, 1:, :]          # (B, N, DIM)
            B, N, _ = feat.shape
            code = F.normalize(
                code_head_ema(feat.reshape(B * N, DIM)), dim=-1
            ).reshape(B, N, REDUCED_DIM)       # (B, N, REDUCED_DIM)
            for img_i in range(B):
                if filled >= HP_POOL_SIZE:
                    break
                ri = np.random.randint(0, N)
                Pool_ag[filled] = F.normalize(feat[img_i, ri].float(), dim=0, eps=1e-6)
                Pool_sp[filled] = F.normalize(code[img_i, ri].float(), dim=0, eps=1e-6)
                filled += 1
    Pool_ag = F.normalize(Pool_ag, dim=1)
    Pool_sp = F.normalize(Pool_sp, dim=1)
    print(f"  Pool_ag: {tuple(Pool_ag.shape)}  Pool_sp: {tuple(Pool_sp.shape)}")

    accum_steps = getattr(args, "accum_steps", 16)
    eff_bs = args.batch_size * accum_steps
    print(f"  Gradient accumulation: {accum_steps} steps "
          f"(physical bs={args.batch_size}, effective bs={eff_bs})")

    total_iters = 0         # counts physical batches
    optim_steps = 0         # counts optimizer.step() calls
    net_optimizer.zero_grad()
    head_optimizer.zero_grad()

    for epoch in range(args.hp_backbone_epochs):
        net.train()
        code_head.train()
        code_head_ema.eval()
        project_head.train()
        epoch_loss = 0.0
        count = 0
        pbar = tqdm(
            train_pair_loader,
            desc=f"HP Backbone Epoch {epoch + 1}/{args.hp_backbone_epochs}",
        )

        for batch in pbar:
            img     = batch["img"].to(device)      # (B, 3, 320, 320)
            img_aug = batch["img_aug"].to(device)  # (B, 3, 320, 320) — different random crop

            # Both views through backbone — GRAD FLOWS
            feat     = net(img)[:, 1:, :]     # (B, N, DIM)
            feat_aug = net(img_aug)[:, 1:, :] # (B, N, DIM)
            B, N, _ = feat.shape
            feat_flat     = feat.reshape(B * N, DIM)
            feat_aug_flat = feat_aug.reshape(B * N, DIM)

            # code_head: linear projection → L2 norm (full gradient flow, no VQ)
            code     = F.normalize(code_head(feat_flat),     dim=-1)  # (B*N, REDUCED_DIM)
            code_aug = F.normalize(code_head(feat_aug_flat), dim=-1)

            # EMA code (no grad, for LHP signal and Pool_sp renewal)
            with torch.no_grad():
                code_ema = F.normalize(
                    code_head_ema(feat_flat.detach()), dim=-1
                )  # (B*N, REDUCED_DIM)

            # HP inputs (all L2-normed)
            modeloutput_f    = F.normalize(feat_flat.detach(), dim=-1)
            modeloutput_s_pr = code_ema

            # project_head: code → z (grad flows through code_head → backbone)
            modeloutput_z     = F.normalize(project_head(code),     dim=-1)
            modeloutput_z_aug = F.normalize(project_head(code_aug), dim=-1)
            with torch.no_grad():
                modeloutput_z_mix = F.normalize(project_head(code_ema), dim=-1)

            # lmbd warmup (based on optimizer steps, not physical batches)
            lmbd = min(
                HP_LAMBDA_MAX,
                HP_LAMBDA_MAX * max(0, optim_steps - HP_WARMUP_ITERS)
                / max(HP_WARMUP_ITERS, 1),
            )

            loss_supcon = hp_loss_fn(
                modeloutput_z, modeloutput_z_mix,
                modeloutput_f, modeloutput_s_pr,
                Pool_ag, Pool_sp, lmbd,
            )
            loss_consistency = torch.mean(pd(modeloutput_z, modeloutput_z_aug))
            # Scale loss by 1/accum_steps so accumulated gradient = mean over accum_steps
            loss = (loss_supcon + HP_ALPHA * loss_consistency) / accum_steps

            if torch.isnan(loss):
                total_iters += 1
                if total_iters % accum_steps == 0:
                    net_optimizer.zero_grad()
                    head_optimizer.zero_grad()
                continue

            loss.backward()

            # EMA update (every physical batch)
            with torch.no_grad():
                for p_s, p_t in zip(code_head.parameters(), code_head_ema.parameters()):
                    p_t.data.mul_(HP_EMA_DECAY).add_(p_s.data * (1 - HP_EMA_DECAY))

            # Renew Pool_ag + Pool_sp circular buffers (based on physical iters)
            # Pool_ag must be renewed with CURRENT backbone features so Rpoint thresholds
            # stay accurate as the backbone changes — stale Pool_ag = wrong GHP positives
            # = supcon stuck flat. Pool_sp renewed the same way for LHP consistency.
            if total_iters % HP_RENEW_EVERY == 0:
                feat_2d = feat.reshape(B, N, DIM)
                idx_list = [np.random.randint(0, N) for _ in range(B)]
                new_ag_entries = F.normalize(
                    torch.stack([feat_2d[i, idx_list[i]].detach().float() for i in range(B)]),
                    dim=-1,
                )
                Pool_ag = torch.cat([Pool_ag[B:], new_ag_entries], dim=0)

                code_ema_2d = code_ema.reshape(B, N, REDUCED_DIM)
                new_entries = torch.stack(
                    [code_ema_2d[i, idx_list[i]] for i in range(B)]
                )
                Pool_sp = torch.cat([Pool_sp[B:], new_entries.detach()], dim=0)

            # Optimizer step every accum_steps physical batches
            if (total_iters + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(backbone_params, HP_GRAD_NORM)
                net_optimizer.step()
                head_optimizer.step()
                net_optimizer.zero_grad()
                head_optimizer.zero_grad()
                optim_steps += 1

            epoch_loss  += (loss.item() * accum_steps)  # un-scale for logging
            count       += 1
            total_iters += 1
            pbar.set_postfix(
                supcon=f"{loss_supcon.item():.4f}",
                cons=f"{loss_consistency.item():.4f}",
                lmbd=f"{lmbd:.3f}",
            )

        print(f"  Epoch {epoch + 1}: avg loss = {epoch_loss / max(count, 1):.6f}  "
              f"(optim_steps={optim_steps})")

    # Save HP-finetuned backbone weights
    hp_ckpt = os.path.join(args.output_dir, "dinov3_hp_finetuned.pth")
    torch.save(net.backbone.state_dict(), hp_ckpt)
    print(f"\nHP backbone checkpoint saved: {hp_ckpt}")

    # Re-freeze backbone for Stages 2, 1, 3
    net.refreeze()
    return net, segment


# ---- Stage 1: HP-style Reference Pool Warm-up ----

def _hp_initialize_pools(net, segment, train_loader, device):
    """Build Pool_ag (backbone, DIM-dim) and Pool_sp (EMA code, REDUCED_DIM-dim).

    Exact protocol from refs/hp/make_reference_pool.py initialize_reference_pool():
      - 1 random spatial patch per image
      - HP_POOL_SIZE different images in total
      - Both pools L2-normalised float32 on device

    Pool_ag stays fixed for the entire Stage 1 (backbone is frozen).
    Pool_sp is renewed at the start of each epoch via _hp_renew_pool_sp().
    """
    Pool_ag = torch.zeros(HP_POOL_SIZE, DIM,         device=device)
    Pool_sp = torch.zeros(HP_POOL_SIZE, REDUCED_DIM, device=device)
    filled  = 0

    net.eval()
    segment.head_ema.eval()

    with torch.no_grad():
        for batch in train_loader:
            if filled >= HP_POOL_SIZE:
                break
            img = batch["img"].to(device)
            feat     = net(img)[:, 1:, :]           # (B, N, DIM)  N=HP_SPLIT=400
            code_ema = segment.head_ema(feat)       # (B, N, REDUCED_DIM)
            B, N, _  = feat.shape

            for img_i in range(B):
                if filled >= HP_POOL_SIZE:
                    break
                randidx = np.random.randint(0, N)   # 1 random patch per image
                Pool_ag[filled] = F.normalize(feat[img_i, randidx].float(),     dim=0, eps=1e-6)
                Pool_sp[filled] = F.normalize(code_ema[img_i, randidx].float(), dim=0, eps=1e-6)
                filled += 1

    Pool_ag = F.normalize(Pool_ag, dim=1)
    Pool_sp = F.normalize(Pool_sp, dim=1)
    print(f"  Pool_ag: {tuple(Pool_ag.shape)}  Pool_sp: {tuple(Pool_sp.shape)}")
    return Pool_ag, Pool_sp


def _hp_renew_pool_sp(net, segment, train_loader, device):
    """Rebuild Pool_sp (EMA code) — called at the start of each Stage 1 epoch.

    Exact protocol from refs/hp/make_reference_pool.py renew_reference_pool():
      - 1 random patch per image, HP_POOL_SIZE images
      - Full rebuild (not a circular buffer update)
    Pool_ag is NOT renewed — backbone is frozen throughout Stage 1.
    """
    Pool_sp = torch.zeros(HP_POOL_SIZE, REDUCED_DIM, device=device)
    filled  = 0

    segment.head_ema.eval()

    with torch.no_grad():
        for batch in train_loader:
            if filled >= HP_POOL_SIZE:
                break
            img      = batch["img"].to(device)
            feat     = net(img)[:, 1:, :]           # (B, N, DIM)
            code_ema = segment.head_ema(feat)       # (B, N, REDUCED_DIM)
            B, N, _  = code_ema.shape

            for img_i in range(B):
                if filled >= HP_POOL_SIZE:
                    break
                randidx = np.random.randint(0, N)
                Pool_sp[filled] = F.normalize(code_ema[img_i, randidx].float(), dim=0, eps=1e-6)
                filled += 1

    Pool_sp = F.normalize(Pool_sp, dim=1)
    return Pool_sp


def train_hp_warmup(net, segment, cluster, train_loader, device, args):
    """Stage 1: HP-style reference pool warm-up (exact HP CVPR 2023 protocol).

    Trains segment.head + segment.projection_head with HPHiddenPositiveLoss:
      - Pool_ag (backbone, fixed) provides GHP positives
      - Pool_sp (EMA code, rebuilt every epoch) provides LHP positives introduced
        via lmbd warmup from 0 → HP_LAMBDA_MAX
      - Double NCE on student projection (z) + teacher projection (z_mix)
      - mini-iter batching: HP_SPLIT=400 anchors per iter
      - mask_neglect_base: keep HP_RHO fraction of negatives in denominator
      - reweighting=1: pnm normalisation by positive count

    After warm-up, head and projection_head reflect DINOv3 semantic structure →
    Stage 3 contrastive loss won't collapse.
    """
    print("\n" + "=" * 60)
    print("  STAGE 1: HP Reference Pool Warm-up (exact CVPR 2023 protocol)")
    print("=" * 60)

    # Inject frozen codebook so TRDecoder VQ step can run
    cb = cluster.codebook.data.clone()
    segment.head.codebook     = cb
    segment.head_ema.codebook = cb

    # Sync EMA ← student before building initial pool
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)

    hp_loss_fn = HPHiddenPositiveLoss(
        temperature=HP_TEMPERATURE,
        base_temperature=HP_BASE_TEMPERATURE,
    ).to(device)

    # Build initial Pool_ag (fixed) and Pool_sp (renewed each epoch)
    Pool_ag, Pool_sp = _hp_initialize_pools(net, segment, train_loader, device)

    # Train both the main head and the projection head
    train_params = (
        list(segment.head.parameters())
        + list(segment.projection_head.parameters())
    )
    optimizer = torch.optim.Adam(train_params, lr=args.stego_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stego_epochs * len(train_loader)
    )

    total_iters = 0
    for epoch in range(args.stego_epochs):
        # Pool_sp renewed each epoch because head_ema weights change.
        # Pool_ag stays fixed — backbone is frozen in Stage 1 so features are identical.
        if epoch > 0:
            print(f"  Renewing Pool_sp (epoch {epoch + 1})…")
            Pool_sp = _hp_renew_pool_sp(net, segment, train_loader, device)

        segment.head.train()
        segment.projection_head.train()
        segment.head_ema.eval()
        segment.projection_head_ema.eval()

        epoch_loss = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"HP Stage1 Epoch {epoch + 1}/{args.stego_epochs}")

        for batch in pbar:
            img = batch["img"].to(device)   # (B, 3, 320, 320)

            with torch.no_grad():
                feat = net(img)[:, 1:, :]           # (B, N, DIM) — frozen backbone

            # Student forward
            code     = segment.head(feat)           # (B, N, REDUCED_DIM) — trainable
            proj     = segment.projection_head(code)  # (B, N, PROJECTION_DIM)

            # Teacher forward (EMA, no grad)
            with torch.no_grad():
                code_ema = segment.head_ema(feat)           # (B, N, REDUCED_DIM)
                proj_ema = segment.projection_head_ema(code_ema)  # (B, N, PROJECTION_DIM)

            B, N, _ = feat.shape
            # Flatten to (B*N, dim) and L2-normalise — exact HP convention
            modeloutput_f    = F.normalize(feat.detach().reshape(B * N, DIM),            dim=-1)
            modeloutput_s_pr = F.normalize(code_ema.detach().reshape(B * N, REDUCED_DIM), dim=-1)
            modeloutput_z    = F.normalize(proj.reshape(B * N, PROJECTION_DIM),           dim=-1)
            modeloutput_z_mix = F.normalize(proj_ema.detach().reshape(B * N, PROJECTION_DIM), dim=-1)

            # lmbd warmup: 0 for first HP_WARMUP_ITERS iters, linear to HP_LAMBDA_MAX
            lmbd = min(
                HP_LAMBDA_MAX,
                HP_LAMBDA_MAX * max(0, total_iters - HP_WARMUP_ITERS)
                / max(HP_WARMUP_ITERS, 1),
            )

            loss = hp_loss_fn(
                modeloutput_z, modeloutput_z_mix,
                modeloutput_f, modeloutput_s_pr,
                Pool_ag, Pool_sp,
                lmbd,
            )

            if torch.isnan(loss):
                optimizer.zero_grad()
                total_iters += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update: student → teacher
            with torch.no_grad():
                ema_update(segment.head,            segment.head_ema,            lamb=HP_EMA_DECAY)
                ema_update(segment.projection_head, segment.projection_head_ema, lamb=HP_EMA_DECAY)

            epoch_loss  += loss.item()
            count       += 1
            total_iters += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lmbd=f"{lmbd:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        print(
            f"  Epoch {epoch + 1}: avg loss = {epoch_loss / max(count, 1):.6f}"
            f"  (count={count}/{len(train_loader)}, nan_skipped={len(train_loader)-count})"
        )

    stego_ckpt = os.path.join(args.output_dir, "segment_tr_stego.pth")
    torch.save(segment.state_dict(), stego_ckpt)
    print(f"\nHP Stage 1 checkpoint saved: {stego_ckpt}")
    return segment


# ---- Codebook Initialization via K-Means ----

def init_codebook_kmeans(net, cluster, train_loader, device, n_images=500):
    """Initialize codebook centroids via k-means on DINOv3 patch features.

    Extracts patch features from n_images training images, runs MiniBatchKMeans
    with k=NUM_CODEBOOK, and assigns centroids to cluster.codebook. This gives
    a semantically-grounded initialization instead of random, which the
    modularity loss alone cannot overcome from random init.
    """
    from sklearn.cluster import MiniBatchKMeans

    print(f"\n  [K-Means Init] Extracting features from {n_images} images…")
    all_feats = []
    collected = 0

    net.eval()
    with torch.no_grad():
        for batch in train_loader:
            if collected >= n_images:
                break
            img = batch["img"].to(device)
            feat = net(img)[:, 1:, :]                     # (B, N, DIM)
            B, N, D = feat.shape
            # Sample 64 random patches per image for better rare-class coverage
            idx = torch.randint(0, N, (B, 64))
            sampled = feat[torch.arange(B).unsqueeze(1), idx]  # (B, 64, D)
            all_feats.append(F.normalize(sampled.float(), dim=-1).reshape(-1, D).cpu().numpy())
            collected += B

    all_feats = np.concatenate(all_feats, axis=0)         # (n_images*64, DIM)
    print(f"  [K-Means Init] Running MiniBatchKMeans(k={cluster.codebook.shape[0]}) "
          f"on {all_feats.shape[0]} patches…")

    kmeans = MiniBatchKMeans(
        n_clusters=cluster.codebook.shape[0],
        batch_size=4096,
        n_init=3,
        max_iter=100,
        random_state=42,
        verbose=0,
    )
    kmeans.fit(all_feats)
    centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
    centroids = F.normalize(centroids, dim=1)
    cluster.codebook.data = centroids
    print(f"  [K-Means Init] Done. Codebook shape: {cluster.codebook.shape}")
    return cluster


# ---- Stage 2: Modularity Codebook ----

def train_codebook(net, cluster, train_loader, device, args):
    """Stage 2: Optimize codebook via modularity maximization.

    First initializes centroids via k-means on DINOv3 features (avoids near-zero
    gradients from random init), then refines with modularity loss.
    """
    print("\n" + "=" * 60)
    print("  STAGE 2: Modularity-based Codebook Learning")
    print("=" * 60)

    # K-means init — critical for non-trivial modularity gradients
    cluster = init_codebook_kmeans(net, cluster, train_loader, device)

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
            img = batch["img"].to(device)

            # Extract features (frozen backbone)
            with torch.no_grad():
                feat = net(img)[:, 1:, :]  # (B, N, 1024)

            # Modularity loss — temp=0.02 sharpens assignments (0.1 too soft → near-zero grads)
            loss = compute_modularity_loss(
                cluster.codebook, feat, device, temp=0.02, grid=True
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

def train_heads(net, segment, cluster, train_loader, device, args, val_loader=None):
    """Stage 3: Train TRDecoder segment head + Cluster probe."""
    print("\n" + "=" * 60)
    print("  STAGE 3: TRDecoder + Cluster Probe Training")
    print("=" * 60)

    # Inject frozen codebook into Decoder heads
    cb = cluster.codebook.data.clone()
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    # Initialize EMA from student
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)

    # Overclustering: reinitialize cluster_probe with N_CLASSES_CLUSTER=54 centroids.
    # Hungarian matching at eval time maps 54 pred clusters → 27 GT classes.
    # CUPS Table 7b: K=54 gives +2.8 PQ over K=27 on COCO; same benefit expected here.
    cluster.cluster_probe = nn.Parameter(
        torch.empty(N_CLASSES_CLUSTER, REDUCED_DIM, device=device)
    )
    reset(cluster.cluster_probe, N_CLASSES_CLUSTER)

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

    best_miou = 0.0
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
            img = batch["img"].to(device)

            # Extract features (frozen backbone)
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

        # Free training tensors and clear MPS cache before bank_compute/validation
        import gc
        gc.collect()
        if device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        # Compute bank for next epoch
        try:
            cluster.bank_compute()
        except Exception as e:
            print(f"  WARNING: bank_compute failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            print("  Skipping bank rebuild — using stale bank for next epoch", flush=True)

        avg_cont = epoch_loss_cont / max(count, 1)
        avg_clust = epoch_loss_clust / max(count, 1)
        avg_total = avg_cont + avg_clust
        print(f"  Epoch {epoch + 1}: contrastive={avg_cont:.4f}, cluster={avg_clust:.4f}, total={avg_total:.4f}", flush=True)

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.head_epochs:
            save_checkpoint(segment, cluster, args.output_dir, epoch + 1)

        # Validation: compute CAUSE-standard mIoU every val_every epochs
        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            miou, pacc = validate(net, segment, cluster, val_loader, device)
            print(f"  [Val] Epoch {epoch + 1}: mIoU={miou:.2f}%  pAcc={pacc:.2f}%"
                  f"  (best mIoU so far: {best_miou:.2f}%)")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(segment, cluster, args.output_dir, "best")
                print(f"  → New best checkpoint saved (mIoU={best_miou:.2f}%)")

    print(f"\nHead training complete. Best val mIoU: {best_miou:.2f}%")
    return segment, cluster


def save_checkpoint(segment, cluster, output_dir, tag):
    """Save segment and cluster checkpoints."""
    seg_path = os.path.join(output_dir, f"segment_tr_{tag}.pth")
    cluster_path = os.path.join(output_dir, f"cluster_tr_{tag}.pth")

    torch.save(segment.state_dict(), seg_path)
    torch.save(cluster.state_dict(), cluster_path)
    print(f"  Checkpoint saved: {seg_path}, {cluster_path}")


# ---- Main ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CAUSE-TR on DINOv3 ViT-L/16 for Cityscapes"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["hp_backbone", "stego", "codebook", "heads", "all"],
                        help="hp_backbone=Stage0 only, stego=Stage1 only, "
                             "codebook=Stage2 only, heads=Stage3 only, all=0+1+2+3")
    parser.add_argument("--model_name", type=str,
                        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="HuggingFace model name for DINOv3 backbone. "
                             "ViT-B/16 (default, 768-dim, matches CAUSE-TR design): "
                             "facebook/dinov3-vitb16-pretrain-lvd1689m. "
                             "ViT-L/16 (1024-dim, stronger but slower): "
                             "facebook/dinov3-vitl16-pretrain-lvd1689m")

    # Stage 0 (HP backbone fine-tuning)
    parser.add_argument("--hp_backbone_epochs", type=int, default=30,
                        help="Number of HP backbone fine-tuning epochs (Stage 0)")
    parser.add_argument("--hp_backbone_lr", type=float, default=5e-4,
                        help="Learning rate for HP backbone fine-tuning (AdamW)")
    parser.add_argument("--hp_backbone_path", type=str, default=None,
                        help="Path to HP-finetuned backbone weights (.pth) for downstream stages")
    parser.add_argument("--accum_steps", type=int, default=16,
                        help="Gradient accumulation steps for Stage 0 (effective_bs = batch_size × accum_steps)")

    # Stage 1 (STEGO warm-up)
    parser.add_argument("--stego_epochs", type=int, default=10,
                        help="Number of STEGO Stage 1 warm-up epochs")
    parser.add_argument("--stego_lr", type=float, default=5e-4,
                        help="Learning rate for STEGO Stage 1")

    # Codebook (Stage 2)
    parser.add_argument("--codebook_epochs", type=int, default=10)
    parser.add_argument("--codebook_lr", type=float, default=1e-3)
    parser.add_argument("--codebook_path", type=str, default=None,
                        help="Path to pre-computed codebook .npy (for --stage heads)")

    # Head training (Stage 3)
    parser.add_argument("--head_epochs", type=int, default=60)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run CAUSE-standard mIoU validation every N head epochs")

    # Data
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--crop_strategy",
        type=str,
        default="rect_then_crop",
        choices=["rect_then_crop", "random_resized_crop"],
        help=(
            "rect_then_crop: resize to (2r, 4r) then RandomCrop(r) — "
            "preserves Cityscapes 2:1 aspect ratio (default). "
            "random_resized_crop: original distorting strategy."
        ),
    )

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

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


def main():
    args = parse_args()
    device = get_device(args)
    print(f"Device: {device}")

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            Path(__file__).resolve().parent.parent, "refs", "cause", "CAUSE_dinov3_v9"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # Load backbone first — DIM depends on backbone embed_dim
    net = DINOv3Backbone(
        model_name=args.model_name, device=device,
        hp_weights=args.hp_backbone_path,
    ).to(device)
    net.eval()

    # Set DIM / PROJECTION_DIM from backbone (supports ViT-B=768 and ViT-L=1024)
    global DIM, PROJECTION_DIM
    DIM = net.embed_dim
    PROJECTION_DIM = DIM * 2
    print(f"  Backbone: embed_dim={DIM}, projection_dim={PROJECTION_DIM}")

    # Save config (after backbone so DIM is correct)
    config = vars(args).copy()
    config["device"] = str(device)
    config["dim"] = DIM
    config["patch_size"] = PATCH_SIZE
    config["train_resolution"] = TRAIN_RESOLUTION
    config["num_patches"] = NUM_PATCHES
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create CAUSE modules
    cause_args = make_cause_args()
    cluster = DeviceAwareCluster(cause_args, device=device).to(device)
    segment = Segment_TR(cause_args).to(device)

    print(f"\nSegment_TR params: {sum(p.numel() for p in segment.parameters()) / 1e6:.2f}M")
    print(f"Cluster params: {sum(p.numel() for p in cluster.parameters()) / 1e6:.2f}M")

    # Datasets
    train_dataset = CityscapesCAUSE(
        args.cityscapes_root, split="train",
        resolution=TRAIN_RESOLUTION, augment=True,
        crop_strategy=args.crop_strategy,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(args.device != "mps"), drop_last=True,
    )

    val_loader = None
    if args.stage in ("heads", "all"):
        val_dataset = CityscapesCAUSEVal(args.cityscapes_root, resolution=TRAIN_RESOLUTION)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=(args.device != "mps"),
        )

    # Stage 0: HP backbone fine-tuning (full 307M params, exact CVPR 2023 protocol)
    if args.stage in ("hp_backbone", "all"):
        # Load a seed codebook for the VQ step in TRDecoder during Stage 0
        seed_codebook_path = args.codebook_path
        if seed_codebook_path is None:
            seed_codebook_path = os.path.join(
                Path(__file__).resolve().parent.parent,
                "refs", "cause", "CAUSE_dinov3_v4", "modular.npy",
            )
        if os.path.exists(seed_codebook_path):
            print(f"Loading seed codebook for Stage 0 from {seed_codebook_path}")
            cb = torch.from_numpy(np.load(seed_codebook_path)).to(device)
            cluster.codebook.data = cb
        else:
            print("  Warning: no seed codebook found — VQ will use random init for Stage 0")
        pair_dataset = CityscapesCAUSEPair(
            args.cityscapes_root, split="train", resolution=TRAIN_RESOLUTION
        )
        pair_loader = DataLoader(
            pair_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        net, segment = train_hp_backbone(net, segment, cluster, pair_loader, device, args)

    # Stage 2: Codebook — must run before Stage 1 (TRDecoder needs codebook for VQ)
    # After Stage 0, backbone features have changed → codebook MUST be retrained here.
    if args.stage in ("codebook", "all"):
        cluster = train_codebook(net, cluster, train_loader, device, args)

    # Load codebook if skipping stage 2; fall back to k-means init if file not found
    if args.stage in ("heads", "stego"):
        codebook_path = args.codebook_path
        if codebook_path is None:
            codebook_path = os.path.join(args.output_dir, "modular.npy")
        if os.path.exists(codebook_path):
            print(f"Loading codebook from {codebook_path}")
            cb = torch.from_numpy(np.load(codebook_path)).to(device)
            cluster.codebook.data = cb
        else:
            print(f"No codebook at {codebook_path} — initializing from K-Means on DINOv3 features")
            cluster = init_codebook_kmeans(net, cluster, train_loader, device)
            np.save(codebook_path, cluster.codebook.detach().cpu().numpy())
            print(f"K-Means codebook saved to {codebook_path}")

    # Stage 1: HP-style reference pool warm-up (after codebook, before contrastive heads)
    if args.stage in ("stego", "all"):
        segment = train_hp_warmup(net, segment, cluster, train_loader, device, args)

    # Load Stage 1 checkpoint if running heads-only after a previous stego run
    if args.stage == "heads":
        stego_ckpt = os.path.join(args.output_dir, "segment_tr_stego.pth")
        if os.path.exists(stego_ckpt):
            state = torch.load(stego_ckpt, map_location=device)
            segment.load_state_dict(state, strict=False)
            print(f"Loaded Stage 1 warm-up checkpoint: {stego_ckpt}")

    # Stage 3: Heads
    if args.stage in ("heads", "all"):
        segment, cluster = train_heads(
            net, segment, cluster, train_loader, device, args, val_loader=val_loader
        )

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
    print("Done!")


if __name__ == "__main__":
    main()
