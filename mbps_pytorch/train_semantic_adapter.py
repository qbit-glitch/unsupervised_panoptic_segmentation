#!/usr/bin/env python3
"""Self-supervised training of LoRA/DoRA adapters on DINOv2 + CAUSE-TR.

Usage:
    python mbps_pytorch/train_semantic_adapter.py \\
        --data_dir /path/to/datasets \\
        --output_dir results/semantic_adapter_dora \\
        --variant dora --rank 4 --alpha 4.0 \\
        --losses distillation,depth_cluster \\
        --epochs 10
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast
from torchvision import transforms as T
from tqdm import tqdm

import yaml
from contextlib import nullcontext

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform

from mbps_pytorch.models.adapters import (
    inject_lora_into_dinov2,
    inject_lora_into_cause_tr,
    freeze_non_adapter_params,
    count_adapter_params,
    set_dinov2_spatial_dims,
)
from mbps_pytorch.losses.feature_consistency import PredictionConsistencyLoss

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_RESOLUTION = 322
PATCH_SIZE = 14
DIM = 768
REDUCED_DIM = 90
PROJECTION_DIM = 2048
NUM_CODEBOOK = 2048
N_CLASSES = 27
EMA_MOMENTUM = 0.99

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor):
    return tensor * IMAGENET_STD + IMAGENET_MEAN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_cause_args():
    return SimpleNamespace(
        dim=DIM, reduced_dim=REDUCED_DIM, projection_dim=PROJECTION_DIM,
        num_codebook=NUM_CODEBOOK, n_classes=N_CLASSES,
        num_queries=23 * 23, crop_size=TRAIN_RESOLUTION, patch_size=PATCH_SIZE,
    )


def ema_update(student_head, teacher_head, lamb=0.99):
    """EMA update matching parameters by name to handle adapter count mismatch."""
    student_state = dict(student_head.named_parameters())
    with torch.no_grad():
        for name, p_t in teacher_head.named_parameters():
            if name in student_state:
                p_s = student_state[name]
                if p_s.shape == p_t.shape:
                    p_t.data = lamb * p_t.data + (1 - lamb) * p_s.data


def patch_cluster_for_device(cluster, device):
    def bank_init_device(self):
        self.prime_bank = {}
        start = torch.empty([0, self.projection_dim], device=device)
        for i in range(self.num_codebook):
            self.prime_bank[i] = start

    def bank_compute_device(self):
        bank_vq_feat = torch.empty([0, self.dim], device=device)
        bank_proj_feat_ema = torch.empty([0, self.projection_dim], device=device)
        for key in self.prime_bank.keys():
            num = self.prime_bank[key].shape[0]
            if num == 0:
                continue
            bank_vq_feat = torch.cat([bank_vq_feat, self.codebook[key].unsqueeze(0).repeat(num, 1)], dim=0)
            bank_proj_feat_ema = torch.cat([bank_proj_feat_ema, self.prime_bank[key]], dim=0)
        self.flat_norm_bank_vq_feat = F.normalize(bank_vq_feat, dim=1)
        self.flat_norm_bank_proj_feat_ema = F.normalize(bank_proj_feat_ema, dim=1)

    import types
    cluster.bank_init = types.MethodType(bank_init_device, cluster)
    cluster.bank_compute = types.MethodType(bank_compute_device, cluster)


def dino_distillation_loss(student_feat, teacher_feat):
    """Cosine-similarity distillation between student and teacher features."""
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1).detach()
    # Per-token cosine similarity; maximize similarity = minimize (1 - cos)
    loss = (1 - (student_feat * teacher_feat).sum(dim=-1)).mean()
    return loss


def cross_view_consistency_loss(feat1, feat2):
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1).detach()  # Stop-gradient on augmented view
    return (1 - (feat1 * feat2).sum(dim=-1)).mean()


def sample_coords(batch_size, feature_samples, device):
    return torch.rand(batch_size, feature_samples, feature_samples, 2, device=device) * 2 - 1


def grid_sample(t, coords):
    return F.grid_sample(t, coords, padding_mode="zeros", align_corners=False, mode="bilinear")


def depth_correlation_loss(code, depth, feature_samples=11, shift=0.0):
    """Depth-aware correlation loss.

    NOTE: This loss correlates segmentation code similarity with depth
    discontinuity. It is a weak geometric signal and may not meaningfully
    improve adaptation. Consider removing if training is unstable.
    """
    B = code.shape[0]
    coords = sample_coords(B, feature_samples, code.device)
    code_sampled = grid_sample(code, coords)
    depth_sampled = grid_sample(depth, coords)

    # Normalize code correlations but NOT depth
    cd = torch.einsum("nchw,ncij->nhwij",
                      F.normalize(code_sampled, dim=1, eps=1e-10),
                      F.normalize(code_sampled, dim=1, eps=1e-10))
    # Use actual depth values (not signs)
    dd = torch.einsum("nchw,ncij->nhwij", depth_sampled, depth_sampled)

    loss = -cd * (dd - shift)
    return loss.mean()


class AdapterTrainingDataset(Dataset):
    def __init__(self, base_dataset, cityscapes_root, depth_subdir="depth_depthpro",
                 split="train", patch_grid=23, use_augmentation=True):
        self.base = base_dataset
        self.cityscapes_root = cityscapes_root
        self.depth_subdir = depth_subdir
        self.split = split
        self.patch_grid = patch_grid
        self.use_augmentation = use_augmentation
        self._depth_cache = {}
        self._build_image_index()
        if use_augmentation:
            self.aug_transform = T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(p=0.5),
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                T.RandomSolarize(threshold=0.5, p=0.2),
            ])
        else:
            self.aug_transform = None

    def _build_image_index(self):
        img_dir = os.path.join(self.cityscapes_root, "leftImg8bit", self.split)
        self.source_images = []
        for city in sorted(os.listdir(img_dir)):
            city_dir = os.path.join(img_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for fname in sorted(os.listdir(city_dir)):
                if fname.endswith("_leftImg8bit.png"):
                    stem = fname.replace("_leftImg8bit.png", "")
                    self.source_images.append({"city": city, "stem": stem})
        self.crops_per_image = max(1, len(self.base) // max(len(self.source_images), 1))

    def _load_depth_for_index(self, idx):
        src_idx = idx // self.crops_per_image
        src_idx = min(src_idx, len(self.source_images) - 1)
        if src_idx in self._depth_cache:
            return self._depth_cache[src_idx]
        entry = self.source_images[src_idx]
        npy_path = os.path.join(self.cityscapes_root, self.depth_subdir, self.split,
                                entry["city"], f"{entry['stem']}.npy")
        if os.path.isfile(npy_path):
            depth = np.load(npy_path).astype(np.float32)
            depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            depth_patch = F.adaptive_avg_pool2d(depth_t, (self.patch_grid, self.patch_grid)).squeeze(0)
        else:
            logger.warning("Depth file missing: %s", npy_path)
            depth_patch = torch.zeros(1, self.patch_grid, self.patch_grid)
        if len(self._depth_cache) < 5000:
            self._depth_cache[src_idx] = depth_patch
        return depth_patch

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        depth = self._load_depth_for_index(idx)
        item["depth"] = depth
        if self.use_augmentation and "img" in item:
            img = item["img"]
            if isinstance(img, torch.Tensor):
                # Denormalize to [0,1], convert to PIL, augment, then re-normalize
                img_denorm = denormalize(img).clamp(0, 1)
                img_pil = T.ToPILImage()(img_denorm)
                img_aug = T.ToTensor()(self.aug_transform(img_pil))
                img_aug = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_aug)
                item["img_aug"] = img_aug
            else:
                # PIL image path: augment first, then ToTensor+Normalize
                img_aug = self.aug_transform(img)
                item["img_aug"] = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(img_aug)
        return item


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def train_semantic_adapter(
    backbone, segment, cluster, train_loader, device, output_dir,
    losses, loss_weights, lr=1e-4, epochs=10, save_every=5,
    lambda_depth=0.05, depth_shift=0.0,
    adapter_config=None, teacher_backbone=None, teacher_segment=None,
    gradient_accumulation_steps=1,
    adapter_lr_mult=5.0,
    use_amp=False,
    confidence_filter=False,
    confidence_p=0.5,
):
    if is_main_process():
        logger.info("=== Semantic Adapter Training ===")
        logger.info("Losses: %s", losses)
        logger.info("Weights: %s", loss_weights)
        logger.info("LR=%.1e, epochs=%d", lr, epochs)

    adapter_params = [p for p in backbone.parameters() if p.requires_grad]
    adapter_params += [p for p in segment.parameters() if p.requires_grad]
    adapter_params += [p for p in cluster.parameters() if p.requires_grad]
    if is_main_process():
        logger.info("Trainable adapter params: %d", sum(p.numel() for p in adapter_params))

    # Uni-UVPT-style LR multiplier: adapters get higher LR
    other_params = [p for n, p in backbone.named_parameters() if p.requires_grad and not any(s in n for s in (".lora_A", ".lora_B", ".lora_magnitude", ".dwconv", ".conv_gate"))]
    other_params += [p for n, p in segment.named_parameters() if p.requires_grad and not any(s in n for s in (".lora_A", ".lora_B", ".lora_magnitude", ".dwconv", ".conv_gate"))]
    other_params += [p for n, p in cluster.named_parameters() if p.requires_grad and not any(s in n for s in (".lora_A", ".lora_B", ".lora_magnitude", ".dwconv", ".conv_gate"))]
    # Deduplicate
    seen_ids = set()
    deduped_other = []
    for p in other_params:
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            deduped_other.append(p)

    param_groups = [
        {"params": adapter_params, "lr": lr * adapter_lr_mult, "weight_decay": 1e-4},
    ]
    if deduped_other:
        param_groups.append({"params": deduped_other, "lr": lr, "weight_decay": 1e-4})

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
    if is_main_process() and scaler is not None:
        logger.info("Using AMP with GradScaler")
    pred_consistency_fn = PredictionConsistencyLoss() if "prediction_consistency" in losses else None
    best_loss = float("inf")

    for epoch in range(epochs):
        backbone.train()
        segment.train()
        cluster.train()
        totals = {k: 0.0 for k in losses + ["total"]}
        count = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(prog):
            img = batch["img"].to(device)
            depth = batch.get("depth")
            if depth is not None:
                depth = depth.to(device)

            set_dinov2_spatial_dims(backbone, h_patches=23, w_patches=23)

            amp_context = autocast() if scaler else nullcontext()
            with amp_context:
                with torch.no_grad():
                    if teacher_backbone is not None and teacher_segment is not None:
                        feat_teacher = teacher_backbone(img)[:, 1:, :]
                        seg_feat_teacher = teacher_segment.head(feat_teacher)
                    else:
                        # Fallback for backward compatibility (should not happen in normal usage)
                        feat_teacher = backbone(img)[:, 1:, :]
                        seg_feat_teacher = segment.head(feat_teacher)

                feat_student = backbone(img)[:, 1:, :]
                seg_feat_student = segment.head(feat_student)

                loss_total = torch.tensor(0.0, device=device)

                if "distillation" in losses:
                    l_dist = dino_distillation_loss(feat_student, feat_teacher)
                    # Optional confidence filtering
                    if confidence_filter and confidence_p < 1.0:
                        from mbps_pytorch.losses.confidence_filtering import select_confident_samples, avg_entropy
                        # Use per-patch feature similarity as proxy for confidence
                        sim = (feat_student * feat_teacher).sum(dim=-1)  # (B, N)
                        conf_logits = torch.stack([sim, 1 - sim], dim=-1)  # (B, N, 2)
                        B, N, _ = conf_logits.shape
                        conf_logits_flat = conf_logits.reshape(B * N, 2)
                        selected_logits, selected_idx = select_confident_samples(conf_logits_flat, confidence_p)
                        if selected_logits.shape[0] > 0:
                            l_dist = (1 - selected_logits[:, 0]).mean()
                        # else keep full loss
                    w = loss_weights.get("distillation", 1.0)
                    loss_total = loss_total + w * l_dist
                    totals["distillation"] += l_dist.item()

                if "cross_view" in losses and "img_aug" in batch:
                    img_aug = batch["img_aug"].to(device)
                    set_dinov2_spatial_dims(backbone, h_patches=23, w_patches=23)
                    feat_aug = backbone(img_aug)[:, 1:, :]
                    l_cv = cross_view_consistency_loss(feat_student, feat_aug)
                    w = loss_weights.get("cross_view", 1.0)
                    loss_total = loss_total + w * l_cv
                    totals["cross_view"] += l_cv.item()

                if "prediction_consistency" in losses and pred_consistency_fn is not None:
                    # Consistency between student and teacher predictions
                    l_pc = pred_consistency_fn(seg_feat_student, seg_feat_teacher)
                    w = loss_weights.get("prediction_consistency", 0.1)
                    loss_total = loss_total + w * l_pc
                    totals["prediction_consistency"] += l_pc.item()

                if "depth_cluster" in losses and depth is not None:
                    code_student = transform(seg_feat_student)
                    l_depth = depth_correlation_loss(code_student, depth, shift=depth_shift)
                    w = loss_weights.get("depth_cluster", lambda_depth)
                    loss_total = loss_total + w * l_depth
                    totals["depth_cluster"] += l_depth.item()

                if "cause_cluster" in losses:
                    logger.warning("cause_cluster loss is deprecated and skipped (zero gradient).")
                    totals["cause_cluster"] += 0.0

                # Gradient accumulation with optional AMP
                loss_total = loss_total / gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(loss_total).backward()
                else:
                    loss_total.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                all_trainable = [p for group in optimizer.param_groups for p in group["params"]]
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                ema_update(segment.head, segment.head_ema, lamb=EMA_MOMENTUM)
                optimizer.zero_grad()

            totals["total"] += loss_total.item() * gradient_accumulation_steps
            count += 1
            prog.set_postfix(total=f"{loss_total.item() * gradient_accumulation_steps:.4f}")

        avg_total = totals["total"] / max(count, 1)
        if is_main_process():
            logger.info("Epoch %d: total=%.4f", epoch + 1, avg_total)
            for k in losses:
                logger.info("  %s: %.4f", k, totals[k] / max(count, 1))
        scheduler.step()

        if is_main_process() and ((epoch + 1) % save_every == 0 or epoch == epochs - 1):
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}.pt")
            state_dict = backbone.module.state_dict() if isinstance(backbone, DDP) else backbone.state_dict()
            torch.save({"backbone": state_dict, "segment": segment.state_dict(),
                        "cluster": cluster.state_dict(), "epoch": epoch + 1,
                        "adapter_config": adapter_config}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        if is_main_process() and avg_total < best_loss:
            best_loss = avg_total
            ckpt_path = os.path.join(output_dir, "best.pt")
            state_dict = backbone.module.state_dict() if isinstance(backbone, DDP) else backbone.state_dict()
            torch.save({"backbone": state_dict, "segment": segment.state_dict(),
                        "cluster": cluster.state_dict(), "epoch": epoch + 1,
                        "adapter_config": adapter_config}, ckpt_path)
            logger.info("New best loss=%.4f, saved to %s", best_loss, ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Train semantic adapters (DINOv2 + CAUSE-TR)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--variant", type=str, default="dora", choices=["lora", "dora", "conv_dora"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--late_block_start", type=int, default=6)
    parser.add_argument("--adapt_cause", action="store_true")
    parser.add_argument("--losses", type=str, default="distillation,depth_cluster")
    parser.add_argument("--loss_weights", type=str, default="")
    parser.add_argument("--lambda_depth", type=float, default=0.05)
    parser.add_argument("--depth_shift", type=float, default=0.0)
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of mini-batches to accumulate before optimizer step")
    parser.add_argument("--adapter_lr_mult", type=float, default=5.0,
                        help="LR multiplier for adapter parameters (Uni-UVPT style)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision (AMP) training")
    parser.add_argument("--confidence_filter", action="store_true",
                        help="Enable entropy-based confidence filtering on distillation loss")
    parser.add_argument("--confidence_p", type=float, default=0.5,
                        help="Fraction of samples to keep in confidence filtering")
    parser.add_argument("--use_prediction_consistency", action="store_true",
                        help="Add prediction consistency loss (Uni-UVPT style)")

    # Pre-parse to get config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if pre_args.config and os.path.isfile(pre_args.config):
        with open(pre_args.config, "r") as f:
            config = yaml.safe_load(f)
        config_defaults = {}
        for section, values in config.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    if k == "names" and isinstance(v, list):
                        config_defaults["losses"] = ",".join(str(x) for x in v)
                    elif k == "weights" and isinstance(v, dict):
                        config_defaults["loss_weights"] = json.dumps(v)
                    elif k == "cause_checkpoint_dir":
                        config_defaults["checkpoint_dir"] = v
                    elif isinstance(v, list):
                        config_defaults[k] = v
                    else:
                        config_defaults[k] = v
            else:
                config_defaults[section] = values
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(remaining_argv)

    # DDP setup
    ddp_enabled = "LOCAL_RANK" in os.environ
    if ddp_enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if is_main_process():
            logger.info("DDP enabled: rank=%d, world_size=%d", dist.get_rank(), dist.get_world_size())
    else:
        if args.device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    if ddp_enabled:
        dist.barrier()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(__file__).resolve().parent.parent / "refs" / "cause")

    cityscapes_root = os.path.join(args.data_dir, "cityscapes")

    logger.info("Loading DINOv2 backbone...")
    backbone_path = os.path.join(args.checkpoint_dir, "checkpoint", "dinov2_vit_base_14.pth")
    backbone = dinov2_vit_base_14()
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    result = backbone.load_state_dict(state, strict=False)
    if result.missing_keys:
        logger.warning("Backbone missing keys: %s", result.missing_keys[:10])
    if result.unexpected_keys:
        logger.warning("Backbone unexpected keys: %s", result.unexpected_keys[:10])

    # Create frozen teacher backbone (original, no adapters)
    logger.info("Creating frozen teacher backbone...")
    teacher_backbone = dinov2_vit_base_14()
    result_teacher = teacher_backbone.load_state_dict(state, strict=False)
    if result_teacher.missing_keys:
        logger.warning("Teacher backbone missing keys: %s", result_teacher.missing_keys[:10])
    if result_teacher.unexpected_keys:
        logger.warning("Teacher backbone unexpected keys: %s", result_teacher.unexpected_keys[:10])
    teacher_backbone.eval()
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    teacher_backbone = teacher_backbone.to(device)

    # Inject adapters into student backbone
    inject_lora_into_dinov2(
        backbone, variant=args.variant, rank=args.rank, alpha=args.alpha,
        dropout=args.dropout, late_block_start=args.late_block_start,
    )
    freeze_non_adapter_params(backbone)

    # Break symmetry: lora_B starts at zero → distillation loss = 0 → no gradients.
    # Random perturbation ensures non-zero loss from step 1.  std=0.01 is small
    # enough that distillation can pull the adapter back, large enough to escape
    # the zero-gradient flat region at init.
    for name, param in backbone.named_parameters():
        if "lora_B" in name:
            param.data.normal_(std=0.01)
            if is_main_process():
                logger.info("Perturbed %s (std=0.01)", name)

    backbone = backbone.to(device)
    if ddp_enabled:
        backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=False)

    logger.info("Loading CAUSE Segment_TR...")
    cause_args = build_cause_args()
    segment = Segment_TR(cause_args).to(device)
    cluster = Cluster(cause_args).to(device)

    seg_path = os.path.join(
        args.checkpoint_dir, "CAUSE", "cityscapes",
        "dinov2_vit_base_14", "2048", "segment_tr.pth",
    )
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    result_seg = segment.load_state_dict(seg_state, strict=False)
    if result_seg.missing_keys:
        logger.warning("Segment missing keys: %s", result_seg.missing_keys[:10])
    if result_seg.unexpected_keys:
        logger.warning("Segment unexpected keys: %s", result_seg.unexpected_keys[:10])

    # Create frozen teacher segment (original, no adapters)
    logger.info("Creating frozen teacher segment...")
    teacher_segment = Segment_TR(cause_args).to(device)
    result_teacher_seg = teacher_segment.load_state_dict(seg_state, strict=False)
    if result_teacher_seg.missing_keys:
        logger.warning("Teacher segment missing keys: %s", result_teacher_seg.missing_keys[:10])
    if result_teacher_seg.unexpected_keys:
        logger.warning("Teacher segment unexpected keys: %s", result_teacher_seg.unexpected_keys[:10])
    teacher_segment.eval()
    for p in teacher_segment.parameters():
        p.requires_grad = False

    mod_path = os.path.join(
        args.checkpoint_dir, "CAUSE", "cityscapes", "modularity",
        "dinov2_vit_base_14", "2048", "modular.npy",
    )
    cb = torch.from_numpy(np.load(mod_path)).to(device)
    cluster.codebook.data = cb
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb
    segment.head_ema.codebook = cb
    teacher_segment.head.codebook = cb
    teacher_segment.head_ema.codebook = cb

    if args.adapt_cause:
        inject_lora_into_cause_tr(
            segment, variant=args.variant, rank=args.rank, alpha=args.alpha,
            dropout=args.dropout, adapt_head=True, adapt_projection=False, adapt_ema=True,
        )
    freeze_non_adapter_params(segment)

    patch_cluster_for_device(cluster, device)
    cluster.bank_init()
    freeze_non_adapter_params(cluster)
    if "cause_cluster" in args.losses:
        logger.warning("cause_cluster is deprecated; cluster_probe will remain frozen.")

    total_trainable = count_adapter_params(backbone) + count_adapter_params(segment) + count_adapter_params(cluster)
    if is_main_process():
        logger.info("Total trainable params: %d", total_trainable)

    if is_main_process():
        logger.info("Building training dataset...")
    if "pydensecrf" not in sys.modules:
        import types
        mock_crf = types.ModuleType("pydensecrf")
        mock_crf.densecrf = types.ModuleType("pydensecrf.densecrf")
        mock_crf.utils = types.ModuleType("pydensecrf.utils")
        sys.modules["pydensecrf"] = mock_crf
        sys.modules["pydensecrf.densecrf"] = mock_crf.densecrf
        sys.modules["pydensecrf.utils"] = mock_crf.utils

    from loader.dataloader import ContrastiveSegDataset

    img_transform = T.Compose([
        T.Resize(TRAIN_RESOLUTION, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(TRAIN_RESOLUTION),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_transform = T.Compose([
        T.Resize(TRAIN_RESOLUTION, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(TRAIN_RESOLUTION),
        T.ToTensor(),
    ])

    crop_dir = os.path.join(args.data_dir, "cityscapes", "cropped", "cityscapes_five_crop_0.5", "img", "train")
    crop_type = "five" if os.path.isdir(crop_dir) else None

    base_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir, dataset_name="cityscapes",
        crop_type=crop_type, image_set="train",
        transform=img_transform, target_transform=label_transform,
    )

    train_dataset = AdapterTrainingDataset(
        base_dataset=base_dataset, cityscapes_root=cityscapes_root,
        depth_subdir=args.depth_subdir, split="train", patch_grid=23,
        use_augmentation="cross_view" in args.losses,
    )

    nw = 0 if device.type == "mps" else args.num_workers
    pin = device.type == "cuda"
    if ddp_enabled:
        sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=nw, pin_memory=pin, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=nw, pin_memory=pin, drop_last=True,
        )

    loss_list = [x.strip() for x in args.losses.split(",")]
    if args.use_prediction_consistency and "prediction_consistency" not in loss_list:
        loss_list.append("prediction_consistency")
    loss_weights = {}
    if args.loss_weights:
        import json
        loss_weights = json.loads(args.loss_weights.replace("'", "\""))

    adapter_config = {
        "variant": args.variant,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "late_block_start": args.late_block_start,
        "adapt_cause": args.adapt_cause,
    }

    train_semantic_adapter(
        backbone, segment, cluster, train_loader, device, args.output_dir,
        losses=loss_list, loss_weights=loss_weights, lr=args.lr, epochs=args.epochs,
        save_every=args.save_every, lambda_depth=args.lambda_depth,
        depth_shift=args.depth_shift, adapter_config=adapter_config,
        teacher_backbone=teacher_backbone, teacher_segment=teacher_segment,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adapter_lr_mult=args.adapter_lr_mult,
        use_amp=args.use_amp,
        confidence_filter=args.confidence_filter,
        confidence_p=args.confidence_p,
    )

    if ddp_enabled:
        dist.destroy_process_group()

    if is_main_process():
        logger.info("Training complete! Checkpoints in %s", args.output_dir)
        logger.info("Next: generate adapted pseudo-labels with:")
        logger.info("  python mbps_pytorch/generate_semantic_pseudolabels_adapted.py \\")
        logger.info("    --checkpoint %s/best.pt --data_dir %s", args.output_dir, args.data_dir)


if __name__ == "__main__":
    main()
