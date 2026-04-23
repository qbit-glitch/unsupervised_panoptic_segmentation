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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def dino_distillation_loss(student_feat, teacher_feat, temp_student=0.1, temp_teacher=0.07):
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1).detach()
    student_logits = student_feat / temp_student
    teacher_probs = F.softmax(teacher_feat / temp_teacher, dim=-1)
    loss = -(teacher_probs * F.log_softmax(student_logits, dim=-1)).sum(dim=-1)
    return loss.mean()


def cross_view_consistency_loss(feat1, feat2):
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)
    return (1 - (feat1 * feat2).sum(dim=-1)).mean()


def sample_coords(batch_size, feature_samples, device):
    return torch.rand(batch_size, feature_samples, feature_samples, 2, device=device) * 2 - 1


def grid_sample(t, coords):
    return F.grid_sample(t, coords, padding_mode="zeros", align_corners=True, mode="bilinear")


def depth_correlation_loss(code, depth, feature_samples=11, shift=0.0):
    B = code.shape[0]
    coords = sample_coords(B, feature_samples, code.device)
    code_sampled = grid_sample(code, coords)
    depth_sampled = grid_sample(depth, coords)

    def norm(t):
        return F.normalize(t, dim=1, eps=1e-10)

    cd = torch.einsum("nchw,ncij->nhwij", norm(code_sampled), norm(code_sampled))
    dd = torch.einsum("nchw,ncij->nhwij", norm(depth_sampled), norm(depth_sampled))
    loss = -cd.clamp(0.0, 0.8) * (dd - shift)
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
                img_pil = T.ToPILImage()(img)
                item["img_aug"] = T.ToTensor()(self.aug_transform(img_pil))
            else:
                item["img_aug"] = self.aug_transform(img)
        return item


def train_semantic_adapter(
    backbone, segment, cluster, train_loader, device, output_dir,
    losses, loss_weights, lr=1e-4, epochs=10, save_every=5,
    lambda_depth=0.05, depth_shift=0.0, temp_student=0.1, temp_teacher=0.07,
    adapter_config=None, teacher_backbone=None, teacher_segment=None,
):
    logger.info("=== Semantic Adapter Training ===")
    logger.info("Losses: %s", losses)
    logger.info("Weights: %s", loss_weights)
    logger.info("LR=%.1e, epochs=%d", lr, epochs)

    adapter_params = [p for p in backbone.parameters() if p.requires_grad]
    adapter_params += [p for p in segment.parameters() if p.requires_grad]
    adapter_params += [p for p in cluster.parameters() if p.requires_grad]
    logger.info("Trainable adapter params: %d", sum(p.numel() for p in adapter_params))

    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float("inf")

    for epoch in range(epochs):
        backbone.train()
        segment.train()
        cluster.train()
        totals = {k: 0.0 for k in losses + ["total"]}
        count = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in prog:
            img = batch["img"].to(device)
            depth = batch.get("depth")
            if depth is not None:
                depth = depth.to(device)

            set_dinov2_spatial_dims(backbone, h_patches=23, w_patches=23)

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
                l_dist = dino_distillation_loss(feat_student, feat_teacher, temp_student, temp_teacher)
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

            if "depth_cluster" in losses and depth is not None:
                code_student = transform(seg_feat_student)
                l_depth = depth_correlation_loss(code_student, depth, shift=depth_shift)
                w = loss_weights.get("depth_cluster", lambda_depth)
                loss_total = loss_total + w * l_depth
                totals["depth_cluster"] += l_depth.item()

            if "cause_cluster" in losses:
                with torch.no_grad():
                    # Use teacher features (frozen original backbone) for EMA head
                    feat_for_ema = teacher_backbone(img)[:, 1:, :] if teacher_backbone is not None else feat_teacher
                    seg_feat_ema = segment.head_ema(feat_for_ema)
                loss_cluster, _ = cluster.forward_centroid(seg_feat_ema)
                w = loss_weights.get("cause_cluster", 1.0)
                loss_total = loss_total + w * loss_cluster
                totals["cause_cluster"] += loss_cluster.item()

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            optimizer.step()
            ema_update(segment.head, segment.head_ema, lamb=EMA_MOMENTUM)

            totals["total"] += loss_total.item()
            count += 1
            prog.set_postfix(total=f"{loss_total.item():.4f}")

        avg_total = totals["total"] / max(count, 1)
        logger.info("Epoch %d: total=%.4f", epoch + 1, avg_total)
        for k in losses:
            logger.info("  %s: %.4f", k, totals[k] / max(count, 1))
        scheduler.step()

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}.pt")
            torch.save({"backbone": backbone.state_dict(), "segment": segment.state_dict(),
                        "cluster": cluster.state_dict(), "epoch": epoch + 1,
                        "adapter_config": adapter_config}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        if avg_total < best_loss:
            best_loss = avg_total
            ckpt_path = os.path.join(output_dir, "best.pt")
            torch.save({"backbone": backbone.state_dict(), "segment": segment.state_dict(),
                        "cluster": cluster.state_dict(), "epoch": epoch + 1,
                        "adapter_config": adapter_config}, ckpt_path)
            logger.info("New best loss=%.4f, saved to %s", best_loss, ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Train semantic adapters (DINOv2 + CAUSE-TR)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
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
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(__file__).resolve().parent.parent / "refs" / "cause")

    cityscapes_root = os.path.join(args.data_dir, "cityscapes")

    logger.info("Loading DINOv2 backbone...")
    backbone_path = os.path.join(args.checkpoint_dir, "checkpoint", "dinov2_vit_base_14.pth")
    backbone = dinov2_vit_base_14()
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=False)

    # Create frozen teacher backbone (original, no adapters)
    logger.info("Creating frozen teacher backbone...")
    teacher_backbone = dinov2_vit_base_14()
    teacher_backbone.load_state_dict(state, strict=False)
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
    backbone = backbone.to(device)

    logger.info("Loading CAUSE Segment_TR...")
    cause_args = build_cause_args()
    segment = Segment_TR(cause_args).to(device)
    cluster = Cluster(cause_args).to(device)

    seg_path = os.path.join(
        args.checkpoint_dir, "CAUSE", "cityscapes",
        "dinov2_vit_base_14", "2048", "segment_tr.pth",
    )
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    segment.load_state_dict(seg_state, strict=False)

    # Create frozen teacher segment (original, no adapters)
    logger.info("Creating frozen teacher segment...")
    teacher_segment = Segment_TR(cause_args).to(device)
    teacher_segment.load_state_dict(seg_state, strict=False)
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

    if args.adapt_cause:
        inject_lora_into_cause_tr(
            segment, variant=args.variant, rank=args.rank, alpha=args.alpha,
            dropout=args.dropout, adapt_head=True, adapt_projection=False, adapt_ema=False,
        )
    freeze_non_adapter_params(segment)

    patch_cluster_for_device(cluster, device)
    cluster.bank_init()
    freeze_non_adapter_params(cluster)
    if "cause_cluster" in args.losses:
        cluster.cluster_probe.requires_grad = True

    total_trainable = count_adapter_params(backbone) + count_adapter_params(segment) + count_adapter_params(cluster)
    logger.info("Total trainable params: %d", total_trainable)

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
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )

    loss_list = [x.strip() for x in args.losses.split(",")]
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
    )

    logger.info("Training complete! Checkpoints in %s", args.output_dir)
    logger.info("Next: generate adapted pseudo-labels with:")
    logger.info("  python mbps_pytorch/generate_semantic_pseudolabels_adapted.py \\")
    logger.info("    --checkpoint %s/best.pt --data_dir %s", args.output_dir, args.data_dir)


if __name__ == "__main__":
    main()
