"""
Train CAUSE-TR with DINOv2 ViT-B/14 on Cityscapes from scratch.

Reverse-engineered from official CAUSE modules (no official training code exists).

Two stages:
  Stage A: Modularity-based codebook learning (~10 epochs)
  Stage B: Segment_TR head + contrastive/cluster loss (~40 epochs)

All hyperparameters matched to official pretrained checkpoint config:
  DINOv2 ViT-B/14, dim=768, patch_size=14, resolution=322
  num_codebook=2048, reduced_dim=90, projection_dim=2048, n_classes=27

Usage:
    python refs/cause/train_cause_tr_dinov2.py \
        --data_dir /Users/qbit-glitch/Desktop/datasets \
        --output_dir refs/cause/CAUSE_dinov2_retrain \
        --device mps

    # Stage A only:
    python refs/cause/train_cause_tr_dinov2.py --stage codebook ...

    # Stage B only (requires codebook from stage A):
    python refs/cause/train_cause_tr_dinov2.py --stage heads \
        --codebook_path refs/cause/CAUSE_dinov2_retrain/modular.npy ...
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add CAUSE repo to path
CAUSE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CAUSE_ROOT)

from modules.segment import Segment_TR
from modules.segment_module import (
    Cluster,
    compute_modularity_based_codebook,
    ema_init,
    ema_update,
    reset,
    stochastic_sampling,
    transform,
    untransform,
)
from loader.dataloader import ContrastiveSegDataset
from utils.utils import (
    ckpt_to_arch,
    ckpt_to_name,
    freeze,
    get_cococity_transform,
)

logger = logging.getLogger(__name__)

# ---- Official config (from pretrained checkpoint analysis) ----
TRAIN_RESOLUTION = 322
PATCH_SIZE = 14
NUM_PATCHES = TRAIN_RESOLUTION ** 2 // PATCH_SIZE ** 2  # 529
DIM = 768
REDUCED_DIM = 90
PROJECTION_DIM = 2048
NUM_CODEBOOK = 2048
N_CLASSES = 27

# Training hyperparams (from module code defaults)
CONTRASTIVE_TEMP = 0.07
POS_THRESH = 0.3
NEG_THRESH = 0.1
BANK_MAX_SIZE = 100
EMA_MOMENTUM = 0.99


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def patch_cluster_for_device(cluster: Cluster, device: torch.device) -> None:
    """Monkey-patch Cluster bank ops to use specified device instead of .cuda()."""

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
            bank_vq_feat = torch.cat(
                [bank_vq_feat, self.codebook[key].unsqueeze(0).repeat(num, 1)], dim=0
            )
            bank_proj_feat_ema = torch.cat(
                [bank_proj_feat_ema, self.prime_bank[key]], dim=0
            )
        self.flat_norm_bank_vq_feat = F.normalize(bank_vq_feat, dim=1)
        self.flat_norm_bank_proj_feat_ema = F.normalize(bank_proj_feat_ema, dim=1)

    def contrastive_ema_mps_safe(
        self, feat, proj_feat, proj_feat_ema,
        temp=0.07, pos_thresh=0.3, neg_thresh=0.1,
    ):
        """MPS-safe contrastive loss: masked mean instead of torch.where indexing."""
        from modules.segment_module import vqt, flatten
        from torch import randperm as perm

        vq_feat = vqt(feat, self.codebook)
        norm_vq_feat = F.normalize(vq_feat, dim=2)
        flat_norm_vq_feat = flatten(norm_vq_feat)

        norm_proj_feat = F.normalize(proj_feat, dim=2)
        norm_proj_feat_ema = F.normalize(proj_feat_ema, dim=2)
        flat_norm_proj_feat_ema = flatten(norm_proj_feat_ema)

        loss_NCE_list = []
        for batch_ind in range(proj_feat.shape[0]):
            anchor_vq_feat = norm_vq_feat[batch_ind]
            anchor_proj_feat = norm_proj_feat[batch_ind]

            cs_st = anchor_proj_feat @ flat_norm_proj_feat_ema.T
            codebook_distance = anchor_vq_feat @ flat_norm_vq_feat.T
            bank_codebook_distance = anchor_vq_feat @ self.flat_norm_bank_vq_feat.T

            pos_mask = (codebook_distance > pos_thresh).float()
            neg_mask = (codebook_distance < neg_thresh).float()

            auto_mask = torch.ones_like(pos_mask)
            n = pos_mask.shape[0]
            auto_mask[:, batch_ind * n:(batch_ind + 1) * n].fill_diagonal_(0)
            pos_mask = pos_mask * auto_mask

            cs_teacher = cs_st / temp
            shifted = cs_teacher - cs_teacher.max(dim=1, keepdim=True)[0].detach()
            denom = (shifted.exp() * (pos_mask + neg_mask)).sum(dim=1, keepdim=True)
            loss_matrix = -shifted + torch.log(denom.clamp(min=1e-9))

            # MPS-safe: masked mean instead of torch.where indexing
            pos_count = pos_mask.sum()
            if pos_count > 0:
                loss_NCE_list.append((loss_matrix * pos_mask).sum() / pos_count)

            # Bank part
            if self.flat_norm_bank_proj_feat_ema.shape[0] != 0:
                cs_bank = anchor_proj_feat @ self.flat_norm_bank_proj_feat_ema.T
                bank_pos = (bank_codebook_distance > pos_thresh).float()
                bank_neg = (bank_codebook_distance < neg_thresh).float()

                cs_bank_t = cs_bank / temp
                shifted_bank = cs_bank_t - cs_bank_t.max(dim=1, keepdim=True)[0].detach()
                denom_bank = (shifted_bank.exp() * (bank_pos + bank_neg)).sum(
                    dim=1, keepdim=True
                )
                loss_bank = -shifted_bank + torch.log(denom_bank.clamp(min=1e-9))

                bank_count = bank_pos.sum()
                if bank_count > 0:
                    loss_NCE_list.append((loss_bank * bank_pos).sum() / bank_count)

        if len(loss_NCE_list) == 0:
            return torch.tensor(0.0, device=feat.device, requires_grad=True)
        return sum(loss_NCE_list) / float(len(loss_NCE_list))

    import types
    cluster.bank_init = types.MethodType(bank_init_device, cluster)
    cluster.bank_compute = types.MethodType(bank_compute_device, cluster)
    cluster.contrastive_ema_with_codebook_bank = types.MethodType(
        contrastive_ema_mps_safe, cluster
    )


def patch_modularity_for_device(device: torch.device):
    """Monkey-patch get_modularity_matrix_and_edge to use specified device."""
    import modules.segment_module as sm

    original_fn = sm.get_modularity_matrix_and_edge

    def patched_fn(x, mode='cos'):
        if mode == 'cos':
            norm = F.normalize(x, dim=2)
            A = (norm @ norm.transpose(2, 1)).clamp(0)
        elif mode == 'l2':
            A = sm.compute_self_distance_batch(x)
        A = A - A * torch.eye(A.shape[1], device=device)
        d = A.sum(dim=2, keepdims=True)
        e = A.sum(dim=(1, 2), keepdims=True)
        W = A - (d / e) @ (d.transpose(2, 1) / e) * e
        return W, e

    sm.get_modularity_matrix_and_edge = patched_fn


def load_backbone(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load and freeze DINOv2 backbone."""
    import models.dinov2vit as model_module
    arch = ckpt_to_arch(ckpt_path)
    net = getattr(model_module, arch)()
    state = torch.load(ckpt_path, map_location=device)
    msg = net.load_state_dict(state, strict=False)
    logger.info(f"Backbone loaded: {msg}")
    net = net.to(device)
    freeze(net)
    return net


def build_args_namespace(args) -> object:
    """Build args namespace compatible with CAUSE module constructors."""
    from types import SimpleNamespace
    ns = SimpleNamespace(
        dim=DIM,
        reduced_dim=REDUCED_DIM,
        projection_dim=PROJECTION_DIM,
        num_codebook=NUM_CODEBOOK,
        n_classes=N_CLASSES,
        num_queries=NUM_PATCHES,
    )
    return ns


# ---------------------------------------------------------------------------
# Stage A: Modularity codebook
# ---------------------------------------------------------------------------


def train_codebook(
    net: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    epochs: int = 10,
    lr: float = 1e-3,
) -> np.ndarray:
    """Stage A: Learn modularity-based codebook."""
    logger.info(f"=== Stage A: Codebook learning ({epochs} epochs, lr={lr}) ===")

    codebook = nn.Parameter(torch.empty(NUM_CODEBOOK, DIM, device=device))
    reset(codebook, NUM_CODEBOOK)
    optimizer = torch.optim.Adam([codebook], lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        prog = tqdm(train_loader, desc=f"Codebook ep{epoch+1}/{epochs}")
        for batch in prog:
            img = batch["img"].to(device)
            with torch.no_grad():
                feat = net(img)[:, 1:, :]  # strip CLS

            loss = compute_modularity_based_codebook(codebook, feat, grid=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            prog.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(count, 1)
        logger.info(f"  Codebook epoch {epoch+1}: avg_loss={avg:.4f}")

    cb_np = codebook.detach().cpu().numpy()
    cb_path = os.path.join(output_dir, "modular.npy")
    np.save(cb_path, cb_np)
    logger.info(f"Codebook saved: {cb_path} shape={cb_np.shape}")
    return cb_np


# ---------------------------------------------------------------------------
# Stage B: Segment head + contrastive + cluster
# ---------------------------------------------------------------------------


def train_heads(
    net: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    codebook_np: np.ndarray,
    epochs: int = 40,
    lr: float = 5e-5,
    val_every: int = 5,
    save_every: int = 10,
    resume_epoch: int = 0,
    resume_dir: str = None,
) -> None:
    """Stage B: Train Segment_TR heads with contrastive + cluster loss."""
    logger.info(f"=== Stage B: Head training ({epochs} epochs, lr={lr}) ===")

    ns = build_args_namespace(None)
    segment = Segment_TR(ns).to(device)
    cluster = Cluster(ns).to(device)

    # Load codebook into cluster
    cb = torch.from_numpy(codebook_np).to(device)
    cluster.codebook.data = cb
    cluster.codebook.requires_grad = False

    # Set codebook on both decoder heads
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    # Init EMA
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)

    # Resume from checkpoint
    if resume_dir and os.path.isdir(resume_dir):
        seg_path = os.path.join(resume_dir, "segment_tr.pth")
        clust_path = os.path.join(resume_dir, "cluster_tr.pth")
        if os.path.exists(seg_path):
            segment.load_state_dict(torch.load(seg_path, map_location=device))
            logger.info(f"Resumed segment from {seg_path}")
        if os.path.exists(clust_path):
            state = torch.load(clust_path, map_location=device)
            cluster.load_state_dict(state, strict=False)
            # Re-set codebook after load
            cluster.codebook.data = cb
            cluster.codebook.requires_grad = False
            logger.info(f"Resumed cluster from {clust_path}")
        # Re-set codebook on decoder heads after loading
        segment.head.codebook = cb
        segment.head_ema.codebook = cb

    # Patch bank for device
    patch_cluster_for_device(cluster, device)
    cluster.bank_init()

    # Optimizer: segment heads + cluster_probe
    params = (
        list(segment.head.parameters())
        + list(segment.projection_head.parameters())
        + [cluster.cluster_probe]
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    # Import eval tool
    from eval_cause_tr_dinov2 import NiceTool

    best_miou = 0.0
    if resume_epoch > 0:
        logger.info(f"Resuming from epoch {resume_epoch}")

    for epoch in range(resume_epoch, epochs):
        segment.train()
        cluster.train()
        total_contrastive = 0.0
        total_cluster = 0.0
        count = 0

        prog = tqdm(train_loader, desc=f"Heads ep{epoch+1}/{epochs}")
        for batch in prog:
            img = batch["img"].to(device)

            with torch.no_grad():
                feat = net(img)[:, 1:, :]  # (B, N, D)

            # Student forward
            seg_feat = segment.head(feat, drop=segment.dropout)
            proj_feat = segment.projection_head(seg_feat)

            # Teacher (EMA) forward — no grad
            with torch.no_grad():
                seg_feat_ema = segment.head_ema(feat)
                proj_feat_ema = segment.projection_head_ema(seg_feat_ema)

            # Bank compute (uses accumulated bank entries)
            cluster.bank_compute()

            # Contrastive loss
            loss_contrastive = cluster.contrastive_ema_with_codebook_bank(
                feat, proj_feat, proj_feat_ema,
                temp=CONTRASTIVE_TEMP,
                pos_thresh=POS_THRESH,
                neg_thresh=NEG_THRESH,
            )

            # Cluster centroid loss
            loss_cluster, _ = cluster.forward_centroid(seg_feat_ema)

            loss = loss_contrastive + loss_cluster

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            ema_update(segment.head, segment.head_ema, lamb=EMA_MOMENTUM)
            ema_update(
                segment.projection_head,
                segment.projection_head_ema,
                lamb=EMA_MOMENTUM,
            )

            # Bank update
            with torch.no_grad():
                cluster.bank_update(feat, proj_feat_ema, max_num=BANK_MAX_SIZE)

            total_contrastive += loss_contrastive.item()
            total_cluster += loss_cluster.item()
            count += 1
            prog.set_postfix(
                nce=f"{loss_contrastive.item():.3f}",
                clust=f"{loss_cluster.item():.3f}",
            )

        avg_nce = total_contrastive / max(count, 1)
        avg_clust = total_cluster / max(count, 1)
        logger.info(
            f"  Heads epoch {epoch+1}: nce={avg_nce:.4f}, cluster={avg_clust:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            _save_checkpoint(segment, cluster, output_dir, epoch + 1)

        # Validation
        if test_loader and ((epoch + 1) % val_every == 0 or epoch == epochs - 1):
            nice = NiceTool(N_CLASSES, device)
            segment.eval()
            cluster.eval()
            miou = _validate(net, segment, cluster, nice, test_loader, device)
            nice.reset()
            logger.info(f"  Val epoch {epoch+1}: mIoU={miou:.1f}%")
            if miou > best_miou:
                best_miou = miou
                _save_checkpoint(segment, cluster, output_dir, epoch=0)
                logger.info(f"  New best mIoU={best_miou:.1f}% — saved as epoch_000 (best)")
            segment.train()
            cluster.train()


def _validate(
    net: nn.Module,
    segment: Segment_TR,
    cluster: Cluster,
    nice,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Run no-CRF validation, return final mIoU."""
    prog = tqdm(test_loader, desc="  Val", leave=False)
    for batch in prog:
        img = batch["img"].to(device)
        label = batch["label"].to(device)
        with torch.no_grad():
            feat = net(img)[:, 1:, :]
            seg_feat_ema = segment.head_ema(feat)
            interp = F.interpolate(
                transform(seg_feat_ema),
                label.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preds = cluster.forward_centroid(untransform(interp), inference=True)
            metrics, desc = nice.eval(preds, label)
        prog.set_postfix_str(desc)
    return metrics["mIoU"]


def _save_checkpoint(
    segment: Segment_TR,
    cluster: Cluster,
    output_dir: str,
    epoch: int,
) -> None:
    """Save segment and cluster checkpoints."""
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    seg_path = os.path.join(epoch_dir, "segment_tr.pth")
    torch.save(segment.state_dict(), seg_path)

    clust_path = os.path.join(epoch_dir, "cluster_tr.pth")
    torch.save(cluster.state_dict(), clust_path)

    logger.info(f"  Saved checkpoint: {epoch_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAUSE-TR DINOv2")
    parser.add_argument(
        "--data_dir",
        default="/Users/qbit-glitch/Desktop/datasets",
    )
    parser.add_argument(
        "--output_dir",
        default="refs/cause/CAUSE_dinov2_retrain",
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoint/dinov2_vit_base_14.pth",
    )
    parser.add_argument("--stage", default="all", choices=["all", "codebook", "heads"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)

    # Stage A
    parser.add_argument("--codebook_epochs", default=10, type=int)
    parser.add_argument("--codebook_lr", default=1e-3, type=float)
    parser.add_argument("--codebook_path", default=None, help="Skip stage A, load this")

    # Stage B
    parser.add_argument("--head_epochs", default=40, type=int)
    parser.add_argument("--head_lr", default=5e-5, type=float)
    parser.add_argument("--val_every", default=5, type=int)
    parser.add_argument("--save_every", default=10, type=int)
    parser.add_argument("--resume_epoch", default=0, type=int, help="Resume from this epoch")
    parser.add_argument("--resume_dir", default=None, help="Dir with segment_tr.pth + cluster_tr.pth")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Resolve paths to absolute BEFORE chdir
    args.output_dir = os.path.abspath(args.output_dir)
    if args.codebook_path:
        args.codebook_path = os.path.abspath(args.codebook_path)
    if args.resume_dir:
        args.resume_dir = os.path.abspath(args.resume_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "train.log"), mode="a"
            ),
        ],
    )

    # Save config
    config = {k: str(v) for k, v in vars(args).items()}
    config.update({
        "train_resolution": TRAIN_RESOLUTION,
        "patch_size": PATCH_SIZE,
        "num_patches": NUM_PATCHES,
        "dim": DIM,
        "num_codebook": NUM_CODEBOOK,
        "n_classes": N_CLASSES,
    })
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Patch modularity for device
    patch_modularity_for_device(device)

    # Change to CAUSE root for relative checkpoint paths
    original_cwd = os.getcwd()
    os.chdir(CAUSE_ROOT)

    logger.info(f"Device: {device}, Seed: {args.seed}")
    logger.info(f"Output: {args.output_dir}")

    # Load backbone
    net = load_backbone(args.ckpt, device)

    # Data loaders
    get_transform = get_cococity_transform
    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir,
        dataset_name="cityscapes",
        crop_type="five",
        image_set="train",
        transform=get_transform(TRAIN_RESOLUTION, False),
        target_transform=get_transform(TRAIN_RESOLUTION, True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    test_loader = None
    try:
        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=args.data_dir,
            dataset_name="cityscapes",
            crop_type=None,
            image_set="val",
            transform=get_transform(TRAIN_RESOLUTION, False),
            target_transform=get_transform(TRAIN_RESOLUTION, True),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        logger.info(f"Train: {len(train_dataset)} crops, Val: {len(test_dataset)} images")
    except RuntimeError:
        logger.warning("Val data unavailable (broken symlink?). Training without validation.")
        logger.info(f"Train: {len(train_dataset)} crops, Val: DISABLED")

    # Stage A: Codebook
    codebook_np = None
    if args.stage in ("all", "codebook"):
        if args.codebook_path and os.path.exists(args.codebook_path):
            logger.info(f"Loading existing codebook: {args.codebook_path}")
            codebook_np = np.load(args.codebook_path)
        else:
            codebook_np = train_codebook(
                net, train_loader, device, args.output_dir,
                epochs=args.codebook_epochs, lr=args.codebook_lr,
            )

    if args.stage == "codebook":
        os.chdir(original_cwd)
        logger.info("Stage A complete. Run with --stage heads to continue.")
        return

    # Stage B: Heads
    if codebook_np is None:
        cb_path = args.codebook_path or os.path.join(args.output_dir, "modular.npy")
        if not os.path.exists(cb_path):
            raise FileNotFoundError(
                f"Codebook not found at {cb_path}. Run --stage codebook first."
            )
        codebook_np = np.load(cb_path)
        logger.info(f"Loaded codebook: {cb_path} shape={codebook_np.shape}")

    train_heads(
        net, train_loader, test_loader, device, args.output_dir,
        codebook_np,
        epochs=args.head_epochs, lr=args.head_lr,
        val_every=args.val_every, save_every=args.save_every,
        resume_epoch=args.resume_epoch, resume_dir=args.resume_dir,
    )

    os.chdir(original_cwd)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
