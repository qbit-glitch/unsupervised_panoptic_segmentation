#!/usr/bin/env python3
"""Train DINOSAUR (Slot Attention on DINOv3 features) for instance discovery.

Based on the DINOSAUR architecture (Seitzer et al., ICLR 2023) with reference
implementations from:
  - https://github.com/gorkaydemir/DINOSAUR (unofficial, PyTorch)
  - https://github.com/amazon-science/object-centric-learning-framework (official)

Architecture:
  - Encoder: LayerNorm + Linear(768 → slot_dim) + ReLU + Linear + positional encoding
  - Slot Attention: K slots compete to explain patches via iterative attention (GRU update)
  - Spatial Broadcast Decoder: replicate slot → all positions + pos embed → MLP → (768+1)
    Each slot predicts (features, mask_logit). Masks = softmax(mask_logits) over slots.
    Reconstruction = sum(features * masks) per position.
  - Loss: MSE(reconstruction, target_features)

Usage:
    python mbps_pytorch/train_dinosaur.py \
        --feature_dir /path/to/cityscapes/dinov3_features/train \
        --val_feature_dir /path/to/cityscapes/dinov3_features/val \
        --output_dir checkpoints/dinosaur/ \
        --num_slots 15 --slot_dim 256 --epochs 200 --device mps
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Grid constants for 512×1024 input with DINOv3 ViT-B/16
GRID_H, GRID_W = 32, 64
N_PATCHES = GRID_H * GRID_W  # 2048
FEAT_DIM = 768


class CityscapesDINOFeatureDataset(Dataset):
    """Load pre-extracted DINOv3 features."""

    def __init__(self, feature_dir, limit=None):
        self.files = sorted(Path(feature_dir).rglob("*.npy"))
        if limit:
            self.files = self.files[:limit]
        logger.info(f"Dataset: {len(self.files)} feature files from {feature_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = np.load(self.files[idx]).astype(np.float32)  # (2048, 768)
        return torch.from_numpy(features)


class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for patch grid."""

    def __init__(self, dim, n_patches=N_PATCHES):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


class SlotAttention(nn.Module):
    """Slot Attention module (Locatello et al., 2020).

    Patches compete for slots: softmax over slot dimension ensures each patch
    is explained by exactly one slot (soft assignment that sums to 1 over slots).
    """

    def __init__(self, num_slots=15, dim=256, iters=3, eps=1e-8, hidden_dim=512):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        self.eps = eps

        # Learnable slot initialization (Gaussian)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim) * (dim ** -0.5))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))

        # Attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Slot update
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs):
        """
        Args:
            inputs: (B, N, D) — encoded patch features

        Returns:
            slots: (B, K, D) — final slot representations
            attn: (B, K, N) — slot attention weights (softmax over slots dim)
        """
        B, N, D = inputs.shape

        # Initialize slots with learned Gaussian
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_logsigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # (B, N, D)
        v = self.to_v(inputs)  # (B, N, D)
        scale = D ** -0.5

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)  # (B, K, D)

            # Attention: softmax over slots (competition for patches)
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * scale  # (B, K, N)
            attn = F.softmax(attn_logits, dim=1)  # normalize over slots

            # Weighted mean of values (normalize over patches)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)  # (B, K, N)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)  # (B, K, D)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D),
            ).reshape(B, -1, D)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Final attention for mask extraction
        slots_final = self.norm_slots(slots)
        q = self.to_q(slots_final)
        attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * scale
        attn = F.softmax(attn_logits, dim=1)  # (B, K, N)

        return slots, attn


class SpatialBroadcastDecoder(nn.Module):
    """Spatial Broadcast Decoder (SBD) — the DINOSAUR decoder.

    For each slot:
      1. Replicate slot vector to all N patch positions
      2. Add learnable positional embeddings
      3. Pass through MLP → output (feat_dim + 1) per position
      4. Split into predicted features (feat_dim) and mask logit (1)

    Final reconstruction:
      masks = softmax(all_mask_logits, dim=slots)  # competition
      recon = sum(masks * features, dim=slots)      # weighted mix
    """

    def __init__(self, slot_dim=256, output_dim=768, hidden_dim=2048,
                 n_layers=3, n_patches=N_PATCHES):
        super().__init__()
        self.n_patches = n_patches

        # Learnable positional embeddings for decoder
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, slot_dim) * 0.02)

        # MLP: slot_dim → hidden → hidden → ... → (output_dim + 1)
        layers = []
        in_dim = slot_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim + 1))  # +1 for mask logit
        self.mlp = nn.Sequential(*layers)

    def forward(self, slots):
        """
        Args:
            slots: (B, K, slot_dim)

        Returns:
            recon_features: (B, K, N, feat_dim) — per-slot predicted features
            mask_logits: (B, K, N, 1) — per-slot mask logits
        """
        B, K, D = slots.shape

        # Spatial broadcast: replicate each slot to all positions
        slots_broadcast = slots.unsqueeze(2).expand(-1, -1, self.n_patches, -1)
        # (B, K, N, D)

        # Add positional embeddings
        pos = self.pos_embed.unsqueeze(1).expand(-1, K, -1, -1)  # (1, K, N, D)
        x = slots_broadcast + pos  # (B, K, N, D)

        # MLP
        x = x.reshape(B * K, self.n_patches, -1)
        x = self.mlp(x)  # (B*K, N, feat_dim+1)
        x = x.reshape(B, K, self.n_patches, -1)

        # Split features and mask logits
        recon_features = x[:, :, :, :-1]  # (B, K, N, feat_dim)
        mask_logits = x[:, :, :, -1:]     # (B, K, N, 1)

        return recon_features, mask_logits


class DINOSAUR(nn.Module):
    """DINOSAUR: Slot Attention on DINOv3 features for scene decomposition.

    Architecture follows the reference implementation:
      1. Encoder projects 768-dim features to slot_dim
      2. Slot Attention groups features into K slots
      3. Spatial Broadcast Decoder reconstructs features per slot
      4. Masks come from decoder (not just attention weights)
    """

    def __init__(self, input_dim=768, slot_dim=256, num_slots=15,
                 num_iters=3, decoder_hidden=2048, decoder_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        # Encoder: project features to slot dimension + positional encoding
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.pos_enc = PositionalEncoding2D(slot_dim)

        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            iters=num_iters,
        )

        # Spatial Broadcast Decoder
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            output_dim=input_dim,
            hidden_dim=decoder_hidden,
            n_layers=decoder_layers,
        )

    def forward(self, features):
        """
        Args:
            features: (B, N, 768) — DINOv3 patch features

        Returns:
            recon: (B, N, 768) — reconstructed features
            masks: (B, K, N) — decoder masks (softmax over slots)
            slots: (B, K, slot_dim) — slot representations
            attn: (B, K, N) — slot attention weights
        """
        B, N, D = features.shape

        # Encode
        x = self.encoder(features)  # (B, N, slot_dim)
        x = self.pos_enc(x)

        # Slot Attention
        slots, attn = self.slot_attention(x)

        # Decode with Spatial Broadcast Decoder
        recon_per_slot, mask_logits = self.decoder(slots)
        # recon_per_slot: (B, K, N, feat_dim)
        # mask_logits: (B, K, N, 1)

        # Compute masks via softmax over slots
        masks = F.softmax(mask_logits.squeeze(-1), dim=1)  # (B, K, N)

        # Reconstruct: weighted combination of per-slot features
        recon = (masks.unsqueeze(-1) * recon_per_slot).sum(dim=1)  # (B, N, feat_dim)

        return recon, masks, slots, attn

    @torch.no_grad()
    def get_masks(self, features, use_decoder_masks=True):
        """Get instance masks from the model.

        Args:
            features: (B, N, 768)
            use_decoder_masks: if True, use decoder masks; else attention masks

        Returns:
            masks: (B, K, GRID_H, GRID_W) — per-slot spatial masks (hard assignment)
            soft_masks: (B, K, N) — soft masks
        """
        self.eval()
        _, dec_masks, _, attn = self.forward(features)

        soft_masks = dec_masks if use_decoder_masks else attn

        # Hard assignment: each patch to its highest-weight slot
        assignment = soft_masks.argmax(dim=1)  # (B, N)
        hard_masks = F.one_hot(assignment, self.num_slots).permute(0, 2, 1).float()

        hard_masks = hard_masks.reshape(-1, self.num_slots, GRID_H, GRID_W)
        return hard_masks, soft_masks


def compute_loss(model, features, entropy_weight=0.0):
    """Compute DINOSAUR training loss.

    Primary loss is MSE reconstruction. Optional entropy regularization
    encourages sharp slot assignments.
    """
    recon, masks, slots, attn = model(features)

    # MSE reconstruction loss (target is detached — no backprop through encoder)
    recon_loss = F.mse_loss(recon, features.detach())

    # Optional: entropy regularization on decoder masks
    entropy_loss = torch.tensor(0.0, device=features.device)
    if entropy_weight > 0:
        # masks: (B, K, N) — per-patch distribution over slots
        masks_per_patch = masks.permute(0, 2, 1)  # (B, N, K)
        entropy = -(masks_per_patch * (masks_per_patch + 1e-10).log()).sum(dim=-1)
        entropy_loss = entropy.mean()

    loss = recon_loss + entropy_weight * entropy_loss

    # Metrics
    with torch.no_grad():
        avg_max_mask = masks.max(dim=1)[0].mean().item()
        # Active slots: slots that are the argmax for at least 1% of patches
        assignment = masks.argmax(dim=1)  # (B, N)
        active_per_image = []
        for b in range(assignment.shape[0]):
            unique_slots = assignment[b].unique()
            active_per_image.append(len(unique_slots))
        active_slots = sum(active_per_image) / len(active_per_image)

    metrics = {
        "loss": loss.item(),
        "recon_loss": recon_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "avg_max_mask": avg_max_mask,
        "active_slots": active_slots,
    }

    return loss, metrics


def train_epoch(model, dataloader, optimizer, device, entropy_weight):
    model.train()
    total_metrics = {}
    n_batches = 0

    for batch in dataloader:
        features = batch.to(device)

        optimizer.zero_grad()
        loss, metrics = compute_loss(model, features, entropy_weight)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


@torch.no_grad()
def eval_epoch(model, dataloader, device, entropy_weight):
    model.eval()
    total_metrics = {}
    n_batches = 0

    for batch in dataloader:
        features = batch.to(device)
        _, metrics = compute_loss(model, features, entropy_weight)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


def main():
    parser = argparse.ArgumentParser("Train DINOSAUR slot attention")
    parser.add_argument("--feature_dir", required=True,
                        help="Directory with DINOv3 train features")
    parser.add_argument("--val_feature_dir", default=None,
                        help="Directory with DINOv3 val features")
    parser.add_argument("--output_dir", default="checkpoints/dinosaur",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_slots", type=int, default=15)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--num_iters", type=int, default=3,
                        help="Slot attention iterations")
    parser.add_argument("--decoder_hidden", type=int, default=2048,
                        help="Decoder MLP hidden dim")
    parser.add_argument("--decoder_layers", type=int, default=3,
                        help="Number of decoder MLP hidden layers")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--entropy_weight", type=float, default=0.0,
                        help="Entropy regularization weight (0 = MSE only, as in paper)")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of training images (for debugging)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. best.pth)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Datasets
    train_dataset = CityscapesDINOFeatureDataset(args.feature_dir, limit=args.limit)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if args.val_feature_dir:
        val_dataset = CityscapesDINOFeatureDataset(args.val_feature_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Model
    model = DINOSAUR(
        input_dim=FEAT_DIM,
        slot_dim=args.slot_dim,
        num_slots=args.num_slots,
        num_iters=args.num_iters,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"DINOSAUR: {args.num_slots} slots, dim={args.slot_dim}, "
                f"iters={args.num_iters}, decoder={args.decoder_layers}x{args.decoder_hidden}, "
                f"params={n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Step scheduler to the correct epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        if "val_metrics" in ckpt:
            best_val_loss = ckpt["val_metrics"]["loss"]
        logger.info(f"Resumed from epoch {ckpt['epoch']}, "
                    f"best_val_loss={best_val_loss:.4f}, "
                    f"resuming at epoch {start_epoch}")

    logger.info(f"Training for epochs {start_epoch}-{args.epochs}, "
                f"{len(train_dataset)} images, batch_size={args.batch_size}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                    args.entropy_weight)
        scheduler.step()

        val_metrics = None
        if val_loader is not None:
            val_metrics = eval_epoch(model, val_loader, device, args.entropy_weight)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # Log
        log_parts = [
            f"Epoch {epoch:3d}/{args.epochs}",
            f"loss={train_metrics['loss']:.4f}",
            f"recon={train_metrics['recon_loss']:.4f}",
            f"max_mask={train_metrics['avg_max_mask']:.3f}",
            f"active={train_metrics['active_slots']:.1f}",
            f"lr={lr:.6f}",
            f"{elapsed:.1f}s",
        ]
        if val_metrics:
            log_parts.append(f"val={val_metrics['loss']:.4f}")
        logger.info("  ".join(log_parts))

        entry = {"epoch": epoch, "lr": lr, "train": train_metrics}
        if val_metrics:
            entry["val"] = val_metrics
        history.append(entry)

        # Save checkpoint
        is_best = False
        if val_metrics and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            is_best = True

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "train_metrics": train_metrics,
            }
            if val_metrics:
                ckpt["val_metrics"] = val_metrics

            if is_best:
                torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
                logger.info(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(ckpt, os.path.join(args.output_dir, f"epoch_{epoch:04d}.pth"))

    # Save history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
