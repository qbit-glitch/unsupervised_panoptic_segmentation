#!/usr/bin/env python
"""
Stage-2: Spectral Initialization Training

This stage implements multi-scale spectral initialization for slot attention.
Uses frozen DINOv3 features and spectral graph theory to find principled
slot prototypes near object boundaries.

Key Components:
1. DINOv3 Feature Extraction (frozen)
2. Multi-Scale Spectral Initialization (8×, 16×, 32×)
3. Standard Slot Attention refinement
4. Spectral Consistency Loss

Based on ICML 2027 Solution: SpectralDiffusion Framework
Stage-2 provides provably better initialization than random (Theorem 1).
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from models.spectral_init import MultiScaleSpectralInit
from models.dinov3 import DINOv3FeatureExtractor, create_dinov3_extractor
from utils.multi_gpu import setup_device, wrap_model, get_model_state, add_multi_gpu_args
from sklearn.metrics import adjusted_rand_score
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2: Spectral Initialization Training")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    
    # Model - Spectral Init
    parser.add_argument("--scales", type=int, nargs="+", default=[8, 16, 32],
                        help="Multi-scale spectral initialization scales")
    parser.add_argument("--slots-per-scale", type=int, default=4,
                        help="Number of slots per scale (total K = len(scales) * slots_per_scale)")
    parser.add_argument("--k-neighbors", type=int, default=20,
                        help="Number of neighbors for k-NN affinity")
    parser.add_argument("--power-iters", type=int, default=50,
                        help="Power iteration steps for eigenvector computation")
    
    # Model - Backbone
    parser.add_argument("--backbone", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="DINOv3 backbone size")
    
    # Model - Slot Attention
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of slot attention iterations")
    parser.add_argument("--use-mamba", action="store_true", default=False,
                        help="Use Mamba-Slot Attention (default: Standard)")
    
    # Loss weights
    parser.add_argument("--lambda-recon", type=float, default=1.0,
                        help="Reconstruction loss weight")
    parser.add_argument("--lambda-spec", type=float, default=0.5,
                        help="Spectral consistency loss weight (curriculum applied)")
    parser.add_argument("--lambda-div", type=float, default=0.01,
                        help="Slot diversity loss weight (increased for better diversity)")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="./datasets",
                        help="Root directory for datasets")
    parser.add_argument("--dataset", type=str, default="clevr",
                        choices=["clevr", "coco", "cityscapes"],
                        help="Dataset to train on")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size (DINOv3 expects 224 or 518)")
    parser.add_argument("--subset-percent", type=float, default=1.0,
                        help="Subset percentage of dataset to use")
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"])
    add_multi_gpu_args(parser)
    
    # Checkpoints
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--stage1-ckpt", type=str, default=None,
                        help="Path to Stage-1 checkpoint (optional, for decoder weights)")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5)
    
    return parser.parse_args()


class SpectralSlotModel(nn.Module):
    """
    Stage-2 Model: DINOv3 + Spectral Init + Slot Attention + Simple Decoder
    
    This is an intermediate model that validates spectral initialization
    before adding Mamba and Diffusion in later stages.
    """
    
    def __init__(
        self,
        backbone: str = "base",
        scales: list = [8, 16, 32],
        slots_per_scale: int = 4,
        k_neighbors: int = 20,
        power_iters: int = 50,
        num_iterations: int = 3,
        image_size: tuple = (224, 224),
        use_mamba: bool = False,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.scales = scales
        self.num_slots = len(scales) * slots_per_scale
        self.image_size = image_size
        
        # 1. DINOv2 Feature Extractor (frozen) - Note: Using DINOv2 since DINOv3 is not yet available
        # DINOv2-base provides 768-dim features
        self.backbone = DINOv3FeatureExtractor(
            model_name="dinov2-base",  # Using DINOv2 as DINOv3 not available on HuggingFace
            scales=scales,
            freeze=freeze_backbone,
        )
        self.feature_dim = self.backbone.feature_dim
        
        # 2. Multi-Scale Spectral Initialization
        self.spectral_init = MultiScaleSpectralInit(
            scales=scales,
            slots_per_scale=slots_per_scale,
            feature_dim=self.feature_dim,
            k_neighbors=k_neighbors,
            num_power_iters=power_iters,
        )
        
        # 3. Slot Attention (Standard or Mamba)
        if use_mamba:
            from models.mamba_slot import MambaSlotAttention
            self.slot_attention = MambaSlotAttention(
                dim=self.feature_dim,
                num_slots=self.num_slots,
                num_iterations=num_iterations,
            )
        else:
            from models.mamba_slot import StandardSlotAttention
            self.slot_attention = StandardSlotAttention(
                dim=self.feature_dim,
                num_slots=self.num_slots,
                num_iterations=num_iterations,
            )
        
        # 4. Simple Spatial Broadcast Decoder (same as Stage-1)
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=self.feature_dim,
            hidden_dim=128,
            output_size=image_size,
        )
    
    def extract_multiscale_features(self, images: torch.Tensor):
        """Extract DINOv3 multi-scale features."""
        # Preprocess
        images = self.backbone.preprocess(images, size=self.image_size)
        
        # Get multi-scale features
        multiscale_features = self.backbone(images, return_multiscale=True)
        
        # Get flat features for slot attention
        flat_features = self.backbone(images, return_multiscale=False)
        
        return multiscale_features, flat_features
    
    def forward(self, images: torch.Tensor, return_loss: bool = True):
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W] input images
            return_loss: Whether to compute losses
            
        Returns:
            dict with reconstructed, masks, slots, losses
        """
        B = images.shape[0]
        
        # 1. Extract DINOv3 features (multi-scale)
        multiscale_features, flat_features = self.extract_multiscale_features(images)
        
        # 2. Spectral initialization
        slots_init = self.spectral_init(multiscale_features, mode="spectral")
        
        # 3. Compute spectral masks for consistency loss (recomputed each forward!)
        with torch.no_grad():
            spectral_masks = self.spectral_init.get_spectral_masks(multiscale_features)  # [B, K, N]
        
        # 4. Slot attention refinement
        slots, attn = self.slot_attention(flat_features, slots_init)
        
        # 5. Decode to masks
        recon, masks, rgb_per_slot = self.decoder(slots)
        
        outputs = {
            'reconstructed': recon,
            'masks': masks,
            'slots': slots,
            'slots_init': slots_init,
            'attn': attn,
            'rgb_per_slot': rgb_per_slot,
            'spectral_masks': spectral_masks,  # Store for debugging
        }
        
        if return_loss:
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, images)
            
            # Spectral consistency loss: ||M_spectral - M_slot||_F^2
            # Per icml_2027_solution.md: compare MASKS not slot embeddings
            spec_loss = self.compute_spectral_consistency(attn, spectral_masks)
            
            # Diversity loss (prevent slot collapse)
            div_loss = self.compute_diversity_loss(slots)
            
            outputs['loss_recon'] = recon_loss
            outputs['loss_spec'] = spec_loss
            outputs['loss_div'] = div_loss
            
            # Keep as 1D tensors for DataParallel
            outputs['loss_dict'] = {
                'recon': recon_loss.unsqueeze(0),
                'spec': spec_loss.unsqueeze(0),
                'div': div_loss.unsqueeze(0),
            }
        
        return outputs
    
    def compute_spectral_consistency(self, slot_masks, spectral_masks):
        """
        Compute spectral consistency loss.
        
        Per icml_2027_solution.md: L_spec = ||M_spectral - M_slot||_F^2
        Compare MASKS (soft assignments), not slot embeddings.
        
        Args:
            slot_masks: [B, N, K] slot attention masks
            spectral_masks: [B, K, N] spectral clustering masks
        
        Returns:
            loss: Frobenius norm of mask difference
        """
        B = slot_masks.shape[0]
        
        # Transpose spectral masks to match slot masks: [B, K, N] -> [B, N, K]
        spectral_masks_t = spectral_masks.transpose(1, 2)  # [B, N, K]
        
        # Handle resolution mismatch between slot attention and spectral masks
        N_slot = slot_masks.shape[1]
        N_spec = spectral_masks_t.shape[1]
        
        if N_slot != N_spec:
            # Interpolate spectral masks to match slot attention resolution
            # Reshape for interpolation: [B, N, K] -> [B, K, N] -> [B, K, sqrt(N), sqrt(N)]
            K = spectral_masks.shape[1]
            H_spec = int(N_spec ** 0.5)
            H_slot = int(N_slot ** 0.5)
            
            spectral_2d = spectral_masks.reshape(B, K, H_spec, H_spec)
            spectral_2d = F.interpolate(spectral_2d, size=(H_slot, H_slot), mode='bilinear', align_corners=False)
            spectral_masks_t = spectral_2d.reshape(B, K, N_slot).transpose(1, 2)  # [B, N_slot, K]
        
        # Frobenius norm: ||M_spec - M_slot||_F^2
        mask_diff = spectral_masks_t - slot_masks
        frobenius_loss = (mask_diff ** 2).sum(dim=-1).mean()  # Sum over K, mean over B and N
        
        # Store debug info
        frobenius_loss.mask_diff_mean = mask_diff.abs().mean().detach()
        frobenius_loss.slot_mask_entropy = -(slot_masks * torch.log(slot_masks + 1e-8)).sum(dim=-1).mean().detach()
        
        return frobenius_loss
    
    def compute_diversity_loss(self, slots):
        """
        Diversity loss to prevent slot collapse.
        
        Penalizes high cosine similarity between slots.
        """
        B, K, D = slots.shape
        
        # Normalize slots
        slots_norm = F.normalize(slots, dim=-1)
        
        # Compute pairwise similarity
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]
        
        # Mask diagonal
        mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
        off_diag = sim[:, mask].view(B, -1)
        
        # Penalize positive similarities only
        loss = F.relu(off_diag).mean()
        
        return loss
    
    def load_stage1_decoder(self, checkpoint_path: str):
        """Load decoder weights from Stage-1 checkpoint."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading Stage-1 decoder from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Filter decoder weights
            decoder_weights = {
                k.replace('decoder.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('decoder.')
            }
            
            if decoder_weights:
                self.decoder.load_state_dict(decoder_weights, strict=False)
                print(f"  Loaded {len(decoder_weights)} decoder parameters")


class SpatialBroadcastDecoder(nn.Module):
    """Simple spatial broadcast decoder for Stage-2."""
    
    def __init__(
        self,
        slot_dim: int = 768,
        hidden_dim: int = 128,
        output_size: tuple = (224, 224),
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.output_size = output_size
        
        # Build positional grid
        self.register_buffer('grid', self._build_grid(*output_size))
        
        # MLP to project slot + position
        self.initial = nn.Sequential(
            nn.Linear(slot_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # CNN for spatial processing
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, 1),  # 3 RGB + 1 alpha
        )
    
    def _build_grid(self, H: int, W: int) -> torch.Tensor:
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        return torch.stack([grid_x, grid_y], dim=-1)
    
    def forward(self, slots: torch.Tensor):
        """
        Args:
            slots: [B, K, D]
        Returns:
            recon: [B, 3, H, W]
            masks: [B, K, H, W]
            rgb: [B, K, 3, H, W]
        """
        B, K, D = slots.shape
        H, W = self.output_size
        
        # Flatten for processing
        slots_flat = slots.reshape(B * K, D)
        
        # Expand grid
        grid = self.grid.unsqueeze(0).expand(B * K, -1, -1, -1)
        grid_flat = grid.reshape(B * K, H * W, 2)
        
        # Broadcast slots
        slots_broadcast = slots_flat.unsqueeze(1).expand(-1, H * W, -1)
        x = torch.cat([slots_broadcast, grid_flat], dim=-1)
        
        # MLP
        x = self.initial(x)
        x = x.reshape(B * K, H, W, -1).permute(0, 3, 1, 2)
        
        # CNN
        x = self.cnn(x)
        x = x.reshape(B, K, 4, H, W)
        
        # Split RGB and alpha
        rgb = torch.sigmoid(x[:, :, :3])
        mask_logits = x[:, :, 3:4]
        
        # Softmax over slots for masks
        masks = F.softmax(mask_logits, dim=1).squeeze(2)
        
        # Composite reconstruction
        recon = (rgb * masks.unsqueeze(2)).sum(dim=1)
        
        return recon, masks, rgb


def create_dataloaders(args):
    """Create dataloaders for training."""
    if args.dataset == "clevr":
        from data.clevr import CLEVRDataset
        
        clevr_root = os.path.join(args.data_dir, "clevr/CLEVR_v1.0")
        
        full_dataset = CLEVRDataset(
            root_dir=clevr_root,
            split="val",  # Using val split for training (larger)
            image_size=(args.image_size, args.image_size),
            max_objects=args.num_slots if hasattr(args, 'num_slots') else 12,
            return_masks=True,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    # Subset
    n = len(full_dataset)
    n_subset = int(n * args.subset_percent)
    
    train_size = int(0.8 * n_subset)
    val_size = n_subset - train_size
    
    indices = torch.randperm(n)[:n_subset].tolist()
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(full_dataset, indices[train_size:])
    
    pin_memory = args.device == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


def warmup_lr(step, warmup_steps, base_lr):
    """Linear warmup schedule."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def get_spectral_weight(epoch, base_weight=0.5):
    """
    Curriculum for spectral loss weight.
    
    Use MILD spectral guidance - too strong causes slot attention to 
    learn pass-through (just copying slots_init unchanged).
    
    Args:
        epoch: Current epoch
        base_weight: Base lambda_spec value
        
    Returns:
        Adjusted spectral weight
    """
    if epoch < 10:
        # Gentle guidance at start
        multiplier = 0.3
    elif epoch < 30:
        # Medium guidance during main learning
        multiplier = 0.2
    else:
        # Very weak - let reconstruction drive fine-tuning
        multiplier = 0.1
    
    return base_weight * multiplier


def compute_ari(pred_masks, true_masks):
    """Compute Adjusted Rand Index."""
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.detach().cpu()
    if torch.is_tensor(true_masks):
        true_masks = true_masks.detach().cpu()
    
    B, K, H, W = pred_masks.shape
    
    # Handle different input formats
    if true_masks.dim() == 4:
        true_masks = true_masks.argmax(dim=1)
    
    # Resize if needed
    if true_masks.shape[1:] != (H, W):
        true_masks = F.interpolate(
            true_masks.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze(1).long()
    
    pred_clusters = pred_masks.argmax(dim=1)
    
    ari_scores = []
    for b in range(B):
        pred_flat = pred_clusters[b].reshape(-1).numpy()
        true_flat = true_masks[b].reshape(-1).numpy()
        
        mask = true_flat > 0
        if mask.sum() > 0:
            ari = adjusted_rand_score(true_flat[mask], pred_flat[mask])
            ari_scores.append(ari)
    
    return np.mean(ari_scores) if ari_scores else 0.0


def train_epoch(model, train_loader, optimizer, epoch, step, args):
    """Train one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_spec = 0.0
    total_div = 0.0
    batch_times = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        batch_start = time.time()
        
        images = batch['image'].to(args.device).float()
        
        # LR warmup
        lr = warmup_lr(step, args.warmup_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        outputs = model(images, return_loss=True)
        
        # Get dynamic spectral weight (curriculum)
        spectral_weight = get_spectral_weight(epoch, args.lambda_spec)
        
        # Compute total loss
        loss = (
            args.lambda_recon * outputs['loss_recon'] +
            spectral_weight * outputs['loss_spec'] +
            args.lambda_div * outputs['loss_div']
        )
        
        # Handle DataParallel
        if loss.dim() > 0:
            loss = loss.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Accumulate losses (handle DataParallel tensors)
        loss_recon = outputs['loss_dict']['recon']
        loss_spec = outputs['loss_dict']['spec']
        loss_div = outputs['loss_dict']['div']
        
        if hasattr(loss_recon, 'dim') and loss_recon.dim() > 0:
            loss_recon = loss_recon.mean()
            loss_spec = loss_spec.mean()
            loss_div = loss_div.mean()
        
        total_loss += loss.item()
        total_recon += loss_recon.item() if hasattr(loss_recon, 'item') else loss_recon
        total_spec += loss_spec.item() if hasattr(loss_spec, 'item') else loss_spec
        total_div += loss_div.item() if hasattr(loss_div, 'item') else loss_div
        
        step += 1
        
        if batch_idx % args.log_interval == 0:
            # Get mask difference for debugging (should vary, not be constant!)
            mask_diff = getattr(outputs['loss_spec'], 'mask_diff_mean', None)
            mask_diff_val = mask_diff.item() if mask_diff is not None else 0
            
            pbar.set_postfix(
                L=f"{loss.item():.3f}",
                rec=f"{loss_recon:.4f}",
                spec=f"{loss_spec:.4f}",
                mΔ=f"{mask_diff_val:.3f}",  # mask difference (should vary!)
                div=f"{loss_div:.3f}",
                sw=f"{spectral_weight:.2f}",
            )
    
    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'spec': total_spec / n,
        'div': total_div / n,
        'avg_batch_ms': np.mean(batch_times) * 1000,
    }, step


@torch.no_grad()
def evaluate(model, val_loader, args):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    all_aris = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['image'].to(args.device).float()
        
        outputs = model(images, return_loss=True)
        
        # Handle DataParallel tensors
        loss_recon = outputs['loss_dict']['recon']
        if hasattr(loss_recon, 'dim') and loss_recon.dim() > 0:
            loss_recon = loss_recon.mean()
        
        total_loss += loss_recon.item() if hasattr(loss_recon, 'item') else loss_recon
        total_recon += loss_recon.item() if hasattr(loss_recon, 'item') else loss_recon
        
        # ARI
        if 'mask' in batch:
            ari = compute_ari(outputs['masks'], batch['mask'])
            all_aris.append(ari)
    
    n = len(val_loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'ari': np.mean(all_aris) if all_aris else 0.0,
    }


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Device setup using multi_gpu utilities
    use_multi_gpu, gpu_ids = setup_device(args)
    
    # Compute total slots
    num_slots = len(args.scales) * args.slots_per_scale
    args.num_slots = num_slots
    
    print("=" * 60)
    print("STAGE-2: SPECTRAL INITIALIZATION TRAINING")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Multi-GPU: {use_multi_gpu}")
    print(f"Backbone: DINOv2-base (frozen)")
    print(f"Scales: {args.scales}")
    print(f"Slots per scale: {args.slots_per_scale}")
    print(f"Total slots: {num_slots}")
    print(f"K-neighbors: {args.k_neighbors}")
    print(f"Power iterations: {args.power_iters}")
    print(f"Use Mamba: {args.use_mamba}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Lambda (recon/spec/div): {args.lambda_recon}/{args.lambda_spec}/{args.lambda_div}")
    print("=" * 60)
    
    # Create model
    model = SpectralSlotModel(
        backbone=args.backbone,
        scales=args.scales,
        slots_per_scale=args.slots_per_scale,
        k_neighbors=args.k_neighbors,
        power_iters=args.power_iters,
        num_iterations=args.num_iterations,
        image_size=(args.image_size, args.image_size),
        use_mamba=args.use_mamba,
        freeze_backbone=True,
    ).to(args.device)
    
    # Load Stage-1 decoder if provided
    if args.stage1_ckpt:
        model.load_stage1_decoder(args.stage1_ckpt)
    
    # Multi-GPU wrapper using utility function
    model = wrap_model(model, use_multi_gpu, gpu_ids)
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Data
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Training loop
    best_ari = -1.0
    step = 0
    history = []
    
    for epoch in range(args.epochs):
        # Train
        train_metrics, step = train_epoch(
            model, train_loader, optimizer, epoch, step, args
        )
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"    Recon: {train_metrics['recon']:.4f}")
        print(f"    Spectral: {train_metrics['spec']:.4f}")
        print(f"    Diversity: {train_metrics['div']:.4f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, args)
            
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val ARI: {val_metrics['ari']:.4f}")
            
            # Save best
            if val_metrics['ari'] > best_ari:
                best_ari = val_metrics['ari']
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(
                    state_dict,
                    f"{args.output_dir}/stage2_best_{timestamp}.pt"
                )
                print(f"  ✓ New best ARI: {best_ari:.4f}")
            
            history.append({
                'epoch': epoch + 1,
                **train_metrics,
                'val_loss': val_metrics['loss'],
                'val_ari': val_metrics['ari'],
            })
    
    # Save final
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict, f"{args.output_dir}/stage2_final_{timestamp}.pt")
    
    # Save history
    with open(f"{args.output_dir}/stage2_history_{timestamp}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("STAGE-2 TRAINING COMPLETE")
    print(f"Best Val ARI: {best_ari:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
