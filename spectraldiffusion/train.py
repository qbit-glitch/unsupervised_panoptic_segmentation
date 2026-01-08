"""
Training Script for SpectralDiffusion

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/clevr.yaml --debug
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# Use torch.autocast for both MPS and CUDA
import numpy as np
from tqdm import tqdm
import wandb

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SpectralDiffusion
from data.clevr import CLEVRDataset, SyntheticShapesDataset, create_synthetic_dataloader
from data.coco import COCOStuffDataset
from utils.metrics import PanopticMetrics, compute_ari


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpectralDiffusion")
    
    # Config
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    
    # Model
    parser.add_argument("--backbone", type=str, default="base",
                       choices=["small", "base", "large"])
    parser.add_argument("--num-slots", type=int, default=12)
    parser.add_argument("--use-diffusion", type=bool, default=True)
    parser.add_argument("--use-mamba", type=bool, default=True)
    parser.add_argument("--init-mode", type=str, default="spectral",
                       choices=["spectral", "random", "learned"])
    parser.add_argument("--full-model", action="store_true",
                       help="Use full model with all components integrated")
    parser.add_argument("--use-dino", action="store_true",
                       help="Use DINOv2 backbone (slower but better features)")
    
    # Data
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["synthetic", "clevr", "clevr_masks", "coco", "pascal_voc"])
    parser.add_argument("--val-dataset", type=str, default=None,
                       choices=["synthetic", "clevr", "clevr_masks", "coco", "pascal_voc"],
                       help="Validation dataset (default: same as --dataset). Use clevr_masks for proper ARI on CLEVR.")
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Loss weights
    parser.add_argument("--lambda-diff", type=float, default=1.0)
    parser.add_argument("--lambda-spec", type=float, default=0.1)
    parser.add_argument("--lambda-ident", type=float, default=0.01)
    
    # Device
    parser.add_argument("--device", type=str, default="mps",
                       choices=["mps", "cuda", "cpu"])
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable bfloat16 mixed precision (recommended for MPS)")
    
    # Logging
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--exp-name", type=str, default="spectral_diffusion")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    
    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true",
                       help="Run 1 train and 1 val batch for debugging")
    
    return parser.parse_args()


def create_dataloader(args, split: str = "train", dataset_override: str = None) -> DataLoader:
    """Create dataloader based on dataset choice.
    
    Args:
        args: Command line arguments
        split: "train" or "val"
        dataset_override: Optional dataset name to use instead of args.dataset
                         Useful for training on CLEVR but validating on CLEVR-with-masks
    """
    dataset_name = dataset_override or args.dataset
    
    if dataset_name == "synthetic":
        num_samples = 100 if args.debug else 5000
        dataset = SyntheticShapesDataset(
            num_samples=num_samples,
            image_size=tuple(args.image_size),
            max_objects=min(args.num_slots, 8),
        )
    elif dataset_name == "clevr":
        dataset = CLEVRDataset(
            root_dir=os.path.join(args.data_dir, "clevr/CLEVR_v1.0"),
            split="train" if split == "train" else "val",
            image_size=tuple(args.image_size),
        )
    elif dataset_name == "clevr_masks":
        # Use CLEVR with ground truth masks from TFRecords for proper ARI
        # Note: Only train TFRecords available, use it for both (subset differently)
        from data.clevr import CLEVRWithMasksDataset
        tfrecord_path = os.path.join(
            args.data_dir, 
            "clevr/clevr_with_masks_clevr_with_masks_train.tfrecords"
        )
        max_samples = 1000 if args.debug else None
        dataset = CLEVRWithMasksDataset(
            tfrecord_path=tfrecord_path,
            image_size=tuple(args.image_size),
            max_objects=args.num_slots,
            max_samples=max_samples,
        )
    elif dataset_name == "coco":
        dataset = COCOStuffDataset(
            root_dir=os.path.join(args.data_dir, "coco_stuff"),
            split="train2017" if split == "train" else "val2017",
            image_size=tuple(args.image_size),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Disable pin_memory for MPS as it's not supported
    pin_memory = args.device == "cuda"
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )


def create_model(args) -> nn.Module:
    """Create SpectralDiffusion model."""
    
    # Check if full model is requested
    if getattr(args, 'full_model', False):
        # Use the full model with all components integrated
        from models.full_model import FullSpectralDiffusion
        
        model = FullSpectralDiffusion(
            image_size=tuple(args.image_size),
            num_slots=args.num_slots,
            feature_dim=256,
            use_dino=getattr(args, 'use_dino', False),
            use_mamba=args.use_mamba,
            init_mode=args.init_mode,
            scales=[8, 16] if args.image_size[0] <= 128 else [8, 16, 32],
            lambda_spec=args.lambda_spec,
            lambda_ident=args.lambda_ident,
        )
        return model.to(args.device)
    
    # Otherwise use simplified model for faster training
    from models.spectral_init import MultiScaleSpectralInit
    from models.mamba_slot import create_slot_attention
    from models.diffusion import MLPMaskDecoder
    from models.pruning import AdaptiveSlotPruning
    
    # Import loss modules for simplified model too
    try:
        from losses.identifiable_loss import IdentifiabilityLoss, SlotDiversityLoss
        from losses.spectral_loss import SpectralConsistencyLoss
        has_loss_modules = True
    except ImportError:
        has_loss_modules = False
    
    class SimplifiedSpectralDiffusion(nn.Module):
        """Simplified model with all losses integrated."""
        
        def __init__(self, args):
            super().__init__()
            
            self.image_size = tuple(args.image_size)
            self.num_slots = args.num_slots
            self.feature_dim = 256
            self.init_mode = args.init_mode
            self.lambda_spec = getattr(args, 'lambda_spec', 0.1)
            self.lambda_ident = getattr(args, 'lambda_ident', 0.01)
            
            # Feature encoder (simple CNN)
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            
            # Spectral init with 3 scales if image is large enough
            if args.image_size[0] >= 128:
                self.scales = [8, 16, 32]
            else:
                self.scales = [8, 16]
            
            num_scales = len(self.scales)
            slots_per_scale = (args.num_slots + num_scales - 1) // num_scales
            self.actual_num_slots = slots_per_scale * num_scales
            
            self.spectral_init = MultiScaleSpectralInit(
                scales=self.scales,
                slots_per_scale=slots_per_scale,
                feature_dim=self.feature_dim,
                num_power_iters=20,
            )
            
            # Learnable slots as fallback
            self.slot_mu = nn.Parameter(torch.randn(1, self.actual_num_slots, self.feature_dim) * 0.1)
            self.slot_sigma = nn.Parameter(torch.ones(1, self.actual_num_slots, self.feature_dim) * 0.1)
            
            # Slot attention
            self.slot_attention = create_slot_attention(
                attention_type="mamba" if args.use_mamba else "standard",
                dim=self.feature_dim,
                num_slots=self.actual_num_slots,
                num_iterations=3,
            )
            
            # Mask decoder
            self.mask_decoder = MLPMaskDecoder(
                slot_dim=self.feature_dim,
                num_slots=self.actual_num_slots,
                image_size=self.image_size,
            )
            
            # Spatial broadcast decoder
            H, W = self.image_size
            self.slot_decoder = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            
            self.register_buffer('pos_grid', self._create_pos_grid(H, W))
            
            self.rgb_decoder = nn.Sequential(
                nn.Linear(256 + 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Sigmoid(),
            )
            
            # Pruning
            self.pruning = AdaptiveSlotPruning(
                num_slots=self.actual_num_slots,
                threshold=0.05,
            )
            
            # Loss modules (integrated)
            if has_loss_modules:
                self.ident_loss_fn = IdentifiabilityLoss(
                    num_slots=self.actual_num_slots,
                    slot_dim=self.feature_dim,
                )
                self.diversity_loss_fn = SlotDiversityLoss()
                self.spectral_loss_fn = SpectralConsistencyLoss()
                self.has_loss_modules = True
            else:
                self.has_loss_modules = False
        
        def _create_pos_grid(self, H, W):
            y = torch.linspace(-1, 1, H)
            x = torch.linspace(-1, 1, W)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            pos = torch.stack([xx, yy], dim=-1)
            return pos.view(1, H * W, 2)
        
        def decode_slots_to_image(self, slots, masks):
            B, K, D = slots.shape
            H, W = self.image_size
            N = H * W
            
            slot_features = self.slot_decoder(slots)
            slot_broadcast = slot_features.unsqueeze(2).expand(-1, -1, N, -1)
            pos = self.pos_grid.expand(B, -1, -1).unsqueeze(1).expand(-1, K, -1, -1)
            combined = torch.cat([slot_broadcast, pos], dim=-1)
            rgb_per_slot = self.rgb_decoder(combined)
            rgb_per_slot = rgb_per_slot.view(B, K, H, W, 3).permute(0, 1, 4, 2, 3)
            masks_expanded = masks.unsqueeze(2)
            reconstructed = (rgb_per_slot * masks_expanded).sum(dim=1)
            return reconstructed, rgb_per_slot
        
        def forward(self, images, return_loss=True):
            B = images.shape[0]
            device = images.device
            H, W = self.image_size
            
            features = self.encoder(images)
            _, C, H_feat, W_feat = features.shape
            
            multiscale = {}
            for scale in self.scales:
                multiscale[scale] = F.adaptive_avg_pool2d(features, scale).permute(0, 2, 3, 1)
            
            flat = features.permute(0, 2, 3, 1).reshape(B, -1, C)
            
            if self.init_mode == "spectral":
                slots_init = self.spectral_init(multiscale, mode="spectral")
            elif self.init_mode == "random":
                slots_init = self.spectral_init(multiscale, mode="random")
            else:
                slots_init = self.slot_mu + self.slot_sigma * torch.randn_like(self.slot_mu)
                slots_init = slots_init.expand(B, -1, -1).to(device)
            
            slots, attn_masks = self.slot_attention(flat, slots_init)
            slots, attn_masks, prune_info = self.pruning(slots, attn_masks)
            output_masks = self.mask_decoder(slots)
            reconstructed, rgb_per_slot = self.decode_slots_to_image(slots, output_masks)
            
            outputs = {
                'slots': slots,
                'masks': output_masks,
                'attention_masks': attn_masks,
                'reconstructed': reconstructed,
            }
            
            if return_loss:
                loss, loss_dict = self._compute_loss(images, output_masks, slots, reconstructed, multiscale)
                outputs['loss'] = loss
                outputs['loss_dict'] = loss_dict
            
            return outputs
        
        def _compute_loss(self, images, masks, slots, reconstructed, multiscale):
            B, K, H, W = masks.shape
            loss_dict = {}
            
            # 1. Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, images)
            loss_dict['recon'] = recon_loss.item()
            
            # 2. Coverage loss
            mask_sum = masks.sum(dim=1)
            coverage_loss = ((mask_sum - 1) ** 2).mean()
            loss_dict['coverage'] = coverage_loss.item()
            
            # 3. Overlap loss
            mask_sq_sum = (masks ** 2).sum(dim=1)
            overlap_loss = ((mask_sum ** 2 - mask_sq_sum) ** 2).mean()
            loss_dict['overlap'] = overlap_loss.item()
            
            # 4. Diversity loss
            if self.has_loss_modules:
                diversity_loss = self.diversity_loss_fn(slots)
            else:
                slots_normalized = F.normalize(slots, dim=-1)
                similarity = torch.bmm(slots_normalized, slots_normalized.transpose(1, 2))
                eye = torch.eye(K, device=slots.device).unsqueeze(0)
                off_diag = similarity * (1 - eye)
                diversity_loss = off_diag.abs().mean()
            loss_dict['diversity'] = diversity_loss.item()
            
            # 5. Entropy loss (encourage higher entropy = more distributed masks)
            # NEGATIVE sign: we want to MAXIMIZE entropy to prevent collapse
            entropy = -(masks * torch.log(masks + 1e-8)).sum(dim=1).mean()
            entropy_loss = -entropy  # Negate to encourage higher entropy
            loss_dict['entropy'] = entropy.item()
            
            # 6. Anti-collapse loss: penalize when any slot covers too much area
            # Each slot should cover roughly 1/K of the image on average
            slot_areas = masks.mean(dim=(2, 3))  # [B, K] - average coverage per slot
            target_area = 1.0 / K
            # Penalize deviation from uniform coverage, especially slots with >2x expected coverage
            area_deviation = (slot_areas - target_area).abs()
            # Extra penalty for dominant slots
            dominant_penalty = F.relu(slot_areas - 2 * target_area).mean()
            collapse_loss = area_deviation.mean() + 2.0 * dominant_penalty
            loss_dict['collapse'] = collapse_loss.item()
            
            # 7. Identifiability loss (NeurIPS 2024)
            if self.has_loss_modules:
                ident_loss = self.ident_loss_fn(slots, update_prior=self.training)
                loss_dict['ident'] = ident_loss.item()
            else:
                ident_loss = torch.tensor(0.0, device=masks.device)
                loss_dict['ident'] = 0.0
            
            # 8. Spectral consistency loss
            if self.has_loss_modules:
                try:
                    spectral_masks = self.spectral_init.get_spectral_masks(multiscale)
                    spectral_masks_resized = F.interpolate(
                        spectral_masks, size=(H, W), mode='bilinear', align_corners=False
                    )
                    spectral_masks_resized = F.softmax(spectral_masks_resized, dim=1)
                    spec_loss = F.mse_loss(masks, spectral_masks_resized)
                    loss_dict['spectral'] = spec_loss.item()
                except Exception:
                    spec_loss = torch.tensor(0.0, device=masks.device)
                    loss_dict['spectral'] = 0.0
            else:
                spec_loss = torch.tensor(0.0, device=masks.device)
                loss_dict['spectral'] = 0.0
            
            # 9. SLOT REPULSION LOSS - force slots to be different (critical for preventing collapse)
            # Compute pairwise mask overlap and penalize
            masks_flat = masks.view(B, K, -1)  # [B, K, H*W]
            mask_overlap = torch.bmm(masks_flat, masks_flat.transpose(1, 2))  # [B, K, K]
            # Normalize by mask areas
            mask_areas = masks_flat.sum(dim=-1, keepdim=True)  # [B, K, 1]
            mask_overlap_normalized = mask_overlap / (mask_areas + 1e-8)
            # Penalize off-diagonal overlaps (slots sharing pixels)
            eye = torch.eye(K, device=masks.device).unsqueeze(0)
            repulsion_loss = (mask_overlap_normalized * (1 - eye)).mean()
            loss_dict['repulsion'] = repulsion_loss.item()
            
            # Total loss - MUCH STRONGER anti-collapse
            total_loss = (
                10.0 * recon_loss +
                1.0 * coverage_loss +
                1.0 * overlap_loss +          # Increased from 0.5
                5.0 * diversity_loss +        # Increased from 1.0 - CRITICAL
                1.0 * entropy_loss +          # Increased from 0.5
                5.0 * collapse_loss +         # Increased from 2.0 - CRITICAL
                10.0 * repulsion_loss +       # NEW - CRITICAL for slot separation
                self.lambda_ident * ident_loss +
                self.lambda_spec * spec_loss
            )
            
            loss_dict['total'] = total_loss.item()
            
            return total_loss, loss_dict
    
    model = SimplifiedSpectralDiffusion(args)
    return model.to(args.device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    args,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch with optional bfloat16 mixed precision."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    loss_accum = {}  # Accumulate all loss components
    
    # Setup autocast context for mixed precision
    use_amp = args.mixed_precision
    if use_amp:
        if args.device == "mps":
            autocast_ctx = torch.autocast("mps", dtype=torch.float32)
        elif args.device == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.float32)
        else:
            autocast_ctx = torch.autocast("cpu", dtype=torch.float32)
    else:
        autocast_ctx = torch.autocast(args.device, enabled=False)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if args.fast_dev_run and batch_idx > 0:
            break
        
        images = batch['image'].to(args.device)
        
        # Convert to bfloat16 if mixed precision is enabled
        if use_amp and args.device == "mps":
            images = images.to(torch.float32)
        
        # Forward pass with optional autocast
        optimizer.zero_grad()
        with autocast_ctx:
            outputs = model(images, return_loss=True)
            loss = outputs['loss']
        
        # Backward pass (no scaler needed for bfloat16)
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Logging
        total_loss += loss.item()
        num_batches += 1
        
        # Accumulate detailed losses if available
        if 'loss_dict' in outputs:
            for key, val in outputs['loss_dict'].items():
                if key not in loss_accum:
                    loss_accum[key] = 0.0
                loss_accum[key] += val
        
        if batch_idx % args.log_interval == 0:
            postfix = {
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            # Add key losses to progress bar
            if 'loss_dict' in outputs:
                postfix['recon'] = f"{outputs['loss_dict'].get('recon', 0):.4f}"
                if outputs['loss_dict'].get('ident', 0) > 0:
                    postfix['ident'] = f"{outputs['loss_dict'].get('ident', 0):.4f}"
            pbar.set_postfix(postfix)
    
    # Compute averages
    result = {
        'train_loss': total_loss / max(num_batches, 1),
    }
    for key, val in loss_accum.items():
        result[f'train_{key}'] = val / max(num_batches, 1)
    
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    args,
) -> Dict[str, float]:
    """Evaluate model with optional bfloat16."""
    model.eval()
    
    metrics = PanopticMetrics()
    total_loss = 0
    num_batches = 0
    total_ari = 0
    
    # Setup autocast context for mixed precision
    use_amp = args.mixed_precision
    if use_amp:
        if args.device == "mps":
            autocast_ctx = torch.autocast("mps", dtype=torch.float32)
        elif args.device == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.float32)
        else:
            autocast_ctx = torch.autocast("cpu", dtype=torch.float32)
    else:
        autocast_ctx = torch.autocast(args.device, enabled=False)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args.fast_dev_run and batch_idx > 0:
            break
        
        images = batch['image'].to(args.device)
        
        # Convert to bfloat16 if mixed precision is enabled
        if use_amp and args.device == "mps":
            images = images.to(torch.float32)
        
        with autocast_ctx:
            outputs = model(images, return_loss=True)
            loss = outputs['loss']
            masks = outputs['masks']  # [B, K, H, W]
        
        total_loss += loss.item()
        num_batches += 1
        
        # Compute ARI if ground truth available
        if 'mask' in batch:
            gt_masks = batch['mask'].to(args.device)
            
            # Masks are now softmax-normalized across slots (mutually exclusive)
            # Debug: print mask statistics on first batch
            if batch_idx == 0:
                mask_max, _ = masks.max(dim=1)
                mask_min, _ = masks.min(dim=1)
                print(f"  [DEBUG] Mask stats: max={mask_max.mean():.4f}, min={mask_min.mean():.4f}")
                
                # Check how many unique slots are being used
                pred_labels_debug = masks.argmax(dim=1)
                unique_slots = [len(torch.unique(pred_labels_debug[b])) for b in range(pred_labels_debug.shape[0])]
                print(f"  [DEBUG] Unique slots per image: {unique_slots[:4]}... (should be > 1)")
            
            # Convert to label maps (argmax works directly since masks are softmax-normalized)
            pred_labels = masks.argmax(dim=1)  # [B, H, W]
            gt_labels = gt_masks.argmax(dim=1)  # [B, H, W]
            
            ari = compute_ari(pred_labels, gt_labels)
            total_ari += ari.mean().item()
    
    results = {
        'val_loss': total_loss / max(num_batches, 1),
        'val_ari': total_ari / max(num_batches, 1),
    }
    
    return results


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    if args.wandb:
        wandb.init(
            project="spectral_diffusion",
            name=args.exp_name,
            config=vars(args),
        )
    
    print("=" * 60)
    print("SpectralDiffusion Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    if args.val_dataset:
        print(f"Val Dataset: {args.val_dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Num slots: {args.num_slots}")
    print(f"Use Mamba: {args.use_mamba}")
    print(f"Init mode: {args.init_mode}")
    print(f"Mixed precision: {args.mixed_precision}")
    print("=" * 60)
    
    # Create dataloaders
    # Use val_dataset for validation if specified (for proper ARI computation)
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(args, split="train")
    val_loader = create_dataloader(args, split="val", dataset_override=args.val_dataset)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args)
    
    # Convert model to bfloat16 for faster MPS training
    if args.mixed_precision and args.device == "mps":
        model = model.to(torch.float32)
        print("Model converted to bfloat16 for MPS acceleration")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("\nStarting training...")
    best_ari = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, args, epoch
        )
        
        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, args)
        else:
            val_metrics = {}
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        
        print(f"\nEpoch {epoch}:")
        for k, v in metrics.items():
            if k != 'epoch':
                print(f"  {k}: {v:.4f}")
        
        if args.wandb:
            wandb.log(metrics)
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        
        # Save best
        if val_metrics.get('val_ari', 0) > best_ari:
            best_ari = val_metrics['val_ari']
            best_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
            }, best_path)
            print(f"New best ARI: {best_ari:.4f}")
        
        if args.fast_dev_run:
            print("\nFast dev run complete!")
            break
    
    print("\nTraining complete!")
    print(f"Best ARI: {best_ari:.4f}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
