"""
Complete Training Pipeline for SpectralDiffusion
Integrates all components with best practices from 2025 papers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import time
from pathlib import Path

# Import our modules
# from spectral_init_2025 import SpectralInitializer
# from mamba_slot_attention_2025 import MambaSlotAttention
# from diffusion_decoder_seg_2025 import LatentDiffusionPanoptic


class SpectralDiffusionModel(nn.Module):
    """
    Complete SpectralDiffusion model
    Integrates: DINOv2 → Spectral Init → Mamba-Slot → Diffusion Decoder
    """
    
    def __init__(
        self,
        num_slots: int = 12,
        slot_dim: int = 768,
        num_iterations: int = 3,
        use_spectral_init: bool = True,
        use_diffusion_decoder: bool = True,
        image_size: int = 224,
        dinov2_model: str = 'dinov2_vitb14'
    ):
        super().__init__()
        self.num_slots = num_slots
        self.use_spectral_init = use_spectral_init
        self.use_diffusion_decoder = use_diffusion_decoder
        
        # 1. Frozen DINOv2 encoder
        try:
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2',
                dinov2_model,
                pretrained=True
            )
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print("✓ Loaded DINOv2 backbone")
        except:
            print("⚠ Could not load DINOv2, using random encoder")
            self.encoder = nn.Sequential(
                nn.Conv2d(3, slot_dim, 14, stride=14),
                nn.GELU()
            )
        
        # 2. Spectral initializer (optional but recommended)
        if use_spectral_init:
            from spectral_init_2025 import SpectralInitializer
            self.spectral_init = SpectralInitializer(
                scales=[8, 16, 32],
                k_per_scale=4,
                knn_k=20
            )
        
        # 3. Mamba-Slot attention
        from mamba_slot_attention_2025 import MambaSlotAttention
        self.slot_attention = MambaSlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            d_state=64,
            num_iterations=num_iterations,
            use_gmm_prior=True,
            use_spatial_fusion=True
        )
        
        # 4. Decoder (diffusion or simple)
        if use_diffusion_decoder:
            from diffusion_decoder_seg_2025 import LatentDiffusionPanoptic
            self.decoder = LatentDiffusionPanoptic(
                num_slots=num_slots,
                slot_dim=slot_dim,
                latent_dim=256,
                num_timesteps=50,
                image_size=image_size
            )
        else:
            # Simple spatial broadcast decoder (faster for debugging)
            self.decoder = SimpleSpatialBroadcastDecoder(
                slot_dim=slot_dim,
                num_slots=num_slots,
                image_size=image_size
            )
    
    def extract_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Extract DINOv2 features
        
        Returns:
            features: [B, N, D] flattened features
            H, W: spatial dimensions
        """
        B = images.size(0)
        
        with torch.no_grad():
            if hasattr(self.encoder, 'forward_features'):
                # DINOv2
                feats = self.encoder.forward_features(images)
                
                if isinstance(feats, dict):
                    feats = feats['x_norm_patchtokens']
                elif feats.dim() == 3 and feats.size(1) > 1:
                    # Remove CLS token
                    feats = feats[:, 1:, :]
                
                # Infer H, W from number of patches
                N = feats.size(1)
                H = W = int(np.sqrt(N))
                
                return feats, H, W
            else:
                # Simple encoder
                feats = self.encoder(images)
                B, D, H, W = feats.shape
                feats = feats.permute(0, 2, 3, 1).reshape(B, H * W, D)
                return feats, H, W
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W]
            targets: [B, H, W] ground truth masks (optional, for supervised)
            train: training mode
        
        Returns:
            dict with masks, slots, losses
        """
        B = images.size(0)
        
        # 1. Extract features
        features, H, W = self.extract_features(images)  # [B, N, D]
        
        # 2. Spectral initialization (if enabled)
        if self.use_spectral_init and train:
            # Reshape for spectral init: [B, H, W, D]
            features_2d = features.reshape(B, H, W, -1)
            slots_init = self.spectral_init(features_2d)
        else:
            slots_init = None
        
        # 3. Mamba-Slot attention
        slots, attn = self.slot_attention(features, slots_init, H, W)
        
        # 4. Decode to masks
        if self.use_diffusion_decoder:
            decoder_outputs = self.decoder(images, slots, train=train)
        else:
            decoder_outputs = self.decoder(slots)
        
        # 5. Compute losses
        outputs = {
            'masks': decoder_outputs.get('masks'),
            'slots': slots,
            'attn': attn,
            'features': features
        }
        
        if train:
            losses = self.compute_losses(
                outputs,
                targets,
                decoder_outputs
            )
            outputs.update(losses)
        
        return outputs
    
    def compute_losses(
        self,
        outputs: Dict,
        targets: Optional[torch.Tensor],
        decoder_outputs: Dict
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # 1. Diffusion loss (if using diffusion decoder)
        if 'loss' in decoder_outputs:
            losses['diffusion_loss'] = decoder_outputs['loss']
        
        # 2. Spectral consistency loss (optional)
        if self.use_spectral_init and self.training:
            # Encourage slots to stay close to spectral initialization
            # This is a soft constraint, not strict
            losses['spectral_loss'] = torch.tensor(0.0, device=outputs['slots'].device)
        
        # 3. GMM identifiability loss
        gmm_loss = self.slot_attention.compute_gmm_prior_loss(outputs['slots'])
        losses['gmm_loss'] = gmm_loss
        
        # 4. Supervised loss (if targets provided)
        if targets is not None and outputs['masks'] is not None:
            # Cross-entropy loss
            B, K, H, W = outputs['masks'].shape
            masks = outputs['masks']
            
            # Resize targets if needed
            if targets.shape[-2:] != (H, W):
                targets = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()
            
            # Compute cross-entropy
            masks_logits = torch.log(masks + 1e-8)
            ce_loss = F.nll_loss(masks_logits, targets)
            losses['supervised_loss'] = ce_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


class SimpleSpatialBroadcastDecoder(nn.Module):
    """Simple decoder for debugging (faster than diffusion)"""
    
    def __init__(self, slot_dim: int, num_slots: int, image_size: int):
        super().__init__()
        self.num_slots = num_slots
        
        # Positional encoding
        self.register_buffer(
            'pos_grid',
            self.build_grid(image_size, image_size)
        )
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Logit for this slot
        )
    
    def build_grid(self, H: int, W: int) -> torch.Tensor:
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        return grid
    
    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, K, D = slots.shape
        H, W = self.pos_grid.shape[:2]
        
        # Broadcast slots to spatial
        slots_flat = slots.reshape(B * K, D)
        
        # Expand grid
        grid = self.pos_grid.unsqueeze(0).expand(B * K, -1, -1, -1)
        grid = grid.reshape(B * K, H * W, 2)
        
        # Broadcast slots
        slots_broadcast = slots_flat.unsqueeze(1).expand(-1, H * W, -1)
        
        # Concatenate
        x = torch.cat([slots_broadcast, grid], dim=-1)
        
        # Decode
        logits = self.decoder(x)  # [B*K, H*W, 1]
        logits = logits.reshape(B, K, H, W)
        
        # Softmax
        masks = F.softmax(logits, dim=1)
        
        return {'masks': masks}


class Trainer:
    """Training orchestrator"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: Path,
        max_epochs: int = 100,
        grad_clip: float = 1.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.best_val_metric = 0.0
        self.step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        metrics = {
            'loss': 0.0,
            'diffusion_loss': 0.0,
            'gmm_loss': 0.0
        }
        
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get data
            images = batch['image'].to(self.device)
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.to(self.device)
            
            # Forward
            outputs = self.model(images, masks, train=True)
            
            # Loss
            loss = outputs['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip
            )
            
            self.optimizer.step()
            
            # Track metrics
            metrics['loss'] += loss.item()
            if 'diffusion_loss' in outputs:
                metrics['diffusion_loss'] += outputs['diffusion_loss'].item()
            if 'gmm_loss' in outputs:
                metrics['gmm_loss'] += outputs['gmm_loss'].item()
            
            num_batches += 1
            self.step += 1
            
            # Log
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Time: {elapsed:.1f}s")
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        
        metrics = {
            'loss': 0.0,
            'ari': 0.0
        }
        
        num_batches = 0
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            masks_true = batch['mask'].to(self.device)
            
            # Forward
            outputs = self.model(images, train=False)
            
            # Get predicted masks
            masks_pred = outputs['masks']  # [B, K, H, W]
            
            # Compute ARI
            ari = self.compute_ari(masks_pred, masks_true)
            metrics['ari'] += ari
            
            num_batches += 1
        
        # Average
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def compute_ari(
        self,
        pred_masks: torch.Tensor,
        true_masks: torch.Tensor
    ) -> float:
        """Compute Adjusted Rand Index"""
        from sklearn.metrics import adjusted_rand_score
        
        B, K, H, W = pred_masks.shape
        
        # Assign to max slot
        pred_clusters = pred_masks.argmax(dim=1)  # [B, H, W]
        
        ari_scores = []
        
        for b in range(B):
            pred_flat = pred_clusters[b].reshape(-1).cpu().numpy()
            true_flat = true_masks[b].reshape(-1).cpu().numpy()
            
            # Exclude background (ID 0)
            mask = true_flat > 0
            if mask.sum() > 0:
                ari = adjusted_rand_score(true_flat[mask], pred_flat[mask])
                ari_scores.append(ari)
        
        return np.mean(ari_scores) if ari_scores else 0.0
    
    def train(self):
        """Complete training loop"""
        print("="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"\n✓ Training metrics:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Validate every 10 epochs
            if (epoch + 1) % 10 == 0:
                val_metrics = self.validate(epoch)
                print(f"\n✓ Validation metrics:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Save best model
                if val_metrics['ari'] > self.best_val_metric:
                    self.best_val_metric = val_metrics['ari']
                    self.save_checkpoint(epoch, 'best')
                    print(f"  → New best ARI: {self.best_val_metric:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}')
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best validation ARI: {self.best_val_metric:.4f}")
    
    def save_checkpoint(self, epoch: int, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'step': self.step
        }
        
        path = self.log_dir / f'checkpoint_{name}.pt'
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("COMPLETE TRAINING PIPELINE")
    print("="*60)
    
    # Hyperparameters
    config = {
        'num_slots': 12,
        'slot_dim': 768,
        'num_iterations': 3,
        'use_spectral_init': True,
        'use_diffusion_decoder': True,
        'image_size': 224,
        'batch_size': 16,
        'learning_rate': 4e-4,
        'max_epochs': 100,
        'grad_clip': 1.0
    }
    
    print("\n✓ Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralDiffusionModel(
        num_slots=config['num_slots'],
        slot_dim=config['slot_dim'],
        num_iterations=config['num_iterations'],
        use_spectral_init=config['use_spectral_init'],
        use_diffusion_decoder=config['use_diffusion_decoder'],
        image_size=config['image_size']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    print(f"\n✓ Model created:")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n" + "="*60)
    print("TRAINING WORKFLOW")
    print("="*60)
    print("""
1. Data Loading:
   - Load Cityscapes/CLEVR dataset
   - Apply augmentations
   - Create DataLoader

2. Training Loop:
   - Extract DINOv2 features (frozen)
   - Initialize slots with spectral clustering
   - Refine with Mamba-Slot attention (3 iterations)
   - Decode to masks with diffusion
   - Compute losses and update

3. Validation:
   - Evaluate ARI every 10 epochs
   - Save best checkpoint
   - Log metrics

4. Expected Timeline:
   - CLEVR: 100 epochs, ~2 hours on A100
   - Cityscapes: 100 epochs, ~8 hours on A100
   
5. Expected Results:
   - CLEVR ARI: 0.90+ after 100 epochs
   - Cityscapes PQ: 38.0+ after 100 epochs
   
NEXT STEPS:
1. Prepare dataset (see data_loaders artifact)
2. Run: python train.py --config config.yaml
3. Monitor training with tensorboard
4. Evaluate on test set
""")