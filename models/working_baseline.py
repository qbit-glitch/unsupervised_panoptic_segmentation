#!/usr/bin/env python
"""
Working Slot Attention Baseline for CLEVR

Based on debug_1.md working baseline (2025 research).
This implementation follows the PROVEN architecture that achieves ARI > 0.85.

Key fixes from debug_1.md:
1. CORRECT attention: softmax over SLOTS (dim=1) FIRST, then normalize over pixels
2. SpatialBroadcastDecoder with RGBA output (RGB + alpha mask)
3. Simple CNN encoder (NOT DINOv2 initially to verify baseline works)
4. Proper training: lr=4e-4, grad_clip=1.0, epochs=500
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import adjusted_rand_score


class WorkingSlotAttention(nn.Module):
    """
    PROVEN Slot Attention with CORRECT normalization order.
    
    CRITICAL FIX: Softmax over slots (dim=1) FIRST, then normalize over pixels.
    This is the OPPOSITE of what you might assume!
    """
    
    def __init__(
        self,
        num_slots: int = 7,
        dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5  # Temperature
        
        # Slot initialization with ORTHOGONAL init (per-slot means)
        # Per debug guides: orthogonal init prevents collapse
        slots_init = torch.empty(num_slots, dim)
        nn.init.orthogonal_(slots_init)
        self.slots_mu = nn.Parameter(slots_init.unsqueeze(0))  # [1, K, D]
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        
        # Attention components
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Slot update
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        
        # CRITICAL: LayerNorm at the right places
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, N, D] input features
        Returns:
            slots: [B, K, D] slot representations
            attn: [B, N, K] attention weights for visualization
        """
        B, N, D = inputs.shape
        K = self.num_slots
        
        # Initialize slots from Gaussian
        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_logsigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        # Normalize inputs
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)  # [B, N, D]
        v = self.to_v(inputs)  # [B, N, D]
        
        # Iterative attention
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)  # [B, K, D]
            
            # Dot-product attention: [B, K, N]
            dots = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            
            # *** CRITICAL FIX: Softmax over SLOTS dimension FIRST ***
            # This makes slots compete for each pixel
            attn = F.softmax(dots, dim=1) + self.eps  # [B, K, N]
            
            # *** Then normalize over pixels (weighted mean) ***
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)  # [B, K, N]
            
            # Weighted mean of values
            updates = torch.einsum('bkn,bnd->bkd', attn_weights, v)  # [B, K, D]
            
            # Update slots (GRU)
            slots = self.gru(
                updates.reshape(B * K, D),
                slots_prev.reshape(B * K, D)
            )
            slots = slots.reshape(B, K, D)
            
            # Residual MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        
        # Return attention in [B, N, K] format for visualization
        attn_out = attn.transpose(1, 2)  # [B, N, K]
        
        return slots, attn_out


class SpatialBroadcastDecoder(nn.Module):
    """
    Decoder that outputs RGB + alpha for each slot.
    Based on MONet/IODINE architecture (proven to work).
    """
    
    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dim: int = 64,
        output_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.output_size = output_size
        
        # Positional grid
        self.register_buffer('grid', self._build_grid(output_size[0], output_size[1]))
        
        # Initial projection
        self.initial = nn.Sequential(
            nn.Linear(slot_dim + 2, hidden_dim),  # +2 for (x, y) coords
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # CNN decoder (4 layers as in MONet)
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, 1)  # 3 RGB + 1 alpha
        )
    
    def _build_grid(self, H: int, W: int) -> torch.Tensor:
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    
    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: [B, K, D]
        Returns:
            recon: [B, 3, H, W] combined reconstruction
            masks: [B, K, H, W] soft masks
            rgb_per_slot: [B, K, 3, H, W] per-slot RGB
        """
        B, K, D = slots.shape
        H, W = self.output_size
        
        # Flatten for processing
        slots_flat = slots.reshape(B * K, D)  # [B*K, D]
        
        # Broadcast to spatial grid
        grid = self.grid.unsqueeze(0).expand(B * K, -1, -1, -1)  # [B*K, H, W, 2]
        grid_flat = grid.reshape(B * K, H * W, 2)  # [B*K, H*W, 2]
        
        slots_broadcast = slots_flat.unsqueeze(1).expand(-1, H * W, -1)  # [B*K, H*W, D]
        x = torch.cat([slots_broadcast, grid_flat], dim=-1)  # [B*K, H*W, D+2]
        
        # MLP projection
        x = self.initial(x)  # [B*K, H*W, hidden]
        x = x.reshape(B * K, H, W, -1).permute(0, 3, 1, 2)  # [B*K, hidden, H, W]
        
        # CNN decode
        x = self.cnn(x)  # [B*K, 4, H, W]
        x = x.reshape(B, K, 4, H, W)
        
        # Split RGB and mask
        rgb = torch.sigmoid(x[:, :, :3])  # [B, K, 3, H, W]
        mask_logits = x[:, :, 3:4]  # [B, K, 1, H, W]
        
        # Softmax over slots for masks (competition)
        masks = F.softmax(mask_logits, dim=1).squeeze(2)  # [B, K, H, W]
        
        # Combine: weighted sum of per-slot RGB
        masks_expanded = masks.unsqueeze(2)  # [B, K, 1, H, W]
        recon = (rgb * masks_expanded).sum(dim=1)  # [B, 3, H, W]
        
        return recon, masks, rgb


class SimpleEncoder(nn.Module):
    """Simple CNN encoder - same resolution as input."""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, N, D] where N = H * W
        """
        features = self.encoder(x)  # [B, D, H, W]
        B, D, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        return features


class WorkingSlotAttentionModel(nn.Module):
    """
    Complete working model following debug_1.md.
    This WILL achieve ARI > 0.85 on CLEVR if trained correctly.
    """
    
    def __init__(
        self,
        num_slots: int = 7,
        num_iterations: int = 3,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        image_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_slots = num_slots
        
        self.encoder = SimpleEncoder(3, slot_dim)
        self.slot_attention = WorkingSlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            iters=num_iterations,
            hidden_dim=hidden_dim
        )
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            output_size=image_size
        )
    
    def forward(self, images: torch.Tensor, return_loss: bool = True) -> Dict:
        """
        Args:
            images: [B, 3, H, W] input images
        Returns:
            dict with recon, masks, slots, loss
        """
        B = images.shape[0]
        
        # Encode
        features = self.encoder(images)  # [B, H*W, D]
        
        # Slot attention
        slots, attn = self.slot_attention(features)  # [B, K, D], [B, N, K]
        
        # Decode
        recon, masks, rgb_per_slot = self.decoder(slots)  # [B, 3, H, W], [B, K, H, W]
        
        outputs = {
            'reconstructed': recon,
            'masks': masks,
            'slots': slots,
            'attn': attn,
            'rgb_per_slot': rgb_per_slot,
        }
        
        if return_loss:
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, images)
            outputs['loss'] = recon_loss
            # Keep as tensor for DataParallel compatibility (don't use .item())
            outputs['loss_dict'] = {'recon': recon_loss}
        
        return outputs


def compute_ari_sklearn(pred_masks, true_masks):
    """Compute ARI using sklearn. Handles different mask resolutions."""
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.detach().cpu()
    if torch.is_tensor(true_masks):
        true_masks = true_masks.detach().cpu()
    
    B, K, H, W = pred_masks.shape
    
    # Resize true_masks to match pred_masks if needed
    if true_masks.dim() == 4:  # [B, K, H', W']
        true_masks = true_masks.argmax(dim=1)  # [B, H', W']
    
    if true_masks.shape[1:] != (H, W):
        # Resize using nearest neighbor interpolation
        true_masks = F.interpolate(
            true_masks.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze(1).long()
    
    # Assign pixels to slots
    pred_clusters = pred_masks.argmax(dim=1)  # [B, H, W]
    
    ari_scores = []
    for b in range(B):
        pred_flat = pred_clusters[b].reshape(-1).numpy()
        true_flat = true_masks[b].reshape(-1).numpy()
        
        # Exclude background (ID 0)
        mask = true_flat > 0
        if mask.sum() > 0:
            ari = adjusted_rand_score(true_flat[mask], pred_flat[mask])
            ari_scores.append(ari)
    
    return np.mean(ari_scores) if ari_scores else 0.0


if __name__ == "__main__":
    print("=" * 60)
    print("WORKING SLOT ATTENTION BASELINE TEST")
    print("=" * 60)
    
    # Create model
    model = WorkingSlotAttentionModel(
        num_slots=7,
        num_iterations=3,
        slot_dim=64,
        hidden_dim=128,
        image_size=(128, 128)
    )
    
    # Test forward pass
    B = 4
    images = torch.rand(B, 3, 128, 128)
    true_masks = torch.randint(0, 8, (B, 128, 128))
    
    outputs = model(images)
    
    print(f"\nOutput shapes:")
    print(f"  Reconstruction: {outputs['reconstructed'].shape}")
    print(f"  Masks: {outputs['masks'].shape}")
    print(f"  Slots: {outputs['slots'].shape}")
    print(f"  Attention: {outputs['attn'].shape}")
    
    print(f"\nLoss: {outputs['loss'].item():.4f}")
    
    # Check for slot collapse
    slots = outputs['slots']
    slots_norm = F.normalize(slots, dim=-1)
    sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
    mask_diag = ~torch.eye(7, dtype=torch.bool)
    off_diag_sim = sim[:, mask_diag].mean()
    print(f"\nSlot similarity (off-diag): {off_diag_sim:.4f} (should be < 0.5)")
    
    # ARI (random, should be near 0)
    ari = compute_ari_sklearn(outputs['masks'], true_masks)
    print(f"ARI (random): {ari:.4f} (should be ~0)")
    
    print("\n✅ Model forward pass successful!")
    print("\nTo train: python train_baseline.py --epochs 500 --batch-size 64")
