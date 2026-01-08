#!/usr/bin/env python
"""
Enhanced Slot Attention Model - Phase 2
Based on debug_1.md SlotAttention2025 (lines 900-1200)

Key improvements from debug_1.md enhanced version:
1. GMM-based slot initialization (per-slot learnable means/sigmas)
2. Learnable temperature for attention
3. norm_after_attn (LayerNorm AFTER aggregation)
4. ContrastiveSlotLoss (optional)
5. Diversity loss to prevent slot collapse

Target: ARI > 0.7 (up from 0.395 baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import adjusted_rand_score


class SlotAttention2025(nn.Module):
    """
    Enhanced Slot Attention from debug_1.md
    Based on SlotContrast (CVPR 2025) + SlotDiffusion (NeurIPS 2023)
    
    Key innovations:
    - GMM-based slot initialization (per-slot learnable means/sigmas)
    - Learnable temperature
    - norm_after_attn (LayerNorm AFTER aggregation) - Krimmel et al. 2024
    """
    
    def __init__(
        self,
        num_slots: int = 11,
        dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        use_gmm: bool = True
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.use_gmm = use_gmm
        
        # GMM-based initialization (per-slot means and sigmas)
        if use_gmm:
            # Each slot has its own learnable mean and sigma
            self.slots_mu = nn.Parameter(torch.randn(num_slots, dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(num_slots, dim))
            self.slots_logpi = nn.Parameter(torch.zeros(num_slots))  # Mixture weights
            nn.init.xavier_uniform_(self.slots_mu)
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.xavier_uniform_(self.slots_mu)
        
        # CRITICAL: Better normalization (Krimmel et al. 2024)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_after_attn = nn.LayerNorm(dim)  # NEW from debug_1.md
        
        # Attention with learnable temperature
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # Learnable temperature (2025 improvement)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Slot update
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        slots_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, N, D] input features
            slots_init: [B, K, D] optional external initialization (e.g., spectral)
        Returns:
            slots: [B, K, D]
            attn: [B, N, K]
        """
        B, N, D = inputs.shape
        K = self.num_slots
        
        # Initialize slots
        if slots_init is not None:
            slots = slots_init
        elif self.use_gmm:
            # GMM-based initialization (per-slot)
            mu = self.slots_mu.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
            sigma = self.slots_logsigma.exp().unsqueeze(0).expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_logsigma.exp().expand(B, K, -1)
            slots = mu + sigma * torch.randn_like(mu)
        
        # Normalize inputs
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        
        # Iterative attention
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)
            
            # Attention with learnable temperature
            dots = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            dots = dots / (self.temperature.exp() + self.eps)  # Learnable temperature
            
            # CRITICAL: Softmax over SLOTS (dim=1) - competition
            attn = F.softmax(dots, dim=1) + self.eps  # [B, K, N]
            
            # Normalize over pixels (weighted mean)
            attn_sum = attn.sum(dim=-1, keepdim=True)
            attn_normalized = attn / (attn_sum + self.eps)  # [B, K, N]
            
            # Aggregate
            updates = torch.einsum('bkn,bnd->bkd', attn_normalized, v)
            
            # NEW: Normalize AFTER aggregation (Krimmel et al. 2024)
            updates = self.norm_after_attn(updates)
            
            # Update slots via GRU
            slots = self.gru(
                updates.reshape(B * K, D),
                slots_prev.reshape(B * K, D)
            )
            slots = slots.reshape(B, K, D)
            
            # Residual MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        
        attn_out = attn.transpose(1, 2)  # [B, N, K]
        return slots, attn_out


class SpectralInitializer(nn.Module):
    """Multi-scale spectral initialization using k-means++."""
    
    def __init__(
        self,
        scales: List[int] = [4, 8, 16],
        k_per_scale: int = 4,
        knn_k: int = 10
    ):
        super().__init__()
        self.scales = scales
        self.k_per_scale = k_per_scale
        self.knn_k = knn_k
        self.num_slots = len(scales) * k_per_scale
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            B, N, D = features.shape
            H = W = int(np.sqrt(N))
            features = features.reshape(B, H, W, D)
        
        B, H, W, D = features.shape
        device = features.device
        
        all_slots = []
        
        for scale in self.scales:
            F_scale = F.adaptive_avg_pool2d(
                features.permute(0, 3, 1, 2),
                (scale, scale)
            ).permute(0, 2, 3, 1)
            
            F_scale_flat = F_scale.reshape(B, scale * scale, D)
            
            batch_slots = []
            for b in range(B):
                centroids = self._kmeans_pp(F_scale_flat[b], self.k_per_scale)
                batch_slots.append(centroids)
            
            scale_slots = torch.stack(batch_slots, dim=0)
            all_slots.append(scale_slots)
        
        return torch.cat(all_slots, dim=1)
    
    def _kmeans_pp(self, features: torch.Tensor, k: int) -> torch.Tensor:
        N, D = features.shape
        device = features.device
        
        centers_idx = [torch.randint(0, N, (1,), device=device).item()]
        
        for _ in range(k - 1):
            centers = features[centers_idx]
            dists = torch.cdist(features, centers)
            min_dists = dists.min(dim=1)[0]
            probs = min_dists ** 2
            probs = probs / (probs.sum() + 1e-8)
            next_center = torch.multinomial(probs, 1).item()
            centers_idx.append(next_center)
        
        return features[centers_idx]


class SpatialBroadcastDecoder(nn.Module):
    """Decoder with RGB + alpha output."""
    
    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dim: int = 64,
        output_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.output_size = output_size
        
        self.register_buffer('grid', self._build_grid(*output_size))
        
        self.initial = nn.Sequential(
            nn.Linear(slot_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, 1)
        )
    
    def _build_grid(self, H: int, W: int) -> torch.Tensor:
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        return torch.stack([grid_x, grid_y], dim=-1)
    
    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, D = slots.shape
        H, W = self.output_size
        
        slots_flat = slots.reshape(B * K, D)
        grid = self.grid.unsqueeze(0).expand(B * K, -1, -1, -1)
        grid_flat = grid.reshape(B * K, H * W, 2)
        
        slots_broadcast = slots_flat.unsqueeze(1).expand(-1, H * W, -1)
        x = torch.cat([slots_broadcast, grid_flat], dim=-1)
        
        x = self.initial(x)
        x = x.reshape(B * K, H, W, -1).permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = x.reshape(B, K, 4, H, W)
        
        rgb = torch.sigmoid(x[:, :, :3])
        mask_logits = x[:, :, 3:4]
        masks = F.softmax(mask_logits, dim=1).squeeze(2)
        masks_expanded = masks.unsqueeze(2)
        recon = (rgb * masks_expanded).sum(dim=1)
        
        return recon, masks, rgb


class SimpleEncoder(nn.Module):
    """Simple CNN encoder."""
    
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
        features = self.encoder(x)
        B, D, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, D)


class EnhancedSlotAttentionModel(nn.Module):
    """
    Enhanced model with SlotAttention2025 from debug_1.md.
    
    Key differences from baseline:
    - GMM-based slot initialization (per-slot learnable means/sigmas)
    - Learnable temperature
    - norm_after_attn (LayerNorm AFTER aggregation)
    - Optional spectral initialization
    - Diversity loss
    """
    
    def __init__(
        self,
        num_slots: int = 11,
        num_iterations: int = 3,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        image_size: Tuple[int, int] = (128, 128),
        use_spectral_init: bool = True,
        use_gmm: bool = True,
        use_mamba: bool = False  # NEW: Use MambaSlotAttention for O(N) complexity
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_slots = num_slots
        self.use_spectral_init = use_spectral_init
        self.use_mamba = use_mamba
        
        self.encoder = SimpleEncoder(3, slot_dim)
        
        if use_spectral_init:
            k_per_scale = num_slots // 3
            self.spectral_init = SpectralInitializer(
                scales=[4, 8, 16],
                k_per_scale=k_per_scale,
                knn_k=10
            )
            self.num_slots = k_per_scale * 3
        else:
            self.spectral_init = None
        
        # Choose slot attention type: Mamba (O(N)) vs Standard (O(N²))
        if use_mamba:
            from .mamba_slot import MambaSlotAttention
            self.slot_attention = MambaSlotAttention(
                dim=slot_dim,
                num_slots=self.num_slots,
                num_iterations=num_iterations,
                d_state=64,
                use_mamba=False,  # Use TransformerBlock for MPS compatibility
                bidirectional=True
            )
            print(f"✓ Using MambaSlotAttention with TransformerBlock (MPS compatible)")
        else:
            # Use SlotAttention2025 with GMM and learnable temperature
            self.slot_attention = SlotAttention2025(
                num_slots=self.num_slots,
                dim=slot_dim,
                iters=num_iterations,
                hidden_dim=hidden_dim,
                use_gmm=use_gmm
            )
            print(f"✓ Using SlotAttention2025 (O(N²) quadratic complexity)")
        
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            output_size=image_size
        )
    
    def forward(self, images: torch.Tensor, return_loss: bool = True) -> Dict:
        B = images.shape[0]
        
        features = self.encoder(images)
        
        if self.use_spectral_init and self.spectral_init is not None:
            H = W = int(np.sqrt(features.shape[1]))
            features_spatial = features.reshape(B, H, W, -1)
            slots_init = self.spectral_init(features_spatial)
        else:
            slots_init = None
        
        slots, attn = self.slot_attention(features, slots_init)
        recon, masks, rgb_per_slot = self.decoder(slots)
        
        outputs = {
            'reconstructed': recon,
            'masks': masks,
            'slots': slots,
            'attn': attn,
            'rgb_per_slot': rgb_per_slot,
        }
        
        if return_loss:
            recon_loss = F.mse_loss(recon, images)
            
            # Diversity loss (prevent slot collapse)
            slots_norm = F.normalize(slots, dim=-1)
            sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
            K = slots.shape[1]
            mask_diag = ~torch.eye(K, dtype=torch.bool, device=slots.device)
            off_diag = sim[:, mask_diag]
            diversity_loss = off_diag.mean()
            
            total_loss = recon_loss + 0.1 * diversity_loss
            
            outputs['loss'] = total_loss
            outputs['loss_dict'] = {
                'recon': recon_loss.item(),
                'diversity': diversity_loss.item(),
                'total': total_loss.item()
            }
        
        return outputs


def compute_ari_sklearn(pred_masks, true_masks):
    """Compute ARI with mask resizing."""
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.detach().cpu()
    if torch.is_tensor(true_masks):
        true_masks = true_masks.detach().cpu()
    
    B, K, H, W = pred_masks.shape
    
    if true_masks.dim() == 4:
        true_masks = true_masks.argmax(dim=1)
    
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


if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED MODEL (SlotAttention2025) TEST")
    print("=" * 60)
    
    for use_spectral in [False, True]:
        print(f"\nSpectral Init: {use_spectral}")
        
        model = EnhancedSlotAttentionModel(
            num_slots=12,
            num_iterations=3,
            slot_dim=64,
            hidden_dim=128,
            image_size=(128, 128),
            use_spectral_init=use_spectral,
            use_gmm=True
        )
        
        B = 4
        images = torch.rand(B, 3, 128, 128)
        outputs = model(images)
        
        print(f"  Reconstruction: {outputs['reconstructed'].shape}")
        print(f"  Masks: {outputs['masks'].shape}")
        print(f"  Slots: {outputs['slots'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        
        # Check slot diversity
        slots = outputs['slots']
        slots_norm = F.normalize(slots, dim=-1)
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        K = slots.shape[1]
        mask_diag = ~torch.eye(K, dtype=torch.bool)
        off_diag = sim[:, mask_diag].mean()
        print(f"  Slot similarity (off-diag): {off_diag:.4f}")
    
    print("\n✅ SlotAttention2025 model ready!")
    print("Key improvements from debug_1.md:")
    print("  - GMM-based slot initialization (per-slot means/sigmas)")
    print("  - Learnable temperature")
    print("  - norm_after_attn (LayerNorm AFTER aggregation)")
