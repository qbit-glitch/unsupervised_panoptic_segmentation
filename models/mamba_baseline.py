#!/usr/bin/env python
"""
Stage-3: Mamba-Slot Attention Baseline for SpectralDiffusion

Integrates Mamba-2 O(N) slot attention with the working baseline from Stage-1.
Key innovations:
- Replace standard slot attention with Mamba-2 SSM
- Linear O(K) complexity for slot processing (vs O(K²))
- Bidirectional context aggregation
- GMM identifiability prior (NeurIPS 2024)

Expected improvements:
- 5× faster training/inference
- Same or better quality than standard slot attention
- Identifiable slot representations

Based on:
- Working Baseline (Stage-1) - SpatialBroadcastDecoder, CNN encoder
- Mamba-2 (Dao & Gu, ICML 2024)
- Identifiable Object-Centric Learning (Willetts & Paige, NeurIPS 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import adjusted_rand_score

# Import Mamba blocks
from .mamba_block import Mamba2Block


class MambaSlotAttentionSmall(nn.Module):
    """
    Mamba-based Slot Attention for small dimensions (64-dim).
    
    Adapted from MambaSlotAttention for the working baseline slot dimension.
    Uses O(K) complexity Mamba for slot refinement.
    """
    
    def __init__(
        self,
        num_slots: int = 7,
        dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        d_state: int = 16,  # Smaller state for dim=64
        use_mamba: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.use_mamba = use_mamba
        
        # Slot initialization (Gaussian)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_logsigma)
        
        # Attention components
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Layer norms
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
        # Mamba blocks for slot refinement
        if use_mamba:
            self.mamba_blocks = nn.ModuleList([
                Mamba2Block(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                    bidirectional=bidirectional,
                )
                for _ in range(iters)
            ])
        else:
            self.mamba_blocks = None
        
        # GRU for slot update (fallback or combined)
        self.gru = nn.GRUCell(dim, dim)
        
        # MLP for residual update
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        
        # GMM prior for identifiability (NeurIPS 2024)
        self.register_buffer('gmm_means', torch.zeros(num_slots, dim))
        self.register_buffer('gmm_vars', torch.ones(num_slots, dim))
        self.gmm_momentum = 0.99
        self.gmm_initialized = False
    
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
        
        # Iterative attention with Mamba refinement
        for t in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)  # [B, K, D]
            
            # Dot-product attention: [B, K, N]
            dots = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            
            # Softmax over SLOTS dimension FIRST (competition)
            attn = F.softmax(dots, dim=1) + self.eps  # [B, K, N]
            
            # Normalize over pixels (weighted mean)
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)  # [B, K, N]
            
            # Weighted mean of values
            updates = torch.einsum('bkn,bnd->bkd', attn_weights, v)  # [B, K, D]
            
            # GRU update
            slots = self.gru(
                updates.reshape(B * K, D),
                slots_prev.reshape(B * K, D)
            ).reshape(B, K, D)
            
            # Mamba refinement (O(K) complexity) - process slots as sequence
            if self.use_mamba and self.mamba_blocks is not None:
                # The Mamba block processes the K slots as a sequence
                slots = self.mamba_blocks[t](slots)  # [B, K, D]
            
            # Residual MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        
        # Update GMM prior during training
        if self.training:
            self._update_gmm_prior(slots.detach())
        
        # Return attention in [B, N, K] format
        attn_out = attn.transpose(1, 2)  # [B, N, K]
        
        return slots, attn_out
    
    def _update_gmm_prior(self, slots: torch.Tensor):
        """Update aggregate GMM prior parameters via EMA."""
        B, K, D = slots.shape
        
        slots_f32 = slots.float()
        batch_means = slots_f32.mean(dim=0)  # [K, D]
        slots_centered = slots_f32 - batch_means.unsqueeze(0)
        batch_vars = (slots_centered ** 2).mean(dim=0) + 1e-6  # [K, D]
        
        if not self.gmm_initialized:
            self.gmm_means.copy_(batch_means)
            self.gmm_vars.copy_(batch_vars)
            self.gmm_initialized = True
        else:
            self.gmm_means = self.gmm_momentum * self.gmm_means + (1 - self.gmm_momentum) * batch_means
            self.gmm_vars = self.gmm_momentum * self.gmm_vars + (1 - self.gmm_momentum) * batch_vars
    
    def compute_identifiability_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence to aggregate GMM prior for identifiability."""
        B, K, D = slots.shape
        
        # Batch slot statistics
        slot_means = slots.mean(dim=0)  # [K, D]
        slots_centered = slots - slot_means.unsqueeze(0)
        slot_vars = (slots_centered ** 2).mean(dim=0) + 1e-6  # [K, D]
        
        # KL divergence to prior (diagonal Gaussians)
        trace_term = (slot_vars / (self.gmm_vars + 1e-6)).sum()
        diff = self.gmm_means - slot_means
        mahal_term = (diff ** 2 / (self.gmm_vars + 1e-6)).sum()
        log_det_term = (torch.log(self.gmm_vars + 1e-6) - torch.log(slot_vars + 1e-6)).sum()
        
        kl = 0.5 * (trace_term + mahal_term - K * D + log_det_term)
        
        return kl / (B * K)


class SpatialBroadcastDecoder(nn.Module):
    """
    Decoder that outputs RGB + alpha for each slot.
    Same as Stage-1 working baseline.
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
        
        self.register_buffer('grid', self._build_grid(output_size[0], output_size[1]))
        
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
        features = self.encoder(x)
        B, D, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        return features


class MambaSlotAttentionModel(nn.Module):
    """
    Stage-3: Mamba-Slot Attention Model
    
    Replaces standard slot attention with Mamba-based O(K) attention.
    Compatible with Stage-1 pretrained weights for encoder and decoder.
    
    Changes from Stage-1:
    - slot_attention: WorkingSlotAttention -> MambaSlotAttentionSmall
    - Adds identifiability loss
    - Adds slot diversity monitoring
    """
    
    def __init__(
        self,
        num_slots: int = 7,
        num_iterations: int = 3,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        image_size: Tuple[int, int] = (128, 128),
        use_mamba: bool = True,
        lambda_ident: float = 0.01,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_slots = num_slots
        self.use_mamba = use_mamba
        self.lambda_ident = lambda_ident
        
        # Same encoder as Stage-1
        self.encoder = SimpleEncoder(3, slot_dim)
        
        # Mamba-based Slot Attention (NEW in Stage-3)
        self.slot_attention = MambaSlotAttentionSmall(
            num_slots=num_slots,
            dim=slot_dim,
            iters=num_iterations,
            hidden_dim=hidden_dim,
            d_state=16,
            use_mamba=use_mamba,
            bidirectional=True,
        )
        
        # Same decoder as Stage-1
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
        
        # Mamba-Slot attention
        slots, attn = self.slot_attention(features)  # [B, K, D], [B, N, K]
        
        # Decode
        recon, masks, rgb_per_slot = self.decoder(slots)
        
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
            
            # Identifiability loss (GMM prior - NeurIPS 2024)
            ident_loss = self.slot_attention.compute_identifiability_loss(slots)
            
            # Total loss
            total_loss = recon_loss + self.lambda_ident * ident_loss
            
            outputs['loss'] = total_loss
            # Keep as 1D tensors for DataParallel compatibility (avoid scalar gather warning)
            outputs['loss_dict'] = {
                'recon': recon_loss.unsqueeze(0),
                'ident': ident_loss.unsqueeze(0),
                'total': total_loss.unsqueeze(0),
            }
        
        return outputs
    
    def load_stage1_weights(self, stage1_checkpoint_path: str):
        """
        Load encoder and decoder weights from Stage-1 checkpoint.
        The slot_attention module is new and will be trained from scratch.
        """
        print(f"Loading Stage-1 weights from {stage1_checkpoint_path}")
        
        state_dict = torch.load(stage1_checkpoint_path, map_location='cpu')
        
        # Get current model state
        current_state = self.state_dict()
        
        # Filter to only encoder and decoder weights
        loaded_count = 0
        for name, param in state_dict.items():
            if name in current_state:
                if 'encoder' in name or 'decoder' in name:
                    if current_state[name].shape == param.shape:
                        current_state[name] = param
                        loaded_count += 1
        
        self.load_state_dict(current_state)
        print(f"Loaded {loaded_count} encoder/decoder parameters from Stage-1")
        print("Slot attention module initialized fresh (Mamba-based)")


def compute_ari_sklearn(pred_masks, true_masks):
    """Compute ARI using sklearn. Handles different mask resolutions."""
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
    print("STAGE-3: MAMBA-SLOT ATTENTION BASELINE TEST")
    print("=" * 60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = MambaSlotAttentionModel(
        num_slots=7,
        num_iterations=3,
        slot_dim=64,
        hidden_dim=128,
        image_size=(128, 128),
        use_mamba=True,
    ).to(device)
    
    # Test forward pass
    B = 4
    images = torch.rand(B, 3, 128, 128, device=device)
    true_masks = torch.randint(0, 8, (B, 128, 128))
    
    outputs = model(images)
    
    print(f"\nOutput shapes:")
    print(f"  Reconstruction: {outputs['reconstructed'].shape}")
    print(f"  Masks: {outputs['masks'].shape}")
    print(f"  Slots: {outputs['slots'].shape}")
    print(f"  Attention: {outputs['attn'].shape}")
    
    print(f"\nLosses:")
    for k, v in outputs['loss_dict'].items():
        print(f"  {k}: {v:.4f}")
    
    # Check for slot collapse
    slots = outputs['slots']
    slots_norm = F.normalize(slots, dim=-1)
    sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
    mask_diag = ~torch.eye(7, dtype=torch.bool, device=device)
    off_diag_sim = sim[:, mask_diag].mean()
    print(f"\nSlot similarity (off-diag): {off_diag_sim:.4f} (should be < 0.5)")
    
    # ARI (random, should be near 0)
    ari = compute_ari_sklearn(outputs['masks'], true_masks)
    print(f"ARI (random): {ari:.4f} (should be ~0)")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")
    
    print("\n✅ Stage-3 model forward pass successful!")
    print("\nTo train: python train_stage3.py --epochs 100 --batch-size 32")
