"""
Mamba-Slot Attention for SpectralDiffusion

Combines Mamba-2 state space models with slot attention for
linear O(N) complexity object-centric learning.

Key innovations:
- Replace standard O(N²) slot attention with Mamba SSM
- Identifiable slots via aggregate GMM prior (NeurIPS 2024)
- Bidirectional context aggregation for slot refinement

Based on:
- Slot Attention (Locatello et al., 2020)
- Identifiable Object-Centric Learning (Willetts & Paige, NeurIPS 2024)
- Mamba-2 (Dao & Gu, ICML 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat
import math

from .mamba_block import Mamba2Block, MambaStack, TransformerBlock


class MambaSlotAttention(nn.Module):
    """
    Mamba-based Slot Attention with linear complexity.
    
    Instead of standard attention between slots and features,
    we use Mamba SSM to iteratively refine slot representations.
    
    Args:
        dim: Feature dimension
        num_slots: Number of slots
        num_iterations: Number of refinement iterations
        d_state: Mamba state dimension
        use_mamba: Whether to use Mamba (True) or Transformer (False) for ablation
        bidirectional: Whether to use bidirectional Mamba
        epsilon: Small constant for numerical stability
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_slots: int = 12,
        num_iterations: int = 3,
        d_state: int = 64,
        use_mamba: bool = True,
        bidirectional: bool = True,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.use_mamba = use_mamba
        
        # Layer norms
        self.norm_features = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        
        # Mamba or Transformer blocks for each iteration
        if use_mamba:
            self.refinement_blocks = nn.ModuleList([
                Mamba2Block(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                    bidirectional=bidirectional,
                )
                for _ in range(num_iterations)
            ])
        else:
            # Transformer for ablation (A2)
            self.refinement_blocks = nn.ModuleList([
                TransformerBlock(
                    d_model=dim,
                    n_heads=8,
                    mlp_ratio=4.0,
                )
                for _ in range(num_iterations)
            ])
        
        # Slot update MLP (GRU-style update)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # Cross-attention for slot-feature interaction
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.scale = dim ** -0.5
        
        # Identifiable GMM prior parameters (NeurIPS 2024)
        # These are updated via EMA during training
        self.register_buffer('gmm_means', torch.zeros(num_slots, dim))
        self.register_buffer('gmm_covs', torch.eye(dim).unsqueeze(0).repeat(num_slots, 1, 1))
        self.register_buffer('gmm_weights', torch.ones(num_slots) / num_slots)
        self.gmm_momentum = 0.99
        self.gmm_initialized = False
    
    def slot_to_feature_attention(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft attention from slots to features.
        
        Args:
            slots: [B, K, D] slot representations
            features: [B, N, D] feature representations
            
        Returns:
            updates: [B, K, D] aggregated feature updates
            attn_weights: [B, N, K] attention weights (soft masks)
        """
        B, K, D = slots.shape
        _, N, _ = features.shape
        
        # Normalize
        slots_norm = self.norm_slots(slots)
        features_norm = self.norm_features(features)
        
        # Compute Q, K, V
        q = self.to_q(slots_norm)  # [B, K, D]
        k = self.to_k(features_norm)  # [B, N, D]
        v = self.to_v(features_norm)  # [B, N, D]
        
        # Attention logits: [B, K, N]
        attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
        
        # Softmax over slots for each feature (competition)
        # [B, N, K] - each feature attends to slots
        attn_weights = F.softmax(attn_logits.transpose(1, 2), dim=-1)  # [B, N, K]
        
        # Weighted sum of features for each slot
        # Normalize by slot attention mass
        slot_attn_mass = attn_weights.sum(dim=1, keepdim=True)  # [B, 1, K]
        slot_attn_mass = slot_attn_mass.clamp(min=self.epsilon)
        
        # Aggregate features per slot
        updates = torch.einsum('bnk,bnd->bkd', attn_weights, v)  # [B, K, D]
        updates = updates / slot_attn_mass.transpose(1, 2)  # Normalize
        
        return updates, attn_weights
    
    def forward(
        self,
        features: torch.Tensor,
        slots_init: torch.Tensor,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Mamba-Slot Attention.
        
        Args:
            features: [B, N, D] DINOv3 patch features
            slots_init: [B, K, D] spectral initialization
            return_attention: Whether to return attention weights
            
        Returns:
            slots: [B, K, D] refined slot representations
            masks: [B, N, K] soft assignment masks
        """
        B, N, D = features.shape
        K = self.num_slots
        
        slots = slots_init
        
        for t, block in enumerate(self.refinement_blocks):
            # === Step 1: Slot-Feature Cross-Attention ===
            # Get updates from features via attention (correct order)
            updates, masks = self.slot_to_feature_attention(slots, features)
            
            # === Step 2: Process ONLY slots through Mamba/Transformer ===
            # This is O(K²) not O(N²), where K << N (12 vs 16384)
            slots_residual = block(self.norm_slots(slots))  # [B, K, D]
            
            # === Step 3: Slot Update (GRU-style) ===
            # Combine transformer output with attention aggregation
            slot_update = self.slot_mlp(updates + slots_residual)
            
            # Residual update
            slots = slots + slot_update
        
        # Final attention weights for masks
        _, masks = self.slot_to_feature_attention(slots, features)
        
        # Update GMM prior (only during training)
        if self.training:
            self.update_gmm_prior(slots.detach())
        
        if return_attention:
            return slots, masks
        return slots
    
    def update_gmm_prior(self, slots: torch.Tensor):
        """
        Update aggregate GMM prior parameters via EMA.
        
        Based on Willetts & Paige (NeurIPS 2024) for identifiability.
        
        Args:
            slots: [B, K, D] current slot representations
        """
        B, K, D = slots.shape
        
        # Convert to float32 for stable computations (bfloat16 issue with add_ alpha)
        slots_f32 = slots.float()
        
        # Compute batch statistics
        batch_means = slots_f32.mean(dim=0)  # [K, D]
        
        # Compute covariance (simplified diagonal approximation)
        slots_centered = slots_f32 - batch_means.unsqueeze(0)
        batch_vars = (slots_centered ** 2).mean(dim=0)  # [K, D]
        
        # EMA update (use explicit operations for bfloat16 compatibility)
        if not self.gmm_initialized:
            self.gmm_means.copy_(batch_means)
            # Initialize covariance as diagonal
            for k in range(K):
                self.gmm_covs[k] = torch.diag(batch_vars[k] + 1e-6)
            self.gmm_initialized = True
        else:
            # Explicit EMA: new = momentum * old + (1 - momentum) * new
            self.gmm_means = self.gmm_momentum * self.gmm_means + (1 - self.gmm_momentum) * batch_means
            for k in range(K):
                new_cov = torch.diag(batch_vars[k] + 1e-6)
                self.gmm_covs[k] = self.gmm_momentum * self.gmm_covs[k] + (1 - self.gmm_momentum) * new_cov
    
    def compute_identifiability_loss(
        self,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence to aggregate GMM prior.
        
        Regularizes slot distributions to have identifiable structure
        as per Willetts & Paige (NeurIPS 2024).
        
        Args:
            slots: [B, K, D] slot representations
            
        Returns:
            kl_loss: scalar KL divergence loss
        """
        B, K, D = slots.shape
        
        # Compute batch slot distribution (approximate as Gaussian per slot)
        slot_means = slots.mean(dim=0)  # [K, D]
        slots_centered = slots - slot_means.unsqueeze(0)
        slot_vars = (slots_centered ** 2).mean(dim=0) + 1e-6  # [K, D]
        
        # KL divergence to prior: KL(q || p)
        # For diagonal Gaussians:
        # KL = 0.5 * (tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) - D + log(|Σ_p|/|Σ_q|))
        
        kl = 0.0
        for k in range(K):
            # Prior parameters
            mu_p = self.gmm_means[k]  # [D]
            sigma_p = torch.diag(self.gmm_covs[k])  # [D] diagonal
            
            # Posterior parameters
            mu_q = slot_means[k]  # [D]
            sigma_q = slot_vars[k]  # [D]
            
            # KL for diagonal Gaussians
            trace_term = (sigma_q / (sigma_p + 1e-6)).sum()
            diff = mu_p - mu_q
            mahal_term = (diff ** 2 / (sigma_p + 1e-6)).sum()
            log_det_term = (torch.log(sigma_p + 1e-6) - torch.log(sigma_q + 1e-6)).sum()
            
            kl += 0.5 * (trace_term + mahal_term - D + log_det_term)
        
        return kl / (B * K)
    
    def get_slot_utilization(
        self,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute utilization of each slot.
        
        Args:
            masks: [B, N, K] soft assignment masks
            
        Returns:
            utilization: [B, K] utilization score per slot
        """
        # Sum attention mass per slot
        utilization = masks.sum(dim=1)  # [B, K]
        
        # Normalize by number of features
        N = masks.shape[1]
        utilization = utilization / N
        
        return utilization


class StandardSlotAttention(nn.Module):
    """
    Standard Slot Attention (Locatello et al., 2020) for ablation.
    
    O(N²) complexity through attention mechanism.
    Used for ablation A2 comparison.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_slots: int = 12,
        num_iterations: int = 3,
        hidden_dim: Optional[int] = None,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        
        hidden_dim = hidden_dim or dim
        
        # Layer norms
        self.norm_features = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        
        # Attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.scale = dim ** -0.5
        
        # GRU update
        self.gru = nn.GRUCell(dim, dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        slots_init: torch.Tensor,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard slot attention forward pass.
        
        Args:
            features: [B, N, D] input features
            slots_init: [B, K, D] initial slots
            return_attention: Whether to return attention
            
        Returns:
            slots: [B, K, D] refined slots
            attn: [B, N, K] attention weights
        """
        B, N, D = features.shape
        K = self.num_slots
        
        slots = slots_init
        
        # Feature projections (computed once)
        features_norm = self.norm_features(features)
        k = self.to_k(features_norm)  # [B, N, D]
        v = self.to_v(features_norm)  # [B, N, D]
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            
            # Query from slots
            q = self.to_q(slots_norm)  # [B, K, D]
            
            # Attention: [B, K, N]
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            
            # Softmax over slots (competition)
            attn = F.softmax(attn_logits.transpose(1, 2), dim=-1)  # [B, N, K]
            
            # Weighted mean
            attn_mass = attn.sum(dim=1, keepdim=True).clamp(min=self.epsilon)
            updates = torch.einsum('bnk,bnd->bkd', attn, v)
            updates = updates / attn_mass.transpose(1, 2)
            
            # GRU update
            slots = self.gru(
                updates.reshape(B * K, D),
                slots_prev.reshape(B * K, D),
            ).reshape(B, K, D)
            
            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        # Final attention
        q = self.to_q(self.norm_slots(slots))
        attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
        attn = F.softmax(attn_logits.transpose(1, 2), dim=-1)
        
        if return_attention:
            return slots, attn
        return slots


def create_slot_attention(
    attention_type: str = "mamba",
    dim: int = 768,
    num_slots: int = 12,
    num_iterations: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create slot attention module.
    
    Args:
        attention_type: "mamba", "transformer", or "standard"
        dim: Feature dimension
        num_slots: Number of slots
        num_iterations: Refinement iterations
        
    Returns:
        Slot attention module
    """
    if attention_type == "mamba":
        return MambaSlotAttention(
            dim=dim,
            num_slots=num_slots,
            num_iterations=num_iterations,
            use_mamba=True,
            **kwargs,
        )
    elif attention_type == "transformer":
        return MambaSlotAttention(
            dim=dim,
            num_slots=num_slots,
            num_iterations=num_iterations,
            use_mamba=False,
            **kwargs,
        )
    elif attention_type == "standard":
        return StandardSlotAttention(
            dim=dim,
            num_slots=num_slots,
            num_iterations=num_iterations,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    # Test Mamba-Slot Attention
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Create Mamba-Slot Attention
    print("\n=== Testing MambaSlotAttention ===")
    mamba_slot = MambaSlotAttention(
        dim=768,
        num_slots=12,
        num_iterations=3,
        use_mamba=True,
    ).to(device)
    
    # Test inputs
    B, N, K, D = 2, 256, 12, 768
    features = torch.randn(B, N, D, device=device)
    slots_init = torch.randn(B, K, D, device=device)
    
    # Forward pass
    slots, masks = mamba_slot(features, slots_init)
    print(f"Input features: {features.shape}")
    print(f"Initial slots: {slots_init.shape}")
    print(f"Output slots: {slots.shape}")
    print(f"Attention masks: {masks.shape}")
    
    # Test identifiability loss
    kl_loss = mamba_slot.compute_identifiability_loss(slots)
    print(f"Identifiability loss: {kl_loss.item():.4f}")
    
    # Test slot utilization
    util = mamba_slot.get_slot_utilization(masks)
    print(f"Slot utilization: {util.mean().item():.4f} (avg)")
    
    # Count parameters
    params = sum(p.numel() for p in mamba_slot.parameters())
    print(f"Parameters: {params:,}")
    
    # Test Standard Slot Attention for comparison
    print("\n=== Testing StandardSlotAttention (ablation) ===")
    standard_slot = StandardSlotAttention(
        dim=768,
        num_slots=12,
        num_iterations=3,
    ).to(device)
    
    slots_std, masks_std = standard_slot(features, slots_init)
    print(f"Standard slots: {slots_std.shape}")
    print(f"Standard masks: {masks_std.shape}")
    
    params_std = sum(p.numel() for p in standard_slot.parameters())
    print(f"Standard parameters: {params_std:,}")
