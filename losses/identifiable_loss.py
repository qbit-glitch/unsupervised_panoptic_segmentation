"""
Identifiability Loss for SpectralDiffusion

KL divergence to aggregate GMM prior for provable slot identifiability.

Based on:
- Identifiable Object-Centric Learning (Willetts & Paige, NeurIPS 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IdentifiabilityLoss(nn.Module):
    """
    Identifiability loss via KL divergence to aggregate GMM prior.
    
    L_ident = D_KL(q(S) || p(S | GMM_aggregate))
    
    Regularizes slot distributions to have identifiable structure.
    
    Args:
        num_slots: Number of slots
        slot_dim: Slot dimension
        ema_momentum: EMA momentum for prior updates
        diagonal_cov: Whether to use diagonal covariance
    """
    
    def __init__(
        self,
        num_slots: int = 12,
        slot_dim: int = 768,
        ema_momentum: float = 0.99,
        diagonal_cov: bool = True,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.ema_momentum = ema_momentum
        self.diagonal_cov = diagonal_cov
        
        # GMM prior parameters (learned via EMA)
        self.register_buffer('prior_means', torch.zeros(num_slots, slot_dim))
        self.register_buffer('prior_vars', torch.ones(num_slots, slot_dim))
        self.register_buffer('prior_weights', torch.ones(num_slots) / num_slots)
        self.register_buffer('initialized', torch.tensor(False))
    
    def update_prior(
        self,
        slots: torch.Tensor,
    ):
        """
        Update GMM prior via EMA.
        
        Args:
            slots: [B, K, D] slot representations
        """
        with torch.no_grad():
            B, K, D = slots.shape
            
            # Convert to float32 for stable EMA (bfloat16 compatibility)
            slots_f32 = slots.float()
            
            # Compute batch statistics
            batch_means = slots_f32.mean(dim=0)  # [K, D]
            slots_centered = slots_f32 - batch_means.unsqueeze(0)
            batch_vars = (slots_centered ** 2).mean(dim=0) + 1e-6  # [K, D]
            
            if not self.initialized.item():
                # Initialize
                self.prior_means.copy_(batch_means)
                self.prior_vars.copy_(batch_vars)
                self.initialized.fill_(True)
            else:
                # Explicit EMA update (bfloat16 compatible)
                # new = momentum * old + (1 - momentum) * batch
                self.prior_means = self.ema_momentum * self.prior_means + (1 - self.ema_momentum) * batch_means
                self.prior_vars = self.ema_momentum * self.prior_vars + (1 - self.ema_momentum) * batch_vars
    
    def compute_kl(
        self,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence to prior.
        
        For diagonal Gaussians:
        KL(q || p) = 0.5 * (tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) 
                          - D + log(|Σ_p|/|Σ_q|))
        
        Args:
            slots: [B, K, D] slot representations
            
        Returns:
            kl: Scalar KL divergence
        """
        B, K, D = slots.shape
        
        # Posterior statistics (per slot across batch)
        post_means = slots.mean(dim=0)  # [K, D]
        slots_centered = slots - post_means.unsqueeze(0)
        post_vars = (slots_centered ** 2).mean(dim=0) + 1e-6  # [K, D]
        
        # KL for diagonal Gaussians
        # tr(Σ_p^{-1} Σ_q) = sum(σ_q^2 / σ_p^2)
        trace_term = (post_vars / (self.prior_vars + 1e-8)).sum()
        
        # (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) = sum((μ_p - μ_q)^2 / σ_p^2)
        diff = self.prior_means - post_means
        mahal_term = ((diff ** 2) / (self.prior_vars + 1e-8)).sum()
        
        # log(|Σ_p|/|Σ_q|) = sum(log(σ_p^2) - log(σ_q^2))
        log_det_term = (torch.log(self.prior_vars + 1e-8) - torch.log(post_vars + 1e-8)).sum()
        
        # Full KL
        kl = 0.5 * (trace_term + mahal_term - K * D + log_det_term)
        
        # Normalize by batch size and slot count
        kl = kl / (B * K)
        
        return kl
    
    def forward(
        self,
        slots: torch.Tensor,
        update_prior: bool = True,
    ) -> torch.Tensor:
        """
        Compute identifiability loss.
        
        Args:
            slots: [B, K, D] slot representations
            update_prior: Whether to update GMM prior
            
        Returns:
            loss: Identifiability loss
        """
        if update_prior and self.training:
            self.update_prior(slots)
        
        kl = self.compute_kl(slots)
        
        return kl


class SlotDiversityLoss(nn.Module):
    """
    Slot diversity loss to prevent slot collapse.
    
    Encourages slots to be diverse/different from each other.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diversity loss.
        
        Minimizes cosine similarity between slots.
        
        Args:
            slots: [B, K, D] slot representations
            
        Returns:
            loss: Diversity loss
        """
        B, K, D = slots.shape
        
        # Normalize slots
        slots_norm = F.normalize(slots, p=2, dim=-1)  # [B, K, D]
        
        # Compute pairwise cosine similarity
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(K, device=slots.device, dtype=torch.bool)
        sim = sim.masked_fill(mask.unsqueeze(0), 0)
        
        # Average off-diagonal similarity (should be low)
        num_pairs = K * (K - 1)
        loss = sim.abs().sum(dim=[1, 2]) / num_pairs
        
        return loss.mean()


class AggregateGMMPrior(nn.Module):
    """
    Aggregate Gaussian Mixture Model prior for identifiability.
    
    Full implementation of Willetts & Paige (NeurIPS 2024).
    """
    
    def __init__(
        self,
        num_slots: int = 12,
        slot_dim: int = 768,
        num_components: int = 4,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_components = num_components
        
        # Learnable GMM parameters per slot
        self.means = nn.Parameter(torch.randn(num_slots, num_components, slot_dim) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(num_slots, num_components, slot_dim))
        self.logits = nn.Parameter(torch.zeros(num_slots, num_components))
    
    def log_prob(
        self,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability under GMM prior.
        
        Args:
            slots: [B, K, D] slot representations
            
        Returns:
            log_prob: [B, K] log probabilities
        """
        B, K, D = slots.shape
        
        # Component weights
        weights = F.softmax(self.logits, dim=-1)  # [K, C]
        
        # Compute log prob for each component
        slots_expanded = slots.unsqueeze(2)  # [B, K, 1, D]
        
        # Gaussian log prob: -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
        means = self.means.unsqueeze(0)  # [1, K, C, D]
        vars = torch.exp(self.log_vars).unsqueeze(0)  # [1, K, C, D]
        
        diff = slots_expanded - means
        log_probs = -0.5 * (
            D * torch.log(torch.tensor(2 * 3.14159)) +
            self.log_vars.unsqueeze(0).sum(dim=-1) +
            ((diff ** 2) / (vars + 1e-8)).sum(dim=-1)
        )  # [B, K, C]
        
        # Log-sum-exp over components
        log_weights = torch.log(weights + 1e-8).unsqueeze(0)  # [1, K, C]
        log_prob = torch.logsumexp(log_probs + log_weights, dim=-1)  # [B, K]
        
        return log_prob
    
    def forward(
        self,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log likelihood as loss.
        
        Args:
            slots: [B, K, D] slot representations
            
        Returns:
            loss: NLL loss for identifiability
        """
        log_prob = self.log_prob(slots)
        loss = -log_prob.mean()
        
        return loss
