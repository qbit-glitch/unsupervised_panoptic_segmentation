"""
Spectral Consistency Loss for SpectralDiffusion

Ensures slot masks align with spectral clustering structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralConsistencyLoss(nn.Module):
    """
    Spectral consistency loss.
    
    L_spec = ||M_spectral - M_slot||_F^2
    
    Encourages slot attention masks to respect spectral partitioning.
    
    Args:
        reduction: "mean", "sum", or "none"
        normalize_masks: Whether to normalize masks before comparison
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        normalize_masks: bool = True,
    ):
        super().__init__()
        self.reduction = reduction
        self.normalize_masks = normalize_masks
    
    def forward(
        self,
        spectral_masks: torch.Tensor,
        slot_masks: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute spectral consistency loss.
        
        Args:
            spectral_masks: [B, K, N] or [B, N, K] spectral clustering masks
            slot_masks: [B, N, K] slot attention masks
            weights: Optional per-sample weights
            
        Returns:
            loss: Spectral consistency loss
        """
        # Ensure same shape
        if spectral_masks.shape[1] != slot_masks.shape[1]:
            # spectral_masks might be [B, K, N], transpose to [B, N, K]
            spectral_masks = spectral_masks.transpose(1, 2)
        
        # Handle dimension mismatch in K
        K_spec = spectral_masks.shape[-1]
        K_slot = slot_masks.shape[-1]
        
        if K_spec != K_slot:
            if K_spec < K_slot:
                # Pad spectral masks
                spectral_masks = F.pad(spectral_masks, (0, K_slot - K_spec))
            else:
                # Truncate spectral masks
                spectral_masks = spectral_masks[..., :K_slot]
        
        # Handle dimension mismatch in N (spatial)
        N_spec = spectral_masks.shape[1]
        N_slot = slot_masks.shape[1]
        
        if N_spec != N_slot:
            # Interpolate slot_masks
            B, N, K = slot_masks.shape
            H = int(N ** 0.5)
            
            if H * H == N:  # Can reshape to square
                slot_masks = slot_masks.view(B, H, H, K).permute(0, 3, 1, 2)
                H_target = int(N_spec ** 0.5)
                slot_masks = F.interpolate(slot_masks, size=(H_target, H_target), mode='bilinear')
                slot_masks = slot_masks.permute(0, 2, 3, 1).view(B, -1, K)
        
        # Normalize masks if requested
        if self.normalize_masks:
            spectral_masks = F.normalize(spectral_masks, p=2, dim=1)
            slot_masks = F.normalize(slot_masks, p=2, dim=1)
        
        # Frobenius norm loss
        diff = spectral_masks - slot_masks
        loss = (diff ** 2).sum(dim=[1, 2])  # [B]
        
        # Apply weights
        if weights is not None:
            loss = loss * weights
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SpectralAlignmentLoss(nn.Module):
    """
    Spectral alignment loss using cosine similarity.
    
    Alternative to Frobenius norm, more robust to scale differences.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        temperature: float = 0.1,
    ):
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
    
    def forward(
        self,
        spectral_masks: torch.Tensor,
        slot_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alignment loss.
        
        Maximizes cosine similarity between spectral and slot masks.
        """
        # Normalize
        spectral_norm = F.normalize(spectral_masks, p=2, dim=1)
        slot_norm = F.normalize(slot_masks, p=2, dim=1)
        
        # Cosine similarity per slot
        sim = (spectral_norm * slot_norm).sum(dim=1)  # [B, K]
        
        # Convert to loss (1 - similarity)
        loss = 1 - sim.mean(dim=1)  # [B]
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
