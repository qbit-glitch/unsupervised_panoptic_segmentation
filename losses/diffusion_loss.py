"""
Diffusion Loss for SpectralDiffusion

DDPM training objective with noise prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiffusionLoss(nn.Module):
    """
    Diffusion training loss (noise prediction).
    
    L = E_{t, x_0, epsilon} [||epsilon - epsilon_theta(x_t, t, c)||^2]
    
    Args:
        loss_type: "l2" (MSE) or "l1" (MAE)
        reduction: "mean", "sum", or "none"
    """
    
    def __init__(
        self,
        loss_type: str = "l2",
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            noise_pred: [B, C, H, W] predicted noise
            noise_target: [B, C, H, W] actual noise
            weights: Optional [B] per-sample weights
            
        Returns:
            loss: Scalar loss value
        """
        if self.loss_type == "l2":
            loss = F.mse_loss(noise_pred, noise_target, reduction='none')
        elif self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise_target, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Reduce spatial dimensions
        loss = loss.mean(dim=[1, 2, 3])  # [B]
        
        # Apply weights
        if weights is not None:
            loss = loss * weights
        
        # Final reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class VLBLoss(nn.Module):
    """
    Variational Lower Bound loss for diffusion.
    
    Combines noise prediction loss with KL terms.
    """
    
    def __init__(
        self,
        lambda_vlb: float = 0.001,
    ):
        super().__init__()
        self.lambda_vlb = lambda_vlb
        self.mse = DiffusionLoss(loss_type="l2")
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        x_0: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VLB loss.
        
        Args:
            noise_pred: Predicted noise
            noise_target: Target noise
            x_0: Original clean sample
            x_0_pred: Predicted clean sample
            t: Timesteps
            
        Returns:
            loss: Total loss
            info: Dictionary with loss components
        """
        # Simple (noise prediction) loss
        loss_simple = self.mse(noise_pred, noise_target)
        
        # x_0 reconstruction loss (helps with edge cases)
        loss_x0 = F.mse_loss(x_0_pred, x_0)
        
        # Total loss
        loss = loss_simple + self.lambda_vlb * loss_x0
        
        info = {
            'loss_simple': loss_simple.item(),
            'loss_x0': loss_x0.item(),
        }
        
        return loss, info
