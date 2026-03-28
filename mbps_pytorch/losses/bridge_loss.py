"""Bridge Loss Functions.

L_bridge = L_recon + lambda_cka * L_cka + lambda_h * L_state

Components:
    - Reconstruction: ||S - f_inv(S_fused)||^2 + ||F - g_inv(F_fused)||^2
    - CKA: Centered Kernel Alignment between fused streams
    - State Regularization: L2 norm of Mamba hidden states
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def reconstruction_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """Compute reconstruction loss after bridge round-trip.

    L_recon = ||X - X_reconstructed||^2

    Args:
        original: Original features of shape (B, N, D).
        reconstructed: Reconstructed features of shape (B, N, D).

    Returns:
        Scalar MSE reconstruction loss.
    """
    return torch.mean((original - reconstructed) ** 2)


def cka_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> torch.Tensor:
    """Compute Centered Kernel Alignment (CKA) loss.

    CKA measures similarity between two sets of representations.
    We minimize negative CKA to encourage cross-modal alignment.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Args:
        features_a: First feature set of shape (B, N, D).
        features_b: Second feature set of shape (B, N, D).

    Returns:
        Scalar negative CKA loss (minimize to increase alignment).
    """
    b = features_a.shape[0]
    total_cka = 0.0

    for i in range(b):
        X = features_a[i]  # (N, D_a)
        Y = features_b[i]  # (N, D_b)

        # Center features
        X = X - torch.mean(X, dim=0, keepdim=True)
        Y = Y - torch.mean(Y, dim=0, keepdim=True)

        # Compute HSIC
        XtY = X.T @ Y  # (D_a, D_b)
        XtX = X.T @ X  # (D_a, D_a)
        YtY = Y.T @ Y  # (D_b, D_b)

        hsic_xy = torch.sum(XtY ** 2)
        hsic_xx = torch.sum(XtX ** 2)
        hsic_yy = torch.sum(YtY ** 2)

        # Safe sqrt to avoid NaN gradients (grad of sqrt(0) = inf)
        denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=1e-12)) + 1e-6
        cka = hsic_xy / denom
        total_cka = total_cka + cka

    # Negative CKA: minimize this to maximize alignment
    return -total_cka / b


def state_regularization_loss(
    state_norms: torch.Tensor,
) -> torch.Tensor:
    """Regularize Mamba hidden state norms.

    Prevents state explosion during training.

    Args:
        state_norms: L2 norms of SSM hidden states.

    Returns:
        Scalar regularization loss.
    """
    return torch.mean(state_norms ** 2)


class BridgeLoss(nn.Module):
    """Combined bridge loss.

    L_bridge = L_recon + lambda_cka * L_cka + lambda_state * L_state

    Args:
        lambda_recon: Weight for reconstruction loss.
        lambda_cka: Weight for CKA alignment loss.
        lambda_state: Weight for state regularization.
    """

    def __init__(
        self,
        lambda_recon: float = 0.5,
        lambda_cka: float = 0.1,
        lambda_state: float = 0.01,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_cka = lambda_cka
        self.lambda_state = lambda_state

    def forward(
        self,
        original_semantic: torch.Tensor,
        original_features: torch.Tensor,
        reconstructed_semantic: torch.Tensor,
        reconstructed_features: torch.Tensor,
        fused_semantic: torch.Tensor,
        fused_features: torch.Tensor,
        align_loss: torch.Tensor,
        state_norms: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined bridge loss.

        Args:
            original_semantic: Original semantic codes (B, N, D_s).
            original_features: Original DINO features (B, N, D_f).
            reconstructed_semantic: Reconstructed semantic (B, N, D_s).
            reconstructed_features: Reconstructed features (B, N, D_f).
            fused_semantic: Fused semantic (B, N, D_b).
            fused_features: Fused features (B, N, D_b).
            align_loss: Alignment regularization from projection.
            state_norms: SSM state norms for regularization.

        Returns:
            Dict with loss components and total.
        """
        losses = {}

        # Reconstruction losses
        l_recon_sem = reconstruction_loss(original_semantic, reconstructed_semantic)
        l_recon_feat = reconstruction_loss(original_features, reconstructed_features)
        l_recon = l_recon_sem + l_recon_feat
        losses["recon_semantic"] = l_recon_sem
        losses["recon_features"] = l_recon_feat
        losses["recon"] = l_recon

        # CKA alignment
        l_cka = cka_loss(fused_semantic, fused_features)
        losses["cka"] = l_cka

        # State regularization
        if state_norms is not None:
            l_state = state_regularization_loss(state_norms)
        else:
            l_state = torch.tensor(0.0, device=original_semantic.device)
        losses["state_reg"] = l_state

        # Alignment from projection bridge
        losses["align"] = align_loss

        # Total
        losses["total"] = (
            self.lambda_recon * l_recon
            + align_loss
            + self.lambda_cka * l_cka
            + self.lambda_state * l_state
        )

        return losses
