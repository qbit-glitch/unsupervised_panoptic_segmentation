"""Bridge Loss Functions.

L_bridge = L_recon + λ_cka · L_cka + λ_h · L_state

Components:
    - Reconstruction: ||S - f_inv(S_fused)||² + ||F - g_inv(F_fused)||²
    - CKA: Centered Kernel Alignment between fused streams
    - State Regularization: L2 norm of Mamba hidden states
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def reconstruction_loss(
    original: jnp.ndarray,
    reconstructed: jnp.ndarray,
) -> jnp.ndarray:
    """Compute reconstruction loss after bridge round-trip.

    L_recon = ||X - X_reconstructed||²

    Args:
        original: Original features of shape (B, N, D).
        reconstructed: Reconstructed features of shape (B, N, D).

    Returns:
        Scalar MSE reconstruction loss.
    """
    return jnp.mean((original - reconstructed) ** 2)


def cka_loss(
    features_a: jnp.ndarray,
    features_b: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Centered Kernel Alignment (CKA) loss.

    CKA measures similarity between two sets of representations.
    We minimize negative CKA to encourage cross-modal alignment.

    CKA(X, Y) = ||Y^T X||_F² / (||X^T X||_F · ||Y^T Y||_F)

    Args:
        features_a: First feature set of shape (B, N, D).
        features_b: Second feature set of shape (B, N, D).

    Returns:
        Scalar negative CKA loss (minimize to increase alignment).
    """
    def _single_cka(X, Y):
        # Center features
        X = X - jnp.mean(X, axis=0, keepdims=True)
        Y = Y - jnp.mean(Y, axis=0, keepdims=True)

        # L2-normalize to bound all matrix operations to [-1, 1] per element.
        # Without this, untrained bridge features with magnitude ~100 cause
        # Frobenius norms ~10^11 and gradient overflow in backprop.
        X = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
        Y = Y / (jnp.linalg.norm(Y, axis=-1, keepdims=True) + 1e-8)

        # Compute HSIC (now bounded since X, Y are unit-norm rows)
        XtY = X.T @ Y  # (D_a, D_b)
        XtX = X.T @ X  # (D_a, D_a)
        YtY = Y.T @ Y  # (D_b, D_b)

        hsic_xy = jnp.sum(XtY ** 2)
        hsic_xx = jnp.sum(XtX ** 2)
        hsic_yy = jnp.sum(YtY ** 2)

        # Safe sqrt to avoid NaN gradients (grad of sqrt(0) = inf)
        denom = jnp.sqrt(jnp.maximum(hsic_xx * hsic_yy, 1e-12)) + 1e-6
        return hsic_xy / denom

    # Vectorize across batch dimension
    cka_values = jax.vmap(_single_cka)(features_a, features_b)

    # Negative CKA: minimize this to maximize alignment
    return -jnp.mean(cka_values)


def state_regularization_loss(
    state_norms: jnp.ndarray,
) -> jnp.ndarray:
    """Regularize Mamba hidden state norms.

    Prevents state explosion during training.

    Args:
        state_norms: L2 norms of SSM hidden states.

    Returns:
        Scalar regularization loss.
    """
    return jnp.mean(state_norms ** 2)


class BridgeLoss:
    """Combined bridge loss.

    L_bridge = L_recon + λ_cka · L_cka + λ_state · L_state

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
        self.lambda_recon = lambda_recon
        self.lambda_cka = lambda_cka
        self.lambda_state = lambda_state

    def __call__(
        self,
        original_semantic: jnp.ndarray,
        original_features: jnp.ndarray,
        reconstructed_semantic: jnp.ndarray,
        reconstructed_features: jnp.ndarray,
        fused_semantic: jnp.ndarray,
        fused_features: jnp.ndarray,
        align_loss: jnp.ndarray,
        state_norms: jnp.ndarray = None,
    ) -> Dict[str, jnp.ndarray]:
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
            l_state = jnp.array(0.0)
        losses["state_reg"] = l_state

        # Alignment from projection bridge
        losses["align"] = align_loss

        # Total — guard each component so a single NaN doesn't kill
        # the entire bridge loss (partial gradients are better than none)
        _safe = lambda x: jnp.where(jnp.isfinite(x), x, 0.0)
        losses["total"] = (
            self.lambda_recon * _safe(l_recon)
            + _safe(align_loss)
            + self.lambda_cka * _safe(l_cka)
            + self.lambda_state * _safe(l_state)
        )

        return losses
