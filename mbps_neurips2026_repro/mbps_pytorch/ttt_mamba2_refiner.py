"""CRF-TTT-Mamba2: Combined CRF + Test-Time Training Pseudo-Label Refinement.

Refines semantic pseudo-labels using Mamba2 state-space models that serve
a dual purpose:
  (A) CRF role: State transition matrix A_t encodes learned pairwise
      potentials — determines how class beliefs propagate between neighbors.
  (B) TTT role: Hidden state acts as a per-image adaptive classifier that
      updates online as it scans across pixels (implicit test-time training).

Key innovations:
  1. CRF-as-SSM: Mamba2 state transition = learned CRF pairwise potential.
     Trained via pairwise feature consistency loss (differentiable CRF energy).
  2. Implicit TTT: Mamba2 state accumulates per-image statistics during scan.
  3. Explicit TTT: optional gradient-based self-supervised adaptation at test
     time using CRF energy + entropy minimization (no GT labels needed).
  4. Confidence-aware gating: high-confidence tokens influence state more.
  5. Cross-modal fusion: DINOv2 features + depth (FiLM) + semantic context.

Architecture:
    DINOv2 (768) ──→ feature_proj ──→ (bridge_dim)
                                        |
    Depth (1+2)  ──→ depth_proj  ──→ (bridge_dim)  ──→ fusion ──→ (bridge_dim)
                                        |                             |
    CAUSE (27)   ──→ context_proj ──→ (bridge_dim)       N × VisionMamba2
                   + confidence gate                     (state = CRF + TTT)
                   + context dropout                          |
                                                        head → (27)

References:
    - TTT-E2E (Sun et al., 2025): Mamba2/GDN as special cases of TTT
    - CRF-RNN (Zheng et al., ICCV 2015): Unrolled CRF as RNN
    - MFuser (CVPR 2025 Highlight): Mamba as a Bridge for segmentation
    - EMPL (MedIA 2024): Pseudo-labeling as EM algorithm
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mamba2 import VisionMamba2


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DepthFiLM(nn.Module):
    """FiLM conditioning: modulate DINOv2 features using depth geometry.

    Encodes depth via sinusoidal bands + Sobel gradients, then applies
    affine modulation (gamma, beta) to projected features.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        depth_freq_bands: int = 6,
    ):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feature_dim, bridge_dim, 1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )
        depth_input_dim = 2 * depth_freq_bands + 3  # sin/cos + raw + grads
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_input_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, bridge_dim * 2, 1),
        )
        self.register_buffer(
            "freq_bands",
            torch.tensor([2**i * math.pi for i in range(depth_freq_bands)]),
        )

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, 768, H, W)
            depth: (B, 1, H, W)
            depth_grads: (B, 2, H, W)
        Returns:
            (B, bridge_dim, H, W)
        """
        feat = self.feat_proj(features)
        d_exp = depth * self.freq_bands[None, :, None, None]
        depth_enc = torch.cat(
            [torch.sin(d_exp), torch.cos(d_exp), depth, depth_grads], dim=1
        )
        film = self.depth_encoder(depth_enc)
        gamma, beta = film.chunk(2, dim=1)
        gamma = gamma.clamp(-2.0, 2.0)
        beta = beta.clamp(-2.0, 2.0)
        return feat * (1.0 + gamma) + beta


class MambaBlock(nn.Module):
    """VisionMamba2 block with pre-norm, residual, and FFN."""

    def __init__(
        self,
        d_model: int = 192,
        scan_mode: str = "cross_scan",
        layer_type: str = "mamba2",
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = VisionMamba2(
            d_model=d_model,
            scan_mode=scan_mode,
            layer_type=layer_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape

        # Mamba2 with pre-norm
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        x_flat = x_flat + self.mamba(
            self.norm1(x_flat).reshape(B, C, H, W)
        ).permute(0, 2, 3, 1).reshape(B, H * W, C)

        # FFN with pre-norm
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        return x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# TTT-Mamba2 Refiner
# ---------------------------------------------------------------------------


class TTTMamba2Refiner(nn.Module):
    """Test-Time Training Mamba2 pseudo-label refinement module.

    The core TTT mechanism: Mamba2's hidden state IS a per-image linear
    model being updated via the SSD state transition at each scan position.
    High-confidence input tokens calibrate the state, which then refines
    low-confidence regions — this is implicit test-time training.

    Args:
        num_classes: Number of semantic classes (27 for CAUSE)
        feature_dim: DINOv2 embedding dimension (768 for ViT-B/14)
        bridge_dim: Internal feature dimension
        num_blocks: Number of VisionMamba2 blocks
        scan_mode: Spatial scan pattern ("cross_scan" | "bidirectional" | "raster")
        layer_type: SSM type ("mamba2" | "gated_delta_net")
        context_dropout: Dropout rate for input logits context (prevents shortcut)
        d_state: SSM state dimension
        d_conv: Causal conv1d kernel width
        expand: Inner dimension expansion factor
        headdim: Dimension per attention head
        chunk_size: SSD chunk size
        gradient_checkpointing: Use gradient checkpointing for memory savings
    """

    def __init__(
        self,
        num_classes: int = 27,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        num_blocks: int = 3,
        scan_mode: str = "cross_scan",
        layer_type: str = "mamba2",
        context_dropout: float = 0.2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bridge_dim = bridge_dim
        self.context_dropout = context_dropout
        self.gradient_checkpointing = gradient_checkpointing

        # --- Input projections ---
        # DINOv2 features → bridge_dim (with depth FiLM)
        self.depth_proj = DepthFiLM(feature_dim, bridge_dim)

        # DINOv2 features → bridge_dim (semantic stream, no depth)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, bridge_dim, 1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )

        # CAUSE logits → bridge_dim (context with confidence gating)
        self.context_proj = nn.Sequential(
            nn.Conv2d(num_classes, bridge_dim, 1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )

        # --- Fusion ---
        self.fusion = nn.Sequential(
            nn.Conv2d(bridge_dim * 3, bridge_dim, 1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )

        # --- Mamba2 blocks ---
        mamba_kwargs = dict(
            d_model=bridge_dim,
            scan_mode=scan_mode,
            layer_type=layer_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self.blocks = nn.ModuleList(
            [MambaBlock(**mamba_kwargs) for _ in range(num_blocks)]
        )

        # --- Output head ---
        self.head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )
        # Small init to prevent identity shortcut
        nn.init.normal_(self.head[1].weight, std=0.01)
        nn.init.zeros_(self.head[1].bias)

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
        cause_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward pass (training + inference).

        Args:
            dinov2_features: (B, 768, H, W) DINOv2 patch features
            depth: (B, 1, H, W) normalized depth [0, 1]
            depth_grads: (B, 2, H, W) Sobel gradients of depth
            cause_probs: (B, 27, H, W) CAUSE softmax probabilities [optional]

        Returns:
            refined_logits: (B, 27, H, W)
        """
        # Feature projection with depth FiLM
        feat_depth = self.depth_proj(dinov2_features, depth, depth_grads)

        # Semantic feature projection (no depth)
        feat_sem = self.feature_proj(dinov2_features)

        # Context from CAUSE predictions
        if cause_probs is not None:
            # Confidence-aware gating: scale by max class probability
            confidence = cause_probs.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
            confidence_scale = 0.5 + 0.5 * confidence  # [0.5, 1.0]

            # Context dropout during training (prevents identity shortcut)
            if self.training and self.context_dropout > 0:
                drop_mask = torch.rand(
                    cause_probs.shape[0], 1, 1, 1,
                    device=cause_probs.device,
                ) > self.context_dropout
                ctx = self.context_proj(cause_probs) * confidence_scale
                ctx = ctx * drop_mask.float()
            else:
                ctx = self.context_proj(cause_probs) * confidence_scale
        else:
            ctx = torch.zeros_like(feat_sem)

        # Fuse all three streams
        x = self.fusion(torch.cat([feat_sem, feat_depth, ctx], dim=1))

        # Mamba2 blocks with residual connections
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return self.head(x)

    @torch.enable_grad()
    def ttt_adapt(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
        cause_probs: Optional[torch.Tensor] = None,
        ttt_steps: int = 3,
        ttt_lr: float = 0.01,
    ) -> torch.Tensor:
        """Explicit Test-Time Training adaptation.

        For each image, perform K gradient steps on self-supervised losses
        to adapt the Mamba2 block parameters, then predict with the adapted
        model. Original parameters are restored after prediction.

        Self-supervised losses (no GT labels needed):
          1. Entropy minimization — encourage confident predictions
          2. Depth-boundary alignment — prediction edges ↔ depth edges

        Args:
            dinov2_features: (B, 768, H, W)
            depth: (B, 1, H, W)
            depth_grads: (B, 2, H, W)
            cause_probs: (B, 27, H, W) optional
            ttt_steps: Number of inner optimization steps
            ttt_lr: Inner learning rate

        Returns:
            refined_logits: (B, 27, H, W) — predictions from adapted model
        """
        # Identify parameters to adapt (Mamba2 blocks only)
        adapt_params = []
        adapt_names = []
        for name, param in self.named_parameters():
            if "blocks" in name:
                adapt_params.append(param)
                adapt_names.append(name)

        # Save original parameters
        orig_data = [p.data.clone() for p in adapt_params]

        was_training = self.training
        self.train()  # Enable training mode for dropout etc.

        for step in range(ttt_steps):
            # Zero gradients
            for p in adapt_params:
                if p.grad is not None:
                    p.grad.zero_()

            # Forward pass
            logits = self.forward(
                dinov2_features, depth, depth_grads, cause_probs
            )

            # Self-supervised CRF-TTT losses
            loss = self._ttt_loss(logits, dinov2_features, depth, depth_grads)
            loss.backward()

            # SGD update on Mamba2 block parameters only
            with torch.no_grad():
                for p in adapt_params:
                    if p.grad is not None:
                        p.data -= ttt_lr * p.grad

        # Final prediction with adapted parameters
        self.eval()
        with torch.no_grad():
            refined = self.forward(
                dinov2_features, depth, depth_grads, cause_probs
            )

        # Restore original parameters
        with torch.no_grad():
            for p, orig in zip(adapt_params, orig_data):
                p.data.copy_(orig)

        if was_training:
            self.train()

        return refined

    def _ttt_loss(
        self,
        logits: torch.Tensor,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> torch.Tensor:
        """Self-supervised TTT loss using Information Maximization (no GT labels).

        Uses IM (SHOT, ICML 2020) instead of raw entropy minimization to
        prevent mode collapse toward majority classes during adaptation.

        1. Information Maximization: min H(Y|X) - max H(Y)
           = per-pixel confidence + class diversity
        2. Depth-boundary alignment: prediction edges ↔ depth edges
        """
        probs = logits.softmax(dim=1)
        log_probs = logits.log_softmax(dim=1)

        # 1. Information Maximization (replaces raw entropy)
        pixel_entropy = -(probs * log_probs).sum(dim=1).mean()
        p_bar = probs.mean(dim=(0, 2, 3))  # (C,)
        marginal_entropy = -(p_bar * (p_bar + 1e-8).log()).sum()
        im_loss = pixel_entropy - marginal_entropy

        # 2. Depth-boundary alignment
        pred_dx = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().sum(dim=1)
        pred_dy = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().sum(dim=1)
        depth_grad_mag = depth_grads.pow(2).sum(dim=1).sqrt()
        align_x = -(pred_dx * depth_grad_mag[:, :, :-1]).mean()
        align_y = -(pred_dy * depth_grad_mag[:, :-1, :]).mean()
        alignment = align_x + align_y

        return im_loss + 0.3 * alignment

    @staticmethod
    def _crf_pairwise_loss(
        probs: torch.Tensor,
        features: torch.Tensor,
        depth: torch.Tensor,
        sigma_feat: float = 0.5,
        sigma_depth: float = 0.1,
    ) -> torch.Tensor:
        """CRF-inspired pairwise consistency loss (differentiable CRF energy).

        For each pair of 4-connected neighbors (i, j):
          weight_ij = exp(-||f_i - f_j||² / 2σ_f²) * exp(-|d_i - d_j|² / 2σ_d²)
          loss += weight_ij * ||p_i - p_j||²

        Penalizes different predictions for pixels with similar DINOv2
        features and continuous depth — exactly the bilateral CRF kernel.

        Operates at patch resolution (32×64) so the 4-connected neighbors
        correspond to spatially adjacent 14×14 patches.
        """
        B, C, H, W = features.shape

        # Feature similarity (horizontal and vertical neighbors)
        # Normalize features for stable cosine-like distance
        feat_norm = F.normalize(features, dim=1)
        feat_diff_x = (feat_norm[:, :, :, 1:] - feat_norm[:, :, :, :-1]).pow(2).sum(1)
        feat_diff_y = (feat_norm[:, :, 1:, :] - feat_norm[:, :, :-1, :]).pow(2).sum(1)
        feat_weight_x = (-feat_diff_x / (2 * sigma_feat**2)).exp()  # (B, H, W-1)
        feat_weight_y = (-feat_diff_y / (2 * sigma_depth**2)).exp()  # (B, H-1, W)

        # Depth continuity
        depth_diff_x = (depth[:, 0, :, 1:] - depth[:, 0, :, :-1]).pow(2)
        depth_diff_y = (depth[:, 0, 1:, :] - depth[:, 0, :-1, :]).pow(2)
        depth_weight_x = (-depth_diff_x / (2 * sigma_depth**2)).exp()
        depth_weight_y = (-depth_diff_y / (2 * sigma_depth**2)).exp()

        # Combined bilateral weight
        weight_x = feat_weight_x * depth_weight_x  # (B, H, W-1)
        weight_y = feat_weight_y * depth_weight_y  # (B, H-1, W)

        # Prediction disagreement (L2 on probabilities)
        pred_diff_x = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).pow(2).sum(1)
        pred_diff_y = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).pow(2).sum(1)

        # Weighted disagreement = differentiable CRF energy
        crf_energy_x = (weight_x * pred_diff_x).mean()
        crf_energy_y = (weight_y * pred_diff_y).mean()

        return crf_energy_x + crf_energy_y

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        blocks = sum(
            p.numel() for n, p in self.named_parameters() if "blocks" in n
        )
        return {
            "total": total,
            "trainable": trainable,
            "mamba_blocks": blocks,
            "projections": total - blocks,
        }
