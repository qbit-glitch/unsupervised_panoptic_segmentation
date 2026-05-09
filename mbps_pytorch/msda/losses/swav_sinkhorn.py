"""SwAV-style online clustering loss with Sinkhorn balanced assignment.

Implements self-supervised clustering via learnable prototypes and
Sinkhorn-Knopp balanced soft assignments. Since the MSDA pipeline
has no data augmentation views, the original DINOv3 features serve
as the "other view" through a small projection MLP.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss

logger = logging.getLogger(__name__)


def _sinkhorn_knopp(
    scores: torch.Tensor,
    iters: int = 3,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Apply Sinkhorn-Knopp normalization for balanced soft assignments.

    Iteratively normalizes rows and columns of the exponentiated score
    matrix so each prototype gets roughly equal assignment mass.

    Args:
        scores: Unnormalized logits of shape (B*N, K).
        iters: Number of Sinkhorn iterations.
        epsilon: Softmax sharpness (lower = sharper assignments).

    Returns:
        Balanced soft assignments of shape (B*N, K), rows sum to 1.
    """
    with torch.no_grad():
        # Numerical stability: subtract max before exp
        max_scores = scores.max(dim=0, keepdim=True).values
        q = torch.exp((scores - max_scores) / epsilon).T  # (K, B*N)
        q_sum = q.sum()
        if q_sum > 0:
            q /= q_sum
        else:
            # Fallback to uniform if all-zero
            return torch.ones_like(scores) / scores.shape[1]

        k, bn = q.shape

        for _ in range(iters):
            # Row normalize: each prototype gets equal mass
            row_sum = q.sum(dim=1, keepdim=True)
            row_sum = row_sum.clamp(min=1e-8)
            q /= row_sum
            q *= bn / k

            # Column normalize: each sample sums to 1
            col_sum = q.sum(dim=0, keepdim=True)
            col_sum = col_sum.clamp(min=1e-8)
            q /= col_sum

        # Final column normalization for proper probabilities
        col_sum = q.sum(dim=0, keepdim=True).clamp(min=1e-8)
        return (q / col_sum).T  # (B*N, K)


@register_loss("swav_sinkhorn")
class SwAVSinkhornLoss(nn.Module):
    """SwAV online clustering with Sinkhorn balanced assignment.

    Uses learnable prototypes to cluster adapter features. Since
    the MSDA pipeline does not produce two augmented views, the
    original DINOv3 features are projected through a small MLP
    to serve as the "other view" for cross-prediction.

    Args:
        num_prototypes: Number of cluster prototypes (K).
        output_dim: Adapter output feature dimension.
        teacher_dim: Original DINOv3 feature dimension.
        temperature: Softmax temperature for score computation.
        sinkhorn_iters: Number of Sinkhorn normalization iterations.
        epsilon: Sinkhorn sharpness parameter.
        projection_hidden: Hidden dim for teacher projection MLP.
    """

    def __init__(
        self,
        num_prototypes: int = 200,
        output_dim: int = 128,
        teacher_dim: int = 1024,
        temperature: float = 0.1,
        sinkhorn_iters: int = 3,
        epsilon: float = 0.05,
        projection_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon

        # Learnable prototypes (L2-normalized before use)
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, output_dim)
        )
        nn.init.xavier_uniform_(self.prototypes.data)

        # Small MLP to project teacher features into adapter space
        # Serves as the "other view" in lieu of data augmentation
        self.teacher_projection = nn.Sequential(
            nn.Linear(teacher_dim, projection_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden, output_dim),
        )

    def forward(
        self,
        adapted_features: torch.Tensor,
        original_features: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SwAV cross-prediction loss.

        Args:
            adapted_features: Adapter output (B, N, D), L2-normalized.
            original_features: Original DINOv3 features (B, N_orig, D_orig).
            depth: Depth map (B, 1, H, W). Unused; kept for interface
                consistency.

        Returns:
            Scalar SwAV loss.
        """
        b, n, d = adapted_features.shape
        n_orig = original_features.shape[1]

        # L2-normalize prototypes
        protos = F.normalize(self.prototypes, dim=-1)  # (K, D)

        # -- View 1: adapted features --
        feat_flat = adapted_features.reshape(b * n, d)  # (B*N, D)
        feat_norm = F.normalize(feat_flat, dim=-1)
        scores_1 = feat_norm @ protos.T  # (B*N, K)

        # -- View 2: projected teacher features --
        teacher_proj = self.teacher_projection(original_features)  # (B, N_orig, D)
        teacher_proj = F.normalize(teacher_proj, dim=-1)

        # Align spatial resolution if needed
        if n_orig != n:
            teacher_proj = self._align_spatial(teacher_proj, target_n=n)

        teacher_flat = teacher_proj.reshape(b * n, d)  # (B*N, D)
        scores_2 = teacher_flat @ protos.T  # (B*N, K)

        # Sinkhorn codes from each view (stop gradient through codes)
        codes_1 = _sinkhorn_knopp(
            scores_1, iters=self.sinkhorn_iters, epsilon=self.epsilon
        )
        codes_2 = _sinkhorn_knopp(
            scores_2, iters=self.sinkhorn_iters, epsilon=self.epsilon
        )

        # Cross-prediction: predict view 2 codes from view 1 scores and vice versa
        loss_12 = self._cross_entropy_loss(scores_1, codes_2)
        loss_21 = self._cross_entropy_loss(scores_2, codes_1)

        return (loss_12 + loss_21) / 2.0

    def _cross_entropy_loss(
        self,
        scores: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft cross-entropy between scores and target codes.

        Args:
            scores: Raw logits (B*N, K).
            codes: Soft target assignments (B*N, K), from Sinkhorn.

        Returns:
            Scalar cross-entropy loss.
        """
        log_probs = F.log_softmax(scores / self.temperature, dim=-1)
        # -sum(Q * log(softmax(scores)))
        loss = -torch.sum(codes * log_probs, dim=-1)
        return loss.mean()

    @staticmethod
    def _align_spatial(
        features: torch.Tensor,
        target_n: int,
    ) -> torch.Tensor:
        """Align feature sequence length via interpolation.

        Args:
            features: (B, N_src, D).
            target_n: Target sequence length.

        Returns:
            Features of shape (B, target_n, D).
        """
        b, n_src, d = features.shape
        if n_src == target_n:
            return features

        # Treat as 1D sequence and interpolate
        # (B, D, N_src) -> (B, D, target_n)
        feat_1d = features.permute(0, 2, 1)
        aligned = F.interpolate(
            feat_1d, size=target_n, mode="linear", align_corners=False
        )
        return aligned.permute(0, 2, 1)
