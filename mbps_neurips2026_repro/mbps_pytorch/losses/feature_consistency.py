"""Multiscale feature consistency loss inspired by Uni-UVPT (NeurIPS 2023).

Ported from: https://github.com/huawei-noah/noah-research/tree/master/uni-uvpt
Reference: "Universal Unsupervised Visual Prompt Tuning for Source-Free Domain
            Adaptive Semantic Segmentation"

Aligns features at different spatial resolutions to enforce consistency
across scales during adapter training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureConsistencyLoss(nn.Module):
    """Multiscale feature consistency loss.

    Encourages features extracted at different resolutions to be consistent
    after projection to a common embedding space.
    """

    def __init__(self, feature_dim: int = 768, num_scales: int = 3, temperature: float = 0.07):
        super().__init__()
        self.num_scales = num_scales
        self.temperature = temperature

        # Projection heads for each scale
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, feature_dim // 4),
            )
            for _ in range(num_scales)
        ])

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss across multiple feature scales.

        Args:
            features_list: List of (B, N_i, C) feature tensors at different scales.

        Returns:
            Scalar consistency loss.
        """
        if len(features_list) < 2:
            return torch.tensor(0.0, device=features_list[0].device)

        # Project each scale
        projected = []
        for feat, proj in zip(features_list, self.projectors):
            # Global average pooling over spatial tokens
            if feat.dim() == 3:
                pooled = feat.mean(dim=1)  # (B, C)
            else:
                pooled = feat
            projected.append(proj(pooled))  # (B, C/4)

        # Compute pairwise consistency via cosine similarity
        loss = 0.0
        count = 0
        for i in range(len(projected)):
            for j in range(i + 1, len(projected)):
                sim = F.cosine_similarity(projected[i], projected[j], dim=-1)
                loss = loss + (1.0 - sim).mean()
                count += 1

        return loss / max(count, 1)


class PredictionConsistencyLoss(nn.Module):
    """Prediction consistency across augmented views / scales.

    Ensures that predictions are stable under input perturbations.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute prediction consistency between two outputs.

        Args:
            pred1: (B, C, H, W) or (B, H, W) first prediction.
            pred2: Same shape as pred1, second prediction (e.g., from augmented view).
            mask: Optional (B, H, W) mask for valid regions.

        Returns:
            Scalar consistency loss.
        """
        if pred1.dim() == 4 and pred2.dim() == 4:
            # Segmentation logits: KL divergence between softmax predictions
            p1 = F.softmax(pred1, dim=1)
            p2 = F.softmax(pred2, dim=1)
            # KL(p1 || p2) + KL(p2 || p1) = symmetric Jensen-Shannon
            log_p1 = F.log_softmax(pred1, dim=1)
            log_p2 = F.log_softmax(pred2, dim=1)
            kl_12 = F.kl_div(log_p2, p1, reduction="none").sum(dim=1)
            kl_21 = F.kl_div(log_p1, p2, reduction="none").sum(dim=1)
            loss = 0.5 * (kl_12 + kl_21)
        else:
            # Regression / dense prediction: MSE
            loss = F.mse_loss(pred1, pred2, reduction="none")

        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)

        return loss.mean() if self.reduction == "mean" else loss.sum()
