"""Stuff-Things MLP Classifier.

Takes DBD, FCC, IDF cue features and classifies each semantic cluster
as either 'stuff' (<=0.5) or 'thing' (>0.5).

Architecture: MLP [3 -> 16 -> 8 -> 1] with sigmoid output.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StuffThingsClassifier(nn.Module):
    """MLP classifier for stuff vs. things discrimination.

    Takes three cue features per cluster (DBD, FCC, IDF) and outputs
    a stuff-things probability.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        threshold: Classification threshold (>threshold = thing).
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (16, 8),
        threshold: float = 0.5,
        input_dim: int = 3,
    ) -> None:
        """Initialize StuffThingsClassifier.

        Args:
            hidden_dims: Hidden layer dimensions.
            threshold: Classification threshold (>threshold = thing).
            input_dim: Input dimension (3 for DBD, FCC, IDF cues).
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.threshold = threshold

        # Build layers
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            in_dim = dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output: 1 logit per cluster
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(
        self,
        cues: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Classify clusters as stuff or things.

        Args:
            cues: Concatenated cue features of shape (B, K, 3).
                  [DBD, FCC, IDF] per cluster.
            deterministic: If True, disable dropout.

        Returns:
            Stuff-things scores of shape (B, K) in [0, 1].
            Score > threshold -> thing, else stuff.
        """
        if deterministic:
            self.hidden_layers.eval()
        else:
            self.hidden_layers.train()

        x = self.hidden_layers(cues)

        # Output: 1 logit per cluster
        x = self.output_layer(x)
        x = torch.sigmoid(x).squeeze(-1)  # (B, K)

        return x

    def get_stuff_thing_sets(
        self,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split clusters into stuff and thing sets.

        Args:
            scores: Classification scores of shape (B, K).

        Returns:
            Tuple of:
                - thing_mask: Boolean mask (B, K) for thing clusters.
                - stuff_mask: Boolean mask (B, K) for stuff clusters.
        """
        thing_mask = scores > self.threshold
        stuff_mask = ~thing_mask
        return thing_mask, stuff_mask


class OracleStuffThings:
    """Oracle stuff-things classifier using ground truth labels.

    Used for ablation studies to evaluate upper-bound performance.
    """

    def __init__(self, thing_class_ids: list[int]):
        """Initialize oracle classifier.

        Args:
            thing_class_ids: List of class IDs that are things.
        """
        self.thing_classes = set(thing_class_ids)

    def __call__(
        self,
        cluster_labels: torch.Tensor,
        gt_semantics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify using ground truth.

        Args:
            cluster_labels: Predicted cluster assignments (B, N).
            gt_semantics: Ground truth semantic labels (B, N).

        Returns:
            Tuple of (thing_mask, stuff_mask) each of shape (B, K).
        """
        k = int(cluster_labels.max().item()) + 1
        b = cluster_labels.shape[0]

        thing_mask = torch.zeros(
            b, k, dtype=torch.bool, device=cluster_labels.device
        )

        for c in range(k):
            mask = cluster_labels == c
            if torch.any(mask):
                # Majority vote from ground truth
                for batch in range(b):
                    cluster_gt = gt_semantics[batch][mask[batch]]
                    if cluster_gt.numel() > 0:
                        majority = torch.bincount(
                            cluster_gt.to(torch.int64),
                            minlength=256,
                        ).argmax()
                        is_thing = int(majority.item()) in self.thing_classes
                        thing_mask[batch, c] = is_thing

        return thing_mask, ~thing_mask
