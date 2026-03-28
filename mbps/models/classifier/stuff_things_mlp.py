"""Stuff-Things MLP Classifier.

Takes DBD, FCC, IDF cue features and classifies each semantic cluster
as either 'stuff' (≤0.5) or 'thing' (>0.5).

Architecture: MLP [3 → 16 → 8 → 1] with sigmoid output.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class StuffThingsClassifier(nn.Module):
    """MLP classifier for stuff vs. things discrimination.

    Takes three cue features per cluster (DBD, FCC, IDF) and outputs
    a stuff-things probability.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        threshold: Classification threshold (>threshold = thing).
    """

    hidden_dims: Tuple[int, ...] = (16, 8)
    threshold: float = 0.5

    @nn.compact
    def __call__(
        self,
        cues: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Classify clusters as stuff or things.

        Args:
            cues: Concatenated cue features of shape (B, K, 3).
                  [DBD, FCC, IDF] per cluster.
            deterministic: If True, disable dropout.

        Returns:
            Stuff-things scores of shape (B, K) in [0, 1].
            Score > threshold → thing, else stuff.
        """
        x = cues

        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"fc{i}")(x)
            x = jax.nn.relu(x)
            x = nn.Dropout(rate=0.1)(x, deterministic=deterministic)

        # Output: 1 logit per cluster
        x = nn.Dense(1, name="output")(x)
        x = jax.nn.sigmoid(x).squeeze(-1)  # (B, K)

        return x

    def get_stuff_thing_sets(
        self,
        scores: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        cluster_labels: jnp.ndarray,
        gt_semantics: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Classify using ground truth.

        Args:
            cluster_labels: Predicted cluster assignments (B, N).
            gt_semantics: Ground truth semantic labels (B, N).

        Returns:
            Tuple of (thing_mask, stuff_mask) each of shape (B, K).
        """
        k = int(jnp.max(cluster_labels)) + 1
        b = cluster_labels.shape[0]

        thing_mask = jnp.zeros((b, k), dtype=jnp.bool_)

        for c in range(k):
            mask = cluster_labels == c
            if jnp.any(mask):
                # Majority vote from ground truth
                for batch in range(b):
                    cluster_gt = gt_semantics[batch][mask[batch]]
                    if cluster_gt.size > 0:
                        majority = jnp.bincount(
                            cluster_gt.astype(jnp.int32),
                            length=256,
                        ).argmax()
                        is_thing = int(majority) in self.thing_classes
                        thing_mask = thing_mask.at[batch, c].set(is_thing)

        return thing_mask, ~thing_mask
