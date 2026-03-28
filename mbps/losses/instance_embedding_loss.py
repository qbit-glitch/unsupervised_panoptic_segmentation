"""Discriminative instance embedding loss for MBPS v2.

Implements the discriminative loss from de Brabandere et al. (2017):
  "Semantic Instance Segmentation with a Discriminative Loss Function"

For each batch element:
  L_pull: Pull same-instance embeddings toward their mean
  L_push: Push different-instance means apart
  L_reg: Small L2 regularization on means

L = alpha * L_pull + beta * L_push + gamma * L_reg
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def discriminative_loss(
    embeddings: jnp.ndarray,
    instance_labels: jnp.ndarray,
    max_instances: int = 20,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.001,
) -> Dict[str, jnp.ndarray]:
    """Discriminative loss for per-token instance embeddings.

    Args:
        embeddings: (B, N, D) per-token instance embeddings.
        instance_labels: (B, N) integer instance IDs (0=background, 1+=instances).
        max_instances: Maximum number of instances to handle.
        delta_v: Pull margin (embeddings within delta_v of mean are not penalized).
        delta_d: Push margin (means within 2*delta_d are penalized).
        alpha: Pull loss weight.
        beta: Push loss weight.
        gamma: Regularization weight.

    Returns:
        Dict with:
            - total: scalar total loss
            - pull: scalar pull loss
            - push: scalar push loss
            - reg: scalar regularization loss
    """
    # Vectorize over batch
    pull, push, reg = jax.vmap(
        _single_discriminative_loss,
        in_axes=(0, 0, None, None, None),
    )(embeddings, instance_labels, max_instances, delta_v, delta_d)

    pull_loss = jnp.mean(pull)
    push_loss = jnp.mean(push)
    reg_loss = jnp.mean(reg)

    total = alpha * pull_loss + beta * push_loss + gamma * reg_loss

    return {
        "total": total,
        "pull": pull_loss,
        "push": push_loss,
        "reg": reg_loss,
    }


def _single_discriminative_loss(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    max_instances: int,
    delta_v: float,
    delta_d: float,
) -> tuple:
    """Discriminative loss for a single image.

    Args:
        embeddings: (N, D) per-token embeddings.
        labels: (N,) integer instance IDs.
        max_instances: Max instances.
        delta_v: Pull margin.
        delta_d: Push margin.

    Returns:
        (pull_loss, push_loss, reg_loss) scalars.
    """
    N, D = embeddings.shape

    # Compute instance means
    # For each instance i in [1, max_instances], compute mean embedding
    means = jnp.zeros((max_instances, D))
    counts = jnp.zeros(max_instances)

    for i in range(max_instances):
        inst_id = i + 1  # Instance IDs start at 1
        mask = (labels == inst_id)  # (N,)
        count = mask.sum()
        # Safe mean: sum / max(count, 1)
        mean = jnp.where(
            mask[:, None],
            embeddings,
            0.0,
        ).sum(axis=0) / jnp.maximum(count, 1.0)
        means = means.at[i].set(mean)
        counts = counts.at[i].set(count)

    valid = counts > 0  # (max_instances,)
    n_instances = jnp.maximum(valid.sum(), 1.0)

    # Pull loss: for each instance, pull embeddings toward mean
    pull_loss = jnp.array(0.0)
    for i in range(max_instances):
        inst_id = i + 1
        mask = (labels == inst_id)  # (N,)
        count = jnp.maximum(counts[i], 1.0)

        # Distance from mean
        diffs = jnp.linalg.norm(embeddings - means[i:i+1], axis=-1)  # (N,)
        # Hinge: max(0, ||e - mu|| - delta_v)^2
        pull_per_token = jnp.maximum(diffs - delta_v, 0.0) ** 2
        pull_per_token = jnp.where(mask, pull_per_token, 0.0)
        pull_loss = pull_loss + jnp.where(
            valid[i], pull_per_token.sum() / count, 0.0
        )

    pull_loss = pull_loss / n_instances

    # Push loss: for each pair of instances, push means apart
    push_loss = jnp.array(0.0)
    n_pairs = jnp.array(0.0)
    for i in range(max_instances):
        for j in range(i + 1, max_instances):
            both_valid = valid[i] & valid[j]
            dist = jnp.linalg.norm(means[i] - means[j])
            # Hinge: max(0, 2*delta_d - ||mu_i - mu_j||)^2
            push_pair = jnp.maximum(2 * delta_d - dist, 0.0) ** 2
            push_loss = push_loss + jnp.where(both_valid, push_pair, 0.0)
            n_pairs = n_pairs + jnp.where(both_valid, 1.0, 0.0)

    push_loss = push_loss / jnp.maximum(n_pairs, 1.0)

    # Regularization: L2 norm of means
    reg_loss = jnp.where(valid[:, None], means, 0.0)
    reg_loss = jnp.sum(jnp.linalg.norm(reg_loss, axis=-1)) / n_instances

    return pull_loss, push_loss, reg_loss
