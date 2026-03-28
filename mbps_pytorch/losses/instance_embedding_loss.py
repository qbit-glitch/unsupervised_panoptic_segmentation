"""Discriminative Instance Embedding Loss (de Brabandere et al., 2017).

Trains per-pixel embeddings so that:
  - Embeddings of same-instance pixels are pulled together (pull loss)
  - Mean embeddings of different instances are pushed apart (push loss)
  - Embedding magnitudes are regularized (reg loss)

Used in v2 training with pseudo-instance labels from MaskCut.
"""

from __future__ import annotations

from typing import Dict

import torch


def discriminative_loss_single(
    embeddings: torch.Tensor,
    instance_labels: torch.Tensor,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.001,
) -> Dict[str, torch.Tensor]:
    """Discriminative loss for a single image.

    Args:
        embeddings: (N, D) per-pixel embeddings.
        instance_labels: (N,) integer instance labels. 0 = background (ignored).
        delta_v: Pull margin.
        delta_d: Push margin.
        alpha: Pull loss weight.
        beta: Push loss weight.
        gamma: Regularization weight.

    Returns:
        Dict with pull, push, reg, total losses.
    """
    device = embeddings.device
    zero = torch.tensor(0.0, device=device)

    # Get unique instance IDs (skip 0 = background)
    unique_ids = torch.unique(instance_labels)
    unique_ids = unique_ids[unique_ids != 0]
    num_instances = len(unique_ids)

    if num_instances == 0:
        return {"pull": zero, "push": zero, "reg": zero, "total": zero}

    # Compute mean embedding per instance
    means = []
    pull_loss = zero
    for inst_id in unique_ids:
        mask = instance_labels == inst_id
        inst_embeds = embeddings[mask]  # (M_i, D)
        mean = inst_embeds.mean(dim=0)  # (D,)
        means.append(mean)

        # Pull: max(0, ||e - mu|| - delta_v)^2
        dists = torch.norm(inst_embeds - mean.unsqueeze(0), dim=1)
        pull = torch.clamp(dists - delta_v, min=0.0) ** 2
        pull_loss = pull_loss + pull.mean()

    pull_loss = pull_loss / num_instances

    # Push: max(0, 2*delta_d - ||mu_i - mu_j||)^2
    push_loss = zero
    if num_instances > 1:
        means_tensor = torch.stack(means, dim=0)  # (C, D)
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                dist = torch.norm(means_tensor[i] - means_tensor[j])
                push = torch.clamp(2.0 * delta_d - dist, min=0.0) ** 2
                push_loss = push_loss + push
        num_pairs = num_instances * (num_instances - 1) / 2
        push_loss = push_loss / num_pairs

    # Regularization: ||mu||
    reg_loss = zero
    for mean in means:
        reg_loss = reg_loss + torch.norm(mean)
    reg_loss = reg_loss / num_instances

    total = alpha * pull_loss + beta * push_loss + gamma * reg_loss

    return {
        "pull": pull_loss,
        "push": push_loss,
        "reg": reg_loss,
        "total": total,
    }


def discriminative_loss(
    embeddings: torch.Tensor,
    instance_labels: torch.Tensor,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.001,
) -> Dict[str, torch.Tensor]:
    """Batched discriminative loss.

    Args:
        embeddings: (B, N, D) per-pixel embeddings.
        instance_labels: (B, N) integer instance labels. 0 = background.
        delta_v: Pull margin.
        delta_d: Push margin.
        alpha: Pull loss weight.
        beta: Push loss weight.
        gamma: Regularization weight.

    Returns:
        Dict with averaged pull, push, reg, total losses.
    """
    batch_size = embeddings.shape[0]
    device = embeddings.device
    zero = torch.tensor(0.0, device=device)

    total_pull = zero
    total_push = zero
    total_reg = zero
    total_loss = zero

    for i in range(batch_size):
        losses = discriminative_loss_single(
            embeddings[i], instance_labels[i],
            delta_v=delta_v, delta_d=delta_d,
            alpha=alpha, beta=beta, gamma=gamma,
        )
        total_pull = total_pull + losses["pull"]
        total_push = total_push + losses["push"]
        total_reg = total_reg + losses["reg"]
        total_loss = total_loss + losses["total"]

    return {
        "pull": total_pull / batch_size,
        "push": total_push / batch_size,
        "reg": total_reg / batch_size,
        "total": total_loss / batch_size,
    }
