"""Depth-aware superpixel-guided mask losses.

Reformulation of the losses from
*Unsupervised Instance Segmentation with Superpixels* (Hoang, 2025) with the
depth-aware edge-weight extension specified in
``reports/depth_aware_sgm_adapter_design.md``.

Three loss components, plus helpers:

* :func:`hard_loss`  -- BCE on superpixels fully inside / fully outside the
  coarse mask supplied by the existing depth-CC pipeline (paper Eq. 5).
* :func:`soft_loss`  -- L1 between the per-superpixel foreground probability
  and the soft label propagated along the minimum spanning tree with
  depth-aware edge weights ``w_{m,n}`` (paper Eq. 6--8).
* :func:`adaptive_self_training_loss` -- holistic-stability weighted L1
  against the average over stored checkpoints (paper Eq. 10).

Computation that requires graph algorithms (adjacency, MST, BFS path-max) runs
on the CPU via NumPy/SciPy; only the differentiable tensor sums stay on the
training device.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


@dataclass(frozen=True)
class SGMLossConfig:
    """Hyper-parameters used by :func:`compute_sgm_losses`."""

    alpha_1: float = 0.05      # colour bandwidth in delta_{k,i}
    alpha_d: float = 0.10      # depth bandwidth in delta_{k,i}
    alpha_2: float = 0.30      # MST temperature in psi_{k,l}
    lambda_d: float = 5.0      # depth weight in w_{m,n}
    lambda_f: float = 1.0      # DINO weight in w_{m,n}
    lambda_ad: float = 1.0     # weight of adaptive self-training loss
    soft_loss_target_detach: bool = True
    hard_pos_weight: Optional[float] = None  # Auto = min(#neg/#pos, 100) when None


# ---------------------------------------------------------------------------
# Per-superpixel reductions (differentiable through M_tilde)
# ---------------------------------------------------------------------------


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, k: int) -> torch.Tensor:
    """``out[k] = mean(values[index == k])`` along the leading axis."""
    out_shape = (k,) + values.shape[1:]
    sums = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    sums.index_add_(0, index, values)
    counts = torch.zeros(k, dtype=values.dtype, device=values.device)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    counts = counts.clamp(min=1.0)
    if values.dim() == 1:
        return sums / counts
    return sums / counts.unsqueeze(-1)


def compute_sp_means(
    image: torch.Tensor,
    depth: torch.Tensor,
    dino_full: torch.Tensor,
    sp_labels: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-superpixel mean colour, depth, and DINO feature.

    Args:
        image: ``(3, H, W)`` RGB in ``[0, 1]`` float.
        depth: ``(H, W)`` float depth in ``[0, 1]``.
        dino_full: ``(D, H, W)`` DINO features upsampled to image resolution.
        sp_labels: ``(H, W)`` int64 superpixel IDs in ``[0, k)``.

    Returns:
        Tuple ``(mu_c (k, 3), bar_d (k,), bar_f (k, D))``.
    """
    flat_sp = sp_labels.reshape(-1)
    mu_c = _scatter_mean(image.reshape(3, -1).T, flat_sp, k)        # (k, 3)
    bar_d = _scatter_mean(depth.reshape(-1), flat_sp, k)            # (k,)
    bar_f = _scatter_mean(dino_full.reshape(dino_full.shape[0], -1).T, flat_sp, k)  # (k, D)
    return mu_c, bar_d, bar_f


def pixel_to_sp_weight(
    image: torch.Tensor,
    depth: torch.Tensor,
    sp_labels: torch.Tensor,
    mu_c: torch.Tensor,
    bar_d: torch.Tensor,
    config: SGMLossConfig,
) -> torch.Tensor:
    """delta_{k,i} = exp(-||mu_c - C_i||^2 / alpha_1 - (bar_d - d_i)^2 / alpha_d).

    Args:
        image: ``(3, H, W)`` float in ``[0, 1]``.
        depth: ``(H, W)`` float in ``[0, 1]``.
        sp_labels: ``(H, W)`` int64 IDs in ``[0, k)``.
        mu_c: ``(k, 3)`` per-SP colour means.
        bar_d: ``(k,)`` per-SP depth means.

    Returns:
        ``(H, W)`` weight map.
    """
    mu_pix = mu_c[sp_labels]                       # (H, W, 3)
    bard_pix = bar_d[sp_labels]                    # (H, W)
    color_diff = ((image.permute(1, 2, 0) - mu_pix) ** 2).sum(dim=-1)
    depth_diff = (depth - bard_pix) ** 2
    return torch.exp(-color_diff / config.alpha_1 - depth_diff / config.alpha_d)


def superpixel_foreground_prob(
    m_tilde: torch.Tensor,
    delta: torch.Tensor,
    sp_labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """``P_k = (1 / nu_k) * sum_i M_tilde_i * delta_{k,i}``.

    Args:
        m_tilde: ``(H, W)`` per-pixel foreground probability for **one** class.
        delta: ``(H, W)`` pixel-to-SP weight.
        sp_labels: ``(H, W)`` int64 IDs in ``[0, k)``.

    Returns:
        ``(k,)`` SP foreground probability.
    """
    flat_sp = sp_labels.reshape(-1)
    weighted = (m_tilde * delta).reshape(-1)
    numer = torch.zeros(k, dtype=m_tilde.dtype, device=m_tilde.device)
    denom = torch.zeros(k, dtype=m_tilde.dtype, device=m_tilde.device)
    numer.index_add_(0, flat_sp, weighted)
    denom.index_add_(0, flat_sp, delta.reshape(-1))
    return numer / denom.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Hard loss (Eq. 5)
# ---------------------------------------------------------------------------


def sp_labels_from_coarse(
    coarse_mask: torch.Tensor,
    sp_labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-SP {-1, 0, 1} labels: ``1`` all-fg, ``0`` all-bg, ``-1`` mixed.

    Args:
        coarse_mask: ``(H, W)`` bool / {0, 1} foreground mask for one class.
        sp_labels: ``(H, W)`` int64 SP IDs.

    Returns:
        ``(k,)`` long tensor with values in ``{-1, 0, 1}``.
    """
    flat_sp = sp_labels.reshape(-1)
    flat_m = coarse_mask.reshape(-1).float()
    pos = torch.zeros(k, dtype=torch.float, device=sp_labels.device)
    total = torch.zeros(k, dtype=torch.float, device=sp_labels.device)
    pos.index_add_(0, flat_sp, flat_m)
    total.index_add_(0, flat_sp, torch.ones_like(flat_m))
    y = torch.full((k,), -1, dtype=torch.long, device=sp_labels.device)
    y[pos == 0] = 0
    y[(pos == total) & (total > 0)] = 1
    return y


def hard_loss(
    prob: torch.Tensor,
    y: torch.Tensor,
    pos_weight: Optional[float] = None,
    pos_weight_clip: float = 100.0,
) -> torch.Tensor:
    """Weighted BCE on superpixels with labels in ``{0, 1}``; ``-1`` rows skipped.

    Args:
        prob: ``(K,)`` per-SP foreground probability in ``[0, 1]``.
        y: ``(K,)`` per-SP label in ``{-1, 0, 1}``.
        pos_weight: weight applied to positive SPs. If ``None``, computed as
            ``min(#neg / #pos, pos_weight_clip)``. Default ``None``.
        pos_weight_clip: upper bound for the auto-computed weight to prevent
            an empty / very rare positive class from blowing the gradient.

    Returns:
        Scalar loss. Returns ``0`` (with grad-graph) when there are no
        labelled SPs or no positive SPs.
    """
    labeled = y >= 0
    if labeled.sum() == 0:
        return prob.sum() * 0.0
    p = prob[labeled].clamp(1e-7, 1.0 - 1e-7)
    yl = y[labeled].float()
    n_pos = int((yl > 0).sum().item())
    n_neg = int((yl == 0).sum().item())
    if n_pos == 0:
        # Image has no foreground SPs for this class — caller should have
        # skipped, but be defensive.
        return prob.sum() * 0.0
    if pos_weight is None:
        pos_weight = float(min(n_neg / max(n_pos, 1), pos_weight_clip))
    weight = torch.where(
        yl > 0,
        torch.tensor(pos_weight, device=p.device, dtype=p.dtype),
        torch.tensor(1.0, device=p.device, dtype=p.dtype),
    )
    bce = -(yl * torch.log(p) + (1.0 - yl) * torch.log(1.0 - p))
    return (weight * bce).mean()


# ---------------------------------------------------------------------------
# Soft loss (Eq. 6--8) with depth-aware MST edge weights
# ---------------------------------------------------------------------------


def superpixel_adjacency(sp_labels: torch.Tensor) -> np.ndarray:
    """4-neighbour superpixel adjacency.

    Args:
        sp_labels: ``(H, W)`` int64 SP IDs.

    Returns:
        ``(E, 2)`` int64 array of unique unordered edges with ``a < b``.
    """
    sp = sp_labels.detach().cpu().numpy()
    pairs: list[Tuple[int, int]] = []
    horiz_a = sp[:, :-1].ravel()
    horiz_b = sp[:, 1:].ravel()
    mask_h = horiz_a != horiz_b
    pairs.extend(zip(np.minimum(horiz_a[mask_h], horiz_b[mask_h]).tolist(),
                     np.maximum(horiz_a[mask_h], horiz_b[mask_h]).tolist()))
    vert_a = sp[:-1, :].ravel()
    vert_b = sp[1:, :].ravel()
    mask_v = vert_a != vert_b
    pairs.extend(zip(np.minimum(vert_a[mask_v], vert_b[mask_v]).tolist(),
                     np.maximum(vert_a[mask_v], vert_b[mask_v]).tolist()))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.array(sorted(set(pairs)), dtype=np.int64)
    return edges


def depth_aware_edge_weights(
    edges: np.ndarray,
    mu_c: torch.Tensor,
    bar_d: torch.Tensor,
    bar_f: torch.Tensor,
    config: SGMLossConfig,
) -> np.ndarray:
    """``w_{m,n} = ||mu_m - mu_n||^2 + lambda_d (d_m - d_n)^2 + lambda_f ||f_m - f_n||^2``.

    Returns NumPy because the next step (MST) is SciPy.
    """
    if edges.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    m_idx = torch.as_tensor(edges[:, 0], dtype=torch.long, device=mu_c.device)
    n_idx = torch.as_tensor(edges[:, 1], dtype=torch.long, device=mu_c.device)
    color = ((mu_c[m_idx] - mu_c[n_idx]) ** 2).sum(-1)
    depth = (bar_d[m_idx] - bar_d[n_idx]) ** 2
    feat = ((bar_f[m_idx] - bar_f[n_idx]) ** 2).sum(-1)
    w = color + config.lambda_d * depth + config.lambda_f * feat
    return w.detach().cpu().numpy().astype(np.float64)


def _mst_adjacency(
    edges: np.ndarray, weights: np.ndarray, k: int
) -> List[List[Tuple[int, float]]]:
    """Return MST as adjacency list ``adj[node] = [(neighbour, edge_weight), ...]``."""
    if edges.shape[0] == 0:
        return [[] for _ in range(k)]
    rows = edges[:, 0]
    cols = edges[:, 1]
    upper = csr_matrix((weights + 1e-12, (rows, cols)), shape=(k, k))
    mst = minimum_spanning_tree(upper).tocoo()
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(k)]
    for r, c, w in zip(mst.row, mst.col, mst.data):
        adj[int(r)].append((int(c), float(w)))
        adj[int(c)].append((int(r), float(w)))
    return adj


def _bfs_path_max(adj: List[List[Tuple[int, float]]], src: int, k: int) -> np.ndarray:
    """Path-max edge weight from ``src`` to every node in the (tree) graph.

    Unreachable nodes return ``+inf`` (which yields ``psi = 0``).
    """
    out = np.full(k, np.inf, dtype=np.float64)
    out[src] = 0.0
    queue: deque = deque([src])
    while queue:
        u = queue.popleft()
        for v, w in adj[u]:
            cand = max(out[u], w)
            if cand < out[v]:
                out[v] = cand
                queue.append(v)
    return out


def mst_soft_labels(
    P: torch.Tensor,
    edges: np.ndarray,
    weights: np.ndarray,
    config: SGMLossConfig,
) -> torch.Tensor:
    """``hat_P_k = sum_l P_l * psi_{k,l} / gamma_k`` along the MST.

    Computation is performed on CPU NumPy (MST + BFS). The gradient path through
    ``P`` is preserved when the caller does **not** detach it.
    """
    k = P.shape[0]
    if k == 0 or edges.shape[0] == 0:
        return P.detach().clone()
    adj = _mst_adjacency(edges, weights, k)
    P_np = P.detach().cpu().numpy().astype(np.float64)
    hat_np = np.zeros(k, dtype=np.float64)
    for node in range(k):
        max_w = _bfs_path_max(adj, node, k)
        psi = np.exp(-max_w / config.alpha_2)
        gamma = psi.sum()
        if gamma < 1e-12:
            hat_np[node] = P_np[node]
        else:
            hat_np[node] = float((P_np * psi).sum() / gamma)
    return torch.from_numpy(hat_np).to(device=P.device, dtype=P.dtype)


def soft_loss(P: torch.Tensor, hat_P: torch.Tensor, detach_target: bool) -> torch.Tensor:
    target = hat_P.detach() if detach_target else hat_P
    return (P - target).abs().mean()


# ---------------------------------------------------------------------------
# Adaptive self-training loss (Eq. 10)
# ---------------------------------------------------------------------------


def adaptive_self_training_loss(
    m_tilde: torch.Tensor,
    checkpoints: Sequence[torch.Tensor],
    threshold: float = 0.5,
) -> torch.Tensor:
    """Holistic-stability weighted L1 toward checkpoint average.

    Args:
        m_tilde: ``(n_thing, H, W)`` current foreground probabilities.
        checkpoints: list of ``(n_thing, H, W)`` predictions from previous
            saved checkpoints (no gradient).
        threshold: binarisation threshold for IoU computation.

    Returns:
        Scalar loss. Returns zero (with grad-graph) when no checkpoints.
    """
    if len(checkpoints) == 0:
        return m_tilde.sum() * 0.0
    curr_bin = (m_tilde > threshold).float()
    iou_sum = m_tilde.new_zeros(())
    target = torch.zeros_like(m_tilde)
    for ckpt in checkpoints:
        ckpt_d = ckpt.detach().to(device=m_tilde.device, dtype=m_tilde.dtype)
        ckpt_bin = (ckpt_d > threshold).float()
        inter = (curr_bin * ckpt_bin).sum()
        union = (curr_bin + ckpt_bin - curr_bin * ckpt_bin).sum().clamp(min=1e-7)
        iou_sum = iou_sum + inter / union
        target = target + ckpt_d
    z = (iou_sum / len(checkpoints)).clamp(0.0, 1.0)
    target = target / len(checkpoints)
    return z * (m_tilde - target).abs().mean()


# ---------------------------------------------------------------------------
# Driver: assemble L_hard + L_soft + lambda_ad * L_ad for one image
# ---------------------------------------------------------------------------


def compute_sgm_losses(
    m_tilde: torch.Tensor,
    coarse_mask: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    dino_full: torch.Tensor,
    sp_labels: torch.Tensor,
    checkpoints: Sequence[torch.Tensor] | None = None,
    config: SGMLossConfig = SGMLossConfig(),
) -> dict:
    """One-image SGM loss bundle, summed over thing classes.

    Args:
        m_tilde: ``(n_thing, H, W)`` predicted foreground probabilities.
        coarse_mask: ``(n_thing, H, W)`` binary supervision from the
            existing depth-CC pipeline.
        image: ``(3, H, W)`` float RGB in ``[0, 1]``.
        depth: ``(H, W)`` float depth in ``[0, 1]``.
        dino_full: ``(D, H, W)`` DINO features upsampled to image resolution.
        sp_labels: ``(H, W)`` int64 superpixel IDs in ``[0, k)``.
        checkpoints: optional sequence of prior ``(n_thing, H, W)`` predictions.

    Returns:
        Dict ``{'loss', 'L_hard', 'L_soft', 'L_ad'}`` of scalar tensors.
    """
    n_thing = m_tilde.shape[0]
    k = int(sp_labels.max().item()) + 1

    mu_c, bar_d, bar_f = compute_sp_means(image, depth, dino_full, sp_labels.long(), k)
    delta = pixel_to_sp_weight(image, depth, sp_labels.long(), mu_c, bar_d, config)
    edges = superpixel_adjacency(sp_labels)
    weights = depth_aware_edge_weights(edges, mu_c, bar_d, bar_f, config)

    l_hard_total = m_tilde.new_zeros(())
    l_soft_total = m_tilde.new_zeros(())
    n_active = 0

    for c in range(n_thing):
        # Fix (2): skip classes with no positive coarse pixels in this image
        # to avoid pulling m_tilde[c] toward the trivial all-zero solution.
        if int(coarse_mask[c].sum().item()) == 0:
            continue
        P = superpixel_foreground_prob(m_tilde[c], delta, sp_labels.long(), k)
        y = sp_labels_from_coarse(coarse_mask[c], sp_labels.long(), k)
        l_hard_total = l_hard_total + hard_loss(P, y, pos_weight=config.hard_pos_weight)
        hat_P = mst_soft_labels(P, edges, weights, config)
        l_soft_total = l_soft_total + soft_loss(P, hat_P, config.soft_loss_target_detach)
        n_active += 1

    denom = max(n_active, 1)
    l_hard_total = l_hard_total / denom
    l_soft_total = l_soft_total / denom
    l_ad = adaptive_self_training_loss(m_tilde, checkpoints or [])
    total = l_hard_total + l_soft_total + config.lambda_ad * l_ad
    return {
        "loss": total,
        "L_hard": l_hard_total.detach(),
        "L_soft": l_soft_total.detach(),
        "L_ad": l_ad.detach() if isinstance(l_ad, torch.Tensor) else l_ad,
    }


__all__ = [
    "SGMLossConfig",
    "compute_sp_means",
    "pixel_to_sp_weight",
    "superpixel_foreground_prob",
    "sp_labels_from_coarse",
    "hard_loss",
    "superpixel_adjacency",
    "depth_aware_edge_weights",
    "mst_soft_labels",
    "soft_loss",
    "adaptive_self_training_loss",
    "compute_sgm_losses",
]
