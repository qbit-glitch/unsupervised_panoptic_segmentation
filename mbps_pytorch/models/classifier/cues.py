"""Multi-Cue Feature Computation for Stuff-Things Classification.

Computes three discriminative cues per semantic cluster:
    1. DBD (Depth Boundary Density): ratio of depth edges within cluster
    2. FCC (Feature Cluster Compactness): intra-cluster feature variance
    3. IDF (Instance Decomposition Frequency): how often cluster is split by instances
"""

from __future__ import annotations

import torch


def compute_depth_boundary_density(
    clusters: torch.Tensor,
    depth: torch.Tensor,
    edge_threshold: float = 0.1,
) -> torch.Tensor:
    """Compute Depth Boundary Density (DBD) per cluster.

    DBD = |{p in cluster : ||nabla D(p)|| > tau}| / |cluster|

    Things have more internal depth boundaries (3D object surfaces),
    stuff has smoother depth (sky, road, grass).

    Args:
        clusters: Cluster assignments of shape (B, N). Values in [0, K).
        depth: Depth values of shape (B, N).
        edge_threshold: Threshold for depth gradient magnitude.

    Returns:
        DBD values of shape (B, K) where K = max(clusters) + 1.
    """
    b, n = clusters.shape
    k = int(clusters.max().item()) + 1

    # Compute depth gradients (approximate via finite differences)
    # Since tokens are in 1D sequence, use adjacent token differences
    depth_padded = torch.cat([depth[:, :1], depth], dim=-1)
    depth_grad = torch.abs(depth_padded[:, 1:] - depth_padded[:, :-1])
    is_boundary = (depth_grad > edge_threshold).float()

    # Per-cluster aggregation
    dbd = torch.zeros(b, k, device=clusters.device, dtype=depth.dtype)
    for c in range(k):
        mask = (clusters == c).float()
        cluster_size = torch.sum(mask, dim=-1) + 1e-8
        boundary_count = torch.sum(mask * is_boundary, dim=-1)
        dbd[:, c] = boundary_count / cluster_size

    return dbd


def compute_feature_cluster_compactness(
    clusters: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Compute Feature Cluster Compactness (FCC) per cluster.

    FCC = 1 / (1 + sigma^2_cluster)

    Where sigma^2 is the mean squared distance of features from cluster centroid.
    Stuff classes tend to be more compact, things more diverse.

    Args:
        clusters: Cluster assignments of shape (B, N).
        features: Feature vectors of shape (B, N, D).

    Returns:
        FCC values of shape (B, K).
    """
    b, n, d = features.shape
    k = int(clusters.max().item()) + 1

    fcc = torch.zeros(b, k, device=clusters.device, dtype=features.dtype)

    for c in range(k):
        mask = (clusters == c).float()  # (B, N)
        mask_sum = torch.sum(mask, dim=-1, keepdim=True) + 1e-8  # (B, 1)

        # Cluster centroid
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
        centroid = torch.sum(
            features * mask_expanded, dim=1
        ) / mask_sum  # (B, D)

        # Variance
        diff = features - centroid.unsqueeze(1)  # (B, N, D)
        sq_dist = torch.sum(diff ** 2, dim=-1)  # (B, N)
        variance = torch.sum(sq_dist * mask, dim=-1) / mask_sum.squeeze(-1)

        fcc[:, c] = 1.0 / (1.0 + variance)

    return fcc


def compute_instance_decomposition_frequency(
    clusters: torch.Tensor,
    instance_masks: torch.Tensor,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute Instance Decomposition Frequency (IDF) per cluster.

    IDF = number of instance masks that significantly overlap with cluster / cluster_size

    Things are decomposed into multiple instances, stuff remains whole.

    Args:
        clusters: Cluster assignments of shape (B, N).
        instance_masks: Instance mask logits of shape (B, M, N).
        mask_threshold: Threshold for binary masks.

    Returns:
        IDF values of shape (B, K).
    """
    b, n = clusters.shape
    m = instance_masks.shape[1]
    k = int(clusters.max().item()) + 1

    binary_masks = (torch.sigmoid(instance_masks) > mask_threshold).float()  # (B, M, N)

    idf = torch.zeros(b, k, device=clusters.device, dtype=instance_masks.dtype)

    for c in range(k):
        cluster_mask = (clusters == c).float()  # (B, N)
        cluster_size = torch.sum(cluster_mask, dim=-1) + 1e-8  # (B,)

        # Count instances that overlap > 10% with this cluster
        overlap = torch.einsum(
            "bmn,bn->bm", binary_masks, cluster_mask
        )  # (B, M)
        overlap_ratio = overlap / cluster_size.unsqueeze(-1)
        num_overlapping = torch.sum(
            (overlap_ratio > 0.1).float(), dim=-1
        )  # (B,)

        idf[:, c] = num_overlapping / cluster_size

    return idf
