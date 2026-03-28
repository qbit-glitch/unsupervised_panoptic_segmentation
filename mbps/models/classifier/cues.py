"""Multi-Cue Feature Computation for Stuff-Things Classification.

Computes three discriminative cues per semantic cluster:
    1. DBD (Depth Boundary Density): ratio of depth edges within cluster
    2. FCC (Feature Cluster Compactness): intra-cluster feature variance
    3. IDF (Instance Decomposition Frequency): how often cluster is split by instances
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_depth_boundary_density(
    clusters: jnp.ndarray,
    depth: jnp.ndarray,
    edge_threshold: float = 0.1,
) -> jnp.ndarray:
    """Compute Depth Boundary Density (DBD) per cluster.

    DBD = |{p ∈ cluster : ||∇D(p)|| > τ}| / |cluster|

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
    k = int(jnp.max(clusters)) + 1

    # Compute depth gradients (approximate via finite differences)
    # Since tokens are in 1D sequence, use adjacent token differences
    depth_grad = jnp.abs(jnp.diff(depth, axis=-1, prepend=depth[:, :1]))
    is_boundary = (depth_grad > edge_threshold).astype(jnp.float32)

    # Per-cluster aggregation
    dbd = jnp.zeros((b, k))
    for c in range(k):
        mask = (clusters == c).astype(jnp.float32)
        cluster_size = jnp.sum(mask, axis=-1) + 1e-8
        boundary_count = jnp.sum(mask * is_boundary, axis=-1)
        dbd = dbd.at[:, c].set(boundary_count / cluster_size)

    return dbd


def compute_feature_cluster_compactness(
    clusters: jnp.ndarray,
    features: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Feature Cluster Compactness (FCC) per cluster.

    FCC = 1 / (1 + σ²_cluster)

    Where σ² is the mean squared distance of features from cluster centroid.
    Stuff classes tend to be more compact, things more diverse.

    Args:
        clusters: Cluster assignments of shape (B, N).
        features: Feature vectors of shape (B, N, D).

    Returns:
        FCC values of shape (B, K).
    """
    b, n, d = features.shape
    k = int(jnp.max(clusters)) + 1

    fcc = jnp.zeros((b, k))

    for c in range(k):
        mask = (clusters == c).astype(jnp.float32)  # (B, N)
        mask_sum = jnp.sum(mask, axis=-1, keepdims=True) + 1e-8  # (B, 1)

        # Cluster centroid
        mask_expanded = mask[:, :, None]  # (B, N, 1)
        centroid = jnp.sum(
            features * mask_expanded, axis=1
        ) / mask_sum  # (B, D)

        # Variance
        diff = features - centroid[:, None, :]  # (B, N, D)
        sq_dist = jnp.sum(diff**2, axis=-1)  # (B, N)
        variance = jnp.sum(sq_dist * mask, axis=-1) / mask_sum.squeeze(-1)

        fcc = fcc.at[:, c].set(1.0 / (1.0 + variance))

    return fcc


def compute_instance_decomposition_frequency(
    clusters: jnp.ndarray,
    instance_masks: jnp.ndarray,
    mask_threshold: float = 0.5,
) -> jnp.ndarray:
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
    k = int(jnp.max(clusters)) + 1

    binary_masks = (jax.nn.sigmoid(instance_masks) > mask_threshold).astype(
        jnp.float32
    )  # (B, M, N)

    idf = jnp.zeros((b, k))

    for c in range(k):
        cluster_mask = (clusters == c).astype(jnp.float32)  # (B, N)
        cluster_size = jnp.sum(cluster_mask, axis=-1) + 1e-8  # (B,)

        # Count instances that overlap > 10% with this cluster
        overlap = jnp.einsum(
            "bmn,bn->bm", binary_masks, cluster_mask
        )  # (B, M)
        overlap_ratio = overlap / cluster_size[:, None]
        num_overlapping = jnp.sum(
            (overlap_ratio > 0.1).astype(jnp.float32), axis=-1
        )  # (B,)

        idf = idf.at[:, c].set(num_overlapping / cluster_size)

    return idf
