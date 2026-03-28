"""Instance mask extraction from per-token embeddings via clustering.

At inference time, the instance embedding head produces per-token embeddings.
This module clusters them into instance masks using:
  1. Cosine similarity thresholding
  2. Connected component labeling
  3. Size filtering

Used during evaluation, NOT during training.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class InstanceResult(NamedTuple):
    """Result of instance clustering."""
    instance_map: jnp.ndarray    # (H_patch, W_patch) int32, instance IDs (0=bg)
    num_instances: int            # Number of discovered instances
    instance_scores: jnp.ndarray  # (max_instances,) confidence per instance


def cluster_embeddings(
    embeddings: jnp.ndarray,
    h_patches: int,
    w_patches: int,
    similarity_threshold: float = 0.7,
    min_patch_count: int = 4,
    max_instances: int = 50,
) -> InstanceResult:
    """Cluster per-token embeddings into instance masks.

    Algorithm:
      1. Compute pairwise cosine similarity between all tokens
      2. Build adjacency: sim > threshold AND spatially adjacent
      3. Greedy flood-fill to extract connected components
      4. Filter by minimum size

    Args:
        embeddings: (N, D) per-token instance embeddings.
        h_patches: Number of patches in height.
        w_patches: Number of patches in width.
        similarity_threshold: Cosine similarity threshold for grouping.
        min_patch_count: Minimum patches for a valid instance.
        max_instances: Maximum instances to return.

    Returns:
        InstanceResult with instance map and metadata.
    """
    N, D = embeddings.shape
    assert N == h_patches * w_patches

    # L2 normalize embeddings
    norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)

    # Compute cosine similarity matrix
    sim = emb_norm @ emb_norm.T  # (N, N)

    # Build spatial adjacency mask (4-connected)
    spatial_adj = _build_spatial_adjacency(h_patches, w_patches)  # (N, N)

    # Combined adjacency: similar AND spatially connected
    adj = (sim > similarity_threshold) & spatial_adj

    # Greedy connected components via BFS
    instance_map = jnp.zeros(N, dtype=jnp.int32)
    visited = jnp.zeros(N, dtype=bool)
    instance_id = 0
    scores = jnp.zeros(max_instances)

    # Note: This is not jit-compatible due to dynamic control flow.
    # For evaluation only (not training).
    instance_map_np = _connected_components_numpy(
        adj, h_patches, w_patches, min_patch_count, max_instances
    )

    return InstanceResult(
        instance_map=jnp.array(instance_map_np.reshape(h_patches, w_patches)),
        num_instances=int((instance_map_np > 0).max()) if instance_map_np.max() > 0 else 0,
        instance_scores=jnp.ones(max_instances),  # Placeholder
    )


def _build_spatial_adjacency(h: int, w: int) -> jnp.ndarray:
    """Build 4-connected spatial adjacency matrix.

    Args:
        h: Height in patches.
        w: Width in patches.

    Returns:
        (N, N) boolean adjacency matrix.
    """
    import numpy as np

    N = h * w
    adj = np.zeros((N, N), dtype=bool)

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            # Right neighbor
            if x + 1 < w:
                adj[idx, idx + 1] = True
                adj[idx + 1, idx] = True
            # Bottom neighbor
            if y + 1 < h:
                adj[idx, idx + w] = True
                adj[idx + w, idx] = True

    return jnp.array(adj)


def _connected_components_numpy(
    adj: jnp.ndarray,
    h: int,
    w: int,
    min_size: int,
    max_instances: int,
) -> "np.ndarray":
    """Extract connected components using numpy BFS.

    Not jit-compatible. Used only at evaluation time.
    """
    import numpy as np
    from collections import deque

    adj_np = np.array(adj)
    N = h * w
    labels = np.zeros(N, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    instance_id = 0

    for start in range(N):
        if visited[start]:
            continue

        # BFS
        queue = deque([start])
        visited[start] = True
        component = [start]

        while queue:
            node = queue.popleft()
            neighbors = np.where(adj_np[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    component.append(nb)
                    queue.append(nb)

        # Filter by size
        if len(component) >= min_size:
            instance_id += 1
            if instance_id > max_instances:
                break
            for idx in component:
                labels[idx] = instance_id

    return labels
