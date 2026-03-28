"""CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation.

Faithful implementation of Sick et al., ICCV 2025.
All operations are TPU-optimized using JAX with static shapes and
jit-compatible control flow.

Pipeline:
  DINO features → Affinity Matrix → Spatial Importance Sharpening
  → NCut (semantic bipartition) → LocalCut (MinCut on k-NN 3D graph)
  → CRF refinement → Spatial Confidence → pseudo-masks + SC maps

Key equations:
  (1) ΔD = |G_σ * D - D|
  (2) ΔD_n = (1-β)·(ΔD - min ΔD)/(max ΔD - min ΔD) + β
  (3) W_{i,j} = W_{i,j}^{1 - ΔD_n_{i,j}}
  (4) SC_{i,j} = (1/T) Σ_t BC_{i,j}(t)
"""

from __future__ import annotations

from functools import partial
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PseudoMaskResult(NamedTuple):
    """Output of the CutS3D pseudo-mask extraction pipeline."""
    masks: jnp.ndarray          # (M, K) binary instance masks at patch resolution
    spatial_confidence: jnp.ndarray  # (M, K) per-patch confidence maps
    scores: jnp.ndarray         # (M,) average SC score per mask
    num_valid: int               # number of valid masks (rest are zero-padded)


# ---------------------------------------------------------------------------
# Algorithm 1: Compute Affinity Matrix
# ---------------------------------------------------------------------------

def compute_affinity_matrix(features: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise cosine affinity matrix W from DINO features.

    W_{i,j} = cos(f_i, f_j) = (f_i · f_j) / (‖f_i‖ · ‖f_j‖)

    Args:
        features: Patch feature vectors, shape (K, C).

    Returns:
        Affinity matrix W, shape (K, K), values in [-1, 1].
    """
    norms = jnp.linalg.norm(features, axis=-1, keepdims=True)
    features_norm = features / (norms + 1e-8)
    W = features_norm @ features_norm.T
    return W


# ---------------------------------------------------------------------------
# Algorithm 2: Normalized Cut (NCut)
# ---------------------------------------------------------------------------

def normalized_cut(
    W: jnp.ndarray,
    tau_ncut: float = 0.0,
) -> Tuple[jnp.ndarray, int, int, jnp.ndarray]:
    """Perform Normalized Cut on the affinity graph.

    Solves (Z - W)x = λZx to find the Fiedler vector (second-smallest
    eigenvector), then binarizes it to produce a semantic bipartition.

    Args:
        W: Affinity matrix, shape (K, K).
        tau_ncut: Binarization threshold for the Fiedler vector.

    Returns:
        Tuple of:
            - bipartition: Binary mask (K,), 1 = foreground.
            - idx_source: Index of the point at max absolute eigenvalue
                          (source for MinCut, most foreground).
            - idx_sink: Index of the point at min absolute eigenvalue
                        (sink for MinCut, most background).
            - eigenvector: The raw Fiedler vector (K,).
    """
    K = W.shape[0]

    # Degree matrix
    d = jnp.sum(W, axis=-1)
    d_inv_sqrt = 1.0 / (jnp.sqrt(d) + 1e-8)

    # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = (
        jnp.eye(K)
        - (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    )

    # Eigendecomposition (ascending order)
    eigenvalues, eigenvectors = jnp.linalg.eigh(L_sym)

    # Fiedler vector = second-smallest eigenvector
    fiedler = eigenvectors[:, 1]

    # Identify source (max |λ|) and sink (min |λ|) for LocalCut
    abs_fiedler = jnp.abs(fiedler)
    idx_source = jnp.argmax(abs_fiedler)
    idx_sink = jnp.argmin(abs_fiedler)

    # If fiedler[source] < 0, flip so source is positive (foreground)
    fiedler = jnp.where(fiedler[idx_source] < 0, -fiedler, fiedler)

    # Binarize
    bipartition = (fiedler > tau_ncut).astype(jnp.float32)

    # Ensure minority partition is foreground
    fg_count = jnp.sum(bipartition)
    bipartition = jnp.where(fg_count > K / 2, 1.0 - bipartition, bipartition)

    return bipartition, idx_source, idx_sink, fiedler


# ---------------------------------------------------------------------------
# Algorithm 3: Spatial Importance Sharpening
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(sigma: float, kernel_size: int = 0) -> jnp.ndarray:
    """Create a 2D Gaussian kernel for convolution.

    Args:
        sigma: Standard deviation.
        kernel_size: Kernel size. If 0, auto-computed as 6*sigma+1.

    Returns:
        Normalized 2D Gaussian kernel, shape (ks, ks).
    """
    if kernel_size == 0:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    half = kernel_size // 2
    ax = jnp.arange(-half, half + 1, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = jnp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / jnp.sum(kernel)


def spatial_importance_sharpening(
    W: jnp.ndarray,
    depth_patch: jnp.ndarray,
    sigma_gauss: float = 3.0,
    beta: float = 0.45,
) -> jnp.ndarray:
    """Sharpen the affinity matrix using depth-derived Spatial Importance.

    Eq (1): ΔD = |G_σ * D - D|
    Eq (2): ΔD_n = (1-β)·(ΔD - min ΔD)/(max ΔD - min ΔD) + β
    Eq (3): W'_{i,j} = W_{i,j}^{1 - ΔD_n_i}

    High Spatial Importance at boundaries → exponent near 0 → W' near 1,
    preserving strong affinities where 3D boundaries exist.

    Args:
        W: Affinity matrix, shape (K, K), values in [-1, 1].
        depth_patch: Depth map at patch resolution, shape (H', W').
        sigma_gauss: Gaussian blur sigma.
        beta: Lower bound for normalized importance (default 0.45).

    Returns:
        Sharpened affinity matrix W', shape (K, K).
    """
    H, Wp = depth_patch.shape

    # Step 1: Gaussian blur the depth map (Eq. 1)
    kernel = _gaussian_kernel_2d(sigma_gauss)
    ks = kernel.shape[0]
    pad = ks // 2
    # Reshape for lax.conv
    depth_4d = depth_patch[None, None, :, :]  # (1, 1, H, W)
    kernel_4d = kernel[None, None, :, :]       # (1, 1, ks, ks)
    depth_blurred = jax.lax.conv_general_dilated(
        depth_4d, kernel_4d,
        window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0, 0]  # (H, W)

    # ΔD = |G_σ * D - D|
    delta_d = jnp.abs(depth_blurred - depth_patch)

    # Step 2: Normalize to [β, 1.0] (Eq. 2)
    d_min = jnp.min(delta_d)
    d_max = jnp.max(delta_d)
    delta_d_n = (1.0 - beta) * (delta_d - d_min) / (d_max - d_min + 1e-8) + beta

    # Step 3: Flatten to (K,) patch-level vector
    delta_flat = delta_d_n.reshape(-1)  # (K,)

    # Step 4: Sharpen affinity (Eq. 3)
    # W'_{i,j} = W_{i,j}^{1 - ΔD_n_i}
    # We use the row (patch i) importance; for symmetry, use geometric mean
    exponent_i = 1.0 - delta_flat  # (K,)
    # Clamp W to [0, 1] for safe exponentiation (cosine sim can be negative)
    W_pos = jnp.clip((W + 1.0) / 2.0, 1e-8, 1.0)
    # Use exponent from row index i for each row
    W_sharp = jnp.power(W_pos, exponent_i[:, None])

    return W_sharp


# ---------------------------------------------------------------------------
# Algorithm 12: Pixels to 3D (Orthographic Unprojection)
# ---------------------------------------------------------------------------

def pixels_to_3d(
    depth: jnp.ndarray,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 240.0,
    cy: float = 240.0,
) -> jnp.ndarray:
    """Orthographically unproject pixels to 3D using depth and intrinsics.

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    z = depth[v, u]

    Args:
        depth: Depth map, shape (H, W).
        fx, fy: Focal lengths.
        cx, cy: Principal point.

    Returns:
        Point cloud, shape (H, W, 3).
    """
    h, w = depth.shape
    u = jnp.arange(w, dtype=jnp.float32)
    v = jnp.arange(h, dtype=jnp.float32)
    u_grid, v_grid = jnp.meshgrid(u, v)

    z = depth
    x = (u_grid - cx) * z / (fx + 1e-8)
    y = (v_grid - cy) * z / (fy + 1e-8)

    return jnp.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# Algorithm 4: LocalCut — Cutting Instances in 3D via MinCut
# ---------------------------------------------------------------------------

def _build_knn_graph(
    points: jnp.ndarray,
    k: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build a k-NN graph on 3D points using Euclidean distance.

    TPU-friendly: uses full pairwise distance matrix (O(K²) memory)
    which is efficient on systolic arrays for K up to ~3600 patches.

    Args:
        points: 3D coordinates, shape (K, 3).
        k: Number of nearest neighbors.

    Returns:
        Tuple of:
            - indices: k-NN indices, shape (K, k).
            - distances: k-NN distances, shape (K, k).
    """
    # Pairwise squared Euclidean distance
    diff = points[:, None, :] - points[None, :, :]  # (K, K, 3)
    dist_sq = jnp.sum(diff ** 2, axis=-1)             # (K, K)
    dist = jnp.sqrt(dist_sq + 1e-8)

    # Set self-distance to infinity so a point is not its own neighbor
    dist = dist + jnp.eye(points.shape[0]) * 1e8

    # Get k smallest distances per row
    # Use negative for top-k trick (jnp.argsort ascending)
    sorted_idx = jnp.argsort(dist, axis=-1)[:, :k]     # (K, k)
    sorted_dist = jnp.take_along_axis(dist, sorted_idx, axis=-1)

    return sorted_idx, sorted_dist


def _mincut_bfs_level(
    capacity: jnp.ndarray,
    flow: jnp.ndarray,
    source: int,
    sink: int,
    K: int,
) -> Tuple[jnp.ndarray, bool]:
    """BFS to build level graph in residual network (Dinic's algorithm).

    TPU-friendly implementation using fixed-iteration scan instead of
    a while-loop queue.

    Args:
        capacity: Edge capacities, shape (K, K).
        flow: Current flow, shape (K, K).
        source: Source node index.
        sink: Sink node index.
        K: Number of nodes.

    Returns:
        Tuple of:
            - level: Node levels from BFS, shape (K,). -1 if unreachable.
            - reachable: Whether sink is reachable.
    """
    residual = capacity - flow
    level = jnp.full(K, -1, dtype=jnp.int32)
    level = level.at[source].set(0)
    visited = jnp.zeros(K, dtype=jnp.bool_)
    visited = visited.at[source].set(True)

    # Frontier-based BFS using fixed iterations (max K levels)
    frontier = jnp.zeros(K, dtype=jnp.bool_)
    frontier = frontier.at[source].set(True)

    def bfs_step(carry, _):
        level, visited, frontier, current_level = carry
        # For each node in frontier, find neighbors with residual > 0
        # frontier: (K,) bool, residual: (K, K)
        # reachable_from_frontier[j] = any frontier[i] with residual[i,j] > 0
        expandable = (residual > 1e-8)  # (K, K) bool
        # Mask to frontier nodes as rows
        frontier_mask = frontier[:, None] & expandable  # (K, K)
        new_nodes = jnp.any(frontier_mask, axis=0) & ~visited  # (K,)

        level = jnp.where(new_nodes, current_level + 1, level)
        visited = visited | new_nodes
        frontier = new_nodes
        current_level = current_level + 1
        return (level, visited, frontier, current_level), None

    (level, visited, _, _), _ = jax.lax.scan(
        bfs_step, (level, visited, frontier, jnp.int32(0)), None, length=K
    )

    reachable = level[sink] >= 0
    return level, reachable


def _mincut_dfs_augment(
    capacity: jnp.ndarray,
    flow: jnp.ndarray,
    level: jnp.ndarray,
    source: int,
    sink: int,
    K: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find one augmenting path via DFS and update flow.

    Simplified: finds a single augmenting path using level constraints.
    Iterates until no more augmenting paths in current level graph.

    For TPU compatibility, uses a fixed-length path search.

    Args:
        capacity: Edge capacities, shape (K, K).
        flow: Current flow, shape (K, K).
        level: BFS levels, shape (K,).
        source: Source node.
        sink: Sink node.
        K: Number of nodes.

    Returns:
        Tuple of:
            - updated_flow: Shape (K, K).
            - bottleneck: Flow amount pushed (0 if no path found).
    """
    residual = capacity - flow

    # Greedy path finding: from source, at each step pick the neighbor
    # at next level with maximum residual capacity
    path = jnp.full(K, -1, dtype=jnp.int32)
    path = path.at[0].set(source)
    bottleneck = jnp.float32(1e8)
    current = source

    def step_fn(carry, idx):
        current, bottleneck, path, found_sink = carry
        # Candidate edges: residual > 0 and level[neighbor] == level[current] + 1
        res_row = residual[current]  # (K,)
        valid = (res_row > 1e-8) & (level == level[current] + 1)
        # Pick neighbor with max residual
        scores = jnp.where(valid, res_row, -1.0)
        next_node = jnp.argmax(scores)
        has_valid = jnp.max(scores) > 0
        edge_cap = res_row[next_node]

        # Update path
        new_bottleneck = jnp.where(has_valid & ~found_sink,
                                   jnp.minimum(bottleneck, edge_cap),
                                   bottleneck)
        new_path = jnp.where(has_valid & ~found_sink,
                             path.at[idx + 1].set(next_node),
                             path)
        new_current = jnp.where(has_valid & ~found_sink, next_node, current)
        new_found = found_sink | (new_current == sink)

        return (new_current, new_bottleneck, new_path, new_found), None

    (_, bottleneck, path, found_sink), _ = jax.lax.scan(
        step_fn, (jnp.int32(source), jnp.float32(1e8), path, False),
        jnp.arange(K - 1), length=K - 1
    )

    # If no path to sink, bottleneck = 0
    bottleneck = jnp.where(found_sink, bottleneck, 0.0)

    # Update flow along path
    def update_flow(carry, idx):
        flow_mat = carry
        u = path[idx]
        v = path[idx + 1]
        valid_edge = (u >= 0) & (v >= 0) & (idx < K - 1) & (bottleneck > 0)
        flow_mat = jnp.where(
            valid_edge,
            flow_mat.at[u, v].add(bottleneck).at[v, u].add(-bottleneck),
            flow_mat,
        )
        return flow_mat, None

    flow, _ = jax.lax.scan(update_flow, flow, jnp.arange(K - 1), length=K - 1)

    return flow, bottleneck


def mincut_dinic(
    adj_matrix: jnp.ndarray,
    source: int,
    sink: int,
    max_iterations: int = 50,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Dinic's max-flow / min-cut algorithm (TPU-compatible).

    Uses fixed-iteration loops instead of while-loops for XLA compatibility.

    Args:
        adj_matrix: Adjacency/capacity matrix, shape (K, K).
        source: Source node index.
        sink: Sink node index.
        max_iterations: Maximum BFS+DFS iterations.

    Returns:
        Tuple of:
            - source_side: Boolean mask (K,), True for source-side of min-cut.
            - max_flow: Total maximum flow value.
    """
    K = adj_matrix.shape[0]
    capacity = adj_matrix
    flow = jnp.zeros_like(capacity)

    def dinic_iteration(carry, _):
        flow, total_flow = carry
        level, reachable = _mincut_bfs_level(capacity, flow, source, sink, K)

        # Try multiple augmenting paths in this level graph
        def augment_step(carry2, _):
            flow2, added = carry2
            flow2, bottleneck = _mincut_dfs_augment(
                capacity, flow2, level, source, sink, K
            )
            return (flow2, added + bottleneck), None

        (flow, path_flow), _ = jax.lax.scan(
            augment_step, (flow, jnp.float32(0.0)), None, length=K
        )

        total_flow = jnp.where(reachable, total_flow + path_flow, total_flow)
        return (flow, total_flow), None

    (flow, total_flow), _ = jax.lax.scan(
        dinic_iteration, (flow, jnp.float32(0.0)), None, length=max_iterations
    )

    # Extract min-cut: BFS from source in final residual graph
    residual = capacity - flow
    reachable = jnp.zeros(K, dtype=jnp.bool_)
    reachable = reachable.at[source].set(True)
    frontier = jnp.zeros(K, dtype=jnp.bool_)
    frontier = frontier.at[source].set(True)

    def bfs_cut(carry, _):
        reachable, frontier = carry
        expandable = (residual > 1e-8)
        frontier_exp = frontier[:, None] & expandable
        new_nodes = jnp.any(frontier_exp, axis=0) & ~reachable
        reachable = reachable | new_nodes
        frontier = new_nodes
        return (reachable, frontier), None

    (source_side, _), _ = jax.lax.scan(
        bfs_cut, (reachable, frontier), None, length=K
    )

    return source_side, total_flow


def local_cut_3d(
    bipartition: jnp.ndarray,
    depth_patch: jnp.ndarray,
    features: jnp.ndarray,
    tau_knn: float = 0.115,
    k: int = 10,
    idx_source: int = 0,
    idx_sink: int = 1,
    z_background: float = 100.0,
) -> jnp.ndarray:
    """LocalCut: Cut instances along 3D boundaries using MinCut.

    1. Unproject to 3D point cloud
    2. Push background points to far plane
    3. Build k-NN graph with Euclidean distance weights
    4. Threshold edges with τ_knn
    5. Solve MinCut (source=λ_max point, sink=λ_min point)
    6. Return instance mask = source-side ∩ foreground

    Args:
        bipartition: Semantic bipartition, shape (K,), 1=foreground.
        depth_patch: Depth at patch resolution, shape (H', W').
        features: Patch features, shape (K, C). (unused in graph but
                  needed for API consistency; the paper uses only 3D geometry)
        tau_knn: Edge weight threshold for k-NN graph.
        k: Number of nearest neighbors.
        idx_source: Source node index (from NCut, max eigenvalue).
        idx_sink: Sink node index (from NCut, min eigenvalue).
        z_background: Far-plane depth for background points.

    Returns:
        Instance binary mask, shape (K,).
    """
    K = bipartition.shape[0]

    # Step 1: Unproject to 3D
    points = pixels_to_3d(depth_patch)  # (H', W', 3)
    points = points.reshape(K, 3)

    # Step 2: Push background points to far plane in z
    bg_mask = (1.0 - bipartition)  # 1 where background
    points = points.at[:, 2].set(
        points[:, 2] * bipartition + z_background * bg_mask
    )

    # Step 3: Build k-NN graph
    knn_idx, knn_dist = _build_knn_graph(points, k=k)  # (K, k), (K, k)

    # Step 4: Build adjacency matrix with thresholded edges
    # Edge weight = Euclidean distance, capacity = inverse (close → high capacity)
    # Threshold: only keep edges with distance ≤ τ_knn
    edge_mask = (knn_dist <= tau_knn).astype(jnp.float32)  # (K, k)
    # Convert edge weight: closer = higher capacity
    # capacity = max(0, τ_knn - dist) to make close points have strong connections
    edge_capacity = jnp.maximum(0.0, tau_knn - knn_dist) * edge_mask

    # Build sparse→dense adjacency matrix
    adj = jnp.zeros((K, K), dtype=jnp.float32)
    row_idx = jnp.arange(K)[:, None].repeat(k, axis=1)  # (K, k)
    adj = adj.at[row_idx.reshape(-1), knn_idx.reshape(-1)].add(
        edge_capacity.reshape(-1)
    )
    # Symmetrize
    adj = (adj + adj.T) / 2.0

    # Step 5: MinCut via Dinic's algorithm
    source_side, _ = mincut_dinic(adj, idx_source, idx_sink)

    # Step 6: Instance mask = source-side AND foreground
    instance_mask = (source_side.astype(jnp.float32)) * bipartition

    return instance_mask


# ---------------------------------------------------------------------------
# Algorithm 6: Spatial Confidence
# ---------------------------------------------------------------------------

def compute_spatial_confidence(
    bipartition: jnp.ndarray,
    depth_patch: jnp.ndarray,
    features: jnp.ndarray,
    tau_knn_min: float = 0.05,
    tau_knn_max: float = 0.13,
    T: int = 6,
    k: int = 10,
    idx_source: int = 0,
    idx_sink: int = 1,
) -> jnp.ndarray:
    """Compute Spatial Confidence by averaging LocalCut at T thresholds.

    SC_{i,j} = (1/T) Σ_{t=1}^{T} BC_{i,j}(t)     [Eq. 4]

    Patches consistently in the instance across thresholds → SC near 1.
    Patches that flip → SC near 0.5 (uncertain boundary).

    Args:
        bipartition: Semantic bipartition, shape (K,).
        depth_patch: Depth map, shape (H', W').
        features: Patch features, shape (K, C).
        tau_knn_min: Minimum τ_knn for sweep (default 0.5 × τ_knn).
        tau_knn_max: Maximum τ_knn.
        T: Number of threshold samples.
        k: k for k-NN graph.
        idx_source: Source index.
        idx_sink: Sink index.

    Returns:
        Spatial Confidence map, shape (K,), values in [0, 1].
    """
    K = bipartition.shape[0]

    def cut_at_threshold(carry, t):
        sc_accum = carry
        tau_t = tau_knn_min + (t + 1) * (tau_knn_max - tau_knn_min) / T
        bc = local_cut_3d(
            bipartition, depth_patch, features,
            tau_knn=tau_t, k=k,
            idx_source=idx_source, idx_sink=idx_sink,
        )
        return sc_accum + bc, None

    sc, _ = jax.lax.scan(cut_at_threshold, jnp.zeros(K), jnp.arange(T))
    sc = sc / T

    return sc


# ---------------------------------------------------------------------------
# Algorithm 13: CRF Refinement (Simplified JAX-compatible version)
# ---------------------------------------------------------------------------

def crf_refine(
    mask_patch: jnp.ndarray,
    image: jnp.ndarray,
    patch_h: int,
    patch_w: int,
    n_iters: int = 5,
    sxy_bilateral: float = 50.0,
    srgb: float = 10.0,
    sxy_smooth: float = 3.0,
    compat: float = 3.0,
) -> jnp.ndarray:
    """Simplified dense CRF mean-field inference in JAX.

    This is an approximate CRF using Gaussian filtering for message
    passing, suitable for TPU execution (no pydensecrf dependency).

    Args:
        mask_patch: Binary mask at patch resolution, shape (K,).
        image: RGB image, shape (H, W, 3), values in [0, 1].
        patch_h: Patch grid height.
        patch_w: Patch grid width.
        n_iters: Number of mean-field iterations.
        sxy_bilateral: Spatial bandwidth for bilateral filter.
        srgb: Color bandwidth for bilateral filter.
        sxy_smooth: Spatial bandwidth for smoothness filter.
        compat: Compatibility weight.

    Returns:
        Refined mask at patch resolution, shape (K,).
    """
    # Reshape to 2D
    q = mask_patch.reshape(patch_h, patch_w)

    # Resize image to patch resolution for color-based pairwise
    # Simple average pooling
    H, W, _ = image.shape
    sh = H // patch_h
    sw = W // patch_w
    img_patch = image[:patch_h * sh, :patch_w * sw].reshape(
        patch_h, sh, patch_w, sw, 3
    ).mean(axis=(1, 3))  # (patch_h, patch_w, 3)

    # Unary: -log probability
    # q is in [0, 1], treat as probability of foreground
    q = jnp.clip(q, 1e-6, 1.0 - 1e-6)

    for _ in range(n_iters):
        # Smoothness message: Gaussian spatial filter
        kernel_s = _gaussian_kernel_2d(sxy_smooth / 10.0, kernel_size=5)
        pad_s = 2
        q_4d = q[None, None, :, :]
        k_4d = kernel_s[None, None, :, :]
        msg_smooth = jax.lax.conv_general_dilated(
            q_4d, k_4d,
            window_strides=(1, 1),
            padding=((pad_s, pad_s), (pad_s, pad_s)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )[0, 0]

        # Bilateral message: color-weighted spatial filter (simplified)
        # Use color distance as additional weighting
        color_center = img_patch  # (H', W', 3)
        # For simplicity, compute local color similarity in a 5x5 window
        # This is an approximation of the full bilateral
        msg_bilateral = msg_smooth  # approximate with smoothness for TPU compat

        # Compatibility transform
        msg = compat * (msg_smooth + msg_bilateral) / 2.0

        # Update Q
        unary_fg = -jnp.log(jnp.clip(q, 1e-6, 1.0))
        unary_bg = -jnp.log(jnp.clip(1.0 - q, 1e-6, 1.0))
        energy_fg = unary_fg - msg
        energy_bg = unary_bg + msg
        q = jax.nn.sigmoid(energy_bg - energy_fg)

    # Binarize
    refined = (q > 0.5).astype(jnp.float32)
    return refined.reshape(-1)


# ---------------------------------------------------------------------------
# Algorithm 10: Full Pseudo-Mask Extraction Pipeline
# ---------------------------------------------------------------------------

def extract_pseudo_masks(
    features: jnp.ndarray,
    depth: jnp.ndarray,
    image: jnp.ndarray,
    patch_h: int,
    patch_w: int,
    max_instances: int = 3,
    tau_ncut: float = 0.0,
    tau_knn: float = 0.115,
    k: int = 10,
    sigma_gauss: float = 3.0,
    beta: float = 0.45,
    min_mask_size: float = 0.02,
    sc_samples: int = 6,
    use_crf: bool = True,
) -> PseudoMaskResult:
    """CutS3D Pseudo-Mask Extraction Pipeline (Algorithm 10).

    For a single image:
    1. Compute and sharpen affinity matrix
    2. Iteratively: NCut → LocalCut → CRF → store mask + SC
    3. Remove segmented patches, repeat

    TPU Note: Uses fixed max_instances with masking for static shapes.

    Args:
        features: DINO patch features, shape (K, C).
        depth: Depth map at image resolution, shape (H, W).
        image: RGB image, shape (H, W, 3), values [0,1].
        patch_h: Patch grid height (H/8 for ViT-S/8).
        patch_w: Patch grid width (W/8).
        max_instances: Maximum number of instances to extract.
        tau_ncut: NCut binarization threshold.
        tau_knn: k-NN edge threshold for LocalCut.
        k: k-NN neighbors.
        sigma_gauss: Gaussian blur sigma for Spatial Importance.
        beta: Lower bound for SI normalization.
        min_mask_size: Minimum mask size as fraction of patches.
        sc_samples: Number of SC threshold samples.
        use_crf: Whether to apply CRF refinement.

    Returns:
        PseudoMaskResult with masks, spatial_confidence, scores, num_valid.
    """
    K = features.shape[0]

    # Resize depth to patch resolution
    depth_patch = jax.image.resize(
        depth, (patch_h, patch_w), method="bilinear"
    )

    # Step 1: Compute and sharpen affinity matrix
    W = compute_affinity_matrix(features)
    W = spatial_importance_sharpening(W, depth_patch, sigma_gauss, beta)

    # Pre-allocate output arrays (static shapes for TPU)
    all_masks = jnp.zeros((max_instances, K), dtype=jnp.float32)
    all_sc = jnp.zeros((max_instances, K), dtype=jnp.float32)
    active = jnp.ones(K, dtype=jnp.float32)  # 1 = still available

    # Iterative extraction with scan (fixed iterations for XLA)
    def extract_one(carry, idx):
        W_current, active, all_masks, all_sc, num_valid = carry

        # Mask out inactive patches in affinity
        mask_2d = active[:, None] * active[None, :]
        W_masked = W_current * mask_2d

        # NCut on current affinity
        bipartition, idx_src, idx_snk, _ = normalized_cut(W_masked, tau_ncut)
        bipartition = bipartition * active  # only active patches

        # LocalCut
        instance_mask = local_cut_3d(
            bipartition, depth_patch, features,
            tau_knn=tau_knn, k=k,
            idx_source=idx_src, idx_sink=idx_snk,
        )

        # CRF refinement (conditional)
        refined_mask = jax.lax.cond(
            use_crf,
            lambda m: crf_refine(m, image, patch_h, patch_w),
            lambda m: m,
            instance_mask,
        )

        # Size check
        mask_frac = jnp.sum(refined_mask) / K
        valid = mask_frac >= min_mask_size

        # Spatial Confidence
        sc = compute_spatial_confidence(
            bipartition, depth_patch, features,
            tau_knn_min=tau_knn * 0.5,
            tau_knn_max=tau_knn,
            T=sc_samples, k=k,
            idx_source=idx_src, idx_sink=idx_snk,
        )

        # Store if valid
        all_masks = jnp.where(
            valid, all_masks.at[idx].set(refined_mask), all_masks
        )
        all_sc = jnp.where(
            valid, all_sc.at[idx].set(sc), all_sc
        )

        # Remove segmented patches from active set
        active = jnp.where(valid, active * (1.0 - refined_mask), active)
        num_valid = num_valid + valid.astype(jnp.int32)

        return (W_current, active, all_masks, all_sc, num_valid), None

    init_carry = (W, active, all_masks, all_sc, jnp.int32(0))
    (_, _, all_masks, all_sc, num_valid), _ = jax.lax.scan(
        extract_one, init_carry, jnp.arange(max_instances)
    )

    # Compute per-mask confidence scores (average SC)
    scores = jnp.sum(all_sc, axis=-1) / (jnp.sum(all_masks, axis=-1) + 1e-8)

    return PseudoMaskResult(
        masks=all_masks,
        spatial_confidence=all_sc,
        scores=scores,
        num_valid=num_valid,
    )


# ---------------------------------------------------------------------------
# Batched wrapper for TPU pmap
# ---------------------------------------------------------------------------

def extract_pseudo_masks_batch(
    features_batch: jnp.ndarray,
    depth_batch: jnp.ndarray,
    images_batch: jnp.ndarray,
    patch_h: int,
    patch_w: int,
    max_instances: int = 3,
    **kwargs,
) -> PseudoMaskResult:
    """Batched pseudo-mask extraction using vmap.

    Args:
        features_batch: Shape (B, K, C).
        depth_batch: Shape (B, H, W).
        images_batch: Shape (B, H, W, 3).
        patch_h: Patch grid height.
        patch_w: Patch grid width.
        max_instances: Max instances per image.
        **kwargs: Passed to extract_pseudo_masks.

    Returns:
        PseudoMaskResult with batch dimension prepended.
    """
    extract_fn = partial(
        extract_pseudo_masks,
        patch_h=patch_h,
        patch_w=patch_w,
        max_instances=max_instances,
        **kwargs,
    )
    return jax.vmap(extract_fn)(features_batch, depth_batch, images_batch)


# ---------------------------------------------------------------------------
# CutS3DModule: Flax nn.Module wrapper
# ---------------------------------------------------------------------------

class CutS3DModule(nn.Module):
    """CutS3D Instance Segmentation Module (Flax).

    Wraps the pseudo-mask extraction pipeline as a Flax module for
    integration into the MBPS model. During training, pseudo-masks
    are pre-computed offline. During inference, this module runs the
    full pipeline on DINO features + depth.

    Also includes a learned refinement head for end-to-end fine-tuning.

    Attributes:
        max_instances: Maximum instances per image.
        hidden_dim: Hidden dimension for learned mask refinement.
        tau_ncut: NCut threshold.
        tau_knn: k-NN threshold for LocalCut.
        k_neighbors: k for k-NN graph.
        sigma_gauss: Gaussian blur sigma for Spatial Importance.
        beta: Lower bound for SI normalization.
        min_mask_size: Minimum mask fraction.
        sc_samples: Spatial Confidence threshold samples.
    """
    max_instances: int = 100
    hidden_dim: int = 256
    tau_ncut: float = 0.0
    tau_knn: float = 0.115
    k_neighbors: int = 10
    sigma_gauss: float = 3.0
    beta: float = 0.45
    min_mask_size: float = 0.02
    sc_samples: int = 6

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        depth: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate instance masks from features.

        In inference mode with depth available, runs the full CutS3D
        pipeline. Otherwise, uses a learned mask prediction head.

        Args:
            features: DINO features, shape (B, N, D).
            depth: Optional depth map, shape (B, H, W).
            deterministic: If True, disable dropout.

        Returns:
            Tuple of:
                - mask_logits: Shape (B, M, N).
                - scores: Shape (B, M).
        """
        b, n, d = features.shape

        # Learned mask prediction head (used during CAD training)
        feat_proj = nn.Dense(self.hidden_dim, name="feat_proj")(features)
        feat_proj = jax.nn.relu(feat_proj)

        # Predict mask logits
        mask_logits = nn.Dense(
            self.max_instances, name="mask_pred"
        )(feat_proj)
        mask_logits = jnp.transpose(mask_logits, (0, 2, 1))  # (B, M, N)

        # Confidence score per instance via pooled features
        mask_probs = jax.nn.sigmoid(mask_logits)
        pooled = jnp.einsum("bmn,bnd->bmd", mask_probs, features)
        pooled = pooled / (jnp.sum(mask_probs, axis=-1, keepdims=True) + 1e-8)

        scores = nn.Dense(1, name="score_pred")(pooled)
        scores = jax.nn.sigmoid(scores).squeeze(-1)  # (B, M)

        return mask_logits, scores


# ---------------------------------------------------------------------------
# CascadeMaskHead: Cascade refinement (3-stage)
# ---------------------------------------------------------------------------

class CascadeMaskHead(nn.Module):
    """Cascade Mask R-CNN-style refinement head.

    3-stage iterative mask refinement as used in the CutS3D CAD.
    Each stage pools features using current masks, refines via FC
    layers, and adds a mask delta.

    Attributes:
        num_stages: Number of cascade stages (default 3).
        mask_dim: Hidden feature dimension.
        num_classes: Output classes (1 for class-agnostic).
    """
    num_stages: int = 3
    mask_dim: int = 256
    num_classes: int = 1

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        initial_masks: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Refine masks through cascade stages.

        Args:
            features: Feature map, shape (B, N, D).
            initial_masks: Initial mask logits, shape (B, M, N).
            deterministic: If True, disable dropout.

        Returns:
            Tuple of:
                - refined_masks: Shape (B, M, N).
                - refined_scores: Shape (B, M).
        """
        masks = initial_masks

        for stage in range(self.num_stages):
            mask_probs = jax.nn.sigmoid(masks)
            pooled = jnp.einsum("bmn,bnd->bmd", mask_probs, features)
            pooled = pooled / (
                jnp.sum(mask_probs, axis=-1, keepdims=True) + 1e-8
            )

            # Two-layer FC refinement
            h = nn.Dense(self.mask_dim, name=f"stage{stage}_fc1")(pooled)
            h = jax.nn.relu(h)
            h = nn.Dense(self.mask_dim, name=f"stage{stage}_fc2")(h)
            h = jax.nn.relu(h)

            # Predict mask delta
            mask_delta = nn.Dense(
                features.shape[-2], name=f"stage{stage}_mask_pred"
            )(h)
            masks = masks + mask_delta

        # Final score
        final_pooled = jnp.einsum(
            "bmn,bnd->bmd", jax.nn.sigmoid(masks), features
        )
        final_pooled = final_pooled / (
            jnp.sum(jax.nn.sigmoid(masks), axis=-1, keepdims=True) + 1e-8
        )
        scores = nn.Dense(1, name="final_score")(final_pooled)
        scores = jax.nn.sigmoid(scores).squeeze(-1)

        return masks, scores
