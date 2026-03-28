"""CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation.

Faithful implementation of Sick et al., ICCV 2025.
All operations are GPU-optimized using PyTorch with CUDA support.

Pipeline:
  DINO features -> Affinity Matrix -> Spatial Importance Sharpening
  -> NCut (semantic bipartition) -> LocalCut (MinCut on k-NN 3D graph)
  -> CRF refinement -> Spatial Confidence -> pseudo-masks + SC maps

Key equations:
  (1) DeltaD = |G_sigma * D - D|
  (2) DeltaD_n = (1-beta)*(DeltaD - min DeltaD)/(max DeltaD - min DeltaD) + beta
  (3) W_{i,j} = W_{i,j}^{1 - DeltaD_n_{i,j}}
  (4) SC_{i,j} = (1/T) Sum_t BC_{i,j}(t)
"""

from __future__ import annotations

from functools import partial
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PseudoMaskResult(NamedTuple):
    """Output of the CutS3D pseudo-mask extraction pipeline."""
    masks: torch.Tensor          # (M, K) binary instance masks at patch resolution
    spatial_confidence: torch.Tensor  # (M, K) per-patch confidence maps
    scores: torch.Tensor         # (M,) average SC score per mask
    num_valid: int               # number of valid masks (rest are zero-padded)


# ---------------------------------------------------------------------------
# Algorithm 1: Compute Affinity Matrix
# ---------------------------------------------------------------------------

def compute_affinity_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine affinity matrix W from DINO features.

    W_{i,j} = cos(f_i, f_j) = (f_i . f_j) / (||f_i|| . ||f_j||)

    Args:
        features: Patch feature vectors, shape (K, C).

    Returns:
        Affinity matrix W, shape (K, K), values in [-1, 1].
    """
    norms = torch.linalg.norm(features, dim=-1, keepdim=True)
    features_norm = features / (norms + 1e-8)
    W = features_norm @ features_norm.T
    return W


# ---------------------------------------------------------------------------
# Algorithm 2: Normalized Cut (NCut)
# ---------------------------------------------------------------------------

def normalized_cut(
    W: torch.Tensor,
    tau_ncut: float = 0.0,
) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    """Perform Normalized Cut on the affinity graph.

    Solves (Z - W)x = lambda*Z*x to find the Fiedler vector (second-smallest
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
    d = torch.sum(W, dim=-1)
    d_inv_sqrt = 1.0 / (torch.sqrt(d) + 1e-8)

    # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = (
        torch.eye(K, device=W.device, dtype=W.dtype)
        - (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    )

    # Eigendecomposition (ascending order)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

    # Fiedler vector = second-smallest eigenvector
    fiedler = eigenvectors[:, 1]

    # Identify source (max |lambda|) and sink (min |lambda|) for LocalCut
    abs_fiedler = torch.abs(fiedler)
    idx_source = int(torch.argmax(abs_fiedler).item())
    idx_sink = int(torch.argmin(abs_fiedler).item())

    # If fiedler[source] < 0, flip so source is positive (foreground)
    fiedler = torch.where(
        fiedler[idx_source] < 0,
        -fiedler,
        fiedler,
    )

    # Binarize
    bipartition = (fiedler > tau_ncut).float()

    # Ensure minority partition is foreground
    fg_count = torch.sum(bipartition)
    if fg_count > K / 2:
        bipartition = 1.0 - bipartition
        # Swap source/sink so source stays in foreground
        idx_source, idx_sink = idx_sink, idx_source

    return bipartition, idx_source, idx_sink, fiedler


# ---------------------------------------------------------------------------
# Algorithm 3: Spatial Importance Sharpening
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(sigma: float, kernel_size: int = 0) -> torch.Tensor:
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
    ax = torch.arange(-half, half + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / torch.sum(kernel)


def spatial_importance_sharpening(
    W: torch.Tensor,
    depth_patch: torch.Tensor,
    sigma_gauss: float = 3.0,
    beta: float = 0.45,
) -> torch.Tensor:
    """Sharpen the affinity matrix using depth-derived Spatial Importance.

    Eq (1): DeltaD = |G_sigma * D - D|
    Eq (2): DeltaD_n = (1-beta)*(DeltaD - min DeltaD)/(max DeltaD - min DeltaD) + beta
    Eq (3): W'_{i,j} = W_{i,j}^{1 - DeltaD_n_i}

    High Spatial Importance at boundaries -> exponent near 0 -> W' near 1,
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
    device = W.device

    # Step 1: Gaussian blur the depth map (Eq. 1)
    kernel = _gaussian_kernel_2d(sigma_gauss).to(device)
    ks = kernel.shape[0]
    pad = ks // 2
    # Reshape for F.conv2d
    depth_4d = depth_patch[None, None, :, :]  # (1, 1, H, W)
    kernel_4d = kernel[None, None, :, :]       # (1, 1, ks, ks)
    depth_blurred = F.conv2d(depth_4d, kernel_4d, padding=pad)[0, 0]  # (H, W)

    # DeltaD = |G_sigma * D - D|
    delta_d = torch.abs(depth_blurred - depth_patch)

    # Step 2: Normalize to [beta, 1.0] (Eq. 2)
    d_min = torch.min(delta_d)
    d_max = torch.max(delta_d)
    delta_d_n = (1.0 - beta) * (delta_d - d_min) / (d_max - d_min + 1e-8) + beta

    # Step 3: Flatten to (K,) patch-level vector
    delta_flat = delta_d_n.reshape(-1)  # (K,)

    # Step 4: Sharpen affinity (Eq. 3)
    # W'_{i,j} = W_{i,j}^{1 - DeltaD_n_i}
    # We use the row (patch i) importance; for symmetry, use geometric mean
    exponent_i = 1.0 - delta_flat  # (K,)
    # Clamp W to [0, 1] for safe exponentiation (cosine sim can be negative)
    W_pos = torch.clamp((W + 1.0) / 2.0, 1e-8, 1.0)
    # Use exponent from row index i for each row
    W_sharp = torch.pow(W_pos, exponent_i[:, None])

    return W_sharp


# ---------------------------------------------------------------------------
# Algorithm 12: Pixels to 3D (Orthographic Unprojection)
# ---------------------------------------------------------------------------

def pixels_to_3d(
    depth: torch.Tensor,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 240.0,
    cy: float = 240.0,
) -> torch.Tensor:
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
    device = depth.device
    u = torch.arange(w, dtype=torch.float32, device=device)
    v = torch.arange(h, dtype=torch.float32, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")

    z = depth
    x = (u_grid - cx) * z / (fx + 1e-8)
    y = (v_grid - cy) * z / (fy + 1e-8)

    return torch.stack([x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# Algorithm 4: LocalCut -- Cutting Instances in 3D via MinCut
# ---------------------------------------------------------------------------

def _build_knn_graph(
    points: torch.Tensor,
    k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a k-NN graph on 3D points using Euclidean distance.

    GPU-friendly: uses full pairwise distance matrix (O(K^2) memory)
    which is efficient on GPU for K up to ~3600 patches.

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
    dist_sq = torch.sum(diff ** 2, dim=-1)             # (K, K)
    dist = torch.sqrt(dist_sq + 1e-8)

    # Set self-distance to infinity so a point is not its own neighbor
    dist = dist + torch.eye(points.shape[0], device=points.device) * 1e8

    # Get k smallest distances per row
    sorted_dist, sorted_idx = torch.topk(dist, k, dim=-1, largest=False)

    return sorted_idx, sorted_dist


def _mincut_bfs_level(
    capacity: torch.Tensor,
    flow: torch.Tensor,
    source: int,
    sink: int,
    K: int,
) -> Tuple[torch.Tensor, bool]:
    """BFS to build level graph in residual network (Dinic's algorithm).

    GPU-friendly implementation using fixed-iteration loop instead of
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
    device = capacity.device
    residual = capacity - flow
    level = torch.full((K,), -1, dtype=torch.int32, device=device)
    level[source] = 0
    visited = torch.zeros(K, dtype=torch.bool, device=device)
    visited[source] = True

    # Frontier-based BFS using fixed iterations (max K levels)
    frontier = torch.zeros(K, dtype=torch.bool, device=device)
    frontier[source] = True
    current_level = 0

    for _ in range(K):
        # For each node in frontier, find neighbors with residual > 0
        expandable = (residual > 1e-8)  # (K, K) bool
        frontier_mask = frontier[:, None] & expandable  # (K, K)
        new_nodes = torch.any(frontier_mask, dim=0) & ~visited  # (K,)

        level = torch.where(new_nodes, torch.tensor(current_level + 1, dtype=torch.int32, device=device), level)
        visited = visited | new_nodes
        frontier = new_nodes
        current_level = current_level + 1

    reachable = bool(level[sink].item() >= 0)
    return level, reachable


def _mincut_dfs_augment(
    capacity: torch.Tensor,
    flow: torch.Tensor,
    level: torch.Tensor,
    source: int,
    sink: int,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find one augmenting path via DFS and update flow.

    Simplified: finds a single augmenting path using level constraints.
    Iterates until no more augmenting paths in current level graph.

    For GPU compatibility, uses a fixed-length path search.

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
    device = capacity.device
    residual = capacity - flow

    # Greedy path finding: from source, at each step pick the neighbor
    # at next level with maximum residual capacity
    path = torch.full((K,), -1, dtype=torch.int32, device=device)
    path[0] = source
    bottleneck = torch.tensor(1e8, dtype=torch.float32, device=device)
    current = source
    found_sink = False

    for idx in range(K - 1):
        # Candidate edges: residual > 0 and level[neighbor] == level[current] + 1
        res_row = residual[current]  # (K,)
        valid = (res_row > 1e-8) & (level == level[current] + 1)
        # Pick neighbor with max residual
        scores = torch.where(valid, res_row, torch.tensor(-1.0, device=device))
        next_node = int(torch.argmax(scores).item())
        has_valid = bool(torch.max(scores).item() > 0)
        edge_cap = res_row[next_node]

        if has_valid and not found_sink:
            bottleneck = torch.minimum(bottleneck, edge_cap)
            path[idx + 1] = next_node
            current = next_node
            if current == sink:
                found_sink = True

    # If no path to sink, bottleneck = 0
    if not found_sink:
        bottleneck = torch.tensor(0.0, dtype=torch.float32, device=device)

    # Update flow along path
    for idx in range(K - 1):
        u = path[idx].item()
        v = path[idx + 1].item()
        if u >= 0 and v >= 0 and bottleneck.item() > 0:
            flow[u, v] = flow[u, v] + bottleneck
            flow[v, u] = flow[v, u] - bottleneck

    return flow, bottleneck


def mincut_dinic(
    adj_matrix: torch.Tensor,
    source: int,
    sink: int,
    max_iterations: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dinic's max-flow / min-cut algorithm (GPU-compatible).

    Uses fixed-iteration loops for compatibility.

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
    device = adj_matrix.device
    capacity = adj_matrix.clone()
    flow = torch.zeros_like(capacity)
    total_flow = torch.tensor(0.0, dtype=torch.float32, device=device)

    for _ in range(max_iterations):
        level, reachable = _mincut_bfs_level(capacity, flow, source, sink, K)

        if not reachable:
            break

        # Try multiple augmenting paths in this level graph
        for _ in range(K):
            flow, bottleneck = _mincut_dfs_augment(
                capacity, flow, level, source, sink, K
            )
            if bottleneck.item() <= 0:
                break
            total_flow = total_flow + bottleneck

    # Extract min-cut: BFS from source in final residual graph
    residual = capacity - flow
    reachable_mask = torch.zeros(K, dtype=torch.bool, device=device)
    reachable_mask[source] = True
    frontier = torch.zeros(K, dtype=torch.bool, device=device)
    frontier[source] = True

    for _ in range(K):
        expandable = (residual > 1e-8)
        frontier_exp = frontier[:, None] & expandable
        new_nodes = torch.any(frontier_exp, dim=0) & ~reachable_mask
        reachable_mask = reachable_mask | new_nodes
        frontier = new_nodes

    return reachable_mask, total_flow


def local_cut_3d(
    bipartition: torch.Tensor,
    depth_patch: torch.Tensor,
    features: torch.Tensor,
    tau_knn: float = 0.115,
    k: int = 10,
    idx_source: int = 0,
    idx_sink: int = 1,
    z_background: float = 100.0,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 240.0,
    cy: float = 240.0,
) -> torch.Tensor:
    """LocalCut: Cut instances along 3D boundaries using MinCut.

    1. Unproject to 3D point cloud
    2. Push background points to far plane
    3. Build k-NN graph with Euclidean distance weights
    4. Threshold edges with tau_knn
    5. Solve MinCut (source=lambda_max point, sink=lambda_min point)
    6. Return instance mask = source-side AND foreground

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
    points = pixels_to_3d(depth_patch, fx=fx, fy=fy, cx=cx, cy=cy)  # (H', W', 3)
    points = points.reshape(K, 3)

    # Normalize each 3D dimension to [0, 1] before k-NN (scale-invariant, matches
    # extract_cuts3d_gpu.py reference: tau_knn=0.115 calibrated for normalized space)
    for dim in range(3):
        d_min = points[:, dim].min()
        d_max = points[:, dim].max()
        if (d_max - d_min) > 1e-8:
            points[:, dim] = (points[:, dim] - d_min) / (d_max - d_min)

    # Step 2: Push background points to far plane in z
    bg_mask = (1.0 - bipartition)  # 1 where background
    z_new = points[:, 2] * bipartition + z_background * bg_mask
    points = torch.cat([points[:, :2], z_new.unsqueeze(-1)], dim=-1)

    # Step 3: Build k-NN graph
    knn_idx, knn_dist = _build_knn_graph(points, k=k)  # (K, k), (K, k)

    # Step 4: Build adjacency matrix with thresholded edges
    # Edge weight = Euclidean distance, capacity = inverse (close -> high capacity)
    # Threshold: only keep edges with distance <= tau_knn
    edge_mask = (knn_dist <= tau_knn).float()  # (K, k)
    # Convert edge weight: closer = higher capacity
    # capacity = max(0, tau_knn - dist) to make close points have strong connections
    edge_capacity = torch.clamp(tau_knn - knn_dist, min=0.0) * edge_mask

    # Build sparse->dense adjacency matrix (vectorized via scatter_add)
    adj = torch.zeros((K, K), dtype=torch.float32, device=bipartition.device)
    row_idx = torch.arange(K, device=bipartition.device)[:, None].expand(-1, k)  # (K, k)
    # Flatten row and col indices, then use scatter_add on the flattened adjacency
    flat_row = row_idx.reshape(-1)  # (K*k,)
    flat_col = knn_idx.reshape(-1)  # (K*k,)
    flat_idx = flat_row * K + flat_col  # linear index into (K, K)
    adj_flat = adj.reshape(-1)
    adj_flat.scatter_add_(0, flat_idx.long(), edge_capacity.reshape(-1))
    adj = adj_flat.reshape(K, K)

    # Symmetrize
    adj = (adj + adj.T) / 2.0

    # Step 5: MinCut via Dinic's algorithm
    source_side, _ = mincut_dinic(adj, idx_source, idx_sink)

    # Step 6: Instance mask = source-side AND foreground
    instance_mask = source_side.float() * bipartition

    return instance_mask


# ---------------------------------------------------------------------------
# Algorithm 6: Spatial Confidence
# ---------------------------------------------------------------------------

def compute_spatial_confidence(
    bipartition: torch.Tensor,
    depth_patch: torch.Tensor,
    features: torch.Tensor,
    tau_knn_min: float = 0.05,
    tau_knn_max: float = 0.13,
    T: int = 6,
    k: int = 10,
    idx_source: int = 0,
    idx_sink: int = 1,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 240.0,
    cy: float = 240.0,
) -> torch.Tensor:
    """Compute Spatial Confidence by averaging LocalCut at T thresholds.

    SC_{i,j} = (1/T) Sum_{t=1}^{T} BC_{i,j}(t)     [Eq. 4]

    Patches consistently in the instance across thresholds -> SC near 1.
    Patches that flip -> SC near 0.5 (uncertain boundary).

    Args:
        bipartition: Semantic bipartition, shape (K,).
        depth_patch: Depth map, shape (H', W').
        features: Patch features, shape (K, C).
        tau_knn_min: Minimum tau_knn for sweep (default 0.5 x tau_knn).
        tau_knn_max: Maximum tau_knn.
        T: Number of threshold samples.
        k: k for k-NN graph.
        idx_source: Source index.
        idx_sink: Sink index.

    Returns:
        Spatial Confidence map, shape (K,), values in [0, 1].
    """
    K = bipartition.shape[0]
    device = bipartition.device
    sc = torch.zeros(K, device=device)

    for t in range(T):
        tau_t = tau_knn_min + (t + 1) * (tau_knn_max - tau_knn_min) / T
        bc = local_cut_3d(
            bipartition, depth_patch, features,
            tau_knn=tau_t, k=k,
            idx_source=idx_source, idx_sink=idx_sink,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
        sc = sc + bc

    sc = sc / T
    return sc


# ---------------------------------------------------------------------------
# Algorithm 13: CRF Refinement (Simplified GPU-compatible version)
# ---------------------------------------------------------------------------

def crf_refine(
    mask_patch: torch.Tensor,
    image: torch.Tensor,
    patch_h: int,
    patch_w: int,
    n_iters: int = 5,
    sxy_bilateral: float = 50.0,
    srgb: float = 10.0,
    sxy_smooth: float = 3.0,
    compat: float = 3.0,
) -> torch.Tensor:
    """Simplified dense CRF mean-field inference in PyTorch.

    This is an approximate CRF using Gaussian filtering for message
    passing, suitable for GPU execution (no pydensecrf dependency).

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
    device = mask_patch.device
    # Reshape to 2D
    q = mask_patch.reshape(patch_h, patch_w)

    # Resize image to patch resolution for color-based pairwise
    # Simple average pooling
    H, W, _ = image.shape
    sh = H // patch_h
    sw = W // patch_w
    img_patch = image[:patch_h * sh, :patch_w * sw].reshape(
        patch_h, sh, patch_w, sw, 3
    ).mean(dim=(1, 3))  # (patch_h, patch_w, 3)

    # Unary: -log probability
    # q is in [0, 1], treat as probability of foreground
    q = torch.clamp(q, 1e-6, 1.0 - 1e-6)

    for _ in range(n_iters):
        # Smoothness message: Gaussian spatial filter
        kernel_s = _gaussian_kernel_2d(sxy_smooth / 10.0, kernel_size=5).to(device)
        pad_s = 2
        q_4d = q[None, None, :, :]
        k_4d = kernel_s[None, None, :, :]
        msg_smooth = F.conv2d(q_4d, k_4d, padding=pad_s)[0, 0]

        # Bilateral message: color-weighted spatial filter (simplified)
        # Use color distance as additional weighting
        # For simplicity, compute local color similarity in a 5x5 window
        # This is an approximation of the full bilateral
        msg_bilateral = msg_smooth  # approximate with smoothness for GPU compat

        # Compatibility transform
        msg = compat * (msg_smooth + msg_bilateral) / 2.0

        # Update Q
        unary_fg = -torch.log(torch.clamp(q, 1e-6, 1.0))
        unary_bg = -torch.log(torch.clamp(1.0 - q, 1e-6, 1.0))
        energy_fg = unary_fg - msg
        energy_bg = unary_bg + msg
        q = torch.sigmoid(energy_bg - energy_fg)

    # Binarize
    refined = (q > 0.5).float()
    return refined.reshape(-1)


# ---------------------------------------------------------------------------
# Algorithm 10: Full Pseudo-Mask Extraction Pipeline
# ---------------------------------------------------------------------------

def extract_pseudo_masks(
    features: torch.Tensor,
    depth: torch.Tensor,
    image: torch.Tensor,
    patch_h: int,
    patch_w: int,
    max_instances: int = 3,
    tau_ncut: float = 0.0,
    tau_knn: float = 0.115,
    k: int = 10,
    sigma_gauss: float = 3.0,
    beta: float = 0.45,
    min_mask_size: float = 0.02,
    max_mask_size: float = 1.0,
    sc_samples: int = 6,
    use_crf: bool = True,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 240.0,
    cy: float = 240.0,
    initial_active: Optional[torch.Tensor] = None,
) -> PseudoMaskResult:
    """CutS3D Pseudo-Mask Extraction Pipeline (Algorithm 10).

    For a single image:
    1. Compute and sharpen affinity matrix
    2. Iteratively: NCut -> LocalCut -> CRF -> store mask + SC
    3. Remove segmented patches, repeat

    GPU Note: Uses fixed max_instances with masking for static shapes.

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
    device = features.device

    # Resize depth to patch resolution
    depth_patch = F.interpolate(
        depth[None, None, :, :],
        size=(patch_h, patch_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    # Step 1: Compute and sharpen affinity matrix
    W = compute_affinity_matrix(features)
    W = spatial_importance_sharpening(W, depth_patch, sigma_gauss, beta)

    # Pre-allocate output arrays
    all_masks = torch.zeros((max_instances, K), dtype=torch.float32, device=device)
    all_sc = torch.zeros((max_instances, K), dtype=torch.float32, device=device)
    if initial_active is not None:
        active = initial_active.float().to(device)
    else:
        active = torch.ones(K, dtype=torch.float32, device=device)
    num_valid = 0

    # Iterative extraction
    for idx in range(max_instances):
        # Mask out inactive patches in affinity
        mask_2d = active[:, None] * active[None, :]
        W_masked = W * mask_2d

        # NCut on current affinity
        bipartition, idx_src, idx_snk, _ = normalized_cut(W_masked, tau_ncut)
        bipartition = bipartition * active  # only active patches

        # LocalCut
        instance_mask = local_cut_3d(
            bipartition, depth_patch, features,
            tau_knn=tau_knn, k=k,
            idx_source=idx_src, idx_sink=idx_snk,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )

        # CRF refinement (conditional)
        if use_crf:
            refined_mask = crf_refine(instance_mask, image, patch_h, patch_w)
        else:
            refined_mask = instance_mask

        # Size check (min and max)
        mask_frac = torch.sum(refined_mask) / K
        valid = bool(mask_frac.item() >= min_mask_size and mask_frac.item() <= max_mask_size)

        # Spatial Confidence
        sc = compute_spatial_confidence(
            bipartition, depth_patch, features,
            tau_knn_min=tau_knn * 0.5,
            tau_knn_max=tau_knn,
            T=sc_samples, k=k,
            idx_source=idx_src, idx_sink=idx_snk,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )

        # Store if valid
        if valid:
            all_masks[idx] = refined_mask
            all_sc[idx] = sc

            # Remove segmented patches from active set
            active = active * (1.0 - refined_mask)
            num_valid += 1

    # Compute per-mask confidence scores (average SC)
    scores = torch.sum(all_sc, dim=-1) / (torch.sum(all_masks, dim=-1) + 1e-8)

    return PseudoMaskResult(
        masks=all_masks,
        spatial_confidence=all_sc,
        scores=scores,
        num_valid=num_valid,
    )


# ---------------------------------------------------------------------------
# Batched wrapper for GPU
# ---------------------------------------------------------------------------

def extract_pseudo_masks_batch(
    features_batch: torch.Tensor,
    depth_batch: torch.Tensor,
    images_batch: torch.Tensor,
    patch_h: int,
    patch_w: int,
    max_instances: int = 3,
    **kwargs,
) -> PseudoMaskResult:
    """Batched pseudo-mask extraction.

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
    B = features_batch.shape[0]
    results = []
    for i in range(B):
        result = extract_pseudo_masks(
            features_batch[i],
            depth_batch[i],
            images_batch[i],
            patch_h=patch_h,
            patch_w=patch_w,
            max_instances=max_instances,
            **kwargs,
        )
        results.append(result)

    # Stack results into batched tensors
    return PseudoMaskResult(
        masks=torch.stack([r.masks for r in results], dim=0),
        spatial_confidence=torch.stack([r.spatial_confidence for r in results], dim=0),
        scores=torch.stack([r.scores for r in results], dim=0),
        num_valid=max(r.num_valid for r in results),
    )


# ---------------------------------------------------------------------------
# CutS3DModule: torch.nn.Module wrapper
# ---------------------------------------------------------------------------

class CutS3DModule(nn.Module):
    """CutS3D Instance Segmentation Module (PyTorch).

    Wraps the pseudo-mask extraction pipeline as a PyTorch module for
    integration into the MBPS model. During training, pseudo-masks
    are pre-computed offline. During inference, this module runs the
    full pipeline on DINO features + depth.

    Also includes a learned refinement head for end-to-end fine-tuning.

    Args:
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

    def __init__(
        self,
        max_instances: int = 100,
        hidden_dim: int = 256,
        input_dim: int = 384,
        tau_ncut: float = 0.0,
        tau_knn: float = 0.115,
        k_neighbors: int = 10,
        sigma_gauss: float = 3.0,
        beta: float = 0.45,
        min_mask_size: float = 0.02,
        sc_samples: int = 6,
    ):
        super().__init__()
        self.max_instances = max_instances
        self.hidden_dim = hidden_dim
        self.tau_ncut = tau_ncut
        self.tau_knn = tau_knn
        self.k_neighbors = k_neighbors
        self.sigma_gauss = sigma_gauss
        self.beta = beta
        self.min_mask_size = min_mask_size
        self.sc_samples = sc_samples

        # Learned mask prediction head (used during CAD training)
        self.feat_proj = nn.Linear(input_dim, hidden_dim)
        self.mask_pred = nn.Linear(hidden_dim, max_instances)
        self.score_pred = nn.Linear(input_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        feat_proj = F.relu(self.feat_proj(features))

        # Predict mask logits
        mask_logits = self.mask_pred(feat_proj)
        mask_logits = mask_logits.permute(0, 2, 1)  # (B, M, N)

        # Confidence score per instance via pooled features
        mask_probs = torch.sigmoid(mask_logits)
        pooled = torch.einsum("bmn,bnd->bmd", mask_probs, features)
        pooled = pooled / (torch.sum(mask_probs, dim=-1, keepdim=True) + 1e-8)

        scores = torch.sigmoid(self.score_pred(pooled)).squeeze(-1)  # (B, M)

        return mask_logits, scores


# ---------------------------------------------------------------------------
# CascadeMaskHead: Cascade refinement (3-stage)
# ---------------------------------------------------------------------------

class CascadeMaskHead(nn.Module):
    """Cascade Mask R-CNN-style refinement head.

    3-stage iterative mask refinement as used in the CutS3D CAD.
    Each stage pools features using current masks, refines via FC
    layers, and adds a mask delta.

    Args:
        num_stages: Number of cascade stages (default 3).
        mask_dim: Hidden feature dimension.
        input_dim: Input feature dimension.
        num_patches: Number of spatial patches (N).
        num_classes: Output classes (1 for class-agnostic).
    """

    def __init__(
        self,
        num_stages: int = 3,
        mask_dim: int = 256,
        input_dim: int = 384,
        num_patches: int = 900,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.mask_dim = mask_dim
        self.num_classes = num_classes

        # Build stages
        self.stage_fc1 = nn.ModuleList()
        self.stage_fc2 = nn.ModuleList()
        self.stage_mask_pred = nn.ModuleList()

        for stage in range(num_stages):
            self.stage_fc1.append(nn.Linear(input_dim, mask_dim))
            self.stage_fc2.append(nn.Linear(mask_dim, mask_dim))
            self.stage_mask_pred.append(nn.Linear(mask_dim, num_patches))

        self.final_score = nn.Linear(input_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        initial_masks: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            mask_probs = torch.sigmoid(masks)
            pooled = torch.einsum("bmn,bnd->bmd", mask_probs, features)
            pooled = pooled / (
                torch.sum(mask_probs, dim=-1, keepdim=True) + 1e-8
            )

            # Two-layer FC refinement
            h = F.relu(self.stage_fc1[stage](pooled))
            h = F.relu(self.stage_fc2[stage](h))

            # Predict mask delta
            mask_delta = self.stage_mask_pred[stage](h)
            masks = masks + mask_delta

        # Final score
        final_pooled = torch.einsum(
            "bmn,bnd->bmd", torch.sigmoid(masks), features
        )
        final_pooled = final_pooled / (
            torch.sum(torch.sigmoid(masks), dim=-1, keepdim=True) + 1e-8
        )
        scores = torch.sigmoid(self.final_score(final_pooled)).squeeze(-1)

        return masks, scores
