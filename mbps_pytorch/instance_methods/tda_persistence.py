"""Method 1: TDA/Persistent Homology instance decomposition.

Uses watershed oversegmentation + persistence-guided merging. The persistence
of a boundary between two basins is defined as the depth difference at the
boundary saddle point. Boundaries with persistence < tau_persist are merged,
keeping only topologically significant structures.

This is equivalent to computing 0-dimensional persistent homology on the depth
field, but implemented efficiently via region adjacency graph + merge.

If gudhi is available, also supports direct cubical complex persistence for
comparison.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.morphology import local_minima
try:
    from skimage import graph as rag_module
except ImportError:
    from skimage.future import graph as rag_module

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def _persistence_guided_watershed(depth, depth_blur_sigma=1.0,
                                  tau_persist=0.05,
                                  filtration_mode="depth_direct"):
    """Compute persistence-guided watershed decomposition.

    1. Initial watershed oversegmentation (no h-minima suppression).
    2. Build region adjacency graph weighted by boundary depth.
    3. Iteratively merge regions whose boundary persistence < tau_persist.

    Args:
        depth: (H,W) float32 [0,1].
        depth_blur_sigma: Gaussian blur sigma.
        tau_persist: persistence threshold. Boundaries weaker than this are merged.
        filtration_mode: "depth_direct" or "gradient_mag" — what scalar field
            to build the filtration on.

    Returns:
        basin_map: (H,W) int32 merged basin labels.
        n_basins: number of final basins.
    """
    # Smooth depth
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64),
                                       sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    # Choose filtration field
    if filtration_mode == "gradient_mag":
        from scipy.ndimage import sobel
        gx = sobel(depth_smooth, axis=1)
        gy = sobel(depth_smooth, axis=0)
        field = np.sqrt(gx ** 2 + gy ** 2)
    else:
        field = depth_smooth

    # Fine watershed oversegmentation (no suppression)
    minima = local_minima(field)
    markers, n_markers = ndimage.label(minima)
    if n_markers == 0:
        return np.ones(depth.shape, dtype=np.int32), 1

    basin_map = watershed(field, markers=markers)

    # Compute per-basin minimum field values efficiently
    basin_ids = np.unique(basin_map)
    basin_min = {}
    for bid in basin_ids:
        basin_min[bid] = float(field[basin_map == bid].min())

    # Build boundary info by scanning 4-connected neighbors
    # For each pair of adjacent basins, collect boundary field values
    H, W = field.shape
    edge_boundary_vals = {}  # (min_id, max_id) -> list of field values

    # Horizontal neighbors
    h_diff = basin_map[:, :-1] != basin_map[:, 1:]
    hy, hx = np.where(h_diff)
    for y, x in zip(hy, hx):
        u, v = int(basin_map[y, x]), int(basin_map[y, x + 1])
        key = (min(u, v), max(u, v))
        val = max(field[y, x], field[y, x + 1])
        edge_boundary_vals.setdefault(key, []).append(val)

    # Vertical neighbors
    v_diff = basin_map[:-1, :] != basin_map[1:, :]
    vy, vx = np.where(v_diff)
    for y, x in zip(vy, vx):
        u, v = int(basin_map[y, x]), int(basin_map[y + 1, x])
        key = (min(u, v), max(u, v))
        val = max(field[y, x], field[y + 1, x])
        edge_boundary_vals.setdefault(key, []).append(val)

    # Build edge list with persistence weights
    import heapq
    edge_heap = []  # min-heap of (persistence, u, v)
    for (u, v), vals in edge_boundary_vals.items():
        boundary_val = np.mean(vals)
        persistence = boundary_val - max(basin_min.get(u, 0), basin_min.get(v, 0))
        persistence = max(persistence, 0)
        heapq.heappush(edge_heap, (persistence, u, v))

    # Union-Find for efficient merging
    parent = {bid: bid for bid in basin_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Hierarchical merge: merge edges below tau_persist
    while edge_heap:
        persistence, u, v = heapq.heappop(edge_heap)
        if persistence >= tau_persist:
            break
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru

    # Build merged map using union-find
    merged_map = basin_map.copy()
    for bid in basin_ids:
        root = find(bid)
        if root != bid:
            merged_map[merged_map == bid] = root

    # Relabel sequentially
    unique_labels = np.unique(merged_map)
    relabel = {old: new for new, old in enumerate(unique_labels)}
    relabeled = np.vectorize(relabel.get)(merged_map).astype(np.int32)

    return relabeled, len(unique_labels)


def _min_weight(graph, src, dst, n):
    """Weight function for RAG merge: keep minimum edge weight."""
    return {"weight": min(
        graph[src][n].get("weight", float("inf")),
        graph[dst][n].get("weight", float("inf")),
    )}


def tda_instances(semantic, depth, thing_ids=THING_IDS,
                  tau_persist=0.05, min_area=1000,
                  dilation_iters=3, depth_blur_sigma=1.0,
                  filtration_mode="depth_direct",
                  features=None):
    """TDA/persistence-guided instance decomposition.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 depth [0,1].
        thing_ids: set of thing class trainIDs.
        tau_persist: persistence threshold for boundary merging.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur sigma.
        filtration_mode: "depth_direct" or "gradient_mag".
        features: unused (kept for interface consistency).

    Returns:
        List of (mask, class_id, score).
    """
    # Compute persistence-guided basins
    basin_map, n_basins = _persistence_guided_watershed(
        depth,
        depth_blur_sigma=depth_blur_sigma,
        tau_persist=tau_persist,
        filtration_mode=filtration_mode,
    )

    # Extract instances: intersect basins with thing-class semantic masks
    raw_instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        cls_basins = np.unique(basin_map[cls_mask])
        for basin_id in cls_basins:
            if basin_id == 0:
                continue
            instance_mask = (basin_map == basin_id) & cls_mask
            area = float(instance_mask.sum())
            if area >= min_area:
                raw_instances.append((instance_mask, cls, area))

    return dilation_reclaim(raw_instances, semantic, thing_ids,
                            min_area=min_area,
                            dilation_iters=dilation_iters)
