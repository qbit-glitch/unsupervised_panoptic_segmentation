"""RAMA MultiCut coarse masks for Stage-1 SGM supervision.

Builds a 4-neighbour patch-affinity graph over frozen DINOv2 patch features,
runs RAMA MultiCut on GPU (Pawel Swoboda et al., 2022), upsamples the
partition to image resolution, then assigns each region to a Cityscapes thing
trainID via majority vote over the existing pseudo-semantic map.

Replaces ``depth_guided_instances`` as the coarse-mask supervision source for
the SGM adapter when the user wants the paper's RAMA path (Hoang 2025) in
place of the depth-CC fallback.

Requires ``rama_py`` built against CUDA — install via the project recipe in
``scripts/remote_rama_santosh.sh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import rama_py
    _HAS_RAMA = True
except ImportError:
    _HAS_RAMA = False


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def _patch_grid_edges(h_p: int, w_p: int) -> Tuple[np.ndarray, np.ndarray]:
    """4-neighbour edge list (i, j) over an H_p x W_p patch grid."""
    n = h_p * w_p
    idx = np.arange(n, dtype=np.int32).reshape(h_p, w_p)
    horiz_a = idx[:, :-1].ravel()
    horiz_b = idx[:, 1:].ravel()
    vert_a = idx[:-1, :].ravel()
    vert_b = idx[1:, :].ravel()
    i = np.concatenate([horiz_a, vert_a]).astype(np.int32)
    j = np.concatenate([horiz_b, vert_b]).astype(np.int32)
    return i, j


def patch_affinity_costs(
    dino_patches: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RAMA edge tuple (i, j, c) with c = cos_sim - threshold.

    Positive c encourages merging; negative encourages splitting.

    Args:
        dino_patches: ``(H_p, W_p, D)`` float patch features.
        threshold: cosine-similarity threshold in ``[-1, 1]``.

    Returns:
        Three ``(E,)`` arrays of dtypes ``int32, int32, float32``.
    """
    h_p, w_p, d = dino_patches.shape
    flat = _l2_normalize_rows(dino_patches.reshape(h_p * w_p, d)).astype(np.float32)
    i, j = _patch_grid_edges(h_p, w_p)
    cos = (flat[i] * flat[j]).sum(axis=-1).astype(np.float32)
    return i, j, cos - float(threshold)


def rama_partition(
    dino_patches: np.ndarray,
    threshold: float = 0.5,
    solver: str = "PD",
) -> np.ndarray:
    """Run RAMA on patch-affinity graph; return per-patch labels.

    Args:
        dino_patches: ``(H_p, W_p, D)`` float DINO features.
        threshold: cosine-similarity threshold.
        solver: RAMA solver mode, e.g. ``"PD"``.

    Returns:
        ``(H_p, W_p)`` int32 label map in ``[0, K)``.
    """
    if not _HAS_RAMA:
        raise ImportError(
            "rama_py not available. Build via scripts/remote_rama_santosh.sh "
            "or `pip install .` against pawelswoboda/RAMA."
        )
    h_p, w_p, _ = dino_patches.shape
    i, j, c = patch_affinity_costs(dino_patches, threshold)
    opts = rama_py.multicut_solver_options(solver)
    # rama_py.rama_cuda returns (node_labels, runtime_info, ...). The first
    # element is always a flat per-node int array of length |V|.
    out = rama_py.rama_cuda(i, j, c, opts)
    node_labels = np.asarray(out[0], dtype=np.int32)
    return node_labels.reshape(h_p, w_p)


def upsample_label_map(
    label_map: np.ndarray, out_hw: Tuple[int, int]
) -> np.ndarray:
    """Nearest-neighbour upsample int label map to ``(H, W)``."""
    t = torch.from_numpy(label_map.astype(np.int64))[None, None].float()
    up = F.interpolate(t, size=out_hw, mode="nearest").squeeze().long()
    return up.numpy().astype(np.int32)


def regions_to_per_class_masks(
    region_labels: np.ndarray,
    semantic_trainid: np.ndarray,
    thing_ids: Sequence[int],
    min_area: int = 1000,
    void_label: int = 255,
) -> np.ndarray:
    """Group RAMA regions by majority trainID and emit per-thing-class masks.

    Args:
        region_labels: ``(H, W)`` int region IDs from :func:`rama_partition`
            after upsampling.
        semantic_trainid: ``(H, W)`` uint8 trainID map (already remapped from
            k=80 clusters by the caller via ``cluster_to_class[semantic]``).
        thing_ids: ordered Cityscapes thing trainIDs, e.g. ``(11, ..., 18)``.
        min_area: minimum pixel area for a region to be kept.
        void_label: value treated as ignore in ``semantic_trainid``.

    Returns:
        ``(T, H, W)`` uint8 binary masks where ``T = len(thing_ids)``.
    """
    h, w = region_labels.shape
    out = np.zeros((len(thing_ids), h, w), dtype=np.uint8)
    thing_index = {t: idx for idx, t in enumerate(thing_ids)}
    for r in np.unique(region_labels):
        mask = region_labels == r
        area = int(mask.sum())
        if area < min_area:
            continue
        ids = semantic_trainid[mask]
        ids = ids[ids != void_label]
        if ids.size == 0:
            continue
        counts = np.bincount(ids, minlength=void_label + 1)
        majority = int(counts.argmax())
        if majority in thing_index:
            out[thing_index[majority]] |= mask.astype(np.uint8)
    return out


def compute_or_load_coarse_masks(
    dino_path: Path,
    semantic_path: Path,
    cluster_to_class: np.ndarray,
    thing_ids: Sequence[int],
    cache_path: Path,
    out_hw: Tuple[int, int],
    threshold: float = 0.5,
    min_area: int = 1000,
) -> np.ndarray:
    """Cache-aware helper used by the trainer dataset.

    The cache file stores ``(T, H, W)`` bool packed into uint8.

    Args:
        dino_path: ``.npy`` patch features.
        semantic_path: ``.png`` cluster-ID map (0..K-1 or 255).
        cluster_to_class: ``(256,)`` LUT cluster -> trainID.
        thing_ids: ordered Cityscapes thing trainIDs.
        cache_path: output cache; written if missing, read if present.
        out_hw: target ``(H, W)`` for the masks.
        threshold: cosine similarity threshold for RAMA.
        min_area: minimum region area in pixels.
    """
    if cache_path.exists():
        return np.load(cache_path).astype(np.uint8)

    dino = np.load(dino_path).astype(np.float32)
    if dino.ndim == 2:
        n = dino.shape[0]
        side = int(np.sqrt(n))
        if side * side == n:
            dino = dino.reshape(side, side, -1)
        else:
            h_p = 32
            w_p = n // h_p
            dino = dino.reshape(h_p, w_p, -1)

    label_map_patch = rama_partition(dino, threshold=threshold)
    label_map_full = upsample_label_map(label_map_patch, out_hw)

    semantic = np.array(Image.open(semantic_path))
    if semantic.shape != out_hw:
        semantic = np.array(
            Image.fromarray(semantic).resize((out_hw[1], out_hw[0]), Image.NEAREST)
        )
    semantic_trainid = cluster_to_class[semantic].astype(np.uint8)

    masks = regions_to_per_class_masks(
        label_map_full, semantic_trainid, thing_ids, min_area=min_area,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, masks.astype(np.uint8))
    return masks


__all__ = [
    "_HAS_RAMA",
    "patch_affinity_costs",
    "rama_partition",
    "upsample_label_map",
    "regions_to_per_class_masks",
    "compute_or_load_coarse_masks",
]
