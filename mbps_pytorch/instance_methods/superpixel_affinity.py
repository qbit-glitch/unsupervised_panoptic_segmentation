"""Superpixel-graph utilities for the lightweight instance-adapter ablation.

This file is a separate ablation path that references the existing MBPS
pipeline instead of modifying it:

* DCFA / k=80 semantics are loaded through
  :mod:`mbps_pytorch.adaptive_instance_semantics`.
* The initial pseudo-instance prior matches
  :func:`mbps_pytorch.convert_to_cups_format.build_instance_map_depth_cc`.
* Outputs are compatible with
  :mod:`mbps_pytorch.evaluate_cascade_pseudolabels`.

The core idea follows the Superpixels paper: train on reliable superpixel
regions and downweight ambiguous boundaries, while allowing frozen DINO/CLIP
features to veto noisy depth-connected components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.special import expit

THING_IDS = set(range(11, 19))
IGNORE_LABEL = 255


@dataclass
class SuperpixelGraph:
    """One image represented as adjacent-superpixel merge candidates."""

    superpixels: np.ndarray
    node_features: np.ndarray
    node_class: np.ndarray
    node_area: np.ndarray
    node_semantic_purity: np.ndarray
    node_instance_id: np.ndarray
    node_instance_purity: np.ndarray
    dino_node_features: np.ndarray
    clip_node_features: np.ndarray
    node_clip_confidence: np.ndarray
    node_proposal_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_targets: np.ndarray
    edge_weights: np.ndarray
    edge_is_hard: np.ndarray
    edge_soft_affinity: np.ndarray


@dataclass
class SuperpixelExtractionConfig:
    """Configuration for extracting superpixel edge samples."""

    n_segments: int = 900
    compactness: float = 12.0
    sigma: float = 0.8
    min_superpixel_area: int = 12
    pseudo_tau: float = 0.20
    pseudo_min_area: int = 1000
    pseudo_depth_sigma: float = 1.0
    pseudo_dilation: int = 3
    semantic_purity_min: float = 0.60
    instance_purity_min: float = 0.55
    positive_affinity_min: float = 0.45
    negative_affinity_max: float = 0.25
    negative_boundary_min: float = 0.08
    intra_instance_negative_affinity_max: float = 0.18
    intra_instance_negative_boundary_min: float = 0.10
    intra_instance_negative_weight: float = 0.75
    balance_hard_negatives: bool = True
    max_hard_pos_to_neg_ratio: float = 4.0
    class_aware_negative_classes: tuple[int, ...] = ()
    class_aware_positive_affinity_min: float | None = None
    class_aware_negative_weight: float = 1.0
    class_aware_positive_weight: float = 1.0
    class_aware_intra_instance_negative_affinity_max: float | None = None
    class_aware_intra_instance_negative_boundary_min: float | None = None
    hard_weight: float = 1.0
    soft_weight: float = 0.10
    cross_class_negative_weight: float = 0.05
    sigma_color: float = 0.08
    sigma_depth: float = 0.04
    dino_temperature: float = 0.20
    clip_temperature: float = 0.20
    proposal_objectness_enabled: bool = False
    proposal_objectness_top_k: int = 100
    proposal_objectness_min_score: float | None = None
    proposal_objectness_support_thresh: float = 0.25
    proposal_soft_affinity_weight: float = 0.0


def resize_nearest(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array with nearest-neighbor interpolation."""
    h, w = hw
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def resize_bilinear(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D float array with bilinear interpolation."""
    h, w = hw
    return np.array(
        Image.fromarray(arr.astype(np.float32)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to [0, 1] while tolerating constant maps."""
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=np.float32)
    lo = float(np.nanmin(depth[finite]))
    hi = float(np.nanmax(depth[finite]))
    if hi <= lo + 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=lo, posinf=hi, neginf=lo)
    return ((depth - lo) / (hi - lo)).astype(np.float32)


def generate_superpixels(
    image_rgb: np.ndarray,
    cfg: SuperpixelExtractionConfig,
) -> np.ndarray:
    """Generate SLIC superpixels, with a dependency-free grid fallback."""
    image_float = image_rgb.astype(np.float32) / 255.0
    try:
        from skimage.segmentation import slic

        labels = slic(
            image_float,
            n_segments=cfg.n_segments,
            compactness=cfg.compactness,
            sigma=cfg.sigma,
            start_label=0,
            convert2lab=True,
            enforce_connectivity=True,
        )
        return labels.astype(np.int32)
    except Exception:
        # Fallback keeps the ablation runnable on lean environments.  It is not
        # as boundary-aware as SLIC, but still gives a graph adapter smoke path.
        h, w = image_rgb.shape[:2]
        target_area = max((h * w) / max(cfg.n_segments, 1), 1.0)
        cell = max(int(np.sqrt(target_area)), 4)
        yy = np.arange(h)[:, None] // cell
        xx = np.arange(w)[None, :] // cell
        labels = yy * (int(np.ceil(w / cell)) + 1) + xx
        _, relabeled = np.unique(labels, return_inverse=True)
        return relabeled.reshape(h, w).astype(np.int32)


def rgb_to_lab_like(image_rgb: np.ndarray) -> np.ndarray:
    """Return Lab if scikit-image is available, otherwise normalized RGB."""
    image_float = image_rgb.astype(np.float32) / 255.0
    try:
        from skimage.color import rgb2lab

        lab = rgb2lab(image_float).astype(np.float32)
        # Roughly normalize to comparable numeric scale.
        lab[..., 0] = lab[..., 0] / 100.0
        lab[..., 1:] = (lab[..., 1:] + 128.0) / 255.0
        return lab
    except Exception:
        return image_float


def find_superpixel_edges(superpixels: np.ndarray) -> np.ndarray:
    """Return unique undirected adjacent-superpixel edges as ``(E, 2)``."""
    edges: set[tuple[int, int]] = set()
    right = superpixels[:, 1:] != superpixels[:, :-1]
    ys, xs = np.where(right)
    for y, x in zip(ys, xs):
        a = int(superpixels[y, x])
        b = int(superpixels[y, x + 1])
        if a != b:
            edges.add((min(a, b), max(a, b)))

    down = superpixels[1:, :] != superpixels[:-1, :]
    ys, xs = np.where(down)
    for y, x in zip(ys, xs):
        a = int(superpixels[y, x])
        b = int(superpixels[y + 1, x])
        if a != b:
            edges.add((min(a, b), max(a, b)))

    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64)


def majority_values(
    superpixels: np.ndarray,
    values: np.ndarray,
    num_nodes: int,
    ignore_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute majority value and purity per superpixel."""
    flat_sp = superpixels.reshape(-1)
    flat_val = values.reshape(-1).astype(np.int64)
    majority = np.full(num_nodes, ignore_value if ignore_value is not None else 0,
                       dtype=np.int64)
    purity = np.zeros(num_nodes, dtype=np.float32)
    for sid in range(num_nodes):
        vals = flat_val[flat_sp == sid]
        if ignore_value is not None:
            vals = vals[vals != ignore_value]
        if vals.size == 0:
            continue
        bincount = np.bincount(vals)
        maj = int(bincount.argmax())
        majority[sid] = maj
        purity[sid] = float(bincount[maj] / max(vals.size, 1))
    return majority, purity


def region_mean_from_grid(
    superpixels: np.ndarray,
    feature_grid: np.ndarray | None,
    num_nodes: int,
) -> np.ndarray:
    """Mean feature vector per superpixel from any ``(h, w, C)`` grid."""
    if feature_grid is None:
        return np.zeros((num_nodes, 0), dtype=np.float32)
    if feature_grid.ndim == 2:
        feature_grid = feature_grid.reshape(1, 1, -1)
    if feature_grid.ndim == 1:
        feature_grid = feature_grid.reshape(1, 1, -1)
    if feature_grid.ndim != 3:
        raise ValueError(f"Expected feature grid with 1/2/3 dims, got {feature_grid.shape}")

    gh, gw, dim = feature_grid.shape
    sp_small = resize_nearest(superpixels.astype(np.int32), (gh, gw))
    out = np.zeros((num_nodes, dim), dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)
    flat_sp = sp_small.reshape(-1)
    flat_feat = feature_grid.reshape(-1, dim).astype(np.float32)
    for sid in range(num_nodes):
        mask = flat_sp == sid
        if mask.any():
            out[sid] = flat_feat[mask].mean(axis=0)
            counts[sid] = float(mask.sum())
    empty = counts == 0
    if empty.any():
        # Superpixels that disappear at patch resolution get zero features.
        out[empty] = 0.0
    return out


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def project_rows(x: np.ndarray, projection: np.ndarray | None) -> np.ndarray:
    """Apply a saved random projection to row features, if provided."""
    if projection is None or x.shape[1] == 0:
        return x.astype(np.float32)
    if x.shape[1] != projection.shape[0]:
        raise ValueError(
            f"Feature dim {x.shape[1]} does not match projection "
            f"{projection.shape}"
        )
    return (x.astype(np.float32) @ projection.astype(np.float32)).astype(np.float32)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.ones(a.shape[0], dtype=np.float32)
    a_n = l2_normalize_rows(a)
    b_n = l2_normalize_rows(b)
    return np.sum(a_n * b_n, axis=1).astype(np.float32)


def build_depth_cc_prior(
    semantic_trainid: np.ndarray,
    depth: np.ndarray,
    cfg: SuperpixelExtractionConfig,
) -> np.ndarray:
    """Build the current tau/A_min depth-CC pseudo-instance prior."""
    from mbps_pytorch.convert_to_cups_format import build_instance_map_depth_cc

    return build_instance_map_depth_cc(
        semantic_trainid.astype(np.uint8),
        normalize_depth(depth),
        THING_IDS,
        min_area=cfg.pseudo_min_area,
        grad_threshold=cfg.pseudo_tau,
        depth_blur_sigma=cfg.pseudo_depth_sigma,
        dilation_iters=cfg.pseudo_dilation,
    ).astype(np.int32)


def depth_gradient(depth: np.ndarray) -> np.ndarray:
    """Sobel magnitude on normalized depth."""
    d = normalize_depth(depth)
    gx = ndimage.sobel(d, axis=1)
    gy = ndimage.sobel(d, axis=0)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _region_stats(superpixels: np.ndarray, value: np.ndarray, num_nodes: int) -> np.ndarray:
    flat_sp = superpixels.reshape(-1)
    flat_v = value.reshape(-1).astype(np.float32)
    out = np.zeros((num_nodes, 4), dtype=np.float32)
    for sid in range(num_nodes):
        vals = flat_v[flat_sp == sid]
        if vals.size == 0:
            continue
        out[sid, 0] = float(vals.mean())
        out[sid, 1] = float(vals.std())
        out[sid, 2] = float(vals.min())
        out[sid, 3] = float(vals.max())
    return out


def _region_mean_std(
    superpixels: np.ndarray,
    value: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    flat_sp = superpixels.reshape(-1)
    flat_v = value.reshape(-1, value.shape[-1]).astype(np.float32)
    dim = flat_v.shape[1]
    out = np.zeros((num_nodes, dim * 2), dtype=np.float32)
    for sid in range(num_nodes):
        vals = flat_v[flat_sp == sid]
        if vals.size == 0:
            continue
        out[sid, :dim] = vals.mean(axis=0)
        out[sid, dim:] = vals.std(axis=0)
    return out


def make_semantic_onehot(classes: np.ndarray, num_classes: int = 19) -> np.ndarray:
    out = np.zeros((len(classes), num_classes), dtype=np.float32)
    valid = (classes >= 0) & (classes < num_classes)
    out[np.where(valid)[0], classes[valid].astype(np.int64)] = 1.0
    return out


def compute_clip_confidence(
    clip_node_features: np.ndarray,
    node_class: np.ndarray,
    clip_prototypes: dict[int, np.ndarray] | None,
) -> np.ndarray:
    """UVIS-style prototype confidence for optional CLIP features."""
    if clip_node_features.shape[1] == 0 or not clip_prototypes:
        return np.ones(len(node_class), dtype=np.float32)

    feats = l2_normalize_rows(clip_node_features)
    conf = np.ones(len(node_class), dtype=np.float32)
    for cls in np.unique(node_class):
        cls = int(cls)
        if cls not in clip_prototypes:
            continue
        idx = np.where(node_class == cls)[0]
        if idx.size == 0:
            continue
        protos = l2_normalize_rows(clip_prototypes[cls].astype(np.float32))
        sim = feats[idx] @ protos.T
        conf[idx] = np.clip(sim.max(axis=1), 0.0, 1.0)
    return conf


def load_proposal_bank(
    path: str | Path | None,
    image_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load a proposal-mask NPZ from either Superpixels or MBPS formats."""
    if path is None or not Path(path).exists():
        return None, None
    with np.load(str(path), allow_pickle=False) as data:
        if "masks" in data:
            masks = data["masks"]
        elif "arr_0" in data:
            masks = data["arr_0"]
        else:
            raise ValueError(f"No masks array found in proposal bank {path}")
        if "scores" in data:
            scores = data["scores"].astype(np.float32)
        else:
            scores = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
        h = int(data["h_patches"]) if "h_patches" in data else None
        w = int(data["w_patches"]) if "w_patches" in data else None

    if masks.ndim == 2:
        if h is None or w is None:
            if image_hw is None:
                raise ValueError(
                    f"Flattened masks in {path} require h_patches/w_patches or image_hw")
            h, w = int(image_hw[0]), int(image_hw[1])
        masks = masks.reshape(masks.shape[0], h, w)
    elif masks.ndim != 3:
        raise ValueError(f"Unsupported proposal mask shape at {path}: {masks.shape}")
    return masks.astype(bool), scores[: masks.shape[0]].astype(np.float32)


def _resize_bool_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == hw:
        return mask.astype(bool)
    h, w = hw
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((w, h), Image.NEAREST)
    return np.asarray(image) > 0


def _normalize_proposal_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    finite = np.isfinite(scores)
    if not finite.any():
        return np.ones_like(scores, dtype=np.float32)
    clean = np.nan_to_num(scores, nan=float(scores[finite].min()))
    lo, hi = np.percentile(clean[finite], [5, 95])
    if hi <= lo + 1.0e-8:
        return np.ones_like(clean, dtype=np.float32)
    return np.clip((clean - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def proposal_objectness_features(
    superpixels: np.ndarray,
    node_area: np.ndarray,
    proposal_masks: np.ndarray | None,
    proposal_scores: np.ndarray | None,
    cfg: SuperpixelExtractionConfig,
) -> dict[str, np.ndarray]:
    """Project proposal-bank objectness/support onto superpixel nodes.

    The features are deliberately generic: they only use proposal masks and
    scores, so the same adapter path can consume RAMA, TokenCut, MaskCut,
    SAM-style banks, or their fusions.
    """
    num_nodes = len(node_area)
    node_features = np.zeros((num_nodes, 4), dtype=np.float32)
    top_proposal = np.full(num_nodes, -1, dtype=np.int32)
    if proposal_masks is None or proposal_scores is None or len(proposal_masks) == 0:
        return {
            "node_features": node_features,
            "top_proposal": top_proposal,
            "coverage": np.zeros((0, num_nodes), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    scores = np.asarray(proposal_scores, dtype=np.float32)
    valid = np.isfinite(scores)
    if cfg.proposal_objectness_min_score is not None:
        valid &= scores >= float(cfg.proposal_objectness_min_score)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return {
            "node_features": node_features,
            "top_proposal": top_proposal,
            "coverage": np.zeros((0, num_nodes), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
        }
    order = idx[np.argsort(-scores[idx])]
    top_k = max(int(cfg.proposal_objectness_top_k), 1)
    order = order[:top_k]
    masks = proposal_masks[order]
    scores = _normalize_proposal_scores(scores[order])

    h, w = superpixels.shape
    flat_sp = superpixels.reshape(-1)
    safe_area = np.maximum(node_area.astype(np.float32), 1.0)
    coverage = np.zeros((len(order), num_nodes), dtype=np.float32)
    for prop_idx, mask in enumerate(masks):
        mask = _resize_bool_mask(mask, (h, w)).reshape(-1)
        if not mask.any():
            continue
        covered = np.bincount(flat_sp[mask], minlength=num_nodes).astype(np.float32)
        coverage[prop_idx] = covered / safe_area

    weighted = coverage * scores[:, None]
    if weighted.size:
        best_idx = weighted.argmax(axis=0)
        best_value = weighted[best_idx, np.arange(num_nodes)]
        has_support = best_value > 0
        top_proposal[has_support] = best_idx[has_support].astype(np.int32)
        node_features[:, 0] = np.clip(best_value, 0.0, 1.0)
        node_features[:, 1] = np.clip(coverage.max(axis=0), 0.0, 1.0)
        node_features[:, 2] = np.clip(weighted.sum(axis=0), 0.0, 1.0)
        support = coverage >= float(cfg.proposal_objectness_support_thresh)
        node_features[:, 3] = support.sum(axis=0).astype(np.float32) / float(len(order))
    return {
        "node_features": node_features.astype(np.float32),
        "top_proposal": top_proposal,
        "coverage": coverage.astype(np.float32),
        "scores": scores.astype(np.float32),
    }


def proposal_edge_features(
    proposal_info: dict[str, np.ndarray],
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Build edge-level proposal support features for adjacent superpixels."""
    node_features = proposal_info["node_features"]
    top_proposal = proposal_info["top_proposal"]
    coverage = proposal_info["coverage"]
    scores = proposal_info["scores"]
    obj_u = node_features[u, 0]
    obj_v = node_features[v, 0]
    top_same = (
        (top_proposal[u] >= 0)
        & (top_proposal[u] == top_proposal[v])
    ).astype(np.float32)
    shared_strength = np.zeros_like(obj_u, dtype=np.float32)
    shared_coverage = np.zeros_like(obj_u, dtype=np.float32)
    if coverage.shape[0] > 0:
        shared = np.minimum(coverage[:, u], coverage[:, v])
        shared_coverage = shared.max(axis=0).astype(np.float32)
        shared_strength = (shared * scores[:, None]).max(axis=0).astype(np.float32)
    return np.stack([
        np.abs(obj_u - obj_v),
        np.minimum(obj_u, obj_v),
        np.maximum(obj_u, obj_v),
        top_same,
        np.clip(shared_strength, 0.0, 1.0),
        np.clip(shared_coverage, 0.0, 1.0),
    ], axis=1).astype(np.float32)


def build_superpixel_graph(
    image_rgb: np.ndarray,
    semantic_trainid: np.ndarray,
    depth: np.ndarray,
    dino_features: np.ndarray | None = None,
    clip_features: np.ndarray | None = None,
    proposal_masks: np.ndarray | None = None,
    proposal_scores: np.ndarray | None = None,
    pseudo_instance_map: np.ndarray | None = None,
    clip_prototypes: dict[int, np.ndarray] | None = None,
    dino_projection: np.ndarray | None = None,
    clip_projection: np.ndarray | None = None,
    cfg: SuperpixelExtractionConfig | None = None,
) -> SuperpixelGraph:
    """Extract a train/inference graph from one image."""
    cfg = cfg or SuperpixelExtractionConfig()
    h, w = image_rgb.shape[:2]
    if semantic_trainid.shape != (h, w):
        semantic_trainid = resize_nearest(semantic_trainid.astype(np.uint8), (h, w))
    if depth.shape != (h, w):
        depth = resize_bilinear(depth, (h, w))
    if pseudo_instance_map is None:
        pseudo_instance_map = build_depth_cc_prior(semantic_trainid, depth, cfg)
    elif pseudo_instance_map.shape != (h, w):
        pseudo_instance_map = resize_nearest(pseudo_instance_map.astype(np.int32), (h, w))

    superpixels = generate_superpixels(image_rgb, cfg)
    unique = np.unique(superpixels)
    remap = np.full(int(unique.max()) + 1, -1, dtype=np.int32)
    remap[unique] = np.arange(len(unique), dtype=np.int32)
    superpixels = remap[superpixels]
    num_nodes = int(superpixels.max()) + 1

    node_area = np.bincount(superpixels.reshape(-1), minlength=num_nodes).astype(np.float32)
    node_class, node_semantic_purity = majority_values(
        superpixels, semantic_trainid, num_nodes, ignore_value=IGNORE_LABEL)
    node_instance_id, node_instance_purity = majority_values(
        superpixels, pseudo_instance_map, num_nodes, ignore_value=0)

    rgb_stats = _region_mean_std(superpixels, image_rgb.astype(np.float32) / 255.0, num_nodes)
    lab = rgb_to_lab_like(image_rgb)
    lab_stats = _region_mean_std(superpixels, lab, num_nodes)
    depth_norm = normalize_depth(depth)
    depth_stats = _region_stats(superpixels, depth_norm, num_nodes)
    grad_mag = depth_gradient(depth)
    grad_stats = _region_stats(superpixels, grad_mag, num_nodes)[:, [0, 3]]
    sem_onehot = make_semantic_onehot(node_class.astype(np.int64), 19)
    geom = np.stack([
        node_area / float(h * w),
        node_semantic_purity,
        node_instance_purity,
        np.isin(node_class, list(THING_IDS)).astype(np.float32),
    ], axis=1).astype(np.float32)

    dino_node_raw = region_mean_from_grid(superpixels, dino_features, num_nodes)
    clip_node_raw = region_mean_from_grid(superpixels, clip_features, num_nodes)
    dino_node = l2_normalize_rows(project_rows(dino_node_raw, dino_projection))
    clip_node = l2_normalize_rows(project_rows(clip_node_raw, clip_projection))
    clip_conf = compute_clip_confidence(clip_node, node_class, clip_prototypes)[:, None]
    if cfg.proposal_objectness_enabled:
        proposal_info = proposal_objectness_features(
            superpixels,
            node_area,
            proposal_masks,
            proposal_scores,
            cfg,
        )
    else:
        proposal_info = {
            "node_features": np.zeros((num_nodes, 4), dtype=np.float32),
            "top_proposal": np.full(num_nodes, -1, dtype=np.int32),
            "coverage": np.zeros((0, num_nodes), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    node_feature_blocks = [
        rgb_stats, lab_stats, depth_stats, grad_stats, sem_onehot, geom,
        dino_node, clip_node, clip_conf,
    ]
    if cfg.proposal_objectness_enabled:
        node_feature_blocks.append(proposal_info["node_features"])
    node_features = np.concatenate(node_feature_blocks, axis=1).astype(np.float32)

    edge_index = find_superpixel_edges(superpixels)
    if edge_index.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        edge_extra_dim = 12 + (6 if cfg.proposal_objectness_enabled else 0)
        return SuperpixelGraph(
            superpixels=superpixels,
            node_features=node_features,
            node_class=node_class,
            node_area=node_area,
            node_semantic_purity=node_semantic_purity,
            node_instance_id=node_instance_id,
            node_instance_purity=node_instance_purity,
            dino_node_features=dino_node,
            clip_node_features=clip_node,
            node_clip_confidence=clip_conf[:, 0],
            node_proposal_features=proposal_info["node_features"],
            edge_index=edge_index,
            edge_features=np.zeros(
                (0, node_features.shape[1] * 3 + edge_extra_dim), dtype=np.float32),
            edge_targets=empty,
            edge_weights=empty,
            edge_is_hard=np.zeros((0,), dtype=bool),
            edge_soft_affinity=empty,
        )

    u = edge_index[:, 0]
    v = edge_index[:, 1]
    nf_u = node_features[u]
    nf_v = node_features[v]

    centroid_yx = np.zeros((num_nodes, 2), dtype=np.float32)
    ys, xs = np.indices((h, w))
    for sid in range(num_nodes):
        mask = superpixels == sid
        if mask.any():
            centroid_yx[sid, 0] = float(ys[mask].mean() / max(h - 1, 1))
            centroid_yx[sid, 1] = float(xs[mask].mean() / max(w - 1, 1))

    lab_mean = lab_stats[:, :3]
    depth_mean = depth_stats[:, 0]
    grad_mean = grad_stats[:, 0]
    dino_sim = cosine_rows(dino_node[u], dino_node[v])
    clip_sim = cosine_rows(clip_node[u], clip_node[v])
    lab_diff = np.linalg.norm(lab_mean[u] - lab_mean[v], axis=1)
    depth_diff = np.abs(depth_mean[u] - depth_mean[v])
    grad_pair = np.maximum(grad_mean[u], grad_mean[v])
    centroid_dist = np.linalg.norm(centroid_yx[u] - centroid_yx[v], axis=1)
    same_pseudo = (
        (node_instance_id[u] > 0)
        & (node_instance_id[u] == node_instance_id[v])
    ).astype(np.float32)
    different_pseudo = (
        (node_instance_id[u] > 0)
        & (node_instance_id[v] > 0)
        & (node_instance_id[u] != node_instance_id[v])
    ).astype(np.float32)
    same_class = (node_class[u] == node_class[v])
    min_sem_purity = np.minimum(node_semantic_purity[u], node_semantic_purity[v])
    min_inst_purity = np.minimum(node_instance_purity[u], node_instance_purity[v])
    min_clip_conf = np.minimum(clip_conf[u, 0], clip_conf[v, 0])
    area_ratio = np.minimum(node_area[u], node_area[v]) / np.maximum(
        np.maximum(node_area[u], node_area[v]), 1.0)

    color_aff = np.exp(-(lab_diff ** 2) / max(cfg.sigma_color, 1e-6))
    depth_aff = np.exp(-(depth_diff ** 2) / max(cfg.sigma_depth, 1e-6))
    dino_aff = expit(dino_sim / max(cfg.dino_temperature, 1e-6))
    clip_aff = expit(clip_sim / max(cfg.clip_temperature, 1e-6))
    has_dino = dino_node.shape[1] > 0
    has_clip = clip_node.shape[1] > 0
    if not has_dino:
        dino_aff = np.ones_like(color_aff)
    if not has_clip:
        clip_aff = np.ones_like(color_aff)
    soft_affinity = np.clip(color_aff * depth_aff * dino_aff * clip_aff, 0.0, 1.0)
    prop_edge_extra = None
    if cfg.proposal_objectness_enabled:
        prop_edge_extra = proposal_edge_features(proposal_info, u, v)
        prop_w = np.clip(float(cfg.proposal_soft_affinity_weight), 0.0, 1.0)
        if prop_w > 0.0:
            prop_aff = prop_edge_extra[:, 4]
            soft_affinity = np.clip(
                (1.0 - prop_w) * soft_affinity + prop_w * prop_aff,
                0.0,
                1.0,
            )

    edge_extra_blocks = [np.stack([
        lab_diff,
        depth_diff,
        grad_pair,
        dino_sim,
        clip_sim,
        same_pseudo,
        different_pseudo,
        centroid_dist,
        area_ratio,
        min_sem_purity,
        min_inst_purity,
        min_clip_conf,
    ], axis=1).astype(np.float32)]
    if prop_edge_extra is not None:
        edge_extra_blocks.append(prop_edge_extra)
    edge_extra = np.concatenate(edge_extra_blocks, axis=1).astype(np.float32)
    edge_features = np.concatenate([
        0.5 * (nf_u + nf_v),
        np.abs(nf_u - nf_v),
        nf_u * nf_v,
        edge_extra,
    ], axis=1).astype(np.float32)

    edge_targets = soft_affinity.astype(np.float32)
    edge_weights = np.full(edge_index.shape[0], cfg.soft_weight, dtype=np.float32)
    edge_is_hard = np.zeros(edge_index.shape[0], dtype=bool)

    both_thing = np.isin(node_class[u], list(THING_IDS)) & np.isin(node_class[v], list(THING_IDS))
    class_aware_classes = np.array(cfg.class_aware_negative_classes, dtype=np.int64)
    class_aware_edge = np.zeros_like(same_class, dtype=bool)
    if class_aware_classes.size > 0:
        class_aware_edge = same_class & np.isin(node_class[u], class_aware_classes)
    positive_affinity_min = np.full(
        edge_index.shape[0], cfg.positive_affinity_min, dtype=np.float32)
    if cfg.class_aware_positive_affinity_min is not None:
        positive_affinity_min[class_aware_edge] = float(
            cfg.class_aware_positive_affinity_min)
    reliable = (
        (min_sem_purity >= cfg.semantic_purity_min)
        & (min_inst_purity >= cfg.instance_purity_min)
        & both_thing
    )
    pos = (
        reliable
        & same_class
        & (same_pseudo > 0)
        & (soft_affinity >= positive_affinity_min)
    )
    neg = (
        reliable
        & same_class
        & (different_pseudo > 0)
        & (
            (soft_affinity <= cfg.negative_affinity_max)
            | (grad_pair >= cfg.negative_boundary_min)
        )
    )
    # Crucial for breaking the depth-only ceiling: allow a reliable cut label
    # inside a coarse pseudo-instance when superpixel evidence strongly rejects
    # a merge. Different-pseudo negatives can only teach merges to stay apart;
    # they cannot teach the adapter to split a bad merged teacher component.
    intra_neg = (
        reliable
        & same_class
        & (same_pseudo > 0)
        & (
            (soft_affinity <= cfg.intra_instance_negative_affinity_max)
            | (grad_pair >= cfg.intra_instance_negative_boundary_min)
        )
    )
    intra_neg &= ~pos
    if class_aware_classes.size > 0:
        cls_affinity_max = (
            cfg.intra_instance_negative_affinity_max
            if cfg.class_aware_intra_instance_negative_affinity_max is None
            else cfg.class_aware_intra_instance_negative_affinity_max
        )
        cls_boundary_min = (
            cfg.intra_instance_negative_boundary_min
            if cfg.class_aware_intra_instance_negative_boundary_min is None
            else cfg.class_aware_intra_instance_negative_boundary_min
        )
        class_intra_neg = (
            reliable
            & class_aware_edge
            & (same_pseudo > 0)
            & (
                (soft_affinity <= cls_affinity_max)
                | (grad_pair >= cls_boundary_min)
            )
        )
        intra_neg |= class_intra_neg & ~pos
    cross = reliable & (~same_class) & (cfg.cross_class_negative_weight > 0)

    edge_targets[pos] = 1.0
    edge_targets[neg | intra_neg | cross] = 0.0
    edge_is_hard[pos | neg | intra_neg | cross] = True
    edge_weights[pos | neg] = cfg.hard_weight
    edge_weights[intra_neg] = cfg.intra_instance_negative_weight
    edge_weights[cross] = cfg.cross_class_negative_weight
    if class_aware_classes.size > 0:
        class_pos = pos & class_aware_edge
        class_neg = (neg | intra_neg) & class_aware_edge
        edge_weights[class_pos] *= max(float(cfg.class_aware_positive_weight), 0.0)
        edge_weights[class_neg] *= max(float(cfg.class_aware_negative_weight), 0.0)

    if cfg.balance_hard_negatives:
        hard_pos = pos
        hard_neg = neg | intra_neg
        n_pos = int(hard_pos.sum())
        n_neg = int(hard_neg.sum())
        if n_pos > 0 and n_neg > 0:
            target_neg = max(float(n_pos) / max(cfg.max_hard_pos_to_neg_ratio, 1.0), 1.0)
            scale = min(max(target_neg / float(n_neg), 1.0), 8.0)
            edge_weights[hard_neg] *= scale
    edge_weights *= np.clip(min_clip_conf, 0.05, 1.0)

    small_node = (
        (node_area[u] < cfg.min_superpixel_area)
        | (node_area[v] < cfg.min_superpixel_area)
    )
    edge_weights[small_node] = 0.0

    return SuperpixelGraph(
        superpixels=superpixels,
        node_features=node_features,
        node_class=node_class,
        node_area=node_area,
        node_semantic_purity=node_semantic_purity,
        node_instance_id=node_instance_id,
        node_instance_purity=node_instance_purity,
        dino_node_features=dino_node,
        clip_node_features=clip_node,
        node_clip_confidence=clip_conf[:, 0],
        node_proposal_features=proposal_info["node_features"],
        edge_index=edge_index,
        edge_features=edge_features,
        edge_targets=edge_targets,
        edge_weights=edge_weights,
        edge_is_hard=edge_is_hard,
        edge_soft_affinity=soft_affinity.astype(np.float32),
    )


def instances_from_edge_probs(
    graph: SuperpixelGraph,
    edge_probs: np.ndarray,
    semantic_trainid: np.ndarray,
    merge_threshold: float = 0.55,
    min_area: int = 1000,
    class_min_area: dict[int, int] | None = None,
    thing_ids: Iterable[int] = THING_IDS,
) -> list[tuple[np.ndarray, int, float]]:
    """Recover instance masks from predicted superpixel merge probabilities."""
    thing_ids = set(int(x) for x in thing_ids)
    num_nodes = graph.node_features.shape[0]
    parent = np.arange(num_nodes, dtype=np.int32)
    rank = np.zeros(num_nodes, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for (u, v), prob in zip(graph.edge_index, edge_probs):
        cls_u = int(graph.node_class[u])
        cls_v = int(graph.node_class[v])
        if cls_u != cls_v or cls_u not in thing_ids:
            continue
        if float(prob) >= merge_threshold:
            union(int(u), int(v))

    groups: dict[tuple[int, int], list[int]] = {}
    for sid in range(num_nodes):
        cls = int(graph.node_class[sid])
        if cls not in thing_ids:
            continue
        root = find(sid)
        groups.setdefault((root, cls), []).append(sid)

    instances: list[tuple[np.ndarray, int, float]] = []
    for (_, cls), members in groups.items():
        member_mask = np.isin(graph.superpixels, np.array(members, dtype=np.int32))
        cls_mask = semantic_trainid == cls
        mask = member_mask & cls_mask
        area = int(mask.sum())
        cls_min_area = min_area
        if class_min_area is not None:
            cls_min_area = int(class_min_area.get(int(cls), min_area))
        if area < cls_min_area:
            continue
        instances.append((mask, cls, float(area)))

    instances.sort(key=lambda item: -item[2])
    if instances:
        max_area = max(instances[0][2], 1.0)
        instances = [(m, c, s / max_area) for m, c, s in instances]
    return instances


def save_instances_npz(
    instances: list[tuple[np.ndarray, int, float]],
    output_path: str | Path,
    h: int,
    w: int,
) -> None:
    """Save instance masks in the repo's standard NPZ + PNG sidecar format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not instances:
        np.savez_compressed(
            str(output_path),
            masks=np.zeros((0, h * w), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int32),
            num_valid=0,
            h_patches=h,
            w_patches=w,
        )
        Image.fromarray(np.zeros((h, w), dtype=np.uint16)).save(
            str(output_path).replace(".npz", "_instance.png"))
        return

    masks = np.zeros((len(instances), h * w), dtype=bool)
    scores = np.zeros((len(instances),), dtype=np.float32)
    class_ids = np.zeros((len(instances),), dtype=np.int32)
    vis = np.zeros((h, w), dtype=np.uint16)
    for idx, (mask, cls, score) in enumerate(instances):
        masks[idx] = mask.reshape(-1)
        scores[idx] = float(score)
        class_ids[idx] = int(cls)
        vis[mask] = idx + 1

    np.savez_compressed(
        str(output_path),
        masks=masks,
        scores=scores,
        class_ids=class_ids,
        num_valid=len(instances),
        h_patches=h,
        w_patches=w,
    )
    Image.fromarray(vis).save(str(output_path).replace(".npz", "_instance.png"))


def load_feature_grid(path: str | Path | None) -> np.ndarray | None:
    """Load a feature grid from common MBPS cache shapes."""
    if path is None or not Path(path).exists():
        return None
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    if arr.ndim == 2:
        # DINO cache convention: (N, C), usually 32*64 patches.
        n, c = arr.shape
        for gh, gw in ((32, 64), (64, 128), (128, 256), (16, 32)):
            if gh * gw == n:
                return arr.reshape(gh, gw, c)
        return arr.reshape(1, n, c)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported feature array shape at {path}: {arr.shape}")
