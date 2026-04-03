#!/usr/bin/env python3
"""MMGD-Cut: Multi-Modal Graph-Diffused K-way Normalized Cut.

Novel unsupervised semantic segmentation combining:
1. Multi-modal feature fusion (DINOv2/v3 + Stable Diffusion)
2. Graph diffusion for affinity denoising (PPR / lazy random walk)
3. Falcon-style K-way NCut optimization

Post-processing options (Rounds 4–6):
- NAMR: Nonlinear Adaptive Mask Refinement (φ(x)=x+1.5·ELU(x), temperature avg)
- Adaptive K: per-image K selection via Laplacian eigengap heuristic
- Multi-scale: coarse 32×32 semantics + fine 64×64 boundary consensus

Usage:
    # Multi-modal with graph diffusion (novel model)
    python mmgd_cut.py --coco_root /path/to/coco --device mps \
        --sources dinov3 sd --diffusion ppr --diff_steps 3

    # Single-modal baseline (matches Falcon)
    python mmgd_cut.py --coco_root /path/to/coco --device mps \
        --sources sd --diffusion none

    # Multi-modal with SSD-1B (best expected)
    python mmgd_cut.py --coco_root /path/to/coco --device mps \
        --sources dinov3 ssd1b --diffusion ppr --diff_steps 3 --step 10

    # With NAMR post-processing (Round 4)
    python mmgd_cut.py --coco_root /path/to/coco --device mps \
        --sources dinov3 ssd1b --diffusion none --namr

    # With adaptive K (Round 5)
    python mmgd_cut.py --coco_root /path/to/coco --device mps \
        --sources dinov3 ssd1b --diffusion none --adaptive_k

    # With multi-scale NCut (Round 6)
    python mmgd_cut.py --coco_root /path/to/coco --device mps \
        --sources dinov3 ssd1b --diffusion none --multiscale
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from diffcut_pseudo_semantics import (
    COCOSTUFF27_CLASSNAMES,
    NUM_CLASSES,
    STUFF_IDS,
    THING_IDS,
    hungarian_miou,
    load_coco_panoptic_gt,
)
from falcon_pseudo_semantics import FalconKwayCut

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Graph Diffusion
# ═══════════════════════════════════════════════════════════════════════


class GraphDiffusion:
    """Graph diffusion for affinity matrix denoising.

    Propagates affinity through the graph via random walks, strengthening
    within-cluster connections and weakening between-cluster connections.
    Applied after the power transform and before diagonal regularization.

    Methods:
        ppr: Truncated Personalized PageRank — S = α Σ (1-α)^k P^k
        lazy_rw: Lazy random walk — W_{t+1} = 0.5(W_t + P·W_t)
    """

    def __init__(
        self,
        method: str = "ppr",
        n_steps: int = 3,
        alpha: float = 0.85,
        device: str = "mps",
    ) -> None:
        self.method = method
        self.n_steps = n_steps
        self.alpha = alpha
        self.device = torch.device(device)

    @torch.no_grad()
    def diffuse(self, W: torch.Tensor) -> torch.Tensor:
        """Apply graph diffusion to affinity matrix W (N, N)."""
        if self.method == "none":
            return W
        if self.method == "ppr":
            return self._ppr(W)
        if self.method == "lazy_rw":
            return self._lazy_random_walk(W)
        raise ValueError(f"Unknown diffusion method: {self.method}")

    def _ppr(self, W: torch.Tensor) -> torch.Tensor:
        """Truncated Personalized PageRank diffusion."""
        N = W.shape[0]
        D_inv = 1.0 / (W.sum(dim=1) + 1e-8)
        P = D_inv.unsqueeze(1) * W  # row-stochastic transition matrix

        a = self.alpha
        I = torch.eye(N, device=self.device, dtype=W.dtype)
        S = a * I
        P_power = I.clone()

        for k in range(1, self.n_steps + 1):
            P_power = P_power @ P
            S = S + a * ((1 - a) ** k) * P_power

        return (S + S.T) / 2  # symmetrize

    def _lazy_random_walk(self, W: torch.Tensor) -> torch.Tensor:
        """Lazy random walk diffusion."""
        W_diff = W.clone()
        for _ in range(self.n_steps):
            D_inv = 1.0 / (W_diff.sum(dim=1) + 1e-8)
            P = D_inv.unsqueeze(1) * W_diff
            W_diff = 0.5 * (W_diff + P @ W_diff)
        return (W_diff + W_diff.T) / 2


# ═══════════════════════════════════════════════════════════════════════
# Multi-Modal Falcon
# ═══════════════════════════════════════════════════════════════════════


class MultiModalFalcon(FalconKwayCut):
    """Falcon K-way NCut with graph diffusion on affinity or features.

    Supports two diffusion modes:
    - "affinity": Diffuse the normalized cosine similarity BEFORE the power
      transform. This propagates raw similarity through the graph.
    - "feature": Propagate features through the initial affinity graph
      (graph neural network message passing), then rebuild the affinity.
      This is more effective because it directly smooths representations.
    """

    def __init__(
        self,
        device: str = "mps",
        diffusion: Optional[GraphDiffusion] = None,
        diff_mode: str = "feature",
    ) -> None:
        super().__init__(device)
        self.diffusion = diffusion
        self.diff_mode = diff_mode

    def build_affinity(
        self,
        features: torch.Tensor,
        alpha: float = 4.5,
        reg_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build affinity with optional graph diffusion.

        Pipeline depends on diff_mode:
          feature:  features → graph propagation → cosine → power → reg
          affinity: cosine → normalize → diffuse → power → reg
        """
        if self.diffusion is not None and self.diff_mode == "feature":
            features = self._feature_diffusion(features)

        W_raw = features @ features.T
        w_min, w_max = W_raw.min(), W_raw.max()
        W_scaled = (W_raw - w_min) / (w_max - w_min + 1e-8)

        if self.diffusion is not None and self.diff_mode == "affinity":
            W_scaled = self.diffusion.diffuse(W_scaled)
            W_scaled = W_scaled.clamp(min=0)

        W = W_scaled.pow(alpha)

        if reg_lambda > 0:
            D_pre = W.sum(dim=1)
            W = W + reg_lambda * torch.diag(D_pre)

        return W, W.sum(dim=1)

    @torch.no_grad()
    def _feature_diffusion(self, features: torch.Tensor) -> torch.Tensor:
        """Propagate features through the initial affinity graph.

        Implements lazy graph message passing:
          F' = 0.5 (F + P @ F) where P = D^{-1} A_init

        This smooths features based on their neighborhood structure,
        making within-cluster features more coherent.
        """
        W_init = features @ features.T
        w_min, w_max = W_init.min(), W_init.max()
        W_norm = ((W_init - w_min) / (w_max - w_min + 1e-8)).clamp(min=0)

        D_inv = 1.0 / (W_norm.sum(dim=1) + 1e-8)
        P = D_inv.unsqueeze(1) * W_norm  # row-stochastic

        for _ in range(self.diffusion.n_steps):
            features = 0.5 * (features + P @ features)

        return F.normalize(features, p=2, dim=1)


# ═══════════════════════════════════════════════════════════════════════
# NAMR Post-Processing (Round 4)
# ═══════════════════════════════════════════════════════════════════════


class NAMRPostProcessor:
    """Nonlinear Adaptive Mask Refinement for NCut label maps.

    Implements NAMR as used in Falcon (ICLR 2026): a bilateral label
    diffusion with nonlinear activation φ(x) = x + 1.5·ELU(x) applied
    at multiple temperatures T, then averaged.

    Unlike standard PAMR (which hurts NCut outputs), NAMR's nonlinear
    activation suppresses dissimilar-pixel cross-talk while amplifying
    coherent label diffusion within uniform regions.

    Algorithm:
        1. For each image pixel i, build soft label from local RGB neighborhood.
        2. Affinity: cos_sim(rgb_i, rgb_j) on mean-centered unit-norm pixels.
        3. Weight: W(i,j,T) = φ(cos_sim / T), φ(x) = x + 1.5·ELU(x), clipped ≥ 0.
        4. Soft label: Q[i] = Σ_j W(i,j) · P_hard[j] / Σ_j W(i,j).
        5. Average Q over temperatures, take argmax.
    """

    def __init__(
        self,
        temperatures: Optional[List[float]] = None,
        window_size: int = 2,
        n_classes: int = 27,
        device: str = "mps",
    ) -> None:
        self.temperatures = temperatures or [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
        self.window_size = window_size  # half-window; full window = 2*ws+1
        self.n_classes = n_classes
        self.device = torch.device(device)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        """Nonlinear activation: φ(x) = x + 1.5·ELU(x).

        For x > 0: φ(x) = 2.5x  (amplifies similar-pixel weights).
        For x → −∞: φ(x) → 0    (suppresses dissimilar-pixel weights).
        """
        return x + 1.5 * F.elu(x)

    @torch.no_grad()
    def refine(self, pred: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Refine a discrete segmentation map via NAMR.

        Args:
            pred:  (H, W) int array with labels [0, n_classes-1]; 255=unknown.
            image: (H_orig, W_orig, 3) uint8 RGB image (any resolution).

        Returns:
            Refined (H, W) int array with same label vocabulary.
        """
        H, W = pred.shape
        ws = self.window_size
        K_w = 2 * ws + 1  # full window side length

        # --- Resize image to (H, W) and compute L2-normalised features ---
        img_resized = np.array(
            Image.fromarray(image).resize((W, H), Image.BILINEAR)
        ).astype(np.float32)  # (H, W, 3) in [0, 255]
        img_t = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )  # (1, 3, H, W)
        # Mean-centre so that cosine similarity captures relative colour
        img_t = img_t - img_t.mean(dim=[2, 3], keepdim=True)
        img_norm = F.normalize(img_t, p=2, dim=1)  # (1, 3, H, W)

        # --- One-hot encode labels ---
        P = torch.zeros(1, self.n_classes, H, W, device=self.device)
        for c in range(self.n_classes):
            P[0, c] = torch.from_numpy(
                (pred == c).astype(np.float32)
            ).to(self.device)

        # --- Extract patches via F.unfold (vectorised over all neighbours) ---
        pad = ws
        img_padded = F.pad(img_norm, (pad, pad, pad, pad), mode="reflect")
        # (1, 3*K_w², H*W)
        img_patches_flat = F.unfold(img_padded, kernel_size=K_w, stride=1)
        # (1, 3, K_w², H, W)
        img_patches = img_patches_flat.reshape(1, 3, K_w * K_w, H, W)

        # Cosine similarity between anchor and each neighbour: (1, K_w², H, W)
        img_anchor = img_norm.unsqueeze(2)  # (1, 3, 1, H, W)
        cos_sim = (img_anchor * img_patches).sum(dim=1)

        # Label patches: (1, n_classes, K_w², H, W)
        P_padded = F.pad(P, (pad, pad, pad, pad), mode="constant", value=0)
        P_patches_flat = F.unfold(P_padded, kernel_size=K_w, stride=1)
        P_patches = P_patches_flat.reshape(1, self.n_classes, K_w * K_w, H, W)

        # --- Temperature averaging ---
        Q_acc = torch.zeros(1, self.n_classes, H, W, device=self.device)
        for T in self.temperatures:
            # (1, 1, K_w², H, W)
            A = self._phi(cos_sim / T).unsqueeze(1).clamp(min=0)
            Q = (A * P_patches).sum(dim=2)         # (1, n_classes, H, W)
            W_sum = A.sum(dim=2).clamp(min=1e-8)   # (1, 1, H, W)
            Q_acc += Q / W_sum

        Q_final = Q_acc / len(self.temperatures)
        refined = Q_final[0].argmax(dim=0).cpu().numpy().astype(np.int32)

        # Preserve unknown pixels
        refined[pred == 255] = 255
        return refined


# ═══════════════════════════════════════════════════════════════════════
# Feature Loading and Fusion
# ═══════════════════════════════════════════════════════════════════════


# (dir_template, expected_tokens, expected_dims)
FEATURE_SOURCES: Dict[str, Tuple[str, int, int]] = {
    "dinov3": ("dinov3_features/val2017", 1024, 1024),
    "dinov3_hires": ("dinov3_features_64x64/val2017", 4096, 1024),
    "dinov3_neco": ("dinov3_neco_features/val2017", 1024, 768),  # Round 7: NeCo-tuned
    "sd": ("sd_features_v14_s{step}/val2017", 256, 1280),
    "ssd1b": ("ssd1b_features_s{step}/val2017", 1024, 1280),
}


def load_multi_modal(
    image_id: str,
    coco_root: str,
    sources: List[str],
    device: str = "mps",
    step: int = 50,
) -> Dict[str, torch.Tensor]:
    """Load features from multiple sources for one image."""
    features: Dict[str, torch.Tensor] = {}
    for source in sources:
        dir_template = FEATURE_SOURCES[source][0].format(step=step)
        feat_path = Path(coco_root) / dir_template / f"{image_id}.npy"
        if feat_path.exists():
            feats = np.load(feat_path)
            features[source] = torch.tensor(
                feats, dtype=torch.float32, device=device
            )
    return features


def align_to_resolution(
    features: torch.Tensor, target_res: int,
) -> torch.Tensor:
    """Interpolate (N, C) feature grid to (target_res², C)."""
    n_tokens, c = features.shape
    grid = int(math.sqrt(n_tokens))
    if grid == target_res:
        return features
    feat_2d = features.reshape(grid, grid, c).permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(
        feat_2d, size=(target_res, target_res),
        mode="bilinear", align_corners=False,
    )
    return out[0].permute(1, 2, 0).reshape(target_res * target_res, c)


def fuse_features(
    feat_dict: Dict[str, torch.Tensor],
    target_res: int = 32,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Weighted concatenation of L2-normalized, resolution-aligned features.

    Cosine similarity of the concatenated result equals the weighted
    average of per-modality cosine similarities (when weights are equal).
    """
    if not weights:
        weights = {k: 1.0 for k in feat_dict}

    parts: List[torch.Tensor] = []
    for source, feats in feat_dict.items():
        aligned = align_to_resolution(feats, target_res)
        normed = F.normalize(aligned, p=2, dim=1)
        parts.append(normed * weights.get(source, 1.0))

    return torch.cat(parts, dim=1)


# ═══════════════════════════════════════════════════════════════════════
# Adaptive K via Eigengap (Round 5)
# ═══════════════════════════════════════════════════════════════════════


def compute_adaptive_k(
    features: torch.Tensor,
    k_min: int = 10,
    k_max: int = 150,
    approx_n: int = 256,
) -> int:
    """Select optimal K per image via Laplacian eigengap heuristic.

    The normalised Laplacian eigenspectrum encodes the natural cluster
    structure: a large gap between λ_k and λ_{k+1} signals that k clusters
    explains the data well. We search for this gap in [k_min, k_max].

    For speed, the affinity is computed on a uniformly-sampled subset of
    approx_n tokens (≪ N=1024), reducing eigh to an approx_n×approx_n call.

    Args:
        features:  (N, D) L2-normalised feature tensor.
        k_min:     Minimum K to consider (default 10).
        k_max:     Maximum K to consider (default 150).
        approx_n:  Downsample to this many tokens for eigenanalysis (default 256).

    Returns:
        Optimal K as integer in [k_min, k_max].
    """
    N = features.shape[0]

    # Uniform sub-sampling for speed (O(approx_n²) instead of O(N²))
    if N > approx_n:
        step = max(1, N // approx_n)
        feat_sub = features[::step][:approx_n]
        feat_sub = F.normalize(feat_sub, p=2, dim=1)
    else:
        feat_sub = features

    n = feat_sub.shape[0]
    k_max_eff = min(k_max, n - 1)
    k_min_eff = max(k_min, 2)

    # Cosine affinity (no power transform — we want eigenvalue geometry)
    W = feat_sub @ feat_sub.T
    W = (W - W.min()) / (W.max() - W.min() + 1e-8)
    W = W.clamp(min=0)

    # Normalised Laplacian: L_sym = I − D^{-½} W D^{-½}
    D = W.sum(dim=1)
    D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)
    L_sym = (
        torch.eye(n, device=features.device, dtype=W.dtype)
        - D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0)
    )

    try:
        eigenvalues = torch.linalg.eigh(L_sym.cpu())[0].float().numpy()
    except Exception:
        logger.warning("eigh failed in compute_adaptive_k; falling back to k_min=%d", k_min_eff)
        return k_min_eff

    # Eigengap: argmax of λ_{k+1} − λ_k for k in [k_min_eff, k_max_eff]
    eigs = eigenvalues[: k_max_eff + 1]
    if len(eigs) < k_min_eff + 1:
        return k_min_eff

    gaps = np.diff(eigs[k_min_eff - 1 : k_max_eff])
    if len(gaps) == 0:
        return k_min_eff

    k_star = int(k_min_eff + int(gaps.argmax()))
    logger.debug("Adaptive K: k_star=%d (gap=%.4f)", k_star, gaps.max())
    return k_star


# ═══════════════════════════════════════════════════════════════════════
# Multi-Scale Segment Merging (Round 6)
# ═══════════════════════════════════════════════════════════════════════


def merge_multiscale_segments(
    global_pred: np.ndarray,
    seg_fine: np.ndarray,
) -> np.ndarray:
    """Sharpen class-label boundaries using fine-grained NCut segments.

    Each fine NCut segment inherits the majority-vote class label from the
    coarse 27-class prediction. Effect: the fine-NCut boundaries subdivide
    regions where the coarse label map is spatially coherent, adding
    sharper edges without introducing new semantic labels.

    Args:
        global_pred: (H, W) coarse label map, values in [0, n_classes-1]
                     or 255 for unknown pixels.
        seg_fine:    (H2, W2) fine NCut segment ID map (arbitrary non-negative IDs).

    Returns:
        (H2, W2) refined label map at fine resolution (uint8 compatible).
    """
    H2, W2 = seg_fine.shape

    # Up-sample coarse labels to fine resolution with NEAREST (preserves labels)
    coarse_at_fine = np.array(
        Image.fromarray(global_pred.astype(np.uint8)).resize(
            (W2, H2), Image.NEAREST
        )
    )

    merged = coarse_at_fine.copy()
    for fine_id in np.unique(seg_fine):
        fine_mask = seg_fine == fine_id
        labels = coarse_at_fine[fine_mask]
        valid = labels < 255  # exclude unknown
        if not valid.any():
            continue
        majority = int(np.bincount(labels[valid].astype(np.int64)).argmax())
        merged[fine_mask] = majority

    return merged.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════


def mmgd_pipeline(
    coco_root: str,
    device: str,
    sources: List[str],
    target_res: int = 32,
    modality_weights: Optional[Dict[str, float]] = None,
    diffusion_method: str = "ppr",
    diff_mode: str = "feature",
    diff_steps: int = 3,
    diff_alpha: float = 0.85,
    K: int = 54,
    alpha: float = 5.5,
    beta: float = 0.5,
    n_iter: int = 15,
    temperature: float = 1.0,
    init: str = "kmeans",
    reg_lambda: float = 0.7,
    step: int = 50,
    mask_size: Tuple[int, int] = (128, 128),
    K_global: int = 27,
    n_images: Optional[int] = None,
    dino_only: bool = True,
    # Round 4: NAMR
    namr: bool = False,
    namr_window: int = 2,
    namr_temperatures: Optional[List[float]] = None,
    # Round 5: Adaptive K
    adaptive_k: bool = False,
    k_min: int = 10,
    k_max: int = 150,
    # Round 6: Multi-scale
    multiscale: bool = False,
    k_fine_multiplier: int = 2,
) -> Dict:
    """Full MMGD-Cut pipeline: fuse → diffuse → NCut → cluster → evaluate.

    Args:
        sources:           Feature sources for NCut affinity (e.g. ["dinov3", "sd"]).
        diffusion_method:  Graph diffusion method ("ppr", "lazy_rw", "none").
        namr:              Apply NAMR post-processing (Round 4).
        adaptive_k:        Select K per image via eigengap (Round 5).
        multiscale:        Run fine 64×64 NCut pass and merge (Round 6).
        Other args:        Same as Falcon K-way NCut.

    Returns:
        Dict with mIoU, things/stuff mIoU, per-class IoU, config metadata.
    """
    coco_p = Path(coco_root)

    # ── Determine image list ──
    if dino_only:
        dino_dir = coco_p / "dinov3_features" / "val2017"
        img_ids = sorted([f.stem for f in dino_dir.glob("*.npy")])
    else:
        img_ids = sorted(
            [f.stem for f in (coco_p / "val2017").glob("*.jpg")]
        )
    if n_images:
        img_ids = img_ids[:n_images]

    # ── Config key for caching ──
    src_str = "+".join(sorted(sources))
    diff_str = (
        f"{diff_mode}_{diffusion_method}{diff_steps}a{diff_alpha}"
        if diffusion_method != "none"
        else "nodiff"
    )
    wt_str = ""
    if modality_weights and any(v != 1.0 for v in modality_weights.values()):
        wt_str = "_w" + "_".join(
            f"{k}{v}" for k, v in sorted(modality_weights.items())
        )

    # seg_key: identifies Phase 1 segment cache.
    # Only adaptive_k changes per-image K, so it affects Phase 1.
    # NAMR and multiscale are pure post-processing — they reuse the base cache.
    seg_post_str = f"_adaptK{k_min}-{k_max}" if adaptive_k else ""
    seg_key = (
        f"mmgd_K{K}_a{alpha}_reg{reg_lambda}"
        f"_{src_str}_{diff_str}_r{target_res}{wt_str}{seg_post_str}"
    )

    # config_key: identifies this full run (including all post-processing flags).
    # Used for results storage in mmgd_results.json.
    post_str = seg_post_str
    if namr:
        # Include window size and temperature count in key to avoid collisions
        n_temps = len(namr_temperatures) if namr_temperatures is not None else 7
        post_str += f"_namrW{namr_window}T{n_temps}"
    if multiscale:
        post_str += "_ms"
    config_key = (
        f"mmgd_K{K}_a{alpha}_reg{reg_lambda}"
        f"_{src_str}_{diff_str}_r{target_res}{wt_str}{post_str}"
    )

    logger.info("═══ MMGD-Cut ═══")
    logger.info(
        "Sources: %s, diffusion: %s, target_res: %d",
        src_str, diff_str, target_res,
    )
    logger.info(
        "Falcon params: K=%d, α=%.1f, β=%.1f, reg_λ=%.1f",
        K, alpha, beta, reg_lambda,
    )
    logger.info(
        "Post-processing: NAMR=%s, AdaptiveK=%s, Multiscale=%s",
        namr, adaptive_k, multiscale,
    )
    logger.info("Images: %d, config: %s", len(img_ids), config_key)

    # ── Build solver ──
    diffusion = None
    if diffusion_method != "none":
        diffusion = GraphDiffusion(
            diffusion_method, diff_steps, diff_alpha, device,
        )
    solver = MultiModalFalcon(
        device=device, diffusion=diffusion, diff_mode=diff_mode,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Segment all images (coarse NCut)
    # ═══════════════════════════════════════════════════════════════════
    # Use seg_key (not config_key) so NAMR/multiscale reuse the base cache.
    seg_dir = coco_p / "mmgd_segments" / seg_key / "val2017"
    seg_dir.mkdir(parents=True, exist_ok=True)

    cached = sum(1 for f in seg_dir.glob("*.npy"))
    logger.info(
        "Phase 1: Multi-modal segmentation (%d cached, %d to do)",
        cached, len(img_ids) - cached,
    )

    t0 = time.time()
    for img_id in tqdm(img_ids, desc="Segmenting"):
        seg_path = seg_dir / f"{img_id}.npy"
        if seg_path.exists():
            continue

        feat_dict = load_multi_modal(
            img_id, coco_root, sources, device, step,
        )
        if not feat_dict:
            continue

        # Fuse or align single modality
        if len(feat_dict) == 1:
            name = list(feat_dict.keys())[0]
            fused = align_to_resolution(feat_dict[name], target_res)
        else:
            fused = fuse_features(feat_dict, target_res, modality_weights)

        # Round 5: per-image adaptive K
        K_img = K
        if adaptive_k:
            feat_normed = F.normalize(fused, p=2, dim=1)
            K_img = compute_adaptive_k(feat_normed, k_min, k_max)

        seg_map, _ = solver.segment(
            fused.unsqueeze(0),
            K=K_img, alpha=alpha, beta=beta,
            n_iter=n_iter, temperature=temperature, init=init,
            reg_lambda=reg_lambda, mask_size=mask_size,
        )
        np.save(seg_path, seg_map.astype(np.int16))

    seg_time = time.time() - t0
    n_total = max(len(img_ids), 1)
    logger.info(
        "Phase 1 done in %.1fs (%.2fs/img)", seg_time, seg_time / n_total,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1b: Fine NCut at 64×64 (only when multiscale=True)
    # ═══════════════════════════════════════════════════════════════════
    fine_seg_dir: Optional[Path] = None
    if multiscale:
        K_fine = K * k_fine_multiplier
        fine_key = f"mmgd_K{K_fine}_a{alpha}_reg{reg_lambda}_dinov3_hires_nodiff_r64_fine"
        fine_seg_dir = coco_p / "mmgd_segments" / fine_key / "val2017"
        fine_seg_dir.mkdir(parents=True, exist_ok=True)

        cached_fine = sum(1 for f in fine_seg_dir.glob("*.npy"))
        logger.info(
            "Phase 1b: Fine 64×64 NCut K=%d (%d cached, %d to do)",
            K_fine, cached_fine, len(img_ids) - cached_fine,
        )
        fine_solver = MultiModalFalcon(device=device)  # no diffusion for fine pass

        t1b = time.time()
        for img_id in tqdm(img_ids, desc="Fine-Segmenting"):
            fine_path = fine_seg_dir / f"{img_id}.npy"
            if fine_path.exists():
                continue

            hires_dict = load_multi_modal(
                img_id, coco_root, ["dinov3_hires"], device, step,
            )
            if "dinov3_hires" not in hires_dict:
                continue

            fused_fine = align_to_resolution(hires_dict["dinov3_hires"], target_res=64)
            seg_fine_map, _ = fine_solver.segment(
                fused_fine.unsqueeze(0),
                K=K_fine, alpha=alpha, beta=beta,
                n_iter=n_iter, temperature=temperature, init=init,
                reg_lambda=reg_lambda, mask_size=(64, 64),
            )
            np.save(fine_path, seg_fine_map.astype(np.int16))

        logger.info(
            "Phase 1b done in %.1fs", time.time() - t1b,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: Pool DINOv3 embeddings per segment
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Phase 2: Pooling DINOv3 embeddings per segment")
    dino_embed_dir = coco_p / "dinov3_features_64x64" / "val2017"
    if not dino_embed_dir.exists():
        dino_embed_dir = coco_p / "dinov3_features" / "val2017"

    all_embeddings: List[np.ndarray] = []
    seg_info: List[Tuple[str, int]] = []

    for img_id in tqdm(img_ids, desc="Pooling"):
        seg_path = seg_dir / f"{img_id}.npy"
        dino_path = dino_embed_dir / f"{img_id}.npy"
        if not seg_path.exists() or not dino_path.exists():
            continue

        seg_map = np.load(seg_path)
        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map

        dino_feats = np.load(dino_path)
        n_tokens = dino_feats.shape[0]
        grid = int(math.sqrt(n_tokens))
        feat_2d = dino_feats.reshape(grid, grid, -1)

        pred_grid = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (grid, grid), Image.NEAREST,
            )
        )

        for seg_id in np.unique(pred_grid):
            mask = pred_grid == seg_id
            if mask.sum() < 2:
                continue
            embed = feat_2d[mask].mean(axis=0)
            if np.isfinite(embed).all():
                all_embeddings.append(embed)
                seg_info.append((img_id, int(seg_id)))

    all_embeddings_arr = np.stack(all_embeddings)
    logger.info(
        "Collected %d segment embeddings (shape %s)",
        len(all_embeddings_arr), all_embeddings_arr.shape,
    )

    # Clean NaN/Inf
    valid_mask = np.isfinite(all_embeddings_arr).all(axis=1)
    if not valid_mask.all():
        n_bad = (~valid_mask).sum()
        logger.warning("Removing %d NaN/Inf embeddings", n_bad)
    valid_embeddings = all_embeddings_arr[valid_mask]

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: Global k-means
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Phase 3: Global k-means (K_global=%d)", K_global)
    kmeans = MiniBatchKMeans(
        n_clusters=K_global,
        batch_size=min(4096, len(valid_embeddings)),
        n_init=5, random_state=42,
    )
    cluster_labels = kmeans.fit_predict(valid_embeddings)

    # Map (image_id, seg_id) → global cluster
    seg_to_cluster: Dict[Tuple[str, int], int] = {}
    valid_idx = 0
    for orig_idx, (img_id, seg_id) in enumerate(seg_info):
        if valid_mask[orig_idx]:
            seg_to_cluster[(img_id, seg_id)] = int(
                cluster_labels[valid_idx]
            )
            valid_idx += 1

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4: Evaluate against GT
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Phase 4: Hungarian matching evaluation")

    # Instantiate NAMR processor if requested
    namr_proc: Optional[NAMRPostProcessor] = None
    if namr:
        namr_proc = NAMRPostProcessor(
            temperatures=namr_temperatures,
            window_size=namr_window,
            n_classes=NUM_CLASSES,
            device=device,
        )
        logger.info(
            "NAMR: window=%d, temperatures=%s",
            namr_window, namr_proc.temperatures,
        )

    per_class_iou_acc = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for img_id in tqdm(img_ids, desc="Evaluating"):
        gt = load_coco_panoptic_gt(coco_root, int(img_id))
        if gt is None:
            continue

        seg_path = seg_dir / f"{img_id}.npy"
        if not seg_path.exists():
            continue

        # ── Load coarse seg and map to cluster labels at mask_size ──
        seg_map = np.load(seg_path)
        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map  # (H_m, W_m)

        global_pred_small = np.full_like(pred, 255, dtype=np.int32)
        for seg_id in np.unique(pred):
            key = (img_id, int(seg_id))
            if key in seg_to_cluster:
                global_pred_small[pred == seg_id] = seg_to_cluster[key]

        # ── Round 6: Multi-scale boundary merging ──
        if multiscale and fine_seg_dir is not None:
            fine_path = fine_seg_dir / f"{img_id}.npy"
            if fine_path.exists():
                seg_fine_raw = np.load(fine_path)
                seg_fine = (
                    seg_fine_raw[0, 0] if seg_fine_raw.ndim == 4 else seg_fine_raw
                )
                global_pred_small = merge_multiscale_segments(
                    global_pred_small.astype(np.uint8), seg_fine,
                ).astype(np.int32)

        # ── Round 4: NAMR post-processing ──
        if namr_proc is not None:
            img_path = coco_p / "val2017" / f"{img_id}.jpg"
            if img_path.exists():
                image = np.array(Image.open(img_path).convert("RGB"))
                global_pred_small = namr_proc.refine(
                    global_pred_small.astype(np.int32), image
                )

        # ── Resize to GT resolution ──
        H_m, W_m = global_pred_small.shape
        global_pred = np.array(
            Image.fromarray(global_pred_small.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST,
            )
        )

        miou_img, iou_per_class, _ = hungarian_miou(global_pred, gt)
        for c in range(NUM_CLASSES):
            if (gt == c).any():
                per_class_iou_acc[c] += iou_per_class[c]
                count_per_class[c] += 1

    # ── Aggregate results ──
    active = count_per_class > 0
    per_class_avg = np.zeros(NUM_CLASSES)
    per_class_avg[active] = per_class_iou_acc[active] / count_per_class[active]

    things = [
        per_class_avg[c] * 100
        for c in range(NUM_CLASSES) if c in THING_IDS and active[c]
    ]
    stuff = [
        per_class_avg[c] * 100
        for c in range(NUM_CLASSES) if c in STUFF_IDS and active[c]
    ]
    miou = per_class_avg[active].mean() * 100

    result = {
        "method": "mmgd_cut",
        "config_key": config_key,
        "sources": sources,
        "diffusion": diffusion_method,
        "diff_mode": diff_mode,
        "diff_steps": diff_steps,
        "diff_alpha": diff_alpha,
        "target_res": target_res,
        "K": K, "alpha": alpha, "beta": beta,
        "reg_lambda": reg_lambda,
        "K_global": K_global,
        "namr": namr,
        "adaptive_k": adaptive_k,
        "multiscale": multiscale,
        "n_images": len(img_ids),
        "n_segments": len(valid_embeddings),
        "miou": round(miou, 2),
        "things_miou": round(np.mean(things), 2) if things else 0.0,
        "stuff_miou": round(np.mean(stuff), 2) if stuff else 0.0,
        "per_class_iou": {
            COCOSTUFF27_CLASSNAMES[c]: round(per_class_avg[c] * 100, 1)
            for c in range(NUM_CLASSES) if count_per_class[c] > 0
        },
    }

    logger.info(
        "MMGD-Cut: mIoU=%.2f%%, Things=%.2f%%, Stuff=%.2f%%",
        miou, result["things_miou"], result["stuff_miou"],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMGD-Cut: Multi-Modal Graph-Diffused K-way NCut",
    )
    p.add_argument("--coco_root", required=True)
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])

    # Multi-modal
    p.add_argument(
        "--sources", nargs="+", default=["dinov3", "sd"],
        choices=["dinov3", "dinov3_hires", "dinov3_neco", "sd", "ssd1b"],
        help="Feature sources for NCut affinity",
    )
    p.add_argument(
        "--target_res", type=int, default=32,
        help="Align all features to this spatial grid size",
    )
    p.add_argument("--w_dinov3", type=float, default=1.0)
    p.add_argument("--w_sd", type=float, default=1.0)
    p.add_argument("--w_ssd1b", type=float, default=1.0)

    # Graph diffusion
    p.add_argument(
        "--diffusion", default="ppr",
        choices=["ppr", "lazy_rw", "none"],
    )
    p.add_argument(
        "--diff_mode", default="feature",
        choices=["feature", "affinity"],
        help="Diffusion on features (GNN-style) or affinity matrix",
    )
    p.add_argument("--diff_steps", type=int, default=3)
    p.add_argument("--diff_alpha", type=float, default=0.85)

    # Falcon NCut params (defaults = best Phase 1 config)
    p.add_argument("--K", type=int, default=54)
    p.add_argument("--alpha", type=float, default=5.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--n_iter", type=int, default=15)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--init", default="kmeans",
        choices=["random", "spectral", "kmeans"],
    )
    p.add_argument("--reg_lambda", type=float, default=0.7)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--mask_size", type=int, default=128)
    p.add_argument("--K_global", type=int, default=27)

    # Round 4: NAMR
    p.add_argument(
        "--namr", action="store_true",
        help="Apply NAMR post-processing (φ(x)=x+1.5·ELU(x), temperature avg)",
    )
    p.add_argument(
        "--namr_window", type=int, default=2,
        help="Half-window size for NAMR neighbourhood (full window = 2*w+1)",
    )
    p.add_argument(
        "--namr_single_temp", type=float, default=None,
        help="Use single temperature for NAMR (overrides default 7-temperature avg)",
    )

    # Round 5: Adaptive K
    p.add_argument(
        "--adaptive_k", action="store_true",
        help="Select K per image via Laplacian eigengap (ignores --K)",
    )
    p.add_argument(
        "--k_min", type=int, default=10,
        help="Minimum K for adaptive K selection",
    )
    p.add_argument(
        "--k_max", type=int, default=150,
        help="Maximum K for adaptive K selection",
    )

    # Round 6: Multi-scale
    p.add_argument(
        "--multiscale", action="store_true",
        help="Run fine 64×64 NCut pass and merge with coarse 32×32",
    )
    p.add_argument(
        "--k_fine_multiplier", type=int, default=2,
        help="K for fine NCut = K * k_fine_multiplier",
    )

    p.add_argument("--n_images", type=int, default=None)
    p.add_argument("--dino_only", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weight_map = {
        "dinov3": args.w_dinov3,
        "dinov3_hires": args.w_dinov3,
        "sd": args.w_sd,
        "ssd1b": args.w_ssd1b,
    }
    mod_weights = {s: weight_map.get(s, 1.0) for s in args.sources}

    namr_temps = None
    if args.namr_single_temp is not None:
        namr_temps = [args.namr_single_temp]

    result = mmgd_pipeline(
        coco_root=args.coco_root,
        device=args.device,
        sources=args.sources,
        target_res=args.target_res,
        modality_weights=mod_weights,
        diffusion_method=args.diffusion,
        diff_mode=args.diff_mode,
        diff_steps=args.diff_steps,
        diff_alpha=args.diff_alpha,
        K=args.K, alpha=args.alpha, beta=args.beta,
        n_iter=args.n_iter, temperature=args.temperature,
        init=args.init, reg_lambda=args.reg_lambda,
        step=args.step,
        mask_size=(args.mask_size, args.mask_size),
        K_global=args.K_global,
        n_images=args.n_images,
        dino_only=args.dino_only,
        namr=args.namr,
        namr_window=args.namr_window,
        namr_temperatures=namr_temps,
        adaptive_k=args.adaptive_k,
        k_min=args.k_min,
        k_max=args.k_max,
        multiscale=args.multiscale,
        k_fine_multiplier=args.k_fine_multiplier,
    )

    out_path = args.output or str(
        Path(args.coco_root) / "mmgd_results.json"
    )
    existing = {}
    if Path(out_path).exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing[result["config_key"]] = result
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
