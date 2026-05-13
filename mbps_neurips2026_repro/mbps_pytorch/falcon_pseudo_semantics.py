#!/usr/bin/env python3
"""Falcon K-way NCut unsupervised semantic segmentation for COCO-Stuff-27.

Implements the Fractional Alternating Cut (Falcon, ICLR 2026) for
simultaneous K-way Normalized Cut without recursive eigendecomposition.

Uses the same SD features, PAMR refinement, and evaluation infrastructure
as diffcut_pseudo_semantics.py.

Usage:
    # Segment + per-image Hungarian evaluation
    python falcon_pseudo_semantics.py --phase segment \
        --coco_root /path/to/coco --K 27 --alpha 4.5 --beta 0.5

    # Global clustering for pseudo-labels
    python falcon_pseudo_semantics.py --phase cluster \
        --coco_root /path/to/coco --K 27 --alpha 4.5 --pamr --K_global 27

    # All phases
    python falcon_pseudo_semantics.py --phase all \
        --coco_root /path/to/coco --K 27 --alpha 4.5 --pamr

References:
    Falcon (ICLR 2026): arXiv 2504.05613
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import median_filter
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# Import shared utilities from DiffCut
from diffcut_pseudo_semantics import (
    COCOSTUFF27_CLASSNAMES,
    NUM_CLASSES,
    STUFF_IDS,
    THING_IDS,
    hungarian_miou,
    load_coco_panoptic_gt,
    pamr_refine,
)

from models.merger.crf_postprocess import crf_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def crf_refine(
    seg_map: np.ndarray,
    image: Image.Image,
    device: str = "mps",
    crf_size: int = 64,
    num_iterations: int = 10,
    unary_strength: float = 10.0,
) -> np.ndarray:
    """Refine segmentation using CRF with bilateral + smoothness potentials.

    Args:
        seg_map: Hard segmentation (1, 1, H, W) int array from Falcon.
        image: RGB PIL Image.
        device: Compute device.
        crf_size: Resolution for CRF computation (NxN).
        num_iterations: CRF mean-field iterations.
        unary_strength: Strength of unary (hard assignment) potentials.

    Returns:
        Refined segmentation as (1, 1, H, W) numpy int array at original resolution.
    """
    # Extract hard prediction at CRF resolution
    pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map[0] if seg_map.ndim == 3 else seg_map
    orig_h, orig_w = pred.shape

    pred_crf = np.array(
        Image.fromarray(pred.astype(np.uint8)).resize(
            (crf_size, crf_size), Image.NEAREST
        )
    )

    # Build one-hot unary potentials with controlled strength
    unique_labels = np.unique(pred_crf)
    K = len(unique_labels)
    label_remap = {old: new for new, old in enumerate(unique_labels)}
    pred_remapped = np.vectorize(label_remap.get)(pred_crf).flatten()

    n_pixels = crf_size * crf_size
    unary = torch.zeros(n_pixels, K, device=device)
    for i, label in enumerate(pred_remapped):
        unary[i, label] = unary_strength
    # Subtract mean so non-assigned classes get negative log-prob
    unary = unary - (unary_strength / K)

    # Prepare RGB image at CRF resolution
    img_resized = image.resize((crf_size, crf_size), Image.BILINEAR)
    img_flat = torch.tensor(
        np.array(img_resized).astype(np.float32).reshape(-1, 3),
        device=device,
    )

    # Run CRF
    refined_probs = crf_inference(
        unary_potentials=unary,
        image=img_flat,
        spatial_h=crf_size,
        spatial_w=crf_size,
        num_iterations=num_iterations,
    )

    # Map back to original label IDs
    refined_labels = refined_probs.argmax(dim=-1).cpu().numpy()
    inv_remap = {new: old for old, new in label_remap.items()}
    refined_orig = np.vectorize(inv_remap.get)(refined_labels).reshape(crf_size, crf_size)

    # Upsample back to original seg_map resolution
    refined_full = np.array(
        Image.fromarray(refined_orig.astype(np.uint8)).resize(
            (orig_w, orig_h), Image.NEAREST
        )
    )

    return refined_full.reshape(1, 1, orig_h, orig_w).astype(int)


# ═══════════════════════════════════════════════════════════════════════
# Falcon K-way Fractional Alternating NCut
# ═══════════════════════════════════════════════════════════════════════

class FalconKwayCut:
    """K-way Normalized Cut via fractional alternating optimization.

    Instead of recursive binary Fiedler-vector splitting (DiffCut),
    optimizes all K segments simultaneously using a fractional quadratic
    transformation. Avoids eigendecomposition entirely.

    Args:
        device: Compute device ('mps', 'cuda', 'cpu').
    """

    def __init__(self, device: str = "mps") -> None:
        self.device = torch.device(device)

    def build_affinity(
        self,
        features: torch.Tensor,
        alpha: float = 4.5,
        reg_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build power-transformed affinity matrix with optional regularization.

        Per Falcon paper: W = W_norm^alpha + lambda * diag(D)
        The diagonal regularization prevents trivial solutions.

        Args:
            features: (N, C) L2-normalized feature vectors.
            alpha: Power exponent for affinity sharpening.
            reg_lambda: Diagonal regularization strength.

        Returns:
            W: (N, N) affinity matrix.
            D: (N,) degree vector.
        """
        W_raw = features @ features.T  # cosine similarity
        w_min = W_raw.min()
        w_max = W_raw.max()
        W_scaled = (W_raw - w_min) / (w_max - w_min + 1e-8)
        W = W_scaled.pow(alpha)

        # Diagonal regularization (Falcon paper Eq. 3)
        if reg_lambda > 0:
            D_pre = W.sum(dim=1)
            W = W + reg_lambda * torch.diag(D_pre)

        D = W.sum(dim=1)
        return W, D

    def initialize_assignments(
        self,
        N: int,
        K: int,
        features: torch.Tensor,
        W: torch.Tensor,
        method: str = "kmeans",
    ) -> torch.Tensor:
        """Initialize soft assignment matrix X.

        Args:
            N: Number of nodes (patches).
            K: Number of clusters.
            features: (N, C) feature vectors.
            W: (N, N) affinity matrix (used for spectral init).
            method: 'random', 'spectral', or 'kmeans'.

        Returns:
            X: (N, K) soft assignment matrix.
        """
        if method == "kmeans":
            feats_np = features.cpu().numpy()
            km = MiniBatchKMeans(
                n_clusters=K, batch_size=min(1024, N), n_init=3, random_state=42,
            )
            labels = km.fit_predict(feats_np)
            X = torch.zeros(N, K, device=self.device)
            for i, lbl in enumerate(labels):
                X[i, lbl] = 1.0
            # Soft relaxation: add small uniform noise
            X = X + 0.05 * torch.rand(N, K, device=self.device)
            X = X / X.sum(dim=1, keepdim=True)

        elif method == "spectral":
            D = W.sum(dim=1)
            d_inv_sqrt = 1.0 / (torch.sqrt(D) + 1e-8)
            L_sym = (
                torch.eye(N, device=self.device)
                - (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
            )
            # Use CPU for eigendecomposition (MPS doesn't support eigh)
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym.cpu())
            # Take K smallest eigenvectors (skip constant eigenvector)
            X = eigenvectors[:, 1 : K + 1].to(self.device)
            # Convert to soft assignments via softmax
            X = torch.softmax(X * 10.0, dim=1)

        else:  # random
            X = torch.rand(N, K, device=self.device)
            X = X / X.sum(dim=1, keepdim=True)

        return X

    def _compute_ncut_objective(
        self, W: torch.Tensor, X: torch.Tensor, D: torch.Tensor
    ) -> float:
        """Compute the NCut objective: Σ_k (x_k^T W x_k) / (x_k^T D x_k).

        Used for convergence monitoring.
        """
        # (K,) numerators: diag(X^T W X)
        WX = W @ X  # (N, K)
        num = (X * WX).sum(dim=0)  # (K,)
        # (K,) denominators: diag(X^T D X) where D is diagonal
        den = (X * (D.unsqueeze(1) * X)).sum(dim=0)  # (K,)
        return (num / (den + 1e-8)).sum().item()

    def alternating_optimization(
        self,
        W: torch.Tensor,
        D: torch.Tensor,
        X: torch.Tensor,
        n_iter: int = 15,
        beta: float = 0.5,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Falcon fractional alternating K-way NCut optimization.

        Args:
            W: (N, N) affinity matrix.
            D: (N,) degree vector.
            X: (N, K) initial soft assignments.
            n_iter: Number of alternating iterations.
            beta: Affinity reweighting strength (larger = softer).
            temperature: Softmax temperature for X update.

        Returns:
            X: (N, K) converged soft assignments.
        """
        N, K = X.shape

        for t in range(n_iter):
            # Step 1: Update auxiliary y_k (closed-form Rayleigh quotient)
            WX = W @ X  # (N, K)
            num_y = (X * WX).sum(dim=0)  # (K,) = x_k^T W x_k
            den_y = (X * (D.unsqueeze(1) * X)).sum(dim=0)  # (K,) = x_k^T D x_k
            y = torch.sqrt(num_y / (den_y + 1e-8))  # (K,)

            # Step 2: Update soft assignments X
            # score_ik = (W @ X)_ik / (Σ_j X_jk * D_j) * y_k
            cluster_degree = (X * D.unsqueeze(1)).sum(dim=0)  # (K,)
            scores = (WX / (cluster_degree.unsqueeze(0) + 1e-8)) * y.unsqueeze(0)

            # Z-score normalize scores per node before softmax to prevent
            # collapse when raw scores are too uniform
            scores = scores - scores.mean(dim=1, keepdim=True)
            scores = scores / (scores.std(dim=1, keepdim=True) + 1e-8)

            X = torch.softmax(scores / temperature, dim=1)

            # Step 3: Dynamic affinity reweighting
            if beta > 0 and t < n_iter - 1:
                # Cosine similarity between assignment vectors
                # For N≤1024, direct computation is fine
                if N <= 1536:
                    X_norm = F.normalize(X, p=2, dim=1)
                    cos_sim = X_norm @ X_norm.T  # (N, N)
                else:
                    # Chunked computation for large N
                    cos_sim = self._chunked_cosine_sim(X, chunk_size=512)

                reweight = torch.exp(-((1.0 - cos_sim) ** 2) / beta)
                W = W * reweight

                # Update degree after reweighting
                D = W.sum(dim=1)

        return X

    def _chunked_cosine_sim(
        self, X: torch.Tensor, chunk_size: int = 512
    ) -> torch.Tensor:
        """Compute cosine similarity matrix in chunks to save memory.

        Args:
            X: (N, K) assignment matrix.
            chunk_size: Rows per chunk.

        Returns:
            cos_sim: (N, N) cosine similarity matrix.
        """
        N = X.shape[0]
        X_norm = F.normalize(X, p=2, dim=1)
        cos_sim = torch.zeros(N, N, device=X.device, dtype=X.dtype)
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            cos_sim[i:end_i] = X_norm[i:end_i] @ X_norm.T
        return cos_sim

    def segment(
        self,
        features: torch.Tensor,
        K: int = 27,
        alpha: float = 4.5,
        beta: float = 0.5,
        n_iter: int = 15,
        temperature: float = 1.0,
        init: str = "kmeans",
        reg_lambda: float = 0.0,
        mask_size: Tuple[int, int] = (128, 128),
        feat_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Full Falcon segmentation pipeline.

        Args:
            features: (1, N, C) patch features.
            K: Number of clusters.
            alpha: Power exponent for affinity.
            beta: Affinity reweighting strength.
            n_iter: Alternating optimization iterations.
            temperature: Softmax temperature.
            init: Initialization method ('random', 'spectral', 'kmeans').
            reg_lambda: Diagonal regularization strength.
            mask_size: Output segmentation resolution.
            feat_hw: Feature grid spatial dims. If None, inferred as sqrt(N).

        Returns:
            (seg_map, K_actual): seg_map is (1,1,H,W) int array, K_actual is
            the number of non-empty clusters.
        """
        _, n_tokens, c = features.shape
        if feat_hw is None:
            h = w = int(math.sqrt(n_tokens))
        else:
            h, w = feat_hw

        feats = features[0]  # (N, C)
        feats_norm = F.normalize(feats, p=2, dim=1)

        # 1. Build affinity
        W, D = self.build_affinity(feats_norm, alpha=alpha, reg_lambda=reg_lambda)

        # 2. Initialize assignments
        X = self.initialize_assignments(
            n_tokens, K, feats_norm, W, method=init,
        )

        # 3. Alternating optimization
        X = self.alternating_optimization(
            W, D, X, n_iter=n_iter, beta=beta, temperature=temperature,
        )

        # 4. Hard assignment
        labels = X.argmax(dim=1)  # (N,)
        unique_labels = labels.unique()
        K_actual = len(unique_labels)

        # Remap to contiguous IDs
        label_map = {old.item(): new for new, old in enumerate(unique_labels)}
        labels_contiguous = torch.tensor(
            [label_map[l.item()] for l in labels],
            device=self.device,
        )

        # 5. Upsample via feature-based cosine assignment
        feats_4d = features.permute(0, 2, 1).reshape(1, c, h, w)
        feats_4d_norm = F.normalize(feats_4d, dim=1)

        # Pool features per cluster
        cluster_embeds = torch.zeros(K_actual, c, device=self.device)
        for k in range(K_actual):
            mask_k = (labels_contiguous == k).float().reshape(1, 1, h, w)
            denom = mask_k.sum() + 1e-8
            pooled = (feats_4d_norm * mask_k).sum(dim=(2, 3)) / denom
            cluster_embeds[k] = pooled[0]

        # Upsample features and assign by cosine similarity
        up_feats = F.interpolate(feats_4d_norm, size=mask_size, mode="bilinear")
        _, c2, hm, wm = up_feats.shape
        sim = up_feats[0].reshape(c2, -1).T @ cluster_embeds.T  # (H*W, K_actual)
        seg = sim.argmax(dim=1).reshape(1, 1, hm, wm)

        return seg.cpu().numpy().astype(int), K_actual


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Segment + Per-Image Hungarian Evaluation
# ═══════════════════════════════════════════════════════════════════════

def falcon_segment_and_evaluate(
    coco_root: str,
    device: str = "mps",
    K: int = 27,
    alpha: float = 4.5,
    beta: float = 0.5,
    n_iter: int = 15,
    temperature: float = 1.0,
    init: str = "kmeans",
    reg_lambda: float = 0.0,
    use_pamr: bool = False,
    use_crf: bool = False,
    step: int = 50,
    mask_size: Tuple[int, int] = (128, 128),
    n_images: Optional[int] = None,
    dino_only: bool = False,
    feature_source: str = "sd",
) -> Dict:
    """Run Falcon segmentation and per-image Hungarian evaluation."""
    if feature_source == "dinov3":
        feat_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        logger.info("Using DINOv3 features for segmentation")
    else:
        feat_dir = Path(coco_root) / f"sd_features_v14_s{step}" / "val2017"
    img_dir = Path(coco_root) / "val2017"

    feat_files = sorted(feat_dir.glob("*.npy"))

    if dino_only:
        dino_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        dino_ids = {f.stem for f in dino_dir.glob("*.npy")}
        feat_files = [f for f in feat_files if f.stem in dino_ids]
        logger.info("Filtered to %d images with DINOv3 features", len(feat_files))

    if n_images:
        feat_files = feat_files[:n_images]

    logger.info(
        "Falcon segmentation: %d images, K=%d, alpha=%.1f, beta=%.1f, "
        "n_iter=%d, temp=%.1f, init=%s, pamr=%s",
        len(feat_files), K, alpha, beta, n_iter, temperature, init, use_pamr,
    )

    falcon = FalconKwayCut(device=device)

    per_class_iou_acc = np.zeros(NUM_CLASSES)
    per_class_count = np.zeros(NUM_CLASSES)
    cluster_counts = []
    times = []

    for feat_path in tqdm(feat_files, desc="Falcon segmentation"):
        image_id = int(feat_path.stem)
        gt = load_coco_panoptic_gt(coco_root, image_id)
        if gt is None:
            continue

        feats = np.load(feat_path)  # (N, C)
        feats_t = torch.tensor(
            feats, dtype=torch.float32, device=device
        ).unsqueeze(0)

        t0 = time.time()
        seg_map, k_actual = falcon.segment(
            feats_t, K=K, alpha=alpha, beta=beta,
            n_iter=n_iter, temperature=temperature, init=init,
            reg_lambda=reg_lambda, mask_size=mask_size,
        )
        cluster_counts.append(k_actual)

        # Optional post-processing refinement
        if use_pamr or use_crf:
            img_path = img_dir / f"{image_id:012d}.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                if use_crf:
                    seg_map = crf_refine(seg_map, img, device=device)
                else:
                    img_t = torch.tensor(
                        np.array(img).astype(np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0)
                    seg_t = torch.tensor(seg_map)
                    refined = pamr_refine(img_t, seg_t, device=device)
                    seg_map = refined  # already (1,1,H,W)

        dt = time.time() - t0
        times.append(dt)

        # Resize prediction to GT resolution
        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map[0]
        pred_resized = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
        )

        miou_img, iou_per_class, _ = hungarian_miou(pred_resized, gt)

        for c in range(NUM_CLASSES):
            if (gt == c).any():
                per_class_iou_acc[c] += iou_per_class[c]
                per_class_count[c] += 1

    # Aggregate results
    valid = per_class_count > 0
    per_class_avg = np.zeros(NUM_CLASSES)
    per_class_avg[valid] = per_class_iou_acc[valid] / per_class_count[valid]

    things = [per_class_avg[c] * 100 for c in range(NUM_CLASSES) if c in THING_IDS and valid[c]]
    stuff = [per_class_avg[c] * 100 for c in range(NUM_CLASSES) if c in STUFF_IDS and valid[c]]

    miou = per_class_avg[valid].mean() * 100

    result = {
        "method": "falcon",
        "K": K,
        "alpha": alpha,
        "beta": beta,
        "n_iter": n_iter,
        "temperature": temperature,
        "init": init,
        "pamr": use_pamr,
        "n_images": len(feat_files),
        "miou": round(miou, 2),
        "things_miou": round(np.mean(things), 2) if things else 0.0,
        "stuff_miou": round(np.mean(stuff), 2) if stuff else 0.0,
        "avg_clusters": round(np.mean(cluster_counts), 1) if cluster_counts else 0,
        "avg_time_s": round(np.mean(times), 3) if times else 0,
        "per_class_iou": {
            COCOSTUFF27_CLASSNAMES[c]: round(per_class_avg[c] * 100, 1)
            for c in range(NUM_CLASSES) if valid[c]
        },
    }

    logger.info(
        "Falcon (K=%d, α=%.1f, β=%.1f, pamr=%s): mIoU=%.2f%%, "
        "Things=%.2f%%, Stuff=%.2f%%, avg_clusters=%.1f, avg_time=%.3fs",
        K, alpha, beta, use_pamr, miou,
        result["things_miou"], result["stuff_miou"],
        result["avg_clusters"], result["avg_time_s"],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Global Clustering for Pseudo-Labels
# ═══════════════════════════════════════════════════════════════════════

def falcon_global_clustering(
    coco_root: str,
    device: str = "mps",
    K: int = 27,
    alpha: float = 4.5,
    beta: float = 0.5,
    n_iter: int = 15,
    temperature: float = 1.0,
    init: str = "kmeans",
    reg_lambda: float = 0.0,
    use_pamr: bool = False,
    use_crf: bool = False,
    K_global: int = 27,
    step: int = 50,
    mask_size: Tuple[int, int] = (128, 128),
    feature_source: str = "dinov3",
    seg_feature_source: str = "sd",
    n_images: Optional[int] = None,
    dino_only: bool = False,
) -> Dict:
    """Pool features per Falcon segment, global k-means, Hungarian to 27."""
    # Segmentation features (for building affinity / NCut)
    if seg_feature_source == "dinov3":
        feat_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        logger.info("Using DINOv3 features for segmentation")
    else:
        feat_dir = Path(coco_root) / f"sd_features_v14_s{step}" / "val2017"
    img_dir = Path(coco_root) / "val2017"

    # Embedding features for global clustering
    if feature_source == "dinov3":
        dino_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        if not dino_dir.exists():
            dino_dir = Path(coco_root) / "dinov3_features" / "val2017"
        logger.info("Using DINOv3 features for embeddings from %s", dino_dir)
    else:
        dino_dir = None
        logger.info("Using SD features for segment embeddings")

    feat_files = sorted(feat_dir.glob("*.npy"))

    if dino_only:
        dino_filter_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        dino_ids = {f.stem for f in dino_filter_dir.glob("*.npy")}
        feat_files = [f for f in feat_files if f.stem in dino_ids]
        logger.info("Filtered to %d images with DINOv3 features", len(feat_files))

    if n_images:
        feat_files = feat_files[:n_images]

    falcon = FalconKwayCut(device=device)

    all_embeddings = []
    segment_info = []

    logger.info("Collecting segment embeddings from %d images...", len(feat_files))

    for feat_path in tqdm(feat_files, desc="Falcon global clustering"):
        image_id = int(feat_path.stem)

        sd_feats = np.load(feat_path)
        sd_feats_t = torch.tensor(
            sd_feats, dtype=torch.float32, device=device
        ).unsqueeze(0)

        seg_map, k_actual = falcon.segment(
            sd_feats_t, K=K, alpha=alpha, beta=beta,
            n_iter=n_iter, temperature=temperature, init=init,
            reg_lambda=reg_lambda, mask_size=mask_size,
        )

        # Optional PAMR / CRF
        if use_pamr or use_crf:
            img_path = img_dir / f"{image_id:012d}.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                if use_crf:
                    seg_map = crf_refine(seg_map, img, device=device)
                else:
                    img_t = torch.tensor(
                        np.array(img).astype(np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0)
                    seg_t = torch.tensor(seg_map)
                    refined = pamr_refine(img_t, seg_t, device=device)
                    seg_map = refined  # already (1,1,H,W)

        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map[0]

        # Load features for embedding
        use_dino = False
        if feature_source == "dinov3" and dino_dir:
            dino_path = dino_dir / f"{image_id:012d}.npy"
            if dino_path.exists():
                use_dino = True

        if use_dino:
            embed_feats = np.load(dino_path)
            n_patches = embed_feats.shape[0]
            grid = int(math.sqrt(n_patches))
            feat_2d = embed_feats.reshape(grid, grid, -1)
        else:
            n_tokens = sd_feats.shape[0]
            grid = int(math.sqrt(n_tokens))
            feat_2d = sd_feats.reshape(grid, grid, -1)

        # Resize pred to feature grid
        pred_grid = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (grid, grid), Image.NEAREST
            )
        )

        for seg_id in np.unique(pred_grid):
            mask = pred_grid == seg_id
            if mask.sum() < 2:
                continue
            embed = feat_2d[mask].mean(axis=0)
            if np.isfinite(embed).all():
                all_embeddings.append(embed)
                segment_info.append((image_id, int(seg_id), int(mask.sum())))

    if not all_embeddings:
        logger.error("No valid segment embeddings collected!")
        return {"miou": 0.0}

    all_embeddings = np.stack(all_embeddings)
    logger.info(
        "Collected %d segment embeddings (shape %s)",
        len(all_embeddings), all_embeddings.shape,
    )

    # Clean NaN/Inf
    valid_mask = np.isfinite(all_embeddings).all(axis=1)
    if not valid_mask.all():
        logger.warning(
            "Removing %d NaN/Inf embeddings", (~valid_mask).sum()
        )
        all_embeddings = all_embeddings[valid_mask]

    # Global k-means clustering
    logger.info("Running MiniBatchKMeans (K=%d)...", K_global)
    kmeans = MiniBatchKMeans(
        n_clusters=K_global, batch_size=min(4096, len(all_embeddings)),
        n_init=5, random_state=42,
    )
    cluster_labels = kmeans.fit_predict(all_embeddings)
    centroids = kmeans.cluster_centers_

    # Assign each image segment to a global cluster
    seg_to_cluster = {}
    for idx, (img_id, seg_id, _) in enumerate(segment_info):
        if valid_mask[idx]:
            seg_to_cluster[(img_id, seg_id)] = cluster_labels[idx]

    # Evaluate with Hungarian matching
    logger.info("Evaluating global clustering on GT...")
    per_class_iou_acc = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    out_dir = Path(coco_root) / f"pseudo_semantic_falcon_K{K}_a{alpha}_b{beta}" / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    for feat_path in tqdm(feat_files, desc="Evaluating global"):
        image_id = int(feat_path.stem)
        gt = load_coco_panoptic_gt(coco_root, image_id)
        if gt is None:
            continue

        # Re-segment (or cache segments above — this is simpler)
        sd_feats = np.load(feat_path)
        sd_feats_t = torch.tensor(
            sd_feats, dtype=torch.float32, device=device
        ).unsqueeze(0)

        seg_map, _ = falcon.segment(
            sd_feats_t, K=K, alpha=alpha, beta=beta,
            n_iter=n_iter, temperature=temperature, init=init,
            reg_lambda=reg_lambda, mask_size=mask_size,
        )

        if use_pamr or use_crf:
            img_path = img_dir / f"{image_id:012d}.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                if use_crf:
                    seg_map = crf_refine(seg_map, img, device=device)
                else:
                    img_t = torch.tensor(
                        np.array(img).astype(np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0)
                    seg_t = torch.tensor(seg_map)
                    refined = pamr_refine(img_t, seg_t, device=device)
                    seg_map = refined  # already (1,1,H,W)

        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map[0]
        pred_resized = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
        )

        # Map segment IDs to global cluster IDs
        global_pred = np.full_like(pred_resized, 255)
        for seg_id in np.unique(pred_resized):
            key = (image_id, int(seg_id))
            if key in seg_to_cluster:
                global_pred[pred_resized == seg_id] = seg_to_cluster[key]

        # Save pseudo-labels
        out_path = out_dir / f"{image_id:012d}.png"
        Image.fromarray(global_pred.astype(np.uint8)).save(out_path)

        # Hungarian matching for eval
        miou_img, iou_per_class, _ = hungarian_miou(global_pred, gt)
        for c in range(NUM_CLASSES):
            if (gt == c).any():
                per_class_iou_acc[c] += iou_per_class[c]
                count_per_class[c] += 1

    valid = count_per_class > 0
    per_class_avg = np.zeros(NUM_CLASSES)
    per_class_avg[valid] = per_class_iou_acc[valid] / count_per_class[valid]

    things = [per_class_avg[c] * 100 for c in range(NUM_CLASSES) if c in THING_IDS and valid[c]]
    stuff = [per_class_avg[c] * 100 for c in range(NUM_CLASSES) if c in STUFF_IDS and valid[c]]
    miou = per_class_avg[valid].mean() * 100

    result = {
        "method": "falcon_global",
        "K": K,
        "alpha": alpha,
        "beta": beta,
        "K_global": K_global,
        "feature_source": feature_source,
        "pamr": use_pamr,
        "n_images": len(feat_files),
        "n_segments": len(all_embeddings),
        "miou": round(miou, 2),
        "things_miou": round(np.mean(things), 2) if things else 0.0,
        "stuff_miou": round(np.mean(stuff), 2) if stuff else 0.0,
        "output_dir": str(out_dir),
        "per_class_iou": {
            COCOSTUFF27_CLASSNAMES[c]: round(per_class_avg[c] * 100, 1)
            for c in range(NUM_CLASSES) if count_per_class[c] > 0
        },
    }

    logger.info(
        "Global clustering (K=%d, K_global=%d, %s): mIoU=%.2f%%, "
        "Things=%.2f%%, Stuff=%.2f%%",
        K, K_global, feature_source, miou,
        result["things_miou"], result["stuff_miou"],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Falcon K-way NCut Unsupervised Semantic Segmentation"
    )
    p.add_argument("--phase", choices=["segment", "cluster", "all"], default="all")
    p.add_argument("--coco_root", required=True, help="Path to COCO dataset root")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])

    # Falcon-specific
    p.add_argument("--K", type=int, default=27, help="Number of NCut clusters")
    p.add_argument("--alpha", type=float, default=4.5, help="Affinity power exponent")
    p.add_argument("--beta", type=float, default=0.5, help="Affinity reweighting strength")
    p.add_argument("--n_iter", type=int, default=15, help="Alternating optimization iterations")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    p.add_argument("--init", default="kmeans", choices=["random", "spectral", "kmeans"])
    p.add_argument("--reg_lambda", type=float, default=0.0,
                    help="Diagonal regularization strength (paper default: try 0.1-1.0)")

    # Shared with DiffCut
    p.add_argument("--step", type=int, default=50, help="SD feature timestep")
    p.add_argument("--pamr", action="store_true", help="Use PAMR post-processing")
    p.add_argument("--crf", action="store_true", help="Use CRF post-processing")
    p.add_argument("--mask_size", type=int, default=128, help="Segmentation output resolution")
    p.add_argument("--K_global", type=int, default=27, help="Global clustering K")
    p.add_argument("--feature_source", default="dinov3", choices=["dinov3", "sd"],
                    help="Features for global clustering embeddings")
    p.add_argument("--seg_feature_source", default="sd", choices=["dinov3", "sd"],
                    help="Features for NCut segmentation (default: SD)")
    p.add_argument("--n_images", type=int, default=None, help="Limit number of images")
    p.add_argument("--dino_only", action="store_true",
                    help="Filter to images with DINOv3 features (500 val)")
    p.add_argument("--output", default=None, help="Output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = {}

    if args.phase in ("segment", "all"):
        result = falcon_segment_and_evaluate(
            coco_root=args.coco_root,
            device=args.device,
            K=args.K,
            alpha=args.alpha,
            beta=args.beta,
            n_iter=args.n_iter,
            temperature=args.temperature,
            init=args.init,
            reg_lambda=args.reg_lambda,
            use_pamr=args.pamr,
            use_crf=args.crf,
            step=args.step,
            mask_size=(args.mask_size, args.mask_size),
            n_images=args.n_images,
            dino_only=args.dino_only,
            feature_source=args.seg_feature_source,
        )
        results["per_image_hungarian"] = result

    if args.phase in ("cluster", "all"):
        result = falcon_global_clustering(
            coco_root=args.coco_root,
            device=args.device,
            K=args.K,
            alpha=args.alpha,
            beta=args.beta,
            n_iter=args.n_iter,
            temperature=args.temperature,
            init=args.init,
            reg_lambda=args.reg_lambda,
            use_pamr=args.pamr,
            use_crf=args.crf,
            K_global=args.K_global,
            step=args.step,
            mask_size=(args.mask_size, args.mask_size),
            feature_source=args.feature_source,
            seg_feature_source=args.seg_feature_source,
            n_images=args.n_images,
            dino_only=args.dino_only,
        )
        results["global_clustering"] = result

    # Save results
    if results:
        out_path = args.output or str(
            Path(args.coco_root) / "falcon_results.json"
        )
        if Path(out_path).exists():
            with open(out_path) as f:
                existing = json.load(f)
        else:
            existing = {}

        config_key = (
            f"falcon_K{args.K}_a{args.alpha}_b{args.beta}"
            f"_i{args.n_iter}_t{args.temperature}_{args.init}"
            f"{'_reg' + str(args.reg_lambda) if args.reg_lambda > 0 else ''}"
            f"{'_pamr' if args.pamr else ''}"
            f"{'_crf' if args.crf else ''}"
        )
        existing[config_key] = results

        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
