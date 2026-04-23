"""Step G -- DCCPR: Depth-Conditioned Class Prior Regularization.

Replaces Step C's binary masking (-> 255) with Bayesian reassignment:

    P(c | d, f) ~ P(d | c) * P(f | c) * P(c)

  P(d|c): GMM with m components fitted per class across all images
  P(f|c): Gaussian centered on per-class feature mean
  P(c):   Pixel-frequency prior

Pixels that would be masked by Step C (> sigma*std from class depth
profile) are instead reassigned to the highest-posterior class when
confidence exceeds a threshold, or masked otherwise.

Target: Recover masked pixels from Step C instead of discarding them.
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

FEAT_H, FEAT_W = 32, 64
NUM_CLASSES = 19


def fit_gmm_1d(
    samples: np.ndarray, k: int = 3, n_iter: int = 20
) -> tuple:
    """Fit 1D Gaussian Mixture Model via EM.

    Args:
        samples: (n,) 1D data.
        k: Number of mixture components.
        n_iter: EM iterations.

    Returns:
        (weights, means, variances) each shape (k,).
    """
    n = len(samples)
    if n < k * 5:
        # Fallback: single Gaussian
        return (
            np.array([1.0]),
            np.array([float(samples.mean())]),
            np.array([max(float(samples.var()), 1e-6)]),
        )

    # Initialize with quantile-based means
    quantiles = np.linspace(0, 1, k + 2)[1:-1]
    means = np.quantile(samples, quantiles).astype(np.float64)
    variances = np.full(k, max(samples.var() / k, 1e-4), dtype=np.float64)
    weights = np.full(k, 1.0 / k, dtype=np.float64)

    for _ in range(n_iter):
        # E-step in log-space to avoid overflow
        log_resp = np.zeros((n, k))
        for j in range(k):
            log_resp[:, j] = (
                np.log(weights[j] + 1e-300)
                - 0.5 * np.log(2 * np.pi * variances[j])
                - 0.5 * (samples - means[j]) ** 2 / variances[j]
            )
        log_resp -= log_resp.max(axis=1, keepdims=True)  # numerical stability
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True) + 1e-10

        # M-step
        Nk = resp.sum(axis=0)
        for j in range(k):
            if Nk[j] < 1e-10:
                continue
            means[j] = (resp[:, j] * samples).sum() / Nk[j]
            variances[j] = max(
                (resp[:, j] * (samples - means[j]) ** 2).sum() / Nk[j],
                1e-4,  # floor prevents degenerate components
            )
        weights = Nk / n

    return weights, means, variances


def _gaussian_pdf(
    x: np.ndarray, mu: float, var: float
) -> np.ndarray:
    """Vectorized 1D Gaussian PDF."""
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def _gmm_log_likelihood_vec(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Vectorized GMM log-likelihood for array of values.

    Args:
        x: (n,) depth values.
        weights, means, variances: GMM parameters, each (k,).

    Returns:
        (n,) log-likelihood values.
    """
    ll = np.zeros(len(x))
    for j in range(len(weights)):
        ll += weights[j] * _gaussian_pdf(x, means[j], variances[j])
    return np.log(np.maximum(ll, 1e-300))


def step_g(
    semantic: np.ndarray,
    depth: np.ndarray,
    features: np.ndarray,
    cluster_to_class: np.ndarray,
    depth_gmms: list,
    class_feat_mean: np.ndarray,
    class_feat_var: np.ndarray,
    class_priors: np.ndarray,
    num_clusters: int = 80,
    sigma_threshold: float = 3.0,
    min_confidence: float = 0.3,
) -> tuple:
    """Bayesian reassignment of depth-outlier pixels.

    Modifies semantic in-place.

    Args:
        semantic: (H, W) uint8 cluster IDs.
        depth: (H, W) float depth map.
        features: (N_patches, D) DINOv3 features.
        cluster_to_class: (256,) cluster -> trainID LUT.
        depth_gmms: List of (weights, means, variances) per class.
        class_feat_mean: (NUM_CLASSES, D) per-class feature mean.
        class_feat_var: (NUM_CLASSES,) per-class feature variance (scalar).
        class_priors: (NUM_CLASSES,) pixel frequency priors.
        num_clusters: Number of clusters.
        sigma_threshold: Sigma threshold to identify outlier pixels.
        min_confidence: Min posterior confidence for reassignment.

    Returns:
        (n_reassigned, n_masked)
    """
    H, W = semantic.shape
    mapped = cluster_to_class[semantic].astype(np.int64)
    feat_2d = features.reshape(FEAT_H, FEAT_W, -1).astype(np.float64)

    # Compute effective class depth mean/std from GMMs
    class_mean = np.zeros(NUM_CLASSES)
    class_std = np.zeros(NUM_CLASSES)
    for cls in range(NUM_CLASSES):
        w, m, v = depth_gmms[cls]
        class_mean[cls] = (w * m).sum()
        raw_var = (w * (v + m**2)).sum() - class_mean[cls] ** 2
        class_std[cls] = np.sqrt(max(raw_var, 1e-8))

    # Collect all outlier pixel coordinates
    all_outlier_y, all_outlier_x = [], []
    for cls in range(NUM_CLASSES):
        if class_std[cls] < 1e-6:
            continue
        mask = mapped == cls
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        deviation = np.abs(depth[ys, xs] - class_mean[cls])
        outlier = deviation > sigma_threshold * class_std[cls]
        if outlier.any():
            all_outlier_y.append(ys[outlier])
            all_outlier_x.append(xs[outlier])

    if not all_outlier_y:
        return 0, 0

    outlier_y = np.concatenate(all_outlier_y)
    outlier_x = np.concatenate(all_outlier_x)
    n_outliers = len(outlier_y)

    # Vectorized posterior computation
    d_vals = depth[outlier_y, outlier_x]
    pr = np.clip((outlier_y * FEAT_H / H).astype(int), 0, FEAT_H - 1)
    pc = np.clip((outlier_x * FEAT_W / W).astype(int), 0, FEAT_W - 1)
    f_vals = feat_2d[pr, pc]  # (n_outliers, D)

    log_posteriors = np.full((n_outliers, NUM_CLASSES), -np.inf)

    for c in range(NUM_CLASSES):
        if class_priors[c] < 1e-10:
            continue

        # P(d|c): GMM log-likelihood
        log_pd = _gmm_log_likelihood_vec(d_vals, *depth_gmms[c])

        # P(f|c): Gaussian log-likelihood
        f_dist = np.sum((f_vals - class_feat_mean[c : c + 1]) ** 2, axis=1)
        log_pf = -0.5 * f_dist / class_feat_var[c]

        # P(c): prior
        log_pc = np.log(class_priors[c])

        log_posteriors[:, c] = log_pd + log_pf + log_pc

    # Normalize to posteriors
    max_lp = log_posteriors.max(axis=1, keepdims=True)
    valid = max_lp.flatten() > -1e30
    posteriors = np.zeros_like(log_posteriors)
    if valid.any():
        posteriors[valid] = np.exp(log_posteriors[valid] - max_lp[valid])
        row_sums = posteriors[valid].sum(axis=1, keepdims=True) + 1e-10
        posteriors[valid] /= row_sums

    best_cls = posteriors.argmax(axis=1)
    confidence = posteriors[np.arange(n_outliers), best_cls]

    # Per-class best cluster LUT
    class_best_cluster = np.full(NUM_CLASSES, 255, dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        cls_mask = cluster_to_class[:num_clusters] == cls
        if cls_mask.any():
            class_best_cluster[cls] = int(np.where(cls_mask)[0][0])

    # Vectorized assignment
    can_reassign = (confidence >= min_confidence) & (
        class_best_cluster[best_cls] != 255
    )

    reassign_idx = np.where(can_reassign)[0]
    clusters = class_best_cluster[best_cls[reassign_idx]]
    semantic[outlier_y[reassign_idx], outlier_x[reassign_idx]] = clusters
    n_reassigned = len(reassign_idx)

    mask_idx = np.where(~can_reassign)[0]
    semantic[outlier_y[mask_idx], outlier_x[mask_idx]] = 255
    n_masked = len(mask_idx)

    return n_reassigned, n_masked
