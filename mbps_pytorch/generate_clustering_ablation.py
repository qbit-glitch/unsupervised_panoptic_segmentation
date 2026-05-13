#!/usr/bin/env python3
"""Clustering ablation: compare overclustering methods for DINOv3 pseudo-labels.

Supports 7 methods via --method flag, all producing k=80 overclustered labels
from pre-extracted DINOv3 ViT-B/16 features (2048 patches × 768 dim per image).

Methods:
    euclidean_kmeans  — Baseline MiniBatchKMeans (reproduces generate_dinov3_kmeans.py)
    spherical_kmeans  — Centroid-renormalized k-means (true spherical)
    vmf               — Von Mises-Fisher EM mixture model
    kmeans_sinkhorn   — Spherical k-means + Sinkhorn equipartition
    vmf_sinkhorn      — vMF-EM + Sinkhorn equipartition
    eagle_spectral    — EAGLE EiCue spectral enrichment + k-means
    cause_codebook    — Learnable codebook with cosine VQ + diversity loss

Usage:
    python mbps_pytorch/generate_clustering_ablation.py \
        --cityscapes_root /path/to/cityscapes \
        --method vmf --k 80 --seed 42
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.linalg import eigh
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUT_H, OUT_W = 512, 1024

PATCH_GRIDS = {
    2048: (32, 64),    # ViT-*/16 on 512×1024
    2738: (37, 74),    # ViT-*/14 on 518×1036
    8192: (64, 128),   # 2x upsampled (shift-avg / FeatUp)
    32768: (128, 256),  # 4x upsampled
}


def detect_feat_dims(feat_path: Path) -> Tuple[int, int]:
    feat = np.load(str(feat_path))
    n_patches = feat.shape[0]
    if n_patches in PATCH_GRIDS:
        return PATCH_GRIDS[n_patches]
    h = int(np.sqrt(n_patches * OUT_H / OUT_W))
    w = n_patches // h
    assert h * w == n_patches, f"Cannot factor {n_patches} patches into grid"
    return h, w


# ─── Feature loading ────────────────────────────────────────────────────────


def find_feature_files(
    cityscapes_root: Path, split: str, feat_subdir: str
) -> List[Dict]:
    feat_dir = cityscapes_root / feat_subdir / split
    files = []
    for city_dir in sorted(feat_dir.iterdir()):
        if city_dir.is_dir():
            for npy in sorted(city_dir.glob("*.npy")):
                stem = npy.stem.replace("_leftImg8bit", "")
                files.append({"feat": npy, "stem": stem, "city": city_dir.name})
    return files


def load_features_normalized(path: Path) -> np.ndarray:
    feat = np.load(str(path)).astype(np.float32)
    norms = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
    return feat / norms


def load_and_subsample_features(
    train_files: List[Dict],
    sample_frac: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_patches = np.load(str(train_files[0]["feat"])).shape[0]
    n_per_image = int(n_patches * sample_frac)
    sampled = []
    for entry in tqdm(train_files, desc="Loading train features"):
        feat = load_features_normalized(entry["feat"])
        idx = rng.choice(len(feat), min(n_per_image, len(feat)), replace=False)
        sampled.append(feat[idx])
    X = np.concatenate(sampled, axis=0)
    logger.info(f"Feature matrix: {X.shape}, dtype={X.dtype}")
    return X


# ─── M0: Euclidean K-Means (baseline) ───────────────────────────────────────


def fit_euclidean_kmeans(
    X: np.ndarray,
    k: int = 80,
    n_init: int = 5,
    max_iter: int = 100,
    batch_size: int = 4096,
    seed: int = 42,
    **kwargs,
) -> Dict:
    logger.info(f"[M0] Euclidean MiniBatchKMeans k={k}")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    kmeans.fit(X)
    logger.info(f"Inertia={kmeans.inertia_:.3f}")
    norms = np.linalg.norm(kmeans.cluster_centers_, axis=1)
    logger.info(
        f"Centroid norms: min={norms.min():.4f}, max={norms.max():.4f}, "
        f"mean={norms.mean():.4f}"
    )
    return {"centers": kmeans.cluster_centers_, "method": "euclidean_kmeans"}


# ─── M1: Spherical K-Means ──────────────────────────────────────────────────


def fit_spherical_kmeans(
    X: np.ndarray,
    k: int = 80,
    n_init: int = 5,
    max_iter: int = 100,
    batch_size: int = 4096,
    seed: int = 42,
    refine_iters: int = 20,
    **kwargs,
) -> Dict:
    logger.info(f"[M1] Spherical K-Means k={k} (+ {refine_iters} renorm iters)")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.copy()
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

    for it in range(refine_iters):
        sims = X @ centers.T
        labels = sims.argmax(axis=1)
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = X[mask].mean(axis=0)
            else:
                new_centers[c] = centers[c]
        new_centers = new_centers / (
            np.linalg.norm(new_centers, axis=1, keepdims=True) + 1e-8
        )
        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < 1e-6:
            logger.info(f"Spherical k-means converged at iter {it+1} (shift={shift:.2e})")
            break

    norms = np.linalg.norm(centers, axis=1)
    logger.info(
        f"Centroid norms: min={norms.min():.4f}, max={norms.max():.4f}, "
        f"mean={norms.mean():.4f}"
    )
    return {"centers": centers, "method": "spherical_kmeans"}


# ─── M2: Von Mises-Fisher EM ────────────────────────────────────────────────


def _estimate_kappa_banerjee(R: np.ndarray, d: int) -> np.ndarray:
    """Banerjee et al. (2005) approximation for vMF concentration."""
    R = np.clip(R, 1e-6, 1.0 - 1e-6)
    kappa = R * (d - R**2) / (1.0 - R**2)
    return np.clip(kappa, 1.0, 10000.0)


def fit_vmf_em(
    X: np.ndarray,
    k: int = 80,
    max_iter: int = 50,
    seed: int = 42,
    kappa_init: float = 100.0,
    batch_size: int = 500_000,
    **kwargs,
) -> Dict:
    logger.info(f"[M2] vMF-EM k={k}, max_iter={max_iter}, kappa_init={kappa_init}")
    N, d = X.shape
    rng = np.random.default_rng(seed)

    # Initialize from spherical k-means
    init_result = fit_spherical_kmeans(X, k=k, seed=seed, refine_iters=10)
    mu = init_result["centers"].copy()
    kappa = np.full(k, kappa_init, dtype=np.float32)
    pi = np.ones(k, dtype=np.float32) / k

    for em_iter in range(max_iter):
        # E-step (batched): compute log-responsibilities
        log_pi = np.log(pi + 1e-30)
        sum_resp = np.zeros(k, dtype=np.float64)
        weighted_sum = np.zeros((k, d), dtype=np.float64)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X[start:end]
            # log p(x|mu_k, kappa_k) ∝ kappa_k * mu_k^T x  (normalizing const cancels in softmax)
            log_probs = X_batch @ (mu * kappa[:, None]).T + log_pi[None, :]
            # Numerically stable softmax
            log_probs -= log_probs.max(axis=1, keepdims=True)
            resp = np.exp(log_probs)
            resp /= resp.sum(axis=1, keepdims=True) + 1e-30
            sum_resp += resp.sum(axis=0)
            weighted_sum += resp.T @ X_batch

        # M-step
        pi = (sum_resp / N).astype(np.float32)
        pi = np.clip(pi, 1e-6, None)
        pi /= pi.sum()

        # Mean direction and resultant length
        R_vec = weighted_sum / (sum_resp[:, None] + 1e-30)
        R_len = np.linalg.norm(R_vec, axis=1).astype(np.float32)
        mu = (R_vec / (np.linalg.norm(R_vec, axis=1, keepdims=True) + 1e-30)).astype(
            np.float32
        )
        kappa = _estimate_kappa_banerjee(R_len, d)

        # Reinitialize collapsed components
        min_size = N / (10 * k)
        collapsed = sum_resp < min_size
        if collapsed.any():
            n_collapsed = collapsed.sum()
            largest = np.argsort(-sum_resp)[:n_collapsed]
            for i, c_idx in enumerate(np.where(collapsed)[0]):
                src = largest[i % len(largest)]
                mu[c_idx] = mu[src] + rng.normal(0, 0.01, d).astype(np.float32)
                mu[c_idx] /= np.linalg.norm(mu[c_idx]) + 1e-8
                kappa[c_idx] = kappa[src]
                pi[c_idx] = pi[src] / 2
                pi[src] /= 2
            logger.info(f"  iter {em_iter}: reinitialized {n_collapsed} collapsed components")

        if em_iter % 10 == 0 or em_iter == max_iter - 1:
            logger.info(
                f"  iter {em_iter}: kappa min={kappa.min():.1f}, max={kappa.max():.1f}, "
                f"mean={kappa.mean():.1f}, pi_min={pi.min():.4f}"
            )

    return {"centers": mu, "kappas": kappa, "mixing_weights": pi, "method": "vmf"}


# ─── Sinkhorn Equipartition ─────────────────────────────────────────────────


def sinkhorn_assign(
    X: np.ndarray,
    centers: np.ndarray,
    n_iters: int = 5,
    temperature: float = 0.1,
    batch_size: int = 200_000,
) -> np.ndarray:
    """Sinkhorn-Knopp equipartitioned assignment. Returns hard labels."""
    N = X.shape[0]
    k = centers.shape[0]

    # Accumulate column sums across batches for global Sinkhorn
    # Two-pass: first compute full Q, then iterate
    # For memory: store Q as float32 (N, k) = N*k*4 bytes
    logger.info(f"Sinkhorn: N={N}, K={k}, temp={temperature}, iters={n_iters}")

    # Compute logits in batches, store full Q
    Q = np.empty((N, k), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        logits = X[start:end] @ centers.T / temperature
        logits -= logits.max(axis=1, keepdims=True)
        Q[start:end] = np.exp(logits)

    Q /= Q.sum() + 1e-30

    for _ in range(n_iters):
        Q /= Q.sum(axis=0, keepdims=True) + 1e-30
        Q /= k
        Q /= Q.sum(axis=1, keepdims=True) + 1e-30
        Q /= N

    Q /= Q.sum(axis=1, keepdims=True) + 1e-30
    return Q.argmax(axis=1)


def _reestimate_centers(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = np.zeros((k, X.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            centers[c] = X[mask].mean(axis=0)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    return centers


# ─── M3: Spherical K-Means + Sinkhorn ───────────────────────────────────────


def fit_kmeans_sinkhorn(
    X: np.ndarray,
    k: int = 80,
    seed: int = 42,
    sinkhorn_iters: int = 5,
    sinkhorn_temp: float = 0.1,
    **kwargs,
) -> Dict:
    logger.info(f"[M3] Spherical K-Means + Sinkhorn k={k}")
    result = fit_spherical_kmeans(X, k=k, seed=seed, **kwargs)
    centers = result["centers"]

    labels = sinkhorn_assign(X, centers, sinkhorn_iters, sinkhorn_temp)
    centers = _reestimate_centers(X, labels, k)

    return {"centers": centers, "method": "kmeans_sinkhorn"}


# ─── M4: vMF + Sinkhorn ─────────────────────────────────────────────────────


def fit_vmf_sinkhorn(
    X: np.ndarray,
    k: int = 80,
    seed: int = 42,
    sinkhorn_iters: int = 5,
    sinkhorn_temp: float = 0.1,
    **kwargs,
) -> Dict:
    logger.info(f"[M4] vMF-EM + Sinkhorn k={k}")
    vmf_result = fit_vmf_em(X, k=k, seed=seed, **kwargs)
    centers = vmf_result["centers"]

    labels = sinkhorn_assign(X, centers, sinkhorn_iters, sinkhorn_temp)
    centers = _reestimate_centers(X, labels, k)

    return {
        "centers": centers,
        "kappas": vmf_result["kappas"],
        "mixing_weights": vmf_result["mixing_weights"],
        "method": "vmf_sinkhorn",
    }


# ─── M5: EAGLE Spectral ─────────────────────────────────────────────────────


def _build_feature_affinity(features: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
    n = features.shape[0]
    sim = features @ features.T
    if k_neighbors < n:
        topk_idx = np.argpartition(-sim, k_neighbors, axis=1)[:, :k_neighbors]
        mask = np.zeros_like(sim, dtype=bool)
        rows = np.arange(n)[:, None]
        mask[rows, topk_idx] = True
        mask = mask | mask.T
        sim = sim * mask
    sim = np.clip(sim, 0.0, None)
    np.fill_diagonal(sim, 0.0)
    return sim


def _build_color_affinity(
    image: np.ndarray,
    patch_h: int,
    patch_w: int,
    sigma_color: float = 0.3,
    sigma_spatial: float = 5.0,
) -> np.ndarray:
    try:
        from skimage.color import rgb2lab
    except ImportError:
        logger.warning("scikit-image not available, using feature-only affinity")
        return None

    img_small = np.array(
        Image.fromarray(image).resize((patch_w, patch_h), Image.BILINEAR)
    )
    lab = rgb2lab(img_small).reshape(-1, 3)
    lab = lab / np.array([100.0, 128.0, 128.0])

    ys, xs = np.mgrid[:patch_h, :patch_w]
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    coords[:, 0] /= patch_h
    coords[:, 1] /= patch_w

    color_dist = np.sum((lab[:, None, :] - lab[None, :, :]) ** 2, axis=-1)
    spatial_dist = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)

    affinity = np.exp(
        -color_dist / (2.0 * sigma_color**2)
        - spatial_dist / (2.0 * sigma_spatial**2)
    )
    np.fill_diagonal(affinity, 0.0)
    return affinity


def _compute_spectral_features(affinity: np.ndarray, n_eig: int = 20) -> np.ndarray:
    n = affinity.shape[0]
    d = np.sum(affinity, axis=1)
    d_inv_sqrt = 1.0 / (np.sqrt(d) + 1e-8)
    l_sym = np.eye(n) - (d_inv_sqrt[:, None] * affinity) * d_inv_sqrt[None, :]
    end_idx = min(1 + n_eig, n)
    _, eigvecs = eigh(l_sym, subset_by_index=[1, end_idx - 1])
    return eigvecs


def _enrich_single_image(
    feat: np.ndarray,
    image_path: Optional[Path],
    alpha: float = 0.7,
    k_neighbors: int = 10,
    n_eig: int = 20,
) -> np.ndarray:
    a_feat = _build_feature_affinity(feat, k_neighbors)
    if image_path is not None and alpha < 1.0:
        img = np.array(Image.open(str(image_path)).convert("RGB"))
        n = feat.shape[0]
        fh, fw = PATCH_GRIDS.get(n, (int(np.sqrt(n)), n // int(np.sqrt(n))))
        a_color = _build_color_affinity(img, fh, fw)
        if a_color is not None:
            affinity = alpha * a_feat + (1.0 - alpha) * a_color
        else:
            affinity = a_feat
    else:
        affinity = a_feat
    spec = _compute_spectral_features(affinity, n_eig)
    spec_norm = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([feat, spec_norm], axis=1)


def fit_eagle_spectral_kmeans(
    X: np.ndarray,
    k: int = 80,
    seed: int = 42,
    train_files: Optional[List[Dict]] = None,
    cityscapes_root: Optional[Path] = None,
    n_eig: int = 20,
    alpha: float = 0.7,
    k_neighbors: int = 10,
    sample_frac: float = 0.5,
    max_images_spectral: int = 200,
    **kwargs,
) -> Dict:
    logger.info(
        f"[M5] EAGLE Spectral + K-Means k={k}, n_eig={n_eig}, alpha={alpha}"
    )
    if train_files is None or cityscapes_root is None:
        raise ValueError("eagle_spectral requires train_files and cityscapes_root")

    rng = np.random.default_rng(seed)
    n_patches = np.load(str(train_files[0]["feat"])).shape[0]
    n_per_image = int(n_patches * sample_frac)

    selected = train_files[:max_images_spectral]
    logger.info(f"Enriching {len(selected)} images with spectral features...")

    sampled = []
    for entry in tqdm(selected, desc="Spectral enrichment"):
        feat = load_features_normalized(entry["feat"])
        img_path = (
            cityscapes_root
            / "leftImg8bit"
            / "train"
            / entry["city"]
            / f"{entry['stem']}_leftImg8bit.png"
        )
        enriched = _enrich_single_image(feat, img_path, alpha, k_neighbors, n_eig)
        enriched = enriched / (np.linalg.norm(enriched, axis=1, keepdims=True) + 1e-8)
        idx = rng.choice(len(enriched), min(n_per_image, len(enriched)), replace=False)
        sampled.append(enriched[idx])

    X_enriched = np.concatenate(sampled, axis=0)
    logger.info(f"Enriched feature matrix: {X_enriched.shape}")

    result = fit_spherical_kmeans(X_enriched, k=k, seed=seed, refine_iters=10)
    return {
        "centers": result["centers"],
        "method": "eagle_spectral",
        "n_eig": n_eig,
        "feat_dim": X_enriched.shape[1],
    }


# ─── M6: CAUSE-Style Learnable Codebook ─────────────────────────────────────


def fit_cause_codebook(
    X: np.ndarray,
    k: int = 80,
    seed: int = 42,
    lr: float = 1e-3,
    num_epochs: int = 10,
    codebook_batch_size: int = 8192,
    diversity_weight: float = 0.1,
    **kwargs,
) -> Dict:
    logger.info(
        f"[M6] CAUSE Codebook k={k}, lr={lr}, epochs={num_epochs}, "
        f"div_w={diversity_weight}"
    )
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise RuntimeError("PyTorch required for cause_codebook method")

    init_result = fit_spherical_kmeans(X, k=k, seed=seed, refine_iters=10)
    codebook = torch.nn.Parameter(
        torch.from_numpy(init_result["centers"].copy()).float()
    )
    optimizer = torch.optim.Adam([codebook], lr=lr)

    X_tensor = torch.from_numpy(X).float()
    N = X_tensor.shape[0]
    rng = np.random.default_rng(seed)

    for epoch in range(num_epochs):
        perm = rng.permutation(N)
        epoch_vq_loss = 0.0
        epoch_div_loss = 0.0
        n_batches = 0

        for start in range(0, N, codebook_batch_size):
            end = min(start + codebook_batch_size, N)
            idx = perm[start:end]
            batch = X_tensor[idx]

            cb_normed = F.normalize(codebook, dim=1)
            sim = F.normalize(batch, dim=1) @ cb_normed.T
            assignments = sim.argmax(dim=1)
            vq_loss = -sim[torch.arange(len(batch)), assignments].mean()

            cc = cb_normed @ cb_normed.T
            div_loss = (cc - torch.eye(k)).pow(2).mean()

            loss = vq_loss + diversity_weight * div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_vq_loss += vq_loss.item()
            epoch_div_loss += div_loss.item()
            n_batches += 1

        with torch.no_grad():
            codebook.data = F.normalize(codebook.data, dim=1)

        if epoch % 3 == 0 or epoch == num_epochs - 1:
            logger.info(
                f"  epoch {epoch}: vq_loss={epoch_vq_loss/n_batches:.4f}, "
                f"div_loss={epoch_div_loss/n_batches:.4f}"
            )

    centers = codebook.detach().cpu().numpy()
    return {"centers": centers, "method": "cause_codebook"}


# ─── Shared: Assignment & Stats ──────────────────────────────────────────────


def assign_clusters_cosine(
    files: List[Dict],
    centers: np.ndarray,
    output_dir: Path,
    split: str,
    method_info: Optional[Dict] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    is_eagle = method_info is not None and method_info.get("method") == "eagle_spectral"
    n_eig = method_info.get("n_eig", 20) if is_eagle else 0
    alpha = method_info.get("alpha", 0.7) if is_eagle else 0.7
    cityscapes_root = method_info.get("cityscapes_root") if is_eagle else None

    for entry in tqdm(files, desc=f"Assigning {split}"):
        feat = load_features_normalized(entry["feat"])

        if is_eagle and cityscapes_root is not None:
            img_path = (
                cityscapes_root
                / "leftImg8bit"
                / split
                / entry["city"]
                / f"{entry['stem']}_leftImg8bit.png"
            )
            feat = _enrich_single_image(feat, img_path, alpha, 10, n_eig)
            feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)

        sims = feat @ centers.T
        cluster_ids = sims.argmax(axis=1).astype(np.uint8)
        n = feat.shape[0]
        fh, fw = PATCH_GRIDS.get(n, detect_feat_dims(entry["feat"]))
        cluster_2d = cluster_ids.reshape(fh, fw)
        label_full = np.array(
            Image.fromarray(cluster_2d).resize((OUT_W, OUT_H), Image.NEAREST)
        )

        city_dir = output_dir / split / entry["city"]
        city_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_full).save(str(city_dir / f"{entry['stem']}.png"))


def compute_cluster_stats(
    files: List[Dict],
    centers: np.ndarray,
    method_info: Optional[Dict] = None,
) -> Dict:
    k = centers.shape[0]
    counts = np.zeros(k, dtype=np.int64)

    is_eagle = method_info is not None and method_info.get("method") == "eagle_spectral"

    for entry in tqdm(files, desc="Computing stats"):
        feat = load_features_normalized(entry["feat"])
        if is_eagle and method_info.get("cityscapes_root") is not None:
            img_path = (
                method_info["cityscapes_root"]
                / "leftImg8bit"
                / "train"
                / entry["city"]
                / f"{entry['stem']}_leftImg8bit.png"
            )
            feat = _enrich_single_image(
                feat, img_path, method_info.get("alpha", 0.7), 10,
                method_info.get("n_eig", 20),
            )
            feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        sims = feat @ centers.T
        labels = sims.argmax(axis=1)
        for c in labels:
            counts[c] += 1

    total = counts.sum()
    probs = counts / (total + 1e-30)
    entropy = -np.sum(probs * np.log(probs + 1e-30)) / np.log(k)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    gini = (2.0 * np.sum((np.arange(1, n + 1)) * sorted_counts) / (n * total + 1e-30)) - (n + 1) / n

    return {
        "sizes": counts.tolist(),
        "entropy": float(entropy),
        "gini": float(gini),
        "min_size": int(counts.min()),
        "max_size": int(counts.max()),
        "median_size": int(np.median(counts)),
        "empty_clusters": int((counts == 0).sum()),
        "total_features": int(total),
    }


# ─── Method registry ─────────────────────────────────────────────────────────

METHODS = {
    "euclidean_kmeans": fit_euclidean_kmeans,
    "spherical_kmeans": fit_spherical_kmeans,
    "vmf": fit_vmf_em,
    "kmeans_sinkhorn": fit_kmeans_sinkhorn,
    "vmf_sinkhorn": fit_vmf_sinkhorn,
    "eagle_spectral": fit_eagle_spectral_kmeans,
    "cause_codebook": fit_cause_codebook,
}


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Clustering ablation for DINOv3 pseudo-labels"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--feat_subdir", type=str, default="dinov3_features")
    parser.add_argument(
        "--method", type=str, required=True, choices=list(METHODS.keys())
    )
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--sample_frac", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    # vMF
    parser.add_argument("--kappa_init", type=float, default=100.0)
    parser.add_argument("--vmf_max_iter", type=int, default=50)
    # Sinkhorn
    parser.add_argument("--sinkhorn_iters", type=int, default=5)
    parser.add_argument("--sinkhorn_temp", type=float, default=0.1)
    # EAGLE
    parser.add_argument("--n_eig", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--max_images_spectral", type=int, default=200)
    # CAUSE codebook
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    # Skip stats
    parser.add_argument("--skip_stats", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Extra suffix for output directory name")
    args = parser.parse_args()

    root = Path(args.cityscapes_root)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_subdir = f"pseudo_semantic_raw_dinov3_k{args.k}_{args.method}{suffix}"
    out_dir = root / out_subdir

    train_files = find_feature_files(root, "train", args.feat_subdir)
    logger.info(f"Found {len(train_files)} train images")

    t0 = time.time()

    fit_fn = METHODS[args.method]
    fit_kwargs = dict(
        k=args.k,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
    )

    if args.method == "eagle_spectral":
        result = fit_fn(
            X=None,
            train_files=train_files,
            cityscapes_root=root,
            k=args.k,
            n_eig=args.n_eig,
            alpha=args.alpha,
            k_neighbors=args.k_neighbors,
            sample_frac=args.sample_frac,
            max_images_spectral=args.max_images_spectral,
            seed=args.seed,
        )
    else:
        X = load_and_subsample_features(train_files, args.sample_frac, args.seed)
        if args.method in ("vmf", "vmf_sinkhorn"):
            fit_kwargs["kappa_init"] = args.kappa_init
            fit_kwargs["max_iter"] = args.vmf_max_iter
        if args.method in ("kmeans_sinkhorn", "vmf_sinkhorn"):
            fit_kwargs["sinkhorn_iters"] = args.sinkhorn_iters
            fit_kwargs["sinkhorn_temp"] = args.sinkhorn_temp
        if args.method == "cause_codebook":
            fit_kwargs["lr"] = args.lr
            fit_kwargs["num_epochs"] = args.num_epochs
            fit_kwargs["diversity_weight"] = args.diversity_weight
            fit_kwargs["codebook_batch_size"] = args.batch_size
        result = fit_fn(X, **fit_kwargs)

    fit_time = time.time() - t0
    logger.info(f"Fitting done in {fit_time:.1f}s")

    centers = result["centers"]
    out_dir.mkdir(parents=True, exist_ok=True)

    save_data = {"centers": centers}
    if "kappas" in result:
        save_data["kappas"] = result["kappas"]
    if "mixing_weights" in result:
        save_data["mixing_weights"] = result["mixing_weights"]
    np.savez(str(out_dir / "centroids.npz"), **save_data)
    logger.info(f"Saved centroids to {out_dir / 'centroids.npz'}")

    method_info = {
        "method": args.method,
        "cityscapes_root": root,
        "n_eig": args.n_eig,
        "alpha": args.alpha,
    }

    for split in args.splits:
        files = find_feature_files(root, split, args.feat_subdir)
        logger.info(f"Assigning {split}: {len(files)} images")
        assign_clusters_cosine(files, centers, out_dir, split, method_info)

    if not args.skip_stats:
        logger.info("Computing cluster statistics on train set...")
        stats = compute_cluster_stats(train_files[:200], centers, method_info)
        stats["fit_time_seconds"] = fit_time
        stats["method"] = args.method
        stats["k"] = args.k
        stats_path = out_dir / "cluster_stats.json"
        with open(str(stats_path), "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(
            f"Stats: entropy={stats['entropy']:.3f}, gini={stats['gini']:.3f}, "
            f"empty={stats['empty_clusters']}, "
            f"sizes=[{stats['min_size']}..{stats['median_size']}..{stats['max_size']}]"
        )

    logger.info(f"All done → {out_dir}")


if __name__ == "__main__":
    main()
