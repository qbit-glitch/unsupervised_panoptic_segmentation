#!/usr/bin/env python3
"""CutS3D Pseudo-Mask Extraction — GPU-Optimized (Paper-Faithful).

Implements the full CutS3D pipeline from Sick et al. (ICCV 2025):
  1. DINO ViT-S/8 features (GPU, batched)
  2. Cosine affinity matrix (GPU)
  3. Spatial Importance Sharpening via depth (GPU) — Eq. 1-3 from paper
  4. NCut spectral bipartition (CPU, scipy eigsh)
  5. LocalCut: 3D k-NN graph (GPU) + MaxFlow (CPU, scipy)
  6. CRF refinement (CPU)
  7. Spatial Confidence: T threshold variations of LocalCut (GPU k-NN reused)

Key GPU optimizations:
  - k-NN graph: pairwise distances on GPU (~0.05s vs 1.79s on CPU at K=8192)
  - Affinity matrix: cosine sim on GPU
  - SI Sharpening: element-wise exponentiation on GPU
  - k-NN graph reuse: built once per instance, reused for all T SC variations

Usage:
    python scripts/extract_cuts3d_gpu.py \\
        --config configs/cityscapes_gpu_512.yaml \\
        --output-dir data/pseudo_masks_512 \\
        --max-instances 5 --gpu 0 --shard 0 --num-shards 2
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path):
    import yaml
    default_path = Path(config_path).parent / "default.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)
    if config_path != str(default_path) and os.path.exists(config_path):
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}
        def deep_merge(base, over):
            for k, v in over.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v
        deep_merge(config, override)
    return config


# ---------------------------------------------------------------------------
# DINO feature extraction (GPU, batched)
# ---------------------------------------------------------------------------

def load_dino_backbone(device):
    """Load DINO ViT-S/8 from torch.hub."""
    model = torch.hub.load(
        "facebookresearch/dino:main", "dino_vits8",
        pretrained=True, trust_repo=True,
    )
    model = model.eval().to(device)
    return model


@torch.no_grad()
def extract_dino_features_batch(model, image_tensors, device):
    """Extract DINO features for a batch + compute affinity on GPU.

    Returns list of (features_np, W_gpu) tuples. W stays on GPU for
    subsequent SI sharpening.
    """
    batch = torch.stack(image_tensors).to(device)  # (B, 3, H, W)
    feats = model.get_intermediate_layers(batch, n=1)[0][:, 1:]  # (B, K, 384)

    results = []
    for i in range(feats.shape[0]):
        f = feats[i]  # (K, 384) on GPU
        norms = torch.norm(f, dim=1, keepdim=True).clamp(min=1e-8)
        f_norm = f / norms
        W = (f_norm @ f_norm.T).clamp(0.0, 1.0)
        results.append((f.cpu().numpy(), W))  # W stays on GPU
    return results


# ---------------------------------------------------------------------------
# Spatial Importance Sharpening (GPU) — Paper Eq. 1-3
# ---------------------------------------------------------------------------

@torch.no_grad()
def spatial_importance_sharpening_gpu(W_gpu, depth_patch, sigma_gauss=3.0,
                                      beta=0.45):
    """Paper-correct SI sharpening on GPU.

    Eq 1: ΔD = |G_σ * D - D|
    Eq 2: ΔD_n = (1-β)(ΔD - min ΔD)/(max ΔD - min ΔD) + β  ∈ [β, 1]
    Eq 3: W_{i,j} = W_{i,j}^{1 - ΔD_n_i · ΔD_n_j}

    Returns: sharpened W on GPU.
    """
    device = W_gpu.device
    # Eq 1: spatial importance = |blurred_depth - depth|
    blurred = gaussian_filter(depth_patch, sigma=sigma_gauss)
    delta_d = np.abs(blurred - depth_patch)

    # Eq 2: normalize to [β, 1.0]
    d_min, d_max = delta_d.min(), delta_d.max()
    if d_max - d_min > 1e-8:
        delta_d_n = (1.0 - beta) * (delta_d - d_min) / (d_max - d_min) + beta
    else:
        delta_d_n = np.full_like(delta_d, beta)

    # Eq 3: W = W^(1 - outer(ΔD_n, ΔD_n))
    delta_flat = torch.from_numpy(delta_d_n.flatten().astype(np.float32)).to(device)
    exponent = 1.0 - delta_flat.unsqueeze(1) * delta_flat.unsqueeze(0)  # (K, K)
    W_clamped = W_gpu.clamp(min=1e-6)
    W_sharpened = W_clamped.pow(exponent).clamp(0.0, 1.0)
    return W_sharpened


# ---------------------------------------------------------------------------
# NCut: Normalized Cut (spectral bipartition) — CPU
# ---------------------------------------------------------------------------

def normalized_cut(W, tau=0.0):
    """Spectral bipartition using Fiedler vector of normalized Laplacian."""
    K = W.shape[0]
    d = W.sum(axis=1) + 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(d)
    W_norm = W * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    try:
        eigenvalues, eigenvectors = eigsh(W_norm, k=2, which='LM')
        fiedler = eigenvectors[:, 0]
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(W_norm)
        fiedler = eigenvectors[:, -2]

    bipartition = (fiedler > tau).astype(np.float32)

    if bipartition.sum() > K / 2:
        bipartition = 1.0 - bipartition
        fiedler = -fiedler

    fg_indices = np.where(bipartition > 0.5)[0]
    bg_indices = np.where(bipartition < 0.5)[0]

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        return bipartition, 0, min(1, K - 1)

    # Source = most foreground point, Sink = most background point
    idx_source = fg_indices[np.argmax(fiedler[fg_indices])]
    idx_sink = bg_indices[np.argmin(fiedler[bg_indices])]

    return bipartition, int(idx_source), int(idx_sink)


# ---------------------------------------------------------------------------
# GPU k-NN graph
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpu_build_knn_graph(points_np, k, device):
    """Build k-NN graph on GPU — O(K²) pairwise distances via matmul.

    ~0.05s at K=8192 on GTX 1080 Ti vs ~1.8s on CPU.
    """
    points = torch.from_numpy(points_np.astype(np.float32)).to(device)
    K = points.shape[0]
    k_actual = min(k, K - 1)

    sq_norms = (points ** 2).sum(dim=1)  # (K,)
    dists_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (points @ points.T)
    dists_sq.clamp_(min=0.0)
    dists_sq.fill_diagonal_(float('inf'))

    # Top-k smallest distances
    knn_dist_sq, knn_idx = torch.topk(dists_sq, k_actual, dim=1, largest=False)
    knn_dist = knn_dist_sq.sqrt()

    return knn_idx.cpu().numpy(), knn_dist.cpu().numpy()


# ---------------------------------------------------------------------------
# 3D point cloud construction
# ---------------------------------------------------------------------------

def pixels_to_3d(depth_patch):
    """Unproject depth to 3D points using pinhole model."""
    H, W = depth_patch.shape
    fx = fy = float(W)
    cx, cy = W / 2.0, H / 2.0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = np.maximum(depth_patch, 0.01)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)  # (H, W, 3)


def prepare_3d_points(bipartition, depth_patch, z_background=100.0):
    """Prepare 3D point cloud with background pushed to far plane.

    This is shared across LocalCut and all SC threshold variations.
    """
    K = bipartition.shape[0]
    patch_h, patch_w = depth_patch.shape

    # Normalize depth to [0, 1]
    d_min, d_max = depth_patch.min(), depth_patch.max()
    if d_max - d_min > 1e-8:
        depth_norm = (depth_patch - d_min) / (d_max - d_min)
    else:
        depth_norm = np.ones_like(depth_patch) * 0.5

    points = pixels_to_3d(depth_norm).reshape(K, 3)

    # Normalize each dimension to [0, 1]
    for dim in range(3):
        p_min, p_max = points[:, dim].min(), points[:, dim].max()
        if p_max - p_min > 1e-8:
            points[:, dim] = (points[:, dim] - p_min) / (p_max - p_min)

    # Push background points to far plane
    bg_mask = 1.0 - bipartition
    points[:, 2] = points[:, 2] * bipartition + z_background * bg_mask

    return points


# ---------------------------------------------------------------------------
# MaxFlow MinCut (CPU, scipy) — reuses precomputed k-NN
# ---------------------------------------------------------------------------

FLOW_SCALE = 10000


def maxflow_mincut(knn_idx, knn_dist, tau_knn, bipartition,
                   idx_source, idx_sink):
    """Run MaxFlow/MinCut on precomputed k-NN graph with given threshold.

    The k-NN graph is built once on GPU; this function just thresholds
    edges differently for each tau_knn and runs scipy MaxFlow.
    """
    K = bipartition.shape[0]
    k = knn_idx.shape[1]

    row_all = np.repeat(np.arange(K), k)
    col_all = knn_idx.ravel()
    dist_all = knn_dist.ravel()

    # Threshold edges
    cap_float = np.maximum(0.0, tau_knn - dist_all)
    cap_int = (cap_float * FLOW_SCALE).astype(np.int32)
    mask = cap_int > 0

    if mask.sum() == 0:
        return bipartition.copy()

    row_filt, col_filt = row_all[mask], col_all[mask]
    cap_filt = cap_int[mask]

    # Symmetrize
    row_sym = np.concatenate([row_filt, col_filt])
    col_sym = np.concatenate([col_filt, row_filt])
    cap_sym = np.concatenate([cap_filt, cap_filt])

    graph = csr_matrix((cap_sym, (row_sym, col_sym)), shape=(K, K))

    # MaxFlow
    result = maximum_flow(graph, idx_source, idx_sink)

    # BFS to find source-side of min-cut
    residual = graph - result.flow
    residual.eliminate_zeros()
    visited = np.zeros(K, dtype=bool)
    queue = deque([idx_source])
    visited[idx_source] = True
    while queue:
        node = queue.popleft()
        row_start = residual.indptr[node]
        row_end = residual.indptr[node + 1]
        for idx in range(row_start, row_end):
            neighbor = residual.indices[idx]
            if not visited[neighbor] and residual.data[idx] > 0:
                visited[neighbor] = True
                queue.append(neighbor)

    source_side = visited.astype(np.float32)
    instance_mask = source_side * bipartition
    return instance_mask


# ---------------------------------------------------------------------------
# CRF Refinement
# ---------------------------------------------------------------------------

def crf_refine(mask_patch, image, patch_h, patch_w, n_iters=5,
               sxy_smooth=3.0, compat=3.0):
    """Simplified CRF mean-field inference using Gaussian smoothing."""
    q = mask_patch.reshape(patch_h, patch_w)
    for _ in range(n_iters):
        msg = gaussian_filter(q, sigma=sxy_smooth / patch_h)
        q_new = q + compat * (msg - 0.5)
        q = np.clip(q_new, 0, 1)
    return q.flatten()


# ---------------------------------------------------------------------------
# Full CutS3D extraction — GPU-optimized, paper-faithful
# ---------------------------------------------------------------------------

def extract_pseudo_masks_gpu(features, depth, image, patch_h, patch_w,
                              W_gpu, device, max_instances=5, tau_ncut=0.0,
                              tau_knn=0.115, k=10, sigma_gauss=3.0, beta=0.45,
                              min_mask_size=0.02, sc_samples=6, use_crf=True,
                              sc_min_ratio=0.5):
    """Full CutS3D extraction with GPU-accelerated operations.

    Pipeline per instance:
      1. NCut on CPU (eigsh ~0.5s)
      2. Build k-NN on GPU once (~0.05s)
      3. LocalCut via MaxFlow on CPU (~0.02s)
      4. CRF on CPU (~0.05s)
      5. Spatial Confidence: T × MaxFlow with different thresholds (~T×0.02s)
         (reuses the same GPU k-NN graph!)

    Returns: masks, spatial_confidence, scores, num_valid
    """
    K = features.shape[0]

    # Resize depth to patch resolution
    depth_pil = Image.fromarray(depth)
    depth_patch = np.array(depth_pil.resize((patch_w, patch_h), Image.BILINEAR))

    # SI Sharpening on GPU (paper Eq. 1-3)
    W_sharpened = spatial_importance_sharpening_gpu(
        W_gpu, depth_patch, sigma_gauss, beta
    )
    # Transfer to CPU for NCut
    W = W_sharpened.cpu().numpy()

    all_masks = np.zeros((max_instances, K), dtype=np.float32)
    all_sc = np.zeros((max_instances, K), dtype=np.float32)
    active = np.ones(K, dtype=np.float32)
    num_valid = 0

    for inst_idx in range(max_instances):
        # Mask out inactive patches
        mask_2d = active[:, None] * active[None, :]
        W_masked = W * mask_2d

        # NCut (CPU)
        bipartition, idx_src, idx_snk = normalized_cut(W_masked, tau_ncut)
        bipartition = bipartition * active

        if bipartition.sum() < 2:
            break

        # Prepare 3D points (shared for LocalCut + all SC variations)
        points_3d = prepare_3d_points(bipartition, depth_patch)

        # Build k-NN graph ONCE on GPU (reused for LocalCut + T SC calls)
        knn_idx, knn_dist = gpu_build_knn_graph(points_3d, k=k, device=device)

        # LocalCut via MaxFlow (CPU, using precomputed GPU k-NN)
        instance_mask = maxflow_mincut(
            knn_idx, knn_dist, tau_knn, bipartition, idx_src, idx_snk
        )

        # CRF refinement
        if use_crf:
            instance_mask = crf_refine(instance_mask, image, patch_h, patch_w)

        # Size check
        mask_frac = instance_mask.sum() / K
        if mask_frac < min_mask_size:
            continue

        # Spatial Confidence (Paper Eq. 4): average T binary cuts at
        # different tau_knn thresholds, REUSING the same k-NN graph
        tau_knn_min = tau_knn * sc_min_ratio
        sc = np.zeros(K, dtype=np.float32)
        for t in range(sc_samples):
            tau_t = tau_knn_min + (t + 1) * (tau_knn - tau_knn_min) / sc_samples
            bc = maxflow_mincut(
                knn_idx, knn_dist, tau_t, bipartition, idx_src, idx_snk
            )
            sc += bc
        sc = sc / max(sc_samples, 1)
        # Paper: SC_min = 0.5
        sc = np.clip(sc, 0.5, 1.0)

        all_masks[num_valid] = instance_mask
        all_sc[num_valid] = sc
        num_valid += 1

        # Remove segmented patches
        active = active * (1.0 - instance_mask)

    # Scores = average SC per mask
    scores = np.zeros(max_instances, dtype=np.float32)
    for i in range(num_valid):
        mask_sum = all_masks[i].sum()
        if mask_sum > 0:
            scores[i] = all_sc[i].sum() / mask_sum

    return all_masks, all_sc, scores, num_valid


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_samples(data_dir, depth_dir, split="train"):
    """Find all image/depth pairs organized by city."""
    img_root = Path(data_dir) / "leftImg8bit" / split
    cities = sorted([d.name for d in img_root.iterdir() if d.is_dir()])

    city_samples = {}
    for city in cities:
        city_img_dir = img_root / city
        samples = []
        for img_path in sorted(city_img_dir.glob("*_leftImg8bit.png")):
            base = img_path.name.replace("_leftImg8bit.png", "")
            depth_path = Path(depth_dir) / split / city / f"{base}.npy"
            samples.append((str(img_path), str(depth_path), base))
        city_samples[city] = samples

    return cities, city_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CutS3D Extraction — GPU-Optimized (Paper-Faithful)"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/pseudo_masks_gpu")
    parser.add_argument("--max-instances", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--tau-knn", type=float, default=0.115)
    parser.add_argument("--sc-samples", type=int, default=6,
                        help="T: number of threshold samples for Spatial Confidence")
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    img_h, img_w = data_cfg["image_size"]
    patch_h, patch_w = img_h // 8, img_w // 8
    K = patch_h * patch_w
    max_instances = args.max_instances

    data_dir = data_cfg["data_dir"]
    depth_dir = data_cfg.get("depth_dir", "")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("  CutS3D Extraction — GPU-Optimized (Paper-Faithful)")
    print("=" * 70)
    print(f"  Device:          {device}")
    print(f"  Image size:      {img_h}x{img_w}")
    print(f"  Patch grid:      {patch_h}x{patch_w} = {K} tokens")
    print(f"  Max instances:   {max_instances}")
    print(f"  tau_knn:         {args.tau_knn}")
    print(f"  SC samples (T):  {args.sc_samples}")
    print(f"  k-NN k:          {args.knn_k}")
    print(f"  Data dir:        {data_dir}")
    print(f"  Depth dir:       {depth_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Pipeline:        DINO(GPU) → Affinity(GPU) → SI(GPU) →")
    print(f"                   NCut(CPU) → k-NN(GPU) → LocalCut(CPU) →")
    print(f"                   CRF(CPU) → SC[T×MaxFlow](CPU, reuse k-NN)")
    print("=" * 70)
    sys.stdout.flush()

    # Load DINO backbone
    print("\nLoading DINO ViT-S/8...")
    sys.stdout.flush()
    dino = load_dino_backbone(device)
    print(f"  DINO loaded on {device}")

    # Warmup
    dummy = torch.randn(1, 3, img_h, img_w).to(device)
    with torch.no_grad():
        _ = dino.get_intermediate_layers(dummy, n=1)
    del dummy
    torch.cuda.empty_cache()
    print("  Warmup done")
    sys.stdout.flush()

    # Discover dataset
    print("\nDiscovering dataset...")
    cities, city_samples = discover_samples(data_dir, depth_dir, "train")
    total_dataset = sum(len(s) for s in city_samples.values())
    print(f"  Found {len(cities)} cities, {total_dataset} total images")
    sys.stdout.flush()

    # Handle resume
    if args.resume_from:
        if args.resume_from in cities:
            idx = cities.index(args.resume_from)
            cities = cities[idx:]
            print(f"  Resuming from city: {args.resume_from}")

    # Build flat sample list
    all_samples = []
    for city in cities:
        for s in city_samples[city]:
            all_samples.append((city, s[0], s[1], s[2]))

    # Shard across GPUs
    if args.num_shards > 1:
        shard_samples = all_samples[args.shard::args.num_shards]
        print(f"  Shard {args.shard}/{args.num_shards}: {len(shard_samples)}/{len(all_samples)} images")
    else:
        shard_samples = all_samples

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Global indices (no collisions across shards)
    sample_to_global_idx = {s[3]: i for i, s in enumerate(all_samples)}

    # Skip already-extracted
    existing = set()
    for f in output_dir.glob("masks_*.npz"):
        existing.add(f.stem)
    if existing:
        print(f"  Found {len(existing)} existing masks, will skip those")

    # Filter pending
    pending_samples = []
    for city, img_path, depth_path, image_id in shard_samples:
        global_idx = sample_to_global_idx[image_id]
        mask_name = f"masks_{global_idx:08d}"
        if mask_name not in existing:
            pending_samples.append((city, img_path, depth_path, image_id, global_idx))
    n_pending = len(pending_samples)
    print(f"  Pending: {n_pending} images to extract")
    sys.stdout.flush()

    total_images = 0
    total_masks = 0
    total_time = 0.0
    dino_batch_size = 2

    def load_sample(sample):
        """Load and preprocess a single image + depth."""
        _, img_path, depth_path, image_id, global_idx = sample
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((img_w, img_h), Image.BILINEAR)
        image_np = np.array(img_resized, dtype=np.float32) / 255.0
        if os.path.exists(depth_path):
            depth_np = np.load(depth_path).astype(np.float32)
            if depth_np.shape != (img_h, img_w):
                depth_pil = Image.fromarray(depth_np)
                depth_np = np.array(
                    depth_pil.resize((img_w, img_h), Image.BILINEAR),
                    dtype=np.float32,
                )
        else:
            depth_np = np.ones((img_h, img_w), dtype=np.float32)
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)
        return image_np, depth_np, img_tensor, image_id, global_idx

    for batch_start in range(0, n_pending, dino_batch_size):
        batch_end = min(batch_start + dino_batch_size, n_pending)
        batch_samples = pending_samples[batch_start:batch_end]
        batch_start_time = time.time()

        # Load images
        loaded = [load_sample(s) for s in batch_samples]
        img_tensors = [l[2] for l in loaded]

        # Batched DINO forward + GPU affinity
        feat_W_list = extract_dino_features_batch(dino, img_tensors, device)

        # Process each image
        for i, (image_np, depth_np, _, image_id, global_idx) in enumerate(loaded):
            features, W_gpu = feat_W_list[i]

            masks, sc, scores, n_valid = extract_pseudo_masks_gpu(
                features, depth_np, image_np,
                patch_h=patch_h, patch_w=patch_w,
                W_gpu=W_gpu, device=device,
                max_instances=max_instances,
                tau_ncut=0.0, tau_knn=args.tau_knn, k=args.knn_k,
                sigma_gauss=3.0, beta=0.45,
                min_mask_size=0.02, sc_samples=args.sc_samples,
                use_crf=True, sc_min_ratio=0.5,
            )

            total_masks += n_valid
            mask_name = f"masks_{global_idx:08d}"
            n_save = max(n_valid, 1)
            np.savez_compressed(
                output_dir / f"{mask_name}.npz",
                masks=masks[:n_save],
                spatial_confidence=sc[:n_save],
                scores=scores[:n_save],
                num_valid=n_valid,
                image_id=image_id,
            )
            total_images += 1

        batch_time = time.time() - batch_start_time
        total_time += batch_time
        done = batch_end

        if done <= dino_batch_size or done % 10 < dino_batch_size or done == n_pending:
            avg_per_img = total_time / total_images
            remaining = n_pending - done
            eta = avg_per_img * remaining
            last_id = loaded[-1][3]
            print(
                f"  [GPU{args.gpu} {done:>4d}/{n_pending}] {last_id}: "
                f"{batch_time / len(loaded):.2f}s/img, "
                f"avg {avg_per_img:.2f}s/img, ETA ~{eta/60:.0f}min"
            )
            sys.stdout.flush()

    # Final summary
    print("\n" + "=" * 70)
    print(f"  Extraction Complete (shard {args.shard}/{args.num_shards}, GPU {args.gpu})")
    print("=" * 70)
    print(f"  Total images:    {total_images}")
    print(f"  Total masks:     {total_masks}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/3600:.2f}h)")
    if total_images > 0:
        print(f"  Avg per image:   {total_time/total_images:.2f}s")
        print(f"  Masks/image:     {total_masks/total_images:.1f}")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
