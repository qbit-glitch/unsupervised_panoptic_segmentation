#!/usr/bin/env python3
"""CutS3D Pseudo-Mask Extraction — COCO/ImageNet, GPU-Optimized.

Full paper-faithful CutS3D pipeline (Sick et al., ICCV 2025) adapted for
flat directory datasets (COCO train2017, ImageNet).

Pipeline per image:
  DINO(GPU) → Affinity(GPU) → SI Sharpening(GPU, Eq 1-3) →
  NCut(CPU, eigsh) → k-NN(GPU) → LocalCut(CPU, scipy maxflow) →
  CRF(CPU) → SC[T×MaxFlow](CPU, reuse k-NN)

Depth is computed on-the-fly using MiDaS DPT-Large (GPU, ~20ms/img).

Multi-GPU: use --shard / --num-shards to split across GPUs.

Usage:
    # Single GPU — COCO
    python scripts/extract_cuts3d_coco_gpu.py \
        --data-dir datasets/coco/train2017 \
        --output-dir data/pseudo_masks_coco \
        --image-size 480 --gpu 0

    # Two GPUs — COCO
    python scripts/extract_cuts3d_coco_gpu.py \
        --data-dir datasets/coco/train2017 \
        --output-dir data/pseudo_masks_coco \
        --image-size 480 --gpu 0 --shard 0 --num-shards 2 &
    python scripts/extract_cuts3d_coco_gpu.py \
        --data-dir datasets/coco/train2017 \
        --output-dir data/pseudo_masks_coco \
        --image-size 480 --gpu 1 --shard 1 --num-shards 2 &

    # ImageNet (paper's training set)
    python scripts/extract_cuts3d_coco_gpu.py \
        --data-dir datasets/imagenet/train \
        --output-dir data/pseudo_masks_imagenet \
        --image-size 480 --gpu 0 --shard 0 --num-shards 2 \
        --recursive
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# DINO feature extraction (GPU, PyTorch)
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
    batch = torch.stack(image_tensors).to(device)
    feats = model.get_intermediate_layers(batch, n=1)[0][:, 1:]  # (B, K, 384)

    results = []
    for i in range(feats.shape[0]):
        f = feats[i]
        norms = torch.norm(f, dim=1, keepdim=True).clamp(min=1e-8)
        f_norm = f / norms
        W = (f_norm @ f_norm.T).clamp(0.0, 1.0)
        results.append((f.cpu().numpy(), W))
    return results


# ---------------------------------------------------------------------------
# MiDaS depth estimation (GPU, on-the-fly)
# ---------------------------------------------------------------------------

def load_depth_model(device, model_type="DPT_Large"):
    """Load MiDaS depth model from torch.hub.

    Args:
        device: torch device.
        model_type: "DPT_Large" (best quality) or "MiDaS_small" (fastest).

    Returns:
        (model, transform) tuple.
    """
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model = model.eval().to(device)
    transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type == "DPT_Large":
        transform = transform.dpt_transform
    else:
        transform = transform.small_transform
    return model, transform


@torch.no_grad()
def estimate_depth_batch(depth_model, depth_transform, images_np, device,
                         target_h, target_w):
    """Estimate monocular depth for a batch of images.

    Args:
        depth_model: MiDaS model.
        depth_transform: MiDaS preprocessing transform.
        images_np: List of numpy images (H, W, 3) float32 [0,1].
        device: torch device.
        target_h, target_w: Output size.

    Returns:
        List of depth maps as numpy (target_h, target_w) float32.
    """
    depths = []
    for img_np in images_np:
        # MiDaS expects uint8 PIL or numpy
        img_uint8 = (img_np * 255).astype(np.uint8)
        input_batch = depth_transform(img_uint8).to(device)

        prediction = depth_model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()
        # Invert MiDaS output (it returns inverse depth)
        depth = depth.max() - depth  # Convert to metric-like depth
        depth = np.clip(depth, 0.01, None)
        depths.append(depth.astype(np.float32))

    return depths


# ---------------------------------------------------------------------------
# Spatial Importance Sharpening (GPU) — Paper Eq. 1-3
# ---------------------------------------------------------------------------

@torch.no_grad()
def spatial_importance_sharpening_gpu(W_gpu, depth_patch, sigma_gauss=3.0,
                                      beta=0.45):
    """Paper Eq. 1-3: sharpen affinity using depth-derived SI.

    Eq 1: ΔD = |G_σ * D - D|
    Eq 2: ΔD_n = (1-β)(ΔD - min ΔD)/(max ΔD - min ΔD) + β  ∈ [β, 1]
    Eq 3: W_{i,j} = W_{i,j}^{1 - ΔD_n_i · ΔD_n_j}
    """
    device = W_gpu.device
    blurred = gaussian_filter(depth_patch, sigma=sigma_gauss)
    delta_d = np.abs(blurred - depth_patch)

    d_min, d_max = delta_d.min(), delta_d.max()
    if d_max - d_min > 1e-8:
        delta_d_n = (1.0 - beta) * (delta_d - d_min) / (d_max - d_min) + beta
    else:
        delta_d_n = np.full_like(delta_d, beta)

    delta_flat = torch.from_numpy(
        delta_d_n.flatten().astype(np.float32)
    ).to(device)
    exponent = 1.0 - delta_flat.unsqueeze(1) * delta_flat.unsqueeze(0)
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

    idx_source = fg_indices[np.argmax(fiedler[fg_indices])]
    idx_sink = bg_indices[np.argmin(fiedler[bg_indices])]

    return bipartition, int(idx_source), int(idx_sink)


# ---------------------------------------------------------------------------
# GPU k-NN graph
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpu_build_knn_graph(points_np, k, device):
    """Build k-NN graph on GPU — O(K²) pairwise distances via matmul."""
    points = torch.from_numpy(points_np.astype(np.float32)).to(device)
    K = points.shape[0]
    k_actual = min(k, K - 1)

    sq_norms = (points ** 2).sum(dim=1)
    dists_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (points @ points.T)
    dists_sq.clamp_(min=0.0)
    dists_sq.fill_diagonal_(float('inf'))

    knn_dist_sq, knn_idx = torch.topk(dists_sq, k_actual, dim=1, largest=False)
    knn_dist = knn_dist_sq.sqrt()

    return knn_idx.cpu().numpy(), knn_dist.cpu().numpy()


# ---------------------------------------------------------------------------
# 3D point cloud
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
    return np.stack([x, y, z], axis=-1)


def prepare_3d_points(bipartition, depth_patch, z_background=100.0):
    """Prepare 3D point cloud with background pushed to far plane."""
    K = bipartition.shape[0]
    d_min, d_max = depth_patch.min(), depth_patch.max()
    if d_max - d_min > 1e-8:
        depth_norm = (depth_patch - d_min) / (d_max - d_min)
    else:
        depth_norm = np.ones_like(depth_patch) * 0.5

    points = pixels_to_3d(depth_norm).reshape(K, 3)

    for dim in range(3):
        p_min, p_max = points[:, dim].min(), points[:, dim].max()
        if p_max - p_min > 1e-8:
            points[:, dim] = (points[:, dim] - p_min) / (p_max - p_min)

    bg_mask = 1.0 - bipartition
    points[:, 2] = points[:, 2] * bipartition + z_background * bg_mask

    return points


# ---------------------------------------------------------------------------
# MaxFlow MinCut (CPU, scipy)
# ---------------------------------------------------------------------------

FLOW_SCALE = 10000


def maxflow_mincut(knn_idx, knn_dist, tau_knn, bipartition,
                   idx_source, idx_sink):
    """Run MaxFlow/MinCut on precomputed k-NN graph."""
    K = bipartition.shape[0]
    k = knn_idx.shape[1]

    row_all = np.repeat(np.arange(K), k)
    col_all = knn_idx.ravel()
    dist_all = knn_dist.ravel()

    cap_float = np.maximum(0.0, tau_knn - dist_all)
    cap_int = (cap_float * FLOW_SCALE).astype(np.int32)
    mask = cap_int > 0

    if mask.sum() == 0:
        return bipartition.copy()

    row_filt, col_filt = row_all[mask], col_all[mask]
    cap_filt = cap_int[mask]

    row_sym = np.concatenate([row_filt, col_filt])
    col_sym = np.concatenate([col_filt, row_filt])
    cap_sym = np.concatenate([cap_filt, cap_filt])

    graph = csr_matrix((cap_sym, (row_sym, col_sym)), shape=(K, K))
    result = maximum_flow(graph, idx_source, idx_sink)

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

def crf_refine(mask_patch, patch_h, patch_w, n_iters=5,
               sxy_smooth=3.0, compat=3.0):
    """Simplified CRF mean-field inference using Gaussian smoothing."""
    q = mask_patch.reshape(patch_h, patch_w)
    for _ in range(n_iters):
        msg = gaussian_filter(q, sigma=sxy_smooth / patch_h)
        q_new = q + compat * (msg - 0.5)
        q = np.clip(q_new, 0, 1)
    return q.flatten()


# ---------------------------------------------------------------------------
# Full CutS3D extraction — paper-faithful
# ---------------------------------------------------------------------------

def extract_pseudo_masks_gpu(features, depth, patch_h, patch_w,
                              W_gpu, device, max_instances=3, tau_ncut=0.0,
                              tau_knn=0.115, k=10, sigma_gauss=3.0, beta=0.45,
                              min_mask_size=0.02, sc_samples=6, use_crf=True,
                              sc_min_ratio=0.5):
    """Full CutS3D extraction with GPU-accelerated operations.

    Pipeline per instance:
      1. NCut on CPU (eigsh)
      2. Build k-NN on GPU once
      3. LocalCut via MaxFlow on CPU
      4. CRF on CPU
      5. Spatial Confidence: T × MaxFlow (reuses k-NN graph)
    """
    K = features.shape[0]

    # Resize depth to patch resolution
    depth_pil = Image.fromarray(depth)
    depth_patch = np.array(depth_pil.resize((patch_w, patch_h), Image.BILINEAR))

    # SI Sharpening on GPU (Eq. 1-3)
    W_sharpened = spatial_importance_sharpening_gpu(
        W_gpu, depth_patch, sigma_gauss, beta
    )
    W = W_sharpened.cpu().numpy()

    all_masks = np.zeros((max_instances, K), dtype=np.float32)
    all_sc = np.zeros((max_instances, K), dtype=np.float32)
    active = np.ones(K, dtype=np.float32)
    num_valid = 0

    for inst_idx in range(max_instances):
        mask_2d = active[:, None] * active[None, :]
        W_masked = W * mask_2d

        bipartition, idx_src, idx_snk = normalized_cut(W_masked, tau_ncut)
        bipartition = bipartition * active

        if bipartition.sum() < 2:
            break

        # 3D points + GPU k-NN (built once, reused for LocalCut + SC)
        points_3d = prepare_3d_points(bipartition, depth_patch)
        knn_idx, knn_dist = gpu_build_knn_graph(points_3d, k=k, device=device)

        # LocalCut via MaxFlow (CPU, using precomputed GPU k-NN)
        instance_mask = maxflow_mincut(
            knn_idx, knn_dist, tau_knn, bipartition, idx_src, idx_snk
        )

        # CRF
        if use_crf:
            instance_mask = crf_refine(instance_mask, patch_h, patch_w)

        mask_frac = instance_mask.sum() / K
        if mask_frac < min_mask_size:
            continue

        # Spatial Confidence (Eq. 4): T binary cuts, reusing k-NN
        tau_knn_min = tau_knn * sc_min_ratio
        sc = np.zeros(K, dtype=np.float32)
        for t in range(sc_samples):
            tau_t = tau_knn_min + (t + 1) * (tau_knn - tau_knn_min) / sc_samples
            bc = maxflow_mincut(
                knn_idx, knn_dist, tau_t, bipartition, idx_src, idx_snk
            )
            sc += bc
        sc = sc / max(sc_samples, 1)
        sc = np.clip(sc, 0.5, 1.0)

        all_masks[num_valid] = instance_mask
        all_sc[num_valid] = sc
        num_valid += 1

        active = active * (1.0 - instance_mask)

    scores = np.zeros(max_instances, dtype=np.float32)
    for i in range(num_valid):
        mask_sum = all_masks[i].sum()
        if mask_sum > 0:
            scores[i] = all_sc[i].sum() / mask_sum

    return all_masks, all_sc, scores, num_valid


# ---------------------------------------------------------------------------
# Dataset discovery (flat or recursive)
# ---------------------------------------------------------------------------

def discover_images(data_dir, recursive=False):
    """Find all images in a directory.

    Args:
        data_dir: Path to image directory.
        recursive: If True, search subdirectories (e.g. ImageNet class dirs).

    Returns:
        Sorted list of (image_path, image_id) tuples.
    """
    extensions = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
    root = Path(data_dir)
    samples = []

    if recursive:
        for ext in extensions:
            for p in root.rglob(f"*{ext}"):
                image_id = str(p.relative_to(root)).replace("/", "_").rsplit(".", 1)[0]
                samples.append((str(p), image_id))
    else:
        for p in sorted(root.iterdir()):
            if p.suffix in extensions:
                image_id = p.stem
                samples.append((str(p), image_id))

    samples.sort(key=lambda x: x[1])
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CutS3D Extraction — COCO/ImageNet, GPU-Optimized"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to image directory (e.g. datasets/coco/train2017)")
    parser.add_argument("--output-dir", type=str, default="data/pseudo_masks_coco")
    parser.add_argument("--image-size", type=int, default=480,
                        help="Image resize dimension (paper: 480)")
    parser.add_argument("--max-instances", type=int, default=3,
                        help="Max instances per image (paper: 3)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--recursive", action="store_true",
                        help="Search subdirectories (for ImageNet class dirs)")
    parser.add_argument("--tau-knn", type=float, default=0.115,
                        help="k-NN edge threshold (paper Table 8a: 0.115)")
    parser.add_argument("--beta", type=float, default=0.45,
                        help="SI normalization lower bound (paper: 0.45)")
    parser.add_argument("--sigma-gauss", type=float, default=3.0,
                        help="Gaussian sigma for SI sharpening (paper: 3.0)")
    parser.add_argument("--sc-samples", type=int, default=6,
                        help="T: Spatial Confidence threshold samples (paper: 6)")
    parser.add_argument("--knn-k", type=int, default=10,
                        help="k for k-NN graph (paper: 10)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="DINO batch size (2-8 depending on GPU memory)")
    parser.add_argument("--depth-model", type=str, default="DPT_Large",
                        choices=["DPT_Large", "MiDaS_small"],
                        help="MiDaS depth model variant")
    parser.add_argument("--no-crf", action="store_true",
                        help="Skip CRF refinement")
    args = parser.parse_args()

    img_h = img_w = args.image_size
    patch_h = patch_w = img_h // 8
    K = patch_h * patch_w

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("  CutS3D Extraction — Paper-Faithful (GPU)")
    print("  Sick et al., 'CutS3D', ICCV 2025")
    print("=" * 70)
    print(f"  Device:          {device}")
    print(f"  Image size:      {img_h}x{img_w}")
    print(f"  Patch grid:      {patch_h}x{patch_w} = {K} patches")
    print(f"  Max instances:   {args.max_instances}")
    print(f"  tau_knn:         {args.tau_knn} (paper Table 8a)")
    print(f"  beta:            {args.beta}")
    print(f"  sigma_gauss:     {args.sigma_gauss}")
    print(f"  SC samples (T):  {args.sc_samples}")
    print(f"  k-NN k:          {args.knn_k}")
    print(f"  Depth model:     {args.depth_model}")
    print(f"  CRF:             {'OFF' if args.no_crf else 'ON'}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Shard:           {args.shard}/{args.num_shards}")
    print(f"  Pipeline:        DINO(GPU) → MiDaS depth(GPU) → Affinity(GPU)")
    print(f"                   → SI(GPU, Eq1-3) → NCut(CPU, eigsh)")
    print(f"                   → k-NN(GPU) → LocalCut(CPU, scipy maxflow)")
    print(f"                   → CRF(CPU) → SC[T×MaxFlow](CPU)")
    print("=" * 70)
    sys.stdout.flush()

    # Load DINO backbone
    print("\nLoading DINO ViT-S/8...")
    sys.stdout.flush()
    dino = load_dino_backbone(device)
    print(f"  DINO loaded on {device}")

    # Load MiDaS depth model
    print(f"Loading MiDaS {args.depth_model}...")
    sys.stdout.flush()
    depth_model, depth_transform = load_depth_model(device, args.depth_model)
    print(f"  MiDaS loaded on {device}")

    # Warmup both models
    dummy = torch.randn(1, 3, img_h, img_w).to(device)
    with torch.no_grad():
        _ = dino.get_intermediate_layers(dummy, n=1)
    del dummy
    torch.cuda.empty_cache()
    print("  GPU warmup done")
    sys.stdout.flush()

    # Discover images
    print(f"\nDiscovering images in {args.data_dir}...")
    all_samples = discover_images(args.data_dir, recursive=args.recursive)
    total = len(all_samples)
    print(f"  Found {total} images")

    # Shard
    if args.num_shards > 1:
        shard_samples = all_samples[args.shard::args.num_shards]
        print(f"  Shard {args.shard}/{args.num_shards}: "
              f"{len(shard_samples)}/{total} images")
    else:
        shard_samples = all_samples

    # Build global index mapping (consistent across shards)
    global_idx_map = {s[1]: i for i, s in enumerate(all_samples)}

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-extracted
    existing = set()
    for f in output_dir.glob("masks_*.npz"):
        existing.add(f.stem)
    if existing:
        print(f"  Found {len(existing)} existing masks, will skip")

    pending = []
    for img_path, image_id in shard_samples:
        global_idx = global_idx_map[image_id]
        mask_name = f"masks_{global_idx:08d}"
        if mask_name not in existing:
            pending.append((img_path, image_id, global_idx))

    n_pending = len(pending)
    print(f"  Pending: {n_pending} images to extract")
    print("=" * 70)
    sys.stdout.flush()

    if n_pending == 0:
        print("Nothing to do!")
        return

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    total_images = 0
    total_masks = 0
    total_time = 0.0
    batch_size = args.batch_size

    for batch_start in range(0, n_pending, batch_size):
        batch_end = min(batch_start + batch_size, n_pending)
        batch_items = pending[batch_start:batch_end]
        batch_t0 = time.time()

        # Load and preprocess images
        images_np = []
        img_tensors = []
        global_indices = []
        image_ids = []

        for img_path, image_id, global_idx in batch_items:
            img_pil = Image.open(img_path).convert("RGB")
            img_resized = img_pil.resize((img_w, img_h), Image.BILINEAR)
            img_np = np.array(img_resized, dtype=np.float32) / 255.0
            images_np.append(img_np)

            # DINO expects ImageNet-normalized input
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
            img_t = (img_t - mean) / std
            img_tensors.append(img_t)

            global_indices.append(global_idx)
            image_ids.append(image_id)

        # Batched DINO forward + GPU affinity
        feat_W_list = extract_dino_features_batch(dino, img_tensors, device)

        # Batched depth estimation
        depths = estimate_depth_batch(
            depth_model, depth_transform, images_np, device,
            target_h=img_h, target_w=img_w,
        )

        # Process each image through the full CutS3D pipeline
        for i in range(len(batch_items)):
            features, W_gpu = feat_W_list[i]
            depth = depths[i]

            masks, sc, scores, n_valid = extract_pseudo_masks_gpu(
                features, depth,
                patch_h=patch_h, patch_w=patch_w,
                W_gpu=W_gpu, device=device,
                max_instances=args.max_instances,
                tau_ncut=0.0, tau_knn=args.tau_knn, k=args.knn_k,
                sigma_gauss=args.sigma_gauss, beta=args.beta,
                min_mask_size=0.02, sc_samples=args.sc_samples,
                use_crf=not args.no_crf, sc_min_ratio=0.5,
            )

            total_masks += n_valid
            n_save = max(n_valid, 1)
            np.savez_compressed(
                output_dir / f"masks_{global_indices[i]:08d}.npz",
                masks=masks[:n_save],
                spatial_confidence=sc[:n_save],
                scores=scores[:n_save],
                num_valid=n_valid,
                image_id=image_ids[i],
            )
            total_images += 1

        batch_time = time.time() - batch_t0
        total_time += batch_time
        done = batch_end

        if done <= batch_size or done % 20 < batch_size or done == n_pending:
            avg = total_time / total_images
            remaining = n_pending - done
            eta_h = avg * remaining / 3600
            last_id = image_ids[-1]
            masks_per = total_masks / total_images if total_images > 0 else 0
            print(
                f"  [GPU{args.gpu} {done:>6d}/{n_pending}] {last_id}: "
                f"{batch_time/len(batch_items):.2f}s/img, "
                f"avg {avg:.2f}s/img, {masks_per:.1f} masks/img, "
                f"ETA ~{eta_h:.1f}h"
            )
            sys.stdout.flush()

    # Summary
    print("\n" + "=" * 70)
    print(f"  Extraction Complete (shard {args.shard}/{args.num_shards}, GPU {args.gpu})")
    print("=" * 70)
    print(f"  Total images:    {total_images}")
    print(f"  Total masks:     {total_masks}")
    if total_images > 0:
        print(f"  Masks/image:     {total_masks/total_images:.1f}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/3600:.2f}h)")
    if total_images > 0:
        print(f"  Avg per image:   {total_time/total_images:.2f}s")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
