"""Instance descriptor computation from pseudo-instance masks.

Computes an 8-dim per-patch descriptor encoding instance membership,
geometry, and local context from precomputed instance masks and depth maps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_instance_descriptor(
    instance_mask: np.ndarray,
    semantic_mask: np.ndarray,
    depth_map: np.ndarray,
    target_h: int = 32,
    target_w: int = 64,
    num_semantic_classes: int = 27,
) -> torch.Tensor:
    """Compute 8-dim instance descriptor per patch.

    Args:
        instance_mask: Instance ID mask (H, W) uint16/uint32. 0 = no instance.
        semantic_mask: Semantic class mask (H, W) uint8. Values 0-26.
        depth_map: Depth map (H_d, W_d) float32 in [0, 1].
        target_h: Patch grid height (32).
        target_w: Patch grid width (64).
        num_semantic_classes: Number of semantic classes (27).

    Returns:
        Instance descriptor of shape (target_h * target_w, 8).
        Channels:
            0: is_thing — 1 if patch belongs to any instance
            1: instance_score — confidence (area-based, normalized)
            2: instance_area_log — log(instance_area / total_area)
            3: depth_offset — patch_depth - mean_instance_depth
            4: instance_depth_var — depth std within instance
            5: boundary_flag — 1 if on instance boundary
            6: local_instance_count — distinct instances in 3x3 neighborhood
            7: semantic_entropy — entropy of semantic class in 3x3 neighborhood
    """
    H, W = instance_mask.shape
    total_area = H * W

    # Resize depth to match instance mask if needed
    if depth_map.shape != (H, W):
        depth_t = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)
        depth_t = F.interpolate(depth_t, size=(H, W), mode="bilinear", align_corners=False)
        depth_full = depth_t.squeeze().numpy()
    else:
        depth_full = depth_map.astype(np.float32)

    # Precompute per-instance statistics
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # skip background

    inst_stats = {}
    max_area = 1
    for iid in unique_ids:
        mask = instance_mask == iid
        area = int(mask.sum())
        if area == 0:
            continue
        mean_depth = float(depth_full[mask].mean())
        std_depth = float(depth_full[mask].std()) if area > 1 else 0.0
        inst_stats[iid] = {
            "area": area,
            "mean_depth": mean_depth,
            "std_depth": std_depth,
        }
        max_area = max(max_area, area)

    # Compute instance scores (area-based, normalized by largest)
    for iid in inst_stats:
        inst_stats[iid]["score"] = inst_stats[iid]["area"] / max_area

    # Compute pixel-level features at full resolution, then pool to patch level
    # Feature 0: is_thing
    is_thing_full = (instance_mask > 0).astype(np.float32)

    # Feature 1: instance_score
    score_full = np.zeros((H, W), dtype=np.float32)
    for iid, stats in inst_stats.items():
        score_full[instance_mask == iid] = stats["score"]

    # Feature 2: instance_area_log
    area_log_full = np.zeros((H, W), dtype=np.float32)
    for iid, stats in inst_stats.items():
        area_log_full[instance_mask == iid] = np.log(stats["area"] / total_area + 1e-8)
    # Normalize to roughly [-1, 1]
    if area_log_full.min() < area_log_full.max():
        area_log_full = (area_log_full - area_log_full.min()) / (area_log_full.max() - area_log_full.min() + 1e-8)
        area_log_full = area_log_full * 2 - 1

    # Feature 3: depth_offset (patch_depth - mean_instance_depth)
    depth_offset_full = np.zeros((H, W), dtype=np.float32)
    for iid, stats in inst_stats.items():
        mask = instance_mask == iid
        depth_offset_full[mask] = depth_full[mask] - stats["mean_depth"]

    # Feature 4: instance_depth_var
    depth_var_full = np.zeros((H, W), dtype=np.float32)
    for iid, stats in inst_stats.items():
        depth_var_full[instance_mask == iid] = stats["std_depth"]

    # Feature 5: boundary_flag (4-connected boundary detection)
    boundary_full = np.zeros((H, W), dtype=np.float32)
    # Shift in 4 directions and check for ID change
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(instance_mask, dy, axis=0), dx, axis=1)
        boundary_full |= (instance_mask != shifted) & (instance_mask > 0)

    # Feature 6: local_instance_count (3x3 neighborhood)
    # Use average pooling of unique-count proxy
    inst_count_full = np.zeros((H, W), dtype=np.float32)
    padded = np.pad(instance_mask, 1, mode="edge")
    for i in range(H):
        for j in range(0, W, 4):  # subsample for speed — full is too slow at pixel level
            end_j = min(j + 4, W)
            for jj in range(j, end_j):
                patch = padded[i:i+3, jj:jj+3]
                inst_count_full[i, jj] = len(np.unique(patch[patch > 0]))

    # Feature 7: semantic_entropy (3x3 neighborhood)
    sem_entropy_full = np.zeros((H, W), dtype=np.float32)
    sem_padded = np.pad(semantic_mask, 1, mode="edge")
    for i in range(0, H, 4):  # subsample for speed
        end_i = min(i + 4, H)
        for j in range(0, W, 4):
            end_j = min(j + 4, W)
            patch = sem_padded[i:i+3, j:j+3]
            counts = np.bincount(patch.ravel(), minlength=num_semantic_classes).astype(np.float32)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            for ii in range(i, end_i):
                for jj in range(j, end_j):
                    if ii < H and jj < W:
                        sem_entropy_full[ii, jj] = entropy

    # Stack all features: (H, W, 8)
    features_full = np.stack([
        is_thing_full,
        score_full,
        area_log_full,
        depth_offset_full,
        depth_var_full,
        boundary_full.astype(np.float32),
        inst_count_full,
        sem_entropy_full,
    ], axis=-1)

    # Pool to patch level via average pooling
    features_t = torch.from_numpy(features_full).float()  # (H, W, 8)
    features_t = features_t.permute(2, 0, 1).unsqueeze(0)  # (1, 8, H, W)
    features_patch = F.adaptive_avg_pool2d(features_t, (target_h, target_w))  # (1, 8, 32, 64)
    features_patch = features_patch.squeeze(0).permute(1, 2, 0)  # (32, 64, 8)

    # Flatten to (N, 8)
    return features_patch.reshape(-1, 8)


def compute_instance_descriptor_fast(
    instance_mask: np.ndarray,
    semantic_mask: np.ndarray,
    depth_map: np.ndarray,
    target_h: int = 32,
    target_w: int = 64,
) -> torch.Tensor:
    """Fast version: compute descriptors directly at patch level.

    Downsamples masks first, then computes features. Less precise but
    ~10x faster than full-resolution computation.

    Args:
        instance_mask: Instance ID mask (H, W).
        semantic_mask: Semantic class mask (H, W).
        depth_map: Depth map (H_d, W_d).
        target_h: Patch grid height.
        target_w: Patch grid width.

    Returns:
        Instance descriptor of shape (target_h * target_w, 8).
    """
    from PIL import Image

    H, W = instance_mask.shape
    total_area = H * W

    # Downsample everything to patch level via nearest (for masks) and bilinear (for depth)
    inst_patch = np.array(
        Image.fromarray(instance_mask.astype(np.uint16)).resize(
            (target_w, target_h), Image.NEAREST
        )
    )
    sem_patch = np.array(
        Image.fromarray(semantic_mask.astype(np.uint8)).resize(
            (target_w, target_h), Image.NEAREST
        )
    )
    depth_t = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)
    depth_patch = F.interpolate(
        depth_t, size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    pH, pW = target_h, target_w
    N = pH * pW

    # Precompute per-instance stats at full resolution
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]

    # Resize depth to match instance mask for stats
    if depth_map.shape != (H, W):
        depth_full_t = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)
        depth_full = F.interpolate(depth_full_t, size=(H, W), mode="bilinear", align_corners=False).squeeze().numpy()
    else:
        depth_full = depth_map

    inst_stats = {}
    max_area = 1
    for iid in unique_ids:
        mask = instance_mask == iid
        area = int(mask.sum())
        if area == 0:
            continue
        mean_d = float(depth_full[mask].mean())
        std_d = float(depth_full[mask].std()) if area > 1 else 0.0
        inst_stats[int(iid)] = {"area": area, "mean_depth": mean_d, "std_depth": std_d}
        max_area = max(max_area, area)

    for iid in inst_stats:
        inst_stats[iid]["score"] = inst_stats[iid]["area"] / max_area

    # Compute features at patch level
    features = np.zeros((pH, pW, 8), dtype=np.float32)

    for i in range(pH):
        for j in range(pW):
            iid = int(inst_patch[i, j])

            # 0: is_thing
            features[i, j, 0] = 1.0 if iid > 0 else 0.0

            if iid > 0 and iid in inst_stats:
                stats = inst_stats[iid]
                # 1: instance_score
                features[i, j, 1] = stats["score"]
                # 2: instance_area_log
                features[i, j, 2] = np.log(stats["area"] / total_area + 1e-8)
                # 3: depth_offset
                features[i, j, 3] = depth_patch[i, j] - stats["mean_depth"]
                # 4: instance_depth_var
                features[i, j, 4] = stats["std_depth"]

            # 5: boundary_flag
            if iid > 0:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < pH and 0 <= nj < pW:
                        if inst_patch[ni, nj] != iid:
                            features[i, j, 5] = 1.0
                            break

            # 6: local_instance_count (3x3)
            i_lo, i_hi = max(0, i-1), min(pH, i+2)
            j_lo, j_hi = max(0, j-1), min(pW, j+2)
            local = inst_patch[i_lo:i_hi, j_lo:j_hi]
            local_ids = np.unique(local[local > 0])
            features[i, j, 6] = len(local_ids)

            # 7: semantic_entropy (3x3)
            local_sem = sem_patch[i_lo:i_hi, j_lo:j_hi].ravel()
            counts = np.bincount(local_sem, minlength=27).astype(np.float32)
            probs = counts / (counts.sum() + 1e-8)
            probs = probs[probs > 0]
            features[i, j, 7] = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize area_log to [-1, 1]
    col2 = features[:, :, 2]
    if col2.min() < col2.max():
        features[:, :, 2] = (col2 - col2.min()) / (col2.max() - col2.min() + 1e-8) * 2 - 1

    return torch.from_numpy(features.reshape(N, 8))
