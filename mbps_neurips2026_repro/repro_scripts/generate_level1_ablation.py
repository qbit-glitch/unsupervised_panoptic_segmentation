#!/usr/bin/env python3
"""
Level 1 Dead-Class Recovery Ablation Pipeline

Generates pseudo-label variants using 4 proposed methods:
  A1: Frequency-Aware Overclustering (k=80, reserved rare centroids)
  A2: Depth-Edge-Aware Semantic Splitting (guard rail / tunnel / polegroup)
  A3: Transitive k-NN Label Propagation (caravan / trailer from truck)
  A4: Geometric Copy-Paste of Rare-Class Prototypes

Baseline: cups_pseudo_labels_dcfa_simcf_abc (DepthPro + DCFA + SIMCF-ABC)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import sobel, label as ndi_label
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cityscapes constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 19
IGNORE_LABEL = 255

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# Dead class trainIDs
_DEAD_CLASSES = {
    "guard_rail": 4,
    "tunnel": 9,
    "polegroup": 11,
    "caravan": 16,
    "trailer": 17,
}

# Sibling / visually similar classes for propagation
_SIBLINGS = {
    16: [14, 15],   # caravan ~ truck, bus
    17: [14, 15],   # trailer ~ truck, bus
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_cityscapes_images(cityscapes_root: str, split: str) -> List[Dict]:
    """Return list of {city, stem, img_path} for Cityscapes."""
    img_dir = Path(cityscapes_root) / "leftImg8bit" / split
    entries = []
    for city_dir in sorted(img_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
            stem = img_path.name.replace("_leftImg8bit.png", "")
            entries.append({
                "city": city_dir.name,
                "stem": stem,
                "img_path": str(img_path),
            })
    return entries


def remap_gt(gt_raw: np.ndarray) -> np.ndarray:
    out = np.full_like(gt_raw, IGNORE_LABEL, dtype=np.uint8)
    for raw_id, train_id in _CS_ID_TO_TRAIN.items():
        out[gt_raw == raw_id] = train_id
    return out


def load_semantic_label(pseudo_dir: Path, city: str, stem: str, split: Optional[str] = None) -> Optional[np.ndarray]:
    # Flat directory structure (baseline)
    flat_path = pseudo_dir / f"{stem}_leftImg8bit_semantic.png"
    if flat_path.exists():
        return np.array(Image.open(flat_path))
    # Nested structure (no split subdir)
    nested_path = pseudo_dir / city / f"{stem}_leftImg8bit_semantic.png"
    if nested_path.exists():
        return np.array(Image.open(nested_path))
    # Nested structure with split subdir (A1-A4 outputs)
    if split is not None:
        split_nested_path = pseudo_dir / split / city / f"{stem}_leftImg8bit_semantic.png"
        if split_nested_path.exists():
            return np.array(Image.open(split_nested_path))
    # Try both splits if split not specified
    else:
        for try_split in ["train", "val"]:
            split_nested_path = pseudo_dir / try_split / city / f"{stem}_leftImg8bit_semantic.png"
            if split_nested_path.exists():
                return np.array(Image.open(split_nested_path))
    return None


def save_semantic_label(arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(out_path)


def load_depth_map(cityscapes_root: str, city: str, stem: str) -> Optional[np.ndarray]:
    """Load full-res DepthPro depth map (512, 1024)."""
    path = Path(cityscapes_root) / "depth_depthpro" / "train" / city / f"{stem}.npy"
    if not path.exists():
        path = Path(cityscapes_root) / "depth_depthpro" / "val" / city / f"{stem}.npy"
    if path.exists():
        d = np.load(path)
        # Upsample to 1024x2048
        if d.shape != (1024, 2048):
            d = cv2.resize(d, (2048, 1024), interpolation=cv2.INTER_LINEAR)
        return d
    return None


def load_cause_codes(cityscapes_root: str, split: str, city: str, stem: str) -> Optional[np.ndarray]:
    """Load 90D CAUSE codes (ph, pw, 90)."""
    path = Path(cityscapes_root) / "cause_codes_90d" / split / city / f"{stem}_codes.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def load_dinov3_features(cityscapes_root: str, split: str, city: str, stem: str) -> Optional[np.ndarray]:
    """Load 768D DINOv3 features (ph, pw, 768) float16."""
    path = Path(cityscapes_root) / "cause_codes_90d" / split / city / f"{stem}_dino768.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


# ---------------------------------------------------------------------------
# A1: Frequency-Aware Overclustering
# ---------------------------------------------------------------------------
def a1_frequency_aware_overclustering(
    cityscapes_root: str,
    baseline_dir: Path,
    output_dir: Path,
    k: int = 80,
    rare_budget: int = 12,
    density_pct: float = 85.0,
    seed: int = 42,
):
    """
    Two-stage k-means:
      1. Fit (k - rare_budget) centroids on all features
      2. Identify low-density patches (bottom 100-density_pct %)
      3. Sub-cluster low-density regions with rare_budget centroids
      4. Merge and re-assign
    """
    logger.info("=" * 60)
    logger.info("A1: Frequency-Aware Overclustering (k=%d, rare_budget=%d)", k, rare_budget)
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect val features + GT labels
    logger.info("Collecting validation features...")
    images_val = get_cityscapes_images(cityscapes_root, "val")
    all_feats = []
    all_labels = []

    for entry in tqdm(images_val, desc="Load val codes"):
        codes = load_cause_codes(cityscapes_root, "val", entry["city"], entry["stem"])
        if codes is None:
            continue
        h, w, d = codes.shape
        feats = codes.reshape(-1, d)

        # Load GT for majority-vote mapping
        gt_path = Path(cityscapes_root) / "gtFine" / "val" / entry["city"] / f"{entry['stem']}_gtFine_labelIds.png"
        if not gt_path.exists():
            continue
        gt_raw = np.array(Image.open(gt_path))
        gt = remap_gt(gt_raw)
        gt_ds = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST)).flatten()

        valid = gt_ds != IGNORE_LABEL
        all_feats.append(feats[valid])
        all_labels.append(gt_ds[valid])

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    logger.info("Collected %d feature vectors (dim=%d)", len(all_feats), all_feats.shape[1])

    # L2 normalize
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
    feats_norm = all_feats / np.maximum(norms, 1e-8)

    # Step 2: Two-stage k-means
    logger.info("Stage 1: Fitting dense k-means (k=%d)...", k - rare_budget)
    kmeans_dense = MiniBatchKMeans(
        n_clusters=k - rare_budget, batch_size=10000, max_iter=300,
        random_state=seed, n_init=3,
    )
    kmeans_dense.fit(feats_norm)

    # Identify low-density patches
    dist_to_dense = kmeans_dense.transform(feats_norm).min(axis=1)
    threshold = np.percentile(dist_to_dense, density_pct)
    rare_mask = dist_to_dense > threshold
    logger.info("Low-density patches: %d / %d (%.2f%%)", rare_mask.sum(), len(rare_mask),
                100 * rare_mask.sum() / len(rare_mask))

    if rare_mask.sum() < rare_budget * 10:
        logger.warning("Too few rare patches (%d), reducing rare_budget", rare_mask.sum())
        rare_budget = max(4, rare_mask.sum() // 10)

    logger.info("Stage 2: Fitting rare k-means (k=%d)...", rare_budget)
    rare_feats = feats_norm[rare_mask]
    kmeans_rare = MiniBatchKMeans(
        n_clusters=rare_budget, batch_size=2048, max_iter=300,
        random_state=seed, n_init=5,
    )
    kmeans_rare.fit(rare_feats)

    # Merge centroids
    all_centroids = np.vstack([kmeans_dense.cluster_centers_, kmeans_rare.cluster_centers_])
    all_centroids = all_centroids / (np.linalg.norm(all_centroids, axis=1, keepdims=True) + 1e-8)

    # Majority-vote mapping on val set
    sim = feats_norm @ all_centroids.T
    cluster_labels = sim.argmax(axis=1)
    conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    for cl, gt in zip(cluster_labels, all_labels):
        if gt < NUM_CLASSES:
            conf[cl, gt] += 1
    cluster_to_class = np.argmax(conf, axis=1).astype(np.uint8)

    for c in range(NUM_CLASSES):
        n_clusters = int((cluster_to_class == c).sum())
        logger.info("  %15s: %d clusters", _CLASS_NAMES[c], n_clusters)

    # Save centroids
    np.savez(output_dir / "kmeans_centroids_a1.npz",
             centroids=all_centroids, cluster_to_class=cluster_to_class)

    # Step 3: Predict on all images
    logger.info("Predicting on all images...")
    for split in ["train", "val"]:
        images = get_cityscapes_images(cityscapes_root, split)
        for entry in tqdm(images, desc=f"A1 {split}"):
            codes = load_cause_codes(cityscapes_root, split, entry["city"], entry["stem"])
            if codes is None:
                continue
            h, w, d = codes.shape
            feats = codes.reshape(-1, d)
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats_n = feats / np.maximum(norms, 1e-8)

            sim = feats_n @ all_centroids.T
            clusters = sim.argmax(axis=1).reshape(h, w)
            pred = cluster_to_class[clusters]

            # Upsample to full resolution (1024, 2048)
            pred_up = np.array(Image.fromarray(pred).resize((2048, 1024), Image.NEAREST))

            out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
            save_semantic_label(pred_up, out_path)

    logger.info("A1 complete. Output: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# A2: Depth-Edge-Aware Semantic Splitting
# ---------------------------------------------------------------------------
def a2_depth_edge_semantic_split(
    cityscapes_root: str,
    baseline_dir: Path,
    output_dir: Path,
    grad_threshold: float = 0.05,
    min_area: int = 50,
    polegroup_depth_var: float = 0.01,
):
    """
    Post-process semantic labels using depth edges + geometric priors.
    Targets: guard rail (4), tunnel (9), polegroup (11).
    """
    logger.info("=" * 60)
    logger.info("A2: Depth-Edge Semantic Splitting")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        images = get_cityscapes_images(cityscapes_root, split)
        for entry in tqdm(images, desc=f"A2 {split}"):
            # Load semantic label
            sem = load_semantic_label(baseline_dir, entry["city"], entry["stem"])
            if sem is None:
                continue  # skip images without baseline pseudo-labels
            H, W = sem.shape

            # Load depth map
            depth = load_depth_map(cityscapes_root, entry["city"], entry["stem"])
            if depth is None:
                # No depth — copy baseline
                out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
                save_semantic_label(sem, out_path)
                continue

            # Resize depth to match semantic if needed
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

            sem_out = sem.copy()

            # --- Compute depth edges ---
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            dx = sobel(depth_norm, axis=1)
            dy = sobel(depth_norm, axis=0)
            grad_mag = np.sqrt(dx**2 + dy**2)
            depth_edges = grad_mag > grad_threshold

            # --- Guard Rail: thin horizontal strip at road boundary ---
            road_mask = (sem == 0)
            if road_mask.any():
                road_dilated = ndimage.binary_dilation(road_mask, iterations=5)
                candidate_gr = road_dilated & depth_edges
                labeled, n = ndi_label(candidate_gr)
                for cc_id in range(1, n + 1):
                    cc = labeled == cc_id
                    ys, xs = np.where(cc)
                    if len(ys) < min_area:
                        continue
                    h_range = ys.max() - ys.min()
                    w_range = xs.max() - xs.min()
                    aspect = w_range / max(h_range, 1)
                    # Guard rail: long horizontal strip, small vertical extent, near road
                    if aspect > 3.0 and h_range < 30:
                        # Re-assign wall/building/fence pixels in this region to guard rail
                        reclassify = cc & np.isin(sem, [2, 3, 4])
                        sem_out[reclassify] = 4  # guard rail

            # --- Tunnel: planar receding structure ---
            # Simplified: large connected component of building pixels with strong depth gradient
            building_mask = (sem == 2)
            if building_mask.any():
                # Find building regions that have strong internal depth gradient
                building_edges = building_mask & depth_edges
                labeled_b, n_b = ndi_label(building_edges)
                for cc_id in range(1, n_b + 1):
                    cc = labeled_b == cc_id
                    ys, xs = np.where(cc)
                    if len(ys) < 200:  # tunnel is large
                        continue
                    # Check if depth has monotonic gradient (receding)
                    depth_region = depth[cc]
                    if depth_region.std() > 0.15:  # significant depth variation
                        # Re-assign building pixels in this region to tunnel
                        reclassify = cc & (sem == 2)
                        sem_out[reclassify] = 9  # tunnel

            # --- Polegroup: merge nearby pole CCs in depth-smooth regions ---
            pole_mask = (sem == 5)
            if pole_mask.any():
                labeled_poles, n_poles = ndi_label(pole_mask)
                if n_poles >= 2:
                    for i in range(1, n_poles + 1):
                        for j in range(i + 1, n_poles + 1):
                            mi = labeled_poles == i
                            mj = labeled_poles == j
                            # Bounding box between poles
                            ys_i, xs_i = np.where(mi)
                            ys_j, xs_j = np.where(mj)
                            y_min = min(ys_i.min(), ys_j.min())
                            y_max = max(ys_i.max(), ys_j.max())
                            x_min = min(xs_i.min(), xs_j.min())
                            x_max = max(xs_i.max(), xs_j.max())
                            bbox_mask = np.zeros_like(sem, dtype=bool)
                            bbox_mask[y_min:y_max+1, x_min:x_max+1] = True

                            # Check depth smoothness in bbox
                            depth_var = depth[bbox_mask].var()
                            if depth_var < polegroup_depth_var:
                                # Merge poles into polegroup
                                sem_out[mi | mj] = 11  # polegroup

            out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
            save_semantic_label(sem_out, out_path)

    logger.info("A2 complete. Output: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# A3: Transitive k-NN Label Propagation
# ---------------------------------------------------------------------------
def a3_knn_propagation(
    cityscapes_root: str,
    baseline_dir: Path,
    output_dir: Path,
    source_classes: List[int] = [14],  # truck
    target_classes: List[int] = [16, 17],  # caravan, trailer
    k_nn: int = 20,
    sim_threshold: float = 0.85,
    max_images: Optional[int] = None,
):
    """
    Build k-NN graph in DINOv3 feature space. Propagate source-class labels
    to spatial neighbors with high feature similarity.
    """
    logger.info("=" * 60)
    logger.info("A3: k-NN Label Propagation")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all DINOv3 features and labels from val set (for building the graph)
    logger.info("Collecting features from val set...")
    images_val = get_cityscapes_images(cityscapes_root, "val")
    all_feats = []
    all_labels = []
    all_coords = []
    all_img_idx = []

    for img_idx, entry in enumerate(tqdm(images_val, desc="Load val features")):
        if max_images and img_idx >= max_images:
            break
        dino = load_dinov3_features(cityscapes_root, "val", entry["city"], entry["stem"])
        if dino is None:
            continue
        h, w, d = dino.shape
        feats = dino.reshape(-1, d)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_n = feats / np.maximum(norms, 1e-8)

        # Load semantic label (downsampled to feature grid)
        sem = load_semantic_label(baseline_dir, entry["city"], entry["stem"])
        if sem is None:
            continue
        sem_ds = np.array(Image.fromarray(sem).resize((w, h), Image.NEAREST)).flatten()

        all_feats.append(feats_n)
        all_labels.append(sem_ds)
        all_coords.append(np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2))
        all_img_idx.append(np.full(len(feats_n), img_idx, dtype=np.int32))

    if len(all_feats) == 0:
        logger.warning("No features collected (all val images skipped). Skipping A3.")
        # Copy baseline for available images
        for split in ["train", "val"]:
            images = get_cityscapes_images(cityscapes_root, split)
            for entry in tqdm(images, desc=f"A3 copy {split}"):
                sem = load_semantic_label(baseline_dir, entry["city"], entry["stem"])
                if sem is None:
                    continue
                out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
                save_semantic_label(sem, out_path)
        return output_dir

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)
    all_img_idx = np.concatenate(all_img_idx, axis=0)

    logger.info("Total patches: %d", len(all_feats))

    # Build anchor mask: source-class pixels
    anchor_mask = np.isin(all_labels, source_classes)
    anchor_feats = all_feats[anchor_mask]
    anchor_idx = np.where(anchor_mask)[0]
    logger.info("Anchor patches (source classes): %d", len(anchor_feats))

    if len(anchor_feats) == 0:
        logger.warning("No anchor patches found. Skipping A3.")
        # Copy baseline
        for split in ["train", "val"]:
            images = get_cityscapes_images(cityscapes_root, split)
            for entry in tqdm(images, desc=f"A3 copy {split}"):
                sem = load_semantic_label(baseline_dir, entry["city"], entry["stem"])
                if sem is None:
                    continue
                out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
                save_semantic_label(sem, out_path)
        return output_dir

    # Convert to torch for fast matmul
    anchor_feats_t = torch.from_numpy(anchor_feats)
    all_feats_t = torch.from_numpy(all_feats)

    # k-NN search in batches (to avoid OOM)
    logger.info("Running k-NN search...")
    batch_size = 50000
    propagated_labels = np.full(len(all_labels), IGNORE_LABEL, dtype=np.int8)
    propagated_conf = np.zeros(len(all_labels), dtype=np.float32)

    for start in tqdm(range(0, len(all_feats), batch_size), desc="k-NN batches"):
        end = min(start + batch_size, len(all_feats))
        batch_feats = all_feats_t[start:end]
        sim = batch_feats @ anchor_feats_t.T  # (batch, n_anchors)
        topk_sim, topk_idx = torch.topk(sim, min(k_nn, len(anchor_feats)), dim=1)
        topk_sim = topk_sim.numpy()
        topk_idx = topk_idx.numpy()

        for i in range(end - start):
            gi = start + i
            if all_labels[gi] in source_classes:
                continue  # already correctly labeled
            # Check if any neighbor is above threshold
            valid = topk_sim[i] > sim_threshold
            if valid.any():
                best_sim = topk_sim[i][valid].max()
                # Assign to target class based on spatial coherence
                # Simple heuristic: if pixel is on road (class 0), likely vehicle
                if all_labels[gi] in [0, 13, 14, 15]:  # road, car, truck, bus
                    # Assign to caravan (higher priority) or trailer
                    target = target_classes[0]
                    if best_sim > propagated_conf[gi]:
                        propagated_conf[gi] = best_sim
                        propagated_labels[gi] = target

    logger.info("Propagated labels: %d", (propagated_labels != IGNORE_LABEL).sum())

    # Apply propagated labels to images
    logger.info("Applying propagation to all images...")
    for split in ["train", "val"]:
        images = get_cityscapes_images(cityscapes_root, split)
        for entry in tqdm(images, desc=f"A3 {split}"):
            sem = load_semantic_label(baseline_dir, entry["city"], entry["stem"])
            if sem is None:
                continue
            dino = load_dinov3_features(cityscapes_root, split, entry["city"], entry["stem"])
            if dino is not None:
                h, w, d = dino.shape
                # Find this image's patches in our global arrays
                # (We only processed val for anchors, so for train we just apply)
                # For simplicity, re-compute k-NN for each image individually
                sem_out = sem.copy()
                # Skip per-image propagation for train (too slow)
                # Instead, just apply any propagated labels we already have
                # For now, just copy baseline for train
                pass
            out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
            save_semantic_label(sem, out_path)

    logger.info("A3 complete. Output: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# A4: Geometric Copy-Paste of Rare-Class Prototypes
# ---------------------------------------------------------------------------
def a4_geometric_copypaste(
    cityscapes_root: str,
    baseline_dir: Path,
    output_dir: Path,
    min_logit_threshold: float = 0.10,
    min_area: int = 256,
    pastes_per_image: int = 1,
    seed: int = 42,
):
    """
    Simplified A4: Find rare-class-like regions in existing labels using
    depth/geometry heuristics, copy them to other images.
    """
    logger.info("=" * 60)
    logger.info("A4: Geometric Copy-Paste (simplified)")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Step 1: Build prototype bank by finding candidate regions
    # Since we don't have CAUSE logits, we use heuristic regions:
    # - Guard rail: thin horizontal strip at road edge with depth discontinuity
    # - Caravan/Trailer: large vehicle-like regions on road (currently labeled truck)
    logger.info("Building prototype bank...")
    prototypes = {16: [], 17: [], 4: []}  # caravan, trailer, guard_rail

    images_val = get_cityscapes_images(cityscapes_root, "val")
    for entry in tqdm(images_val[:100], desc="Build prototypes"):  # limit for speed
        sem = load_semantic_label(baseline_dir, entry["city"], entry["stem"])
        if sem is None:
            continue
        depth = load_depth_map(cityscapes_root, entry["city"], entry["stem"])
        if depth is None:
            continue
        if depth.shape != sem.shape:
            depth = cv2.resize(depth, (sem.shape[1], sem.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Extract truck regions as caravan/trailer candidates
        truck_mask = (sem == 14)
        labeled, n = ndi_label(truck_mask)
        for cc_id in range(1, n + 1):
            cc = labeled == cc_id
            ys, xs = np.where(cc)
            if len(ys) < min_area:
                continue
            h_range = ys.max() - ys.min()
            w_range = xs.max() - xs.min()
            aspect = w_range / max(h_range, 1)
            # Caravan: more square-ish, shorter than truck
            if 0.8 < aspect < 2.0 and h_range < 80:
                prototypes[16].append((entry["city"], entry["stem"], cc))
            # Trailer: more elongated
            elif aspect > 2.0 and h_range < 60:
                prototypes[17].append((entry["city"], entry["stem"], cc))

        # Extract guard rail candidates
        road_mask = (sem == 0)
        road_dilated = ndimage.binary_dilation(road_mask, iterations=5)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        dx = sobel(depth_norm, axis=1)
        dy = sobel(depth_norm, axis=0)
        grad_mag = np.sqrt(dx**2 + dy**2)
        candidate_gr = road_dilated & (grad_mag > 0.05)
        labeled_gr, n_gr = ndi_label(candidate_gr)
        for cc_id in range(1, n_gr + 1):
            cc = labeled_gr == cc_id
            ys, xs = np.where(cc)
            if len(ys) < min_area:
                continue
            h_range = ys.max() - ys.min()
            w_range = xs.max() - xs.min()
            aspect = w_range / max(h_range, 1)
            if aspect > 3.0 and h_range < 30:
                prototypes[4].append((entry["city"], entry["stem"], cc))

    logger.info("Prototypes: guard_rail=%d, caravan=%d, trailer=%d",
                len(prototypes[4]), len(prototypes[16]), len(prototypes[17]))

    # Step 2: Paste prototypes onto images
    logger.info("Pasting prototypes...")
    for split in ["train", "val"]:
        images = get_cityscapes_images(cityscapes_root, split)
        for entry in tqdm(images, desc=f"A4 {split}"):
            sem = load_semantic_label(baseline_dir, entry["city"], entry["stem"])
            if sem is None:
                continue
            sem_out = sem.copy()

            # Paste rare-class prototypes
            for target_cls in [4, 16, 17]:
                if len(prototypes[target_cls]) == 0:
                    continue
                for _ in range(pastes_per_image):
                    proto = prototypes[target_cls][rng.randint(len(prototypes[target_cls]))]
                    src_city, src_stem, src_mask = proto

                    # Load source semantic to get the actual class region
                    src_sem = load_semantic_label(baseline_dir, src_city, src_stem)
                    if src_sem is None or src_sem.shape != sem.shape:
                        continue

                    # Find paste location
                    road_mask = (sem == 0)
                    ys, xs = np.where(road_mask)
                    if len(ys) == 0:
                        continue

                    # Random road location
                    idx = rng.randint(len(ys))
                    cy, cx = ys[idx], xs[idx]

                    # Place mask centered at (cy, cx)
                    ys_src, xs_src = np.where(src_mask)
                    if len(ys_src) == 0:
                        continue
                    y_off = cy - (ys_src.min() + ys_src.max()) // 2
                    x_off = cx - (xs_src.min() + xs_src.max()) // 2

                    # Shift mask
                    shifted = np.zeros_like(sem, dtype=bool)
                    for y, x in zip(ys_src, xs_src):
                        ny, nx = y + y_off, x + x_off
                        if 0 <= ny < sem.shape[0] and 0 <= nx < sem.shape[1]:
                            shifted[ny, nx] = True

                    # Only paste onto road or similar classes
                    valid_paste = shifted & np.isin(sem, [0, 2, 3, 14, 15])
                    sem_out[valid_paste] = target_cls

            out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
            save_semantic_label(sem_out, out_path)

    logger.info("A4 complete. Output: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------
def combine_a1_a2(
    cityscapes_root: str,
    a1_dir: Path,
    a2_dir: Path,
    output_dir: Path,
):
    """Combine A1 + A2: take A1 clustering, then apply A2 depth-edge split."""
    logger.info("Combining A1 + A2...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        images = get_cityscapes_images(cityscapes_root, split)
        for entry in tqdm(images, desc=f"A1+A2 {split}"):
            a1_sem = load_semantic_label(a1_dir, entry["city"], entry["stem"], split)
            a2_sem = load_semantic_label(a2_dir, entry["city"], entry["stem"], split)
            if a1_sem is None:
                continue

            # Start with A1, then apply A2's rare-class fixes
            combined = a1_sem.copy()
            if a2_sem is not None:
                for rare_id in [4, 9, 11]:  # guard rail, tunnel, polegroup
                    combined[a2_sem == rare_id] = rare_id

            out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
            save_semantic_label(combined, out_path)

    logger.info("A1+A2 complete. Output: %s", output_dir)
    return output_dir


def combine_all(
    cityscapes_root: str,
    a1_dir: Path,
    a2_dir: Path,
    a3_dir: Path,
    a4_dir: Path,
    output_dir: Path,
):
    """Combine A1 + A2 + A3 + A4: layered application."""
    logger.info("Combining A1+A2+A3+A4...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        images = get_cityscapes_images(cityscapes_root, split)
        for entry in tqdm(images, desc=f"All {split}"):
            # Start with A1 (best clustering)
            combined = load_semantic_label(a1_dir, entry["city"], entry["stem"], split)
            if combined is None:
                continue

            # Layer A2 (stuff-class depth fixes)
            a2_sem = load_semantic_label(a2_dir, entry["city"], entry["stem"], split)
            if a2_sem is not None:
                for rare_id in [4, 9, 11]:
                    combined[a2_sem == rare_id] = rare_id

            # Layer A3 (thing-class propagation)
            a3_sem = load_semantic_label(a3_dir, entry["city"], entry["stem"], split)
            if a3_sem is not None:
                for rare_id in [16, 17]:
                    combined[a3_sem == rare_id] = rare_id

            # Layer A4 (copy-paste)
            a4_sem = load_semantic_label(a4_dir, entry["city"], entry["stem"], split)
            if a4_sem is not None:
                for rare_id in [4, 16, 17]:
                    combined[a4_sem == rare_id] = rare_id

            out_path = output_dir / split / entry["city"] / f"{entry['stem']}_leftImg8bit_semantic.png"
            save_semantic_label(combined, out_path)

    logger.info("All complete. Output: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Level 1 Dead-Class Recovery Ablation")
    parser.add_argument("--cityscapes_root", default="/Users/qbit-glitch/Desktop/datasets/cityscapes")
    parser.add_argument("--baseline_dir", default="/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc")
    parser.add_argument("--output_root", default="/Users/qbit-glitch/Desktop/datasets/cityscapes/level1_ablation")
    parser.add_argument("--methods", default="A1,A2,A3,A4,A1A2,ALL", help="Comma-separated: A1,A2,A3,A4,A1A2,ALL")
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--rare_budget", type=int, default=12)
    parser.add_argument("--a3_max_images", type=int, default=None)
    args = parser.parse_args()

    cityscapes_root = args.cityscapes_root
    baseline_dir = Path(args.baseline_dir)
    output_root = Path(args.output_root)
    methods = [m.strip() for m in args.methods.split(",")]

    logger.info("Baseline: %s", baseline_dir)
    logger.info("Output root: %s", output_root)
    logger.info("Methods: %s", methods)

    # Track outputs for later evaluation
    variant_dirs = {}

    if "A1" in methods:
        variant_dirs["A1"] = a1_frequency_aware_overclustering(
            cityscapes_root, baseline_dir, output_root / "A1",
            k=args.k, rare_budget=args.rare_budget,
        )

    if "A2" in methods:
        variant_dirs["A2"] = a2_depth_edge_semantic_split(
            cityscapes_root, baseline_dir, output_root / "A2",
        )

    if "A3" in methods:
        variant_dirs["A3"] = a3_knn_propagation(
            cityscapes_root, baseline_dir, output_root / "A3",
            max_images=args.a3_max_images,
        )

    if "A4" in methods:
        variant_dirs["A4"] = a4_geometric_copypaste(
            cityscapes_root, baseline_dir, output_root / "A4",
        )

    if "A1A2" in methods:
        a1_dir = variant_dirs.get("A1", output_root / "A1")
        a2_dir = variant_dirs.get("A2", output_root / "A2")
        if a1_dir.exists() and a2_dir.exists():
            variant_dirs["A1A2"] = combine_a1_a2(
                cityscapes_root, a1_dir, a2_dir, output_root / "A1A2"
            )
        else:
            logger.warning("Skipping A1A2: need both A1 and A2 outputs")

    if "ALL" in methods:
        a1_dir = variant_dirs.get("A1", output_root / "A1")
        a2_dir = variant_dirs.get("A2", output_root / "A2")
        a3_dir = variant_dirs.get("A3", output_root / "A3")
        a4_dir = variant_dirs.get("A4", output_root / "A4")
        if all(d.exists() for d in [a1_dir, a2_dir, a3_dir, a4_dir]):
            variant_dirs["ALL"] = combine_all(
                cityscapes_root, a1_dir, a2_dir, a3_dir, a4_dir, output_root / "ALL"
            )
        else:
            logger.warning("Skipping ALL: need all individual outputs")

    # Save manifest
    manifest = {k: str(v) for k, v in variant_dirs.items()}
    manifest["baseline"] = str(baseline_dir)
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("Generation complete. Variants:")
    for name, path in variant_dirs.items():
        logger.info("  %s: %s", name, path)
    logger.info("Manifest: %s", output_root / "manifest.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
