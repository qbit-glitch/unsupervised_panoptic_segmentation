"""Self-enhanced copy-paste augmentation for MBPS v2 training.

Adapted from CUPS (Hahn et al., "Scene-Centric Unsupervised Panoptic
Segmentation", CVPR 2025) for our JAX/TPU pipeline. Operates at
token-level for labels and pixel-level for images/depth.

Two modes:
    - Pseudo-label mode (default): Paste instances extracted from
      pseudo-label annotations in the current batch.
    - Self-enhanced mode: Paste instances from cached model predictions
      instead of pseudo-labels (activates after a configurable number of
      steps, following CUPS Section 3.2).

Reference:
    Hahn et al., CVPR 2025 — cups/augmentation.py (CopyPasteAugmentation)
    Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation
    Method for Instance Segmentation", CVPR 2021
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _resize_nearest(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize a 2D or 3D array using nearest-neighbor interpolation.

    Pure numpy implementation — no scipy/cv2/torch dependencies.
    Works for 2D (h, w) and 3D (h, w, c) arrays.

    Args:
        arr: Input array of shape (h, w) or (h, w, c).
        new_h: Target height (>= 1).
        new_w: Target width (>= 1).

    Returns:
        Resized array.
    """
    old_h, old_w = arr.shape[:2]
    row_idx = np.clip(
        (np.arange(new_h) * old_h / new_h).astype(np.intp), 0, old_h - 1
    )
    col_idx = np.clip(
        (np.arange(new_w) * old_w / new_w).astype(np.intp), 0, old_w - 1
    )
    return arr[np.ix_(row_idx, col_idx)]


def _extract_instances(
    pseudo_instance: np.ndarray,
    pseudo_semantic: np.ndarray,
    image: np.ndarray,
    depth: np.ndarray,
    patch_size: int = 16,
    min_tokens: int = 4,
) -> List[Dict[str, np.ndarray]]:
    """Extract individual object instances from a single sample.

    Each instance is cropped to its bounding box in token coordinates,
    with corresponding pixel-level image and depth crops.

    Args:
        pseudo_instance: (N,) int32 token-level instance IDs (0=background).
        pseudo_semantic: (N,) int32 token-level semantic class labels.
        image: (H, W, 3) float32 pixel-level image.
        depth: (H, W) float32 pixel-level depth map.
        patch_size: Backbone patch size (16 for DINOv3 ViT-B/16).
        min_tokens: Minimum token count for a valid instance.

    Returns:
        List of instance dicts with keys:
            mask: (h_tok, w_tok) bool — token-level binary mask (cropped).
            semantic: (h_tok, w_tok) int32 — semantic labels (cropped).
            image: (h_tok*ps, w_tok*ps, 3) float32 — image crop.
            depth: (h_tok*ps, w_tok*ps) float32 — depth crop.
    """
    H, W = image.shape[:2]
    H_p = H // patch_size
    W_p = W // patch_size
    ps = patch_size

    inst_2d = pseudo_instance.reshape(H_p, W_p)
    sem_2d = pseudo_semantic.reshape(H_p, W_p)

    instances: List[Dict[str, np.ndarray]] = []
    unique_ids = np.unique(inst_2d)

    for inst_id in unique_ids:
        if inst_id == 0:
            continue

        mask = inst_2d == inst_id
        if mask.sum() < min_tokens:
            continue

        # Bounding box in token coordinates
        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1

        instances.append({
            "mask": mask[r0:r1, c0:c1].copy(),
            "semantic": sem_2d[r0:r1, c0:c1].copy(),
            "image": image[r0 * ps:r1 * ps, c0 * ps:c1 * ps].copy(),
            "depth": depth[r0 * ps:r1 * ps, c0 * ps:c1 * ps].copy(),
        })

    return instances


def copy_paste_augment(
    batch: Dict[str, np.ndarray],
    rng: np.random.RandomState,
    patch_size: int = 16,
    max_paste_objects: int = 5,
    min_instance_tokens: int = 4,
    flip_prob: float = 0.5,
    scale_range: Tuple[float, float] = (1.0, 1.0),
    source_batch: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Apply copy-paste augmentation to a batch.

    For each target image in the batch, randomly selects instances and
    pastes them at random positions. Updates both pixel-level data
    (image, depth) and token-level labels (semantic, instance).

    Following CUPS (Hahn et al., CVPR 2025):
    - Random selection of 1..max_paste_objects instances per image
    - Random horizontal flip with probability flip_prob
    - Random scale from uniform(scale_range[0], scale_range[1])
    - Random spatial placement (uniform over valid token positions)
    - Occlusion: pasted objects overwrite existing labels

    Args:
        batch: Dict with keys:
            image: (B, H, W, 3) float32 pixel-level RGB.
            depth: (B, H, W) float32 pixel-level depth.
            pseudo_semantic: (B, N) int32 token-level class labels.
            pseudo_instance: (B, N) int32 token-level instance IDs.
        rng: NumPy RandomState for reproducible augmentation.
        patch_size: Backbone patch size (16 for DINOv3).
        max_paste_objects: Max instances to paste per target image.
        min_instance_tokens: Min tokens for a pasteable instance.
        flip_prob: Probability of horizontally flipping pasted objects.
        scale_range: (min_scale, max_scale) for random instance scaling.
            Default (1.0, 1.0) means no scaling. CUPS uses (0.25, 1.5).
        source_batch: Optional separate batch to extract paste instances
            from. When None, instances come from ``batch`` itself.
            Used for self-enhanced copy-paste (CUPS Section 3.2).

    Returns:
        Augmented batch dict (same keys and shapes as input).
    """
    # Validate required fields
    required = {"image", "depth", "pseudo_semantic", "pseudo_instance"}
    if not required.issubset(batch.keys()):
        return batch

    images = batch["image"]       # (B, H, W, 3)
    depths = batch["depth"]       # (B, H, W)
    semantics = batch["pseudo_semantic"]  # (B, N)
    instances = batch["pseudo_instance"]  # (B, N)

    B, H, W = images.shape[0], images.shape[1], images.shape[2]
    H_p = H // patch_size
    W_p = W // patch_size
    ps = patch_size

    # ---- Step 1: Collect all instances from source ----
    # Self-enhanced mode: extract from source_batch (model predictions).
    # Standard mode: extract from the target batch itself.
    source = source_batch if source_batch is not None else batch
    src_images = source["image"]
    src_depths = source["depth"]
    src_semantics = source["pseudo_semantic"]
    src_instances = source["pseudo_instance"]

    all_instances: List[Dict[str, np.ndarray]] = []
    for b in range(src_images.shape[0]):
        inst_list = _extract_instances(
            src_instances[b], src_semantics[b], src_images[b], src_depths[b],
            patch_size=patch_size, min_tokens=min_instance_tokens,
        )
        all_instances.extend(inst_list)

    if len(all_instances) == 0:
        return batch

    # ---- Step 2: Copy arrays for in-place modification ----
    out_images = images.copy()
    out_depths = depths.copy()
    out_semantics = semantics.copy()
    out_instances = instances.copy()

    # ---- Step 3: Paste random instances onto each target ----
    for b in range(B):
        inst_2d = out_instances[b].reshape(H_p, W_p).copy()
        sem_2d = out_semantics[b].reshape(H_p, W_p).copy()
        img = out_images[b]  # (H, W, 3), a view into out_images
        dep = out_depths[b]  # (H, W), a view into out_depths

        max_id = int(inst_2d.max())
        n_paste = rng.randint(1, max_paste_objects + 1)

        for _ in range(n_paste):
            # Sample random source instance
            src = all_instances[rng.randint(len(all_instances))]
            mask_crop = src["mask"].copy()    # (h_tok, w_tok)
            sem_crop = src["semantic"].copy() # (h_tok, w_tok)
            img_crop = src["image"].copy()    # (h_tok*ps, w_tok*ps, 3)
            dep_crop = src["depth"].copy()    # (h_tok*ps, w_tok*ps)

            h_tok, w_tok = mask_crop.shape

            # Random horizontal flip
            if rng.random() < flip_prob:
                mask_crop = mask_crop[:, ::-1].copy()
                sem_crop = sem_crop[:, ::-1].copy()
                img_crop = img_crop[:, ::-1, :].copy()
                dep_crop = dep_crop[:, ::-1].copy()

            # Random scale (CUPS: uniform over scale_range)
            if scale_range[0] < scale_range[1]:
                scale = rng.uniform(scale_range[0], scale_range[1])
            else:
                scale = scale_range[0]

            if scale != 1.0:
                new_h_tok = max(1, round(h_tok * scale))
                new_w_tok = max(1, round(w_tok * scale))

                # Skip if scaled instance exceeds target grid
                if new_h_tok > H_p or new_w_tok > W_p:
                    continue

                # Resize token-level arrays (nearest-neighbor)
                mask_crop = _resize_nearest(
                    mask_crop.astype(np.uint8), new_h_tok, new_w_tok,
                ).astype(bool)
                sem_crop = _resize_nearest(sem_crop, new_h_tok, new_w_tok)

                # Resize pixel-level arrays to exact token-aligned size
                new_h_px = new_h_tok * ps
                new_w_px = new_w_tok * ps
                img_crop = _resize_nearest(img_crop, new_h_px, new_w_px)
                dep_crop = _resize_nearest(dep_crop, new_h_px, new_w_px)

                h_tok, w_tok = new_h_tok, new_w_tok

            # Random placement in token coordinates
            max_r = max(H_p - h_tok, 0)
            max_c = max(W_p - w_tok, 0)
            r_off = rng.randint(0, max_r + 1)
            c_off = rng.randint(0, max_c + 1)

            # Clip to grid boundaries
            r_end = min(r_off + h_tok, H_p)
            c_end = min(c_off + w_tok, W_p)
            h_actual = r_end - r_off
            w_actual = c_end - c_off

            # Trim source crops to fit
            m = mask_crop[:h_actual, :w_actual]
            s = sem_crop[:h_actual, :w_actual]
            ic = img_crop[:h_actual * ps, :w_actual * ps]
            dc = dep_crop[:h_actual * ps, :w_actual * ps]

            if m.sum() == 0:
                continue

            # Assign new unique instance ID to pasted object
            max_id += 1
            new_id = max_id

            # ---- Token-level label update ----
            target_inst = inst_2d[r_off:r_end, c_off:c_end]
            inst_2d[r_off:r_end, c_off:c_end] = np.where(
                m, new_id, target_inst,
            )

            target_sem = sem_2d[r_off:r_end, c_off:c_end]
            sem_2d[r_off:r_end, c_off:c_end] = np.where(
                m, s, target_sem,
            )

            # ---- Pixel-level image/depth update ----
            # Upsample token mask to pixel resolution via repeat
            pixel_mask = np.repeat(
                np.repeat(m, ps, axis=0), ps, axis=1,
            )  # (h_actual*ps, w_actual*ps)

            pr0, pr1 = r_off * ps, r_end * ps
            pc0, pc1 = c_off * ps, c_end * ps

            # Blend: where mask is True, take from source; else keep target
            img[pr0:pr1, pc0:pc1] = np.where(
                pixel_mask[:, :, np.newaxis], ic, img[pr0:pr1, pc0:pc1],
            )
            dep[pr0:pr1, pc0:pc1] = np.where(
                pixel_mask, dc, dep[pr0:pr1, pc0:pc1],
            )

        # Write modified labels back (images/depths modified in-place)
        out_instances[b] = inst_2d.reshape(-1)
        out_semantics[b] = sem_2d.reshape(-1)

    # ---- Return augmented batch, preserving extra fields ----
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v  # default: keep original
    out["image"] = out_images
    out["depth"] = out_depths
    out["pseudo_semantic"] = out_semantics
    out["pseudo_instance"] = out_instances
    return out


def create_self_enhanced_source(
    images: np.ndarray,
    depths: np.ndarray,
    semantic_preds: np.ndarray,
    instance_preds: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.75,
) -> Optional[Dict[str, np.ndarray]]:
    """Create a source batch from model predictions for self-enhanced mode.

    Converts model inference outputs into the batch dict format expected
    by ``copy_paste_augment``'s ``source_batch`` parameter.

    Following CUPS (Hahn et al., CVPR 2025, Section 3.2): after a warmup
    period, the model's own confident predictions replace pseudo-labels
    as the instance source for copy-paste augmentation.

    Args:
        images: (B, H, W, 3) float32 original images.
        depths: (B, H, W) float32 depth maps.
        semantic_preds: (B, N) int32 predicted class per token.
        instance_preds: (B, N) int32 predicted instance ID per token (0=bg).
        confidence: (B, N) float32 per-token confidence (max softmax).
        confidence_threshold: Minimum confidence to keep a prediction.

    Returns:
        Source batch dict, or None if no confident instances remain.
    """
    mask = confidence >= confidence_threshold
    filtered_semantic = np.where(mask, semantic_preds, 0).astype(np.int32)
    filtered_instance = np.where(mask, instance_preds, 0).astype(np.int32)

    if not np.any(filtered_instance > 0):
        return None

    return {
        "image": images,
        "depth": depths,
        "pseudo_semantic": filtered_semantic,
        "pseudo_instance": filtered_instance,
    }
