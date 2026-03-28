#!/usr/bin/env python3
"""Post-process instance pseudo-labels to improve panoptic quality.

4 incremental steps (training-free, uses RGB images for guidance):
  1. Morphological cleanup (fill holes, remove tiny fragments, merge adjacent CC)
  2. Guided filter boundary refinement (He et al. 2010, numpy implementation)
  3. Superpixel boundary snapping (cv2.watershed-based)
  4. Fragment merging via color+spatial proximity

Usage:
    PYTHONPATH=. python mbps_pytorch/postprocess_instances.py \\
        --cityscapes_root /path/to/cityscapes \\
        --split val --cause27 --evaluate \\
        --steps 1,2,3,4
"""

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    label,
    uniform_filter,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WORK_H, WORK_W = 512, 1024
NUM_CLASSES = 19
IGNORE_LABEL = 255
_THING_IDS = set(range(11, 19))
_STUFF_IDS = set(range(0, 11))

_CAUSE27_TO_TRAINID = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_instances(npz_path):
    """Load instance masks from NPZ → (M, H, W) bool, (M,) float32 scores."""
    data = np.load(npz_path)
    masks = data["masks"]
    scores = data["scores"] if "scores" in data else None
    num_valid = int(data["num_valid"]) if "num_valid" in data else masks.shape[0]
    masks = masks[:num_valid]
    if scores is not None:
        scores = scores[:num_valid]

    if masks.shape[0] == 0:
        return np.zeros((0, WORK_H, WORK_W), dtype=bool), np.array([], dtype=np.float32)

    if masks.ndim == 2:
        M, N = masks.shape
        if "h_patches" in data and "w_patches" in data:
            hp, wp = int(data["h_patches"]), int(data["w_patches"])
        else:
            hp, wp = None, None
            for hp_c, wp_c in [(512, 1024), (128, 64), (64, 128), (32, 64)]:
                if hp_c * wp_c == N:
                    hp, wp = hp_c, wp_c
                    break
            if hp is None:
                return np.zeros((0, WORK_H, WORK_W), dtype=bool), np.array([], dtype=np.float32)
        masks = masks.reshape(M, hp, wp)

    # Resize to WORK_H x WORK_W if needed
    M = masks.shape[0]
    if masks.shape[1:] != (WORK_H, WORK_W):
        resized = np.zeros((M, WORK_H, WORK_W), dtype=bool)
        for i in range(M):
            m_img = Image.fromarray(masks[i].astype(np.uint8) * 255)
            resized[i] = np.array(m_img.resize((WORK_W, WORK_H), Image.NEAREST)) > 127
        masks = resized
    else:
        masks = masks.astype(bool)

    if scores is None:
        scores = masks.sum(axis=(1, 2)).astype(np.float32)
    else:
        scores = scores[:M].astype(np.float32)

    return masks, scores


def save_instances(masks, scores, output_path):
    """Save in flat (M, H*W) format compatible with evaluate_cascade_pseudolabels."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    M = masks.shape[0]
    flat = masks.reshape(M, -1) if M > 0 else np.zeros((0, WORK_H * WORK_W), dtype=bool)
    np.savez_compressed(
        output_path,
        masks=flat,
        scores=scores,
        num_valid=M,
        h_patches=WORK_H,
        w_patches=WORK_W,
    )


def load_semantic(sem_path, cause27=True):
    """Load semantic pseudo-labels → (H, W) uint8 trainIDs at WORK resolution."""
    sem = np.array(Image.open(sem_path))
    if cause27:
        sem = _CAUSE27_TO_TRAINID[sem]
    if sem.shape != (WORK_H, WORK_W):
        sem = np.array(
            Image.fromarray(sem).resize((WORK_W, WORK_H), Image.NEAREST)
        )
    return sem


def load_rgb(rgb_path):
    """Load RGB image → (H, W, 3) uint8 at WORK resolution."""
    img = np.array(Image.open(rgb_path).convert("RGB"))
    if img.shape[:2] != (WORK_H, WORK_W):
        img = np.array(
            Image.fromarray(img).resize((WORK_W, WORK_H), Image.BILINEAR)
        )
    return img


def classify_mask(mask, semantic):
    """Majority-vote class assignment for an instance mask."""
    vals = semantic[mask]
    vals = vals[vals != IGNORE_LABEL]
    if len(vals) == 0:
        return IGNORE_LABEL
    return int(np.bincount(vals, minlength=NUM_CLASSES).argmax())


def _remove_empty(masks, scores):
    """Filter out masks with zero area."""
    keep = [i for i in range(len(masks)) if masks[i].sum() > 0]
    if len(keep) == len(masks):
        return masks, scores
    if len(keep) == 0:
        return np.zeros((0, WORK_H, WORK_W), dtype=bool), np.array([], dtype=np.float32)
    return masks[keep], scores[keep]


# ---------------------------------------------------------------------------
# Step 1: Morphological Cleanup
# ---------------------------------------------------------------------------

def morphological_cleanup(masks, scores, semantic,
                          closing_radius=3, min_fragment_area=200,
                          merge_gap_size=5):
    """Fill holes, remove tiny fragments, merge adjacent same-class instances.

    Args:
        closing_radius: Disk structuring element radius for binary closing.
        min_fragment_area: Remove masks with area below this threshold.
        merge_gap_size: Dilation radius for detecting adjacent same-class masks.
    """
    if masks.shape[0] == 0:
        return masks, scores

    M, H, W = masks.shape

    # (a) Binary closing to fill small holes
    y, x = np.ogrid[-closing_radius:closing_radius + 1,
                     -closing_radius:closing_radius + 1]
    disk = (x ** 2 + y ** 2) <= closing_radius ** 2
    for i in range(M):
        masks[i] = binary_closing(masks[i], structure=disk)

    # (b) Remove small fragments
    keep = [i for i in range(M) if masks[i].sum() >= min_fragment_area]
    if len(keep) == 0:
        return np.zeros((0, H, W), dtype=bool), np.array([], dtype=np.float32)
    masks = masks[keep]
    scores = scores[keep]
    M = len(masks)

    # (c) Dilation-based merge of same-class adjacent masks
    classes = np.array([classify_mask(masks[i], semantic) for i in range(M)])
    dilate_struct = np.ones((2 * merge_gap_size + 1, 2 * merge_gap_size + 1), dtype=bool)
    merged_out = set()

    for cls in _THING_IDS:
        cls_idx = [i for i in range(M) if classes[i] == cls and i not in merged_out]
        if len(cls_idx) <= 1:
            continue

        # Build dilated union for this class
        union_dilated = np.zeros((H, W), dtype=np.int32)
        for idx in cls_idx:
            dilated = binary_dilation(masks[idx], structure=dilate_struct)
            union_dilated[dilated] = 1

        # Connected components on the dilated union
        labeled, n_groups = label(union_dilated)

        # For each CC group, merge all masks that overlap with it
        for group_id in range(1, n_groups + 1):
            group_mask = labeled == group_id
            members = [i for i in cls_idx if (masks[i] & group_mask).any()]
            if len(members) <= 1:
                continue

            # Merge all into the first member
            merged = masks[members[0]].copy()
            best_score = scores[members[0]]
            for mi in members[1:]:
                merged |= masks[mi]
                best_score = max(best_score, scores[mi])
                merged_out.add(mi)

            masks[members[0]] = merged
            scores[members[0]] = best_score

    # Remove merged masks
    keep = [i for i in range(M) if i not in merged_out]
    masks, scores = masks[keep], scores[keep]
    return _remove_empty(masks, scores)


# ---------------------------------------------------------------------------
# Step 2: Guided Filter (He et al. 2010)
# ---------------------------------------------------------------------------

def _guided_filter_color(guide, src, radius, eps):
    """Guided filter with 3-channel color guide.

    Args:
        guide: (H, W, 3) float64 in [0, 1]
        src: (H, W) float64 in [0, 1]
        radius: box filter radius
        eps: regularization

    Returns:
        (H, W) float64 filtered output
    """
    size = 2 * radius + 1

    def fmean(x):
        return uniform_filter(x.astype(np.float64), size=size, mode="reflect")

    I_r, I_g, I_b = guide[:, :, 0], guide[:, :, 1], guide[:, :, 2]

    mean_I_r = fmean(I_r)
    mean_I_g = fmean(I_g)
    mean_I_b = fmean(I_b)
    mean_p = fmean(src)

    # Covariance of (I, p)
    cov_Ip_r = fmean(I_r * src) - mean_I_r * mean_p
    cov_Ip_g = fmean(I_g * src) - mean_I_g * mean_p
    cov_Ip_b = fmean(I_b * src) - mean_I_b * mean_p

    # 3x3 covariance matrix of I (with eps regularization on diagonal)
    var_rr = fmean(I_r * I_r) - mean_I_r * mean_I_r + eps
    var_rg = fmean(I_r * I_g) - mean_I_r * mean_I_g
    var_rb = fmean(I_r * I_b) - mean_I_r * mean_I_b
    var_gg = fmean(I_g * I_g) - mean_I_g * mean_I_g + eps
    var_gb = fmean(I_g * I_b) - mean_I_g * mean_I_b
    var_bb = fmean(I_b * I_b) - mean_I_b * mean_I_b + eps

    # Explicit 3x3 matrix inverse via cofactors
    inv_rr = var_gg * var_bb - var_gb * var_gb
    inv_rg = var_gb * var_rb - var_rg * var_bb
    inv_rb = var_rg * var_gb - var_gg * var_rb
    inv_gg = var_rr * var_bb - var_rb * var_rb
    inv_gb = var_rb * var_rg - var_rr * var_gb
    inv_bb = var_rr * var_gg - var_rg * var_rg

    det = var_rr * inv_rr + var_rg * inv_rg + var_rb * inv_rb
    det = np.maximum(det, 1e-10)

    a_r = (inv_rr * cov_Ip_r + inv_rg * cov_Ip_g + inv_rb * cov_Ip_b) / det
    a_g = (inv_rg * cov_Ip_r + inv_gg * cov_Ip_g + inv_gb * cov_Ip_b) / det
    a_b = (inv_rb * cov_Ip_r + inv_gb * cov_Ip_g + inv_bb * cov_Ip_b) / det

    b = mean_p - a_r * mean_I_r - a_g * mean_I_g - a_b * mean_I_b

    q = (fmean(a_r) * I_r + fmean(a_g) * I_g + fmean(a_b) * I_b + fmean(b))
    return np.clip(q, 0.0, 1.0)


def guided_filter_refine(masks, scores, rgb,
                         radius=8, eps=0.01, threshold=0.5):
    """Refine instance boundaries using guided filter with RGB as guide.

    Each binary mask → float → guided filter → threshold. Overlaps
    resolved by assigning each pixel to the highest-response instance.
    """
    if masks.shape[0] == 0:
        return masks, scores

    M, H, W = masks.shape
    guide = rgb.astype(np.float64) / 255.0

    # Apply guided filter to each mask
    filtered = np.zeros((M, H, W), dtype=np.float64)
    for i in range(M):
        filtered[i] = _guided_filter_color(guide, masks[i].astype(np.float64),
                                           radius, eps)

    # Resolve overlaps: each pixel → instance with highest filtered value
    new_masks = np.zeros((M, H, W), dtype=bool)
    max_vals = np.max(filtered, axis=0)
    argmax_vals = np.argmax(filtered, axis=0)

    for i in range(M):
        new_masks[i] = (argmax_vals == i) & (filtered[i] > threshold)

    return _remove_empty(new_masks, scores)


# ---------------------------------------------------------------------------
# Step 3: Superpixel Boundary Snapping
# ---------------------------------------------------------------------------

def _generate_superpixels(rgb, n_markers=2000):
    """Generate superpixels via watershed on RGB gradient magnitude.

    Returns:
        labels: (H, W) int32, region labels (>= 1). -1 = boundary.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_uint8 = np.clip(grad_mag / grad_mag.max() * 255, 0, 255).astype(np.uint8)

    # Find flat regions (low gradient) as seeds
    _, thresh = cv2.threshold(grad_uint8, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv2.erode(thresh, kernel, iterations=2)

    # Label connected components as markers
    n_labels, markers = cv2.connectedComponents(sure_fg)

    # Subsample if too many markers
    if n_labels - 1 > n_markers:
        areas = np.bincount(markers.ravel())
        areas[0] = 0  # ignore background
        top_k = np.argsort(-areas)[:n_markers]
        new_markers = np.zeros_like(markers)
        for new_id, old_id in enumerate(top_k, 1):
            new_markers[markers == old_id] = new_id
        markers = new_markers

    # Ensure markers are int32, background (0) = unknown for watershed
    markers = markers.astype(np.int32)

    # Run watershed
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    labels = cv2.watershed(rgb_bgr, markers)
    return labels


def superpixel_snap(masks, scores, rgb,
                    n_markers=2000, majority_thresh=0.5):
    """Snap instance boundaries to superpixel boundaries.

    For each superpixel, assign it to the instance that covers the
    majority of its area. Boundary pixels keep original assignment.
    """
    if masks.shape[0] == 0:
        return masks, scores

    M, H, W = masks.shape
    sp_labels = _generate_superpixels(rgb, n_markers)

    sp_ids = np.unique(sp_labels)
    sp_ids = sp_ids[sp_ids > 0]  # exclude boundary (-1) and background (0)

    new_masks = np.zeros((M, H, W), dtype=bool)

    # Vectorized: build instance ID map for fast lookup
    # inst_map[y, x] = instance index (0-based) or -1 if no instance
    inst_map = np.full((H, W), -1, dtype=np.int32)
    for i in range(M):
        inst_map[masks[i]] = i

    for sp_id in sp_ids:
        sp_mask = sp_labels == sp_id
        sp_area = sp_mask.sum()
        if sp_area == 0:
            continue

        # Count pixels per instance within this superpixel
        inst_vals = inst_map[sp_mask]
        inst_vals = inst_vals[inst_vals >= 0]
        if len(inst_vals) == 0:
            continue

        counts = np.bincount(inst_vals, minlength=M)
        best_inst = int(counts.argmax())
        best_coverage = counts[best_inst] / sp_area

        if best_coverage >= majority_thresh:
            new_masks[best_inst] |= sp_mask

    # Handle boundary pixels (-1): keep original assignment
    boundary = sp_labels == -1
    if boundary.any():
        for i in range(M):
            new_masks[i] |= (masks[i] & boundary)

    return _remove_empty(new_masks, scores)


# ---------------------------------------------------------------------------
# Step 4: Fragment Merging via Color + Spatial Proximity
# ---------------------------------------------------------------------------

def merge_fragments(masks, scores, semantic, rgb,
                    spatial_thresh=50.0, color_thresh=25.0,
                    min_area=100):
    """Merge over-segmented same-class fragments using color+spatial cues.

    For each thing class, greedily merge the closest pair of fragments
    if both spatial distance (centroid) and color distance (mean LAB)
    are below threshold.
    """
    if masks.shape[0] <= 1:
        return masks, scores

    M, H, W = masks.shape

    # Convert RGB → LAB for perceptual color distance
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    classes = np.array([classify_mask(masks[i], semantic) for i in range(M)])
    merged_out = set()

    for cls in _THING_IDS:
        cls_idx = [i for i in range(M)
                   if classes[i] == cls and masks[i].sum() >= min_area
                   and i not in merged_out]
        if len(cls_idx) <= 1:
            continue

        # Compute centroids and mean LAB colors
        n = len(cls_idx)
        centroids = np.zeros((n, 2), dtype=np.float64)
        colors = np.zeros((n, 3), dtype=np.float64)
        for j, i in enumerate(cls_idx):
            ys, xs = np.where(masks[i])
            centroids[j] = [ys.mean(), xs.mean()]
            colors[j] = lab[masks[i]].mean(axis=0)

        # Greedy merge
        active = list(range(n))
        changed = True
        while changed and len(active) > 1:
            changed = False
            best_combined = float("inf")
            best_pair = None

            for ai in range(len(active)):
                for bi in range(ai + 1, len(active)):
                    a, b = active[ai], active[bi]
                    sdist = np.linalg.norm(centroids[a] - centroids[b])
                    cdist = np.linalg.norm(colors[a] - colors[b])
                    if sdist < spatial_thresh and cdist < color_thresh:
                        combined = sdist + cdist
                        if combined < best_combined:
                            best_combined = combined
                            best_pair = (ai, bi)

            if best_pair is not None:
                ai, bi = best_pair
                a, b = active[ai], active[bi]
                idx_a, idx_b = cls_idx[a], cls_idx[b]

                # Merge b into a
                masks[idx_a] |= masks[idx_b]
                scores[idx_a] = max(scores[idx_a], scores[idx_b])
                merged_out.add(idx_b)

                # Update centroid and color
                ys, xs = np.where(masks[idx_a])
                centroids[a] = [ys.mean(), xs.mean()]
                colors[a] = lab[masks[idx_a]].mean(axis=0)

                active.pop(bi)
                changed = True

    keep = [i for i in range(M) if i not in merged_out]
    masks, scores = masks[keep], scores[keep]
    return _remove_empty(masks, scores)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_images(cityscapes_root, split, instance_subdir):
    """Find all (city, stem) pairs from the instance directory."""
    inst_dir = os.path.join(cityscapes_root, instance_subdir, split)
    image_list = []
    for city in sorted(os.listdir(inst_dir)):
        city_path = os.path.join(inst_dir, city)
        if not os.path.isdir(city_path):
            continue
        for fname in sorted(os.listdir(city_path)):
            if fname.endswith(".npz"):
                stem = fname.replace(".npz", "").replace("_leftImg8bit", "")
                image_list.append((city, stem))
    return image_list


def find_npz(base_dir, city, stem):
    """Find NPZ with or without _leftImg8bit suffix."""
    for suffix in ["", "_leftImg8bit"]:
        path = os.path.join(base_dir, city, stem + suffix + ".npz")
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

def run_evaluation(cityscapes_root, split, semantic_subdir, instance_subdir,
                   cause27, step_label):
    """Run panoptic evaluation using existing infrastructure."""
    from mbps_pytorch.evaluate_cascade_pseudolabels import (
        discover_pairs,
        evaluate_panoptic,
    )
    pairs = discover_pairs(cityscapes_root, split, semantic_subdir, instance_subdir)
    results = evaluate_panoptic(
        pairs, (WORK_H, WORK_W),
        thing_mode="hybrid",
        cause27=cause27,
        cc_min_area=500,
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-process instance pseudo-labels for panoptic quality"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause_crf")
    parser.add_argument("--instance_subdir", type=str,
                        default="pseudo_instance_spidepth")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output subdir prefix (default: {instance_subdir}_pp)")
    parser.add_argument("--steps", type=str, default="1,2,3,4",
                        help="Comma-separated steps to apply (e.g. '1,2')")
    parser.add_argument("--cause27", action="store_true")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run panoptic eval after each step combination")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    # Step 1 params
    parser.add_argument("--closing_radius", type=int, default=3)
    parser.add_argument("--min_fragment_area", type=int, default=200)
    parser.add_argument("--merge_gap_size", type=int, default=5)
    # Step 2 params
    parser.add_argument("--gf_radius", type=int, default=8)
    parser.add_argument("--gf_eps", type=float, default=0.01)
    parser.add_argument("--gf_threshold", type=float, default=0.5)
    # Step 3 params
    parser.add_argument("--sp_n_markers", type=int, default=2000)
    parser.add_argument("--sp_majority_thresh", type=float, default=0.5)
    # Step 4 params
    parser.add_argument("--merge_spatial_thresh", type=float, default=50.0)
    parser.add_argument("--merge_color_thresh", type=float, default=25.0)
    parser.add_argument("--merge_min_area", type=int, default=100)

    args = parser.parse_args()

    steps = sorted(set(int(s) for s in args.steps.split(",")))
    out_prefix = args.output_prefix or f"{args.instance_subdir}_pp"

    # Discover images
    image_list = discover_images(args.cityscapes_root, args.split,
                                 args.instance_subdir)
    if args.max_images:
        image_list = image_list[:args.max_images]
    log.info(f"Found {len(image_list)} images in {args.instance_subdir}/{args.split}")

    all_results = {}

    # Process each cumulative step combination
    for step_max in steps:
        active_steps = [s for s in steps if s <= step_max]
        step_label = "+".join(str(s) for s in active_steps)
        output_subdir = f"{out_prefix}_s{''.join(str(s) for s in active_steps)}"

        log.info(f"\n{'=' * 60}")
        log.info(f"Running steps: {step_label}")
        log.info(f"Output: {output_subdir}")
        log.info(f"{'=' * 60}")

        t0 = time.time()
        total_in, total_out = 0, 0
        n_images = 0

        for city, stem in tqdm(image_list, desc=f"Steps {step_label}",
                                ncols=100):
            # Load instances
            npz_path = find_npz(
                os.path.join(args.cityscapes_root, args.instance_subdir,
                             args.split),
                city, stem,
            )
            if npz_path is None:
                continue

            masks, scores = load_instances(npz_path)
            total_in += len(masks)

            # Load semantic
            sem_path = os.path.join(
                args.cityscapes_root, args.semantic_subdir, args.split,
                city, f"{stem}.png",
            )
            if not os.path.exists(sem_path):
                sem_path = os.path.join(
                    args.cityscapes_root, args.semantic_subdir, args.split,
                    city, f"{stem}_leftImg8bit.png",
                )
            semantic = load_semantic(sem_path, args.cause27)

            # Load RGB (only if steps 2/3/4 are active)
            rgb = None
            if any(s in active_steps for s in [2, 3, 4]):
                rgb_path = os.path.join(
                    args.cityscapes_root, "leftImg8bit", args.split,
                    city, f"{stem}_leftImg8bit.png",
                )
                rgb = load_rgb(rgb_path)

            # Apply steps in order
            if 1 in active_steps:
                masks, scores = morphological_cleanup(
                    masks, scores, semantic,
                    closing_radius=args.closing_radius,
                    min_fragment_area=args.min_fragment_area,
                    merge_gap_size=args.merge_gap_size,
                )

            if 2 in active_steps:
                masks, scores = guided_filter_refine(
                    masks, scores, rgb,
                    radius=args.gf_radius,
                    eps=args.gf_eps,
                    threshold=args.gf_threshold,
                )

            if 3 in active_steps:
                masks, scores = superpixel_snap(
                    masks, scores, rgb,
                    n_markers=args.sp_n_markers,
                    majority_thresh=args.sp_majority_thresh,
                )

            if 4 in active_steps:
                masks, scores = merge_fragments(
                    masks, scores, semantic, rgb,
                    spatial_thresh=args.merge_spatial_thresh,
                    color_thresh=args.merge_color_thresh,
                    min_area=args.merge_min_area,
                )

            # Save
            out_path = os.path.join(
                args.cityscapes_root, output_subdir, args.split,
                city, f"{stem}.npz",
            )
            save_instances(masks, scores, out_path)
            total_out += len(masks)
            n_images += 1

        elapsed = time.time() - t0
        avg_inst = total_out / max(n_images, 1)
        log.info(f"  Processed {n_images} images in {elapsed:.1f}s "
                 f"({elapsed / max(n_images, 1):.2f}s/img)")
        log.info(f"  Instances: {total_in} -> {total_out} "
                 f"({avg_inst:.1f} avg/img)")

        # Evaluate
        if args.evaluate:
            log.info(f"  Evaluating...")
            results = run_evaluation(
                args.cityscapes_root, args.split,
                args.semantic_subdir, output_subdir,
                args.cause27, step_label,
            )
            all_results[step_label] = results
            log.info(
                f"  PQ={results['PQ']:.2f} | "
                f"PQ_stuff={results['PQ_stuff']:.2f} | "
                f"PQ_things={results['PQ_things']:.2f} | "
                f"SQ={results['SQ']:.2f} | RQ={results['RQ']:.2f}"
            )

    # Summary table
    if all_results:
        print(f"\n{'=' * 70}")
        print("POST-PROCESSING RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Steps':<15} {'PQ':>6} {'PQ_st':>7} {'PQ_th':>7} "
              f"{'SQ':>6} {'RQ':>6}")
        print(f"{'-' * 15} {'-' * 6} {'-' * 7} {'-' * 7} "
              f"{'-' * 6} {'-' * 6}")
        for lbl, r in sorted(all_results.items()):
            print(f"{lbl:<15} {r['PQ']:>6.1f} {r['PQ_stuff']:>7.1f} "
                  f"{r['PQ_things']:>7.1f} {r['SQ']:>6.1f} "
                  f"{r['RQ']:>6.1f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
