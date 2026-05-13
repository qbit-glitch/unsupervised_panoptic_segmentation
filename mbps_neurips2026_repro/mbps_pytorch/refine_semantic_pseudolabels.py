#!/usr/bin/env python3
"""Refine CAUSE-TR 27-class semantic pseudo-labels using DINOv2 features + depth.

Three incremental steps (all training-free):
  S1: DINOv2 prototype denoising — flip noisy patches to nearest class prototype
  S2: DINOv2 k-NN label diffusion — smooth labels via feature-graph propagation
  S3: Depth-aware DenseCRF — pixel-level refinement using RGB + depth pairwise

Starts from RAW CAUSE (not CRF) to avoid double-CRF.

Usage:
    PYTHONPATH=. python mbps_pytorch/refine_semantic_pseudolabels.py \
        --cityscapes_root /path/to/cityscapes --split val \
        --output eval_semantic_refinement.json
"""

import argparse
import json
import logging
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CAUSE_CLASSES = 27
PATCH_H, PATCH_W = 32, 64
FULL_H, FULL_W = 1024, 2048
WORK_H, WORK_W = 512, 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data discovery & loading
# ---------------------------------------------------------------------------

def discover_images(cityscapes_root, split):
    """Enumerate all Cityscapes images for a split."""
    img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
    entries = []
    for city in sorted(os.listdir(img_dir)):
        city_path = os.path.join(img_dir, city)
        if not os.path.isdir(city_path):
            continue
        for fname in sorted(os.listdir(city_path)):
            if fname.endswith("_leftImg8bit.png"):
                stem = fname.replace("_leftImg8bit.png", "")
                entries.append({"stem": stem, "city": city})
    return entries


def load_all_features(cityscapes_root, entries, split):
    """Load all DINOv2 features into memory (float16 to save RAM)."""
    features = {}
    for entry in tqdm(entries, desc="Loading DINOv2 features"):
        path = os.path.join(
            cityscapes_root, "dinov2_features", split,
            entry["city"], f"{entry['stem']}_leftImg8bit.npy",
        )
        features[entry["stem"]] = np.load(path)  # (2048, 768) float16
    return features


def load_cause_labels_patch(cityscapes_root, entries, split,
                            semantic_subdir="pseudo_semantic_cause"):
    """Load raw CAUSE labels downsampled to 32x64 patch resolution."""
    labels = {}
    for entry in tqdm(entries, desc="Loading CAUSE labels"):
        path = os.path.join(
            cityscapes_root, semantic_subdir, split,
            entry["city"], f"{entry['stem']}.png",
        )
        sem = np.array(Image.open(path))  # (1024, 2048) uint8
        sem_patch = np.array(
            Image.fromarray(sem).resize((PATCH_W, PATCH_H), Image.NEAREST)
        )  # (32, 64) uint8
        labels[entry["stem"]] = sem_patch
    return labels


# ---------------------------------------------------------------------------
# S1: DINOv2 Prototype Denoising
# ---------------------------------------------------------------------------

def compute_global_prototypes(all_features, all_labels, num_classes=NUM_CAUSE_CLASSES):
    """Compute L2-normalized per-class prototypes from all images.

    Returns:
        prototypes: (K, 768) float32, L2-normalized
        class_counts: (K,) int64
    """
    embed_dim = 768
    proto_sum = np.zeros((num_classes, embed_dim), dtype=np.float64)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for stem in tqdm(all_features, desc="S1: computing prototypes"):
        feats = np.nan_to_num(all_features[stem].astype(np.float32))  # (2048, 768)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_norm = feats / (norms + 1e-8)

        patch_labels = all_labels[stem].flatten()  # (2048,)

        for c in range(num_classes):
            mask = patch_labels == c
            count = mask.sum()
            if count > 0:
                proto_sum[c] += feats_norm[mask].astype(np.float64).sum(axis=0)
                class_counts[c] += count

    # Average and L2-normalize
    prototypes = np.zeros((num_classes, embed_dim), dtype=np.float32)
    for c in range(num_classes):
        if class_counts[c] > 0:
            prototypes[c] = (proto_sum[c] / class_counts[c]).astype(np.float32)
            prototypes[c] /= np.linalg.norm(prototypes[c]) + 1e-8

    active = (class_counts > 0).sum()
    log.info(f"S1: {active}/{num_classes} classes have patches")
    for c in range(num_classes):
        if class_counts[c] > 0:
            log.info(f"  class {c:2d}: {class_counts[c]:>8d} patches")

    return prototypes, class_counts


def apply_prototype_correction(all_features, all_labels, prototypes, class_counts,
                                threshold=0.3):
    """Flip noisy patches to the nearest prototype class.

    For each patch: if nearest prototype != CAUSE label AND cosine sim to
    CAUSE class < threshold, flip to the prototype class.
    """
    valid_classes = class_counts > 0
    refined = {}
    total_flipped = 0
    total_patches = 0

    for stem in all_features:
        feats = np.nan_to_num(all_features[stem].astype(np.float32))
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_norm = feats / (norms + 1e-8)

        cause_labels = all_labels[stem].flatten()  # (2048,)

        # Cosine similarity to all prototypes
        sim = feats_norm @ prototypes.T  # (2048, 27)
        sim[:, ~valid_classes] = -np.inf

        proto_class = np.argmax(sim, axis=1)  # (2048,)
        cause_sim = sim[np.arange(len(cause_labels)), cause_labels]  # (2048,)

        # Flip condition
        disagree = proto_class != cause_labels
        low_conf = cause_sim < threshold
        flip_mask = disagree & low_conf

        new_labels = cause_labels.copy()
        new_labels[flip_mask] = proto_class[flip_mask]

        refined[stem] = new_labels.reshape(PATCH_H, PATCH_W).astype(np.uint8)
        total_flipped += flip_mask.sum()
        total_patches += len(cause_labels)

    pct = total_flipped / total_patches * 100
    log.info(f"S1: Flipped {total_flipped}/{total_patches} patches ({pct:.2f}%)")
    return refined


# ---------------------------------------------------------------------------
# S2: DINOv2 k-NN Label Diffusion
# ---------------------------------------------------------------------------

def step_s2_knn_diffusion(all_features, all_labels, k=10, alpha=0.5,
                           num_iterations=15):
    """Per-image k-NN label diffusion at patch resolution."""
    num_classes = NUM_CAUSE_CLASSES
    n_patches = PATCH_H * PATCH_W
    refined = {}

    for stem in tqdm(all_features, desc="S2: k-NN diffusion"):
        feats = np.nan_to_num(all_features[stem].astype(np.float32))  # (2048, 768)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_norm = feats / (norms + 1e-8)

        patch_labels = all_labels[stem].flatten()  # (2048,)

        # Cosine similarity matrix
        sim_matrix = feats_norm @ feats_norm.T  # (2048, 2048)
        np.fill_diagonal(sim_matrix, -np.inf)  # exclude self

        # Top-k neighbors per patch
        knn_indices = np.argpartition(-sim_matrix, k, axis=1)[:, :k]  # (2048, k)

        # Initialize label distribution: one-hot
        label_dist = np.zeros((n_patches, num_classes), dtype=np.float32)
        label_dist[np.arange(n_patches), patch_labels] = 1.0

        # Iterative diffusion (vectorized)
        for _ in range(num_iterations):
            neighbor_dists = label_dist[knn_indices]  # (2048, k, 27)
            neighbor_mean = neighbor_dists.mean(axis=1)  # (2048, 27)
            label_dist = alpha * label_dist + (1 - alpha) * neighbor_mean

        refined_labels = np.argmax(label_dist, axis=1).astype(np.uint8)
        refined[stem] = refined_labels.reshape(PATCH_H, PATCH_W)

    return refined


# ---------------------------------------------------------------------------
# S3: Depth-Aware DenseCRF
# ---------------------------------------------------------------------------

def step_s3_depth_crf(patch_labels, cityscapes_root, entries, split,
                       num_classes=NUM_CAUSE_CLASSES,
                       crf_iterations=10,
                       sxy_bilateral=80, srgb=13, compat_bilateral=10,
                       sxy_gaussian=3, compat_gaussian=3,
                       sxy_depth=80, sdepth=0.3, compat_depth=5,
                       gt_prob=0.9):
    """Upsample patch labels to 512x1024, apply depth-aware DenseCRF."""
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils

    refined = {}

    for entry in tqdm(entries, desc="S3: Depth-CRF"):
        stem, city = entry["stem"], entry["city"]

        # 1. Upsample patch labels (32,64) -> (512,1024) nearest
        patch_lbl = patch_labels[stem]  # (32, 64) uint8
        lbl_up = np.array(
            Image.fromarray(patch_lbl).resize((WORK_W, WORK_H), Image.NEAREST)
        )  # (512, 1024) uint8

        # 2. Build unary with label smoothing
        rest_prob = (1.0 - gt_prob) / (num_classes - 1)
        unary = np.full((num_classes, WORK_H, WORK_W), rest_prob, dtype=np.float32)
        for c in range(num_classes):
            unary[c][lbl_up == c] = gt_prob

        U = -np.log(np.clip(unary, 1e-10, 1.0))
        U = U.reshape(num_classes, -1)
        U = np.ascontiguousarray(U.astype(np.float32))

        # 3. Load RGB at 512x1024
        img_path = os.path.join(
            cityscapes_root, "leftImg8bit", split, city,
            f"{stem}_leftImg8bit.png",
        )
        img = np.array(
            Image.open(img_path).convert("RGB").resize(
                (WORK_W, WORK_H), Image.BILINEAR
            )
        ).astype(np.uint8)
        img = np.ascontiguousarray(img)

        # 4. Load depth at 512x1024
        depth_path = os.path.join(
            cityscapes_root, "depth_spidepth", split, city,
            f"{stem}.npy",
        )
        depth = np.load(depth_path).astype(np.float32)  # (512, 1024)
        if depth.shape != (WORK_H, WORK_W):
            depth = np.array(
                Image.fromarray(depth).resize((WORK_W, WORK_H), Image.BILINEAR)
            )

        # 5. Build CRF
        d = dcrf.DenseCRF2D(WORK_W, WORK_H, num_classes)
        d.setUnaryEnergy(U)

        # Pairwise 1: RGB bilateral
        d.addPairwiseBilateral(
            sxy=sxy_bilateral, srgb=srgb,
            rgbim=img, compat=compat_bilateral,
        )

        # Pairwise 2: Gaussian smoothness
        d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)

        # Pairwise 3: Depth bilateral
        # Reshape depth to (H, W, 1) for create_pairwise_bilateral
        depth_3d = depth[:, :, np.newaxis]
        depth_feats = crf_utils.create_pairwise_bilateral(
            sdims=(sxy_depth, sxy_depth),
            schan=(sdepth,),
            img=depth_3d,
            chdim=2,
        )
        d.addPairwiseEnergy(depth_feats, compat=compat_depth)

        # 6. Inference
        Q = d.inference(crf_iterations)
        Q = np.array(Q).reshape(num_classes, WORK_H, WORK_W)
        refined_lbl = np.argmax(Q, axis=0).astype(np.uint8)

        refined[stem] = refined_lbl

    return refined


# ---------------------------------------------------------------------------
# Save & evaluate helpers
# ---------------------------------------------------------------------------

def save_patch_labels(patch_labels, cityscapes_root, subdir, entries, split):
    """Save 32x64 patch labels as 1024x2048 PNGs."""
    out_base = os.path.join(cityscapes_root, subdir, split)
    for entry in entries:
        stem, city = entry["stem"], entry["city"]
        lbl = patch_labels[stem]  # (32, 64)
        lbl_full = np.array(
            Image.fromarray(lbl).resize((FULL_W, FULL_H), Image.NEAREST)
        )
        out_dir = os.path.join(out_base, city)
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(lbl_full.astype(np.uint8)).save(
            os.path.join(out_dir, f"{stem}.png")
        )
    log.info(f"Saved {len(entries)} images to {out_base}")


def save_fullres_labels(full_labels, cityscapes_root, subdir, entries, split):
    """Save full-res (512x1024) labels as 1024x2048 PNGs."""
    out_base = os.path.join(cityscapes_root, subdir, split)
    for entry in entries:
        stem, city = entry["stem"], entry["city"]
        lbl = full_labels[stem]  # (512, 1024) from CRF
        lbl_full = np.array(
            Image.fromarray(lbl).resize((FULL_W, FULL_H), Image.NEAREST)
        )
        out_dir = os.path.join(out_base, city)
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(lbl_full.astype(np.uint8)).save(
            os.path.join(out_dir, f"{stem}.png")
        )
    log.info(f"Saved {len(entries)} images to {out_base}")


def run_evaluation(cityscapes_root, split, semantic_subdir, instance_subdir,
                   thing_mode, cc_min_area, eval_hw):
    """Run semantic + panoptic evaluation for a given semantic subdir."""
    import sys
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from mbps_pytorch.evaluate_cascade_pseudolabels import (
        discover_pairs,
        evaluate_panoptic,
        evaluate_semantic,
    )

    pairs = discover_pairs(cityscapes_root, split, semantic_subdir, instance_subdir)
    log.info(f"Evaluating {semantic_subdir}: {len(pairs)} pairs")

    sem_results = evaluate_semantic(pairs, eval_hw, cause27=True)
    pan_results = evaluate_panoptic(
        pairs, eval_hw,
        thing_mode=thing_mode,
        cause27=True,
        cc_min_area=cc_min_area,
    )

    return {
        "mIoU": sem_results["miou"],
        "pixel_acc": sem_results["pixel_accuracy"],
        "PQ": pan_results["PQ"],
        "PQ_stuff": pan_results["PQ_stuff"],
        "PQ_things": pan_results["PQ_things"],
        "SQ": pan_results.get("SQ", 0),
        "RQ": pan_results.get("RQ", 0),
    }


def print_comparison_table(results):
    """Print a summary table of all evaluated steps."""
    print("\n" + "=" * 70)
    print("SEMANTIC REFINEMENT RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Step':<25s} {'mIoU':>6s} {'PxAcc':>6s} {'PQ':>6s} {'PQ_st':>6s} {'PQ_th':>6s}"
    print(header)
    print("-" * 70)
    for label, res in results.items():
        row = (
            f"{label:<25s} "
            f"{res['mIoU']:>6.1f} "
            f"{res['pixel_acc']:>6.1f} "
            f"{res['PQ']:>6.1f} "
            f"{res['PQ_stuff']:>6.1f} "
            f"{res['PQ_things']:>6.1f}"
        )
        print(row)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Refine CAUSE-TR semantic pseudo-labels (S1+S2+S3)"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 1024],
                        metavar=("H", "W"))
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause",
                        help="Source CAUSE labels (default: raw, not CRF)")

    # S1 parameters
    parser.add_argument("--s1_threshold", type=float, default=0.3,
                        help="Cosine sim threshold for prototype flipping")

    # S2 parameters
    parser.add_argument("--s2_k", type=int, default=10,
                        help="Number of nearest neighbors")
    parser.add_argument("--s2_alpha", type=float, default=0.5,
                        help="Self-weight in diffusion (higher = more conservative)")
    parser.add_argument("--s2_iterations", type=int, default=15,
                        help="Number of diffusion iterations")

    # S3 parameters
    parser.add_argument("--s3_sxy_bilateral", type=int, default=80)
    parser.add_argument("--s3_srgb", type=int, default=13)
    parser.add_argument("--s3_compat_bilateral", type=int, default=10)
    parser.add_argument("--s3_sxy_gaussian", type=int, default=3)
    parser.add_argument("--s3_compat_gaussian", type=int, default=3)
    parser.add_argument("--s3_sxy_depth", type=int, default=80)
    parser.add_argument("--s3_sdepth", type=float, default=0.3)
    parser.add_argument("--s3_compat_depth", type=int, default=5)
    parser.add_argument("--s3_gt_prob", type=float, default=0.9)
    parser.add_argument("--s3_iterations", type=int, default=10)

    # Control
    parser.add_argument("--steps", type=str, default="s1,s2,s3",
                        help="Comma-separated steps to run")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation (just generate labels)")
    parser.add_argument("--instance_subdir", type=str,
                        default="pseudo_instance_spidepth")
    parser.add_argument("--thing_mode", type=str, default="hybrid")
    parser.add_argument("--cc_min_area", type=int, default=500)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path for results")

    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",")]
    eval_hw = tuple(args.eval_size)

    log.info(f"Cityscapes root: {args.cityscapes_root}")
    log.info(f"Split: {args.split}, Steps: {steps}")
    log.info(f"Source semantics: {args.semantic_subdir}")

    # 1. Discover images
    entries = discover_images(args.cityscapes_root, args.split)
    log.info(f"Found {len(entries)} images")

    # 2. Load DINOv2 features (~1.5 GB for 500 images in float16)
    t0 = time.time()
    features = load_all_features(args.cityscapes_root, entries, args.split)
    log.info(f"Features loaded in {time.time() - t0:.1f}s")

    # 3. Load CAUSE labels at patch resolution
    cause_labels = load_cause_labels_patch(
        args.cityscapes_root, entries, args.split, args.semantic_subdir
    )

    results = {}

    # 4. Evaluate baselines
    if not args.skip_eval:
        log.info("Evaluating baseline: raw CAUSE")
        results["baseline_cause"] = run_evaluation(
            args.cityscapes_root, args.split,
            "pseudo_semantic_cause", args.instance_subdir,
            args.thing_mode, args.cc_min_area, eval_hw,
        )
        log.info("Evaluating baseline: CAUSE-CRF")
        results["baseline_cause_crf"] = run_evaluation(
            args.cityscapes_root, args.split,
            "pseudo_semantic_cause_crf", args.instance_subdir,
            args.thing_mode, args.cc_min_area, eval_hw,
        )

    current_labels = cause_labels  # Start from RAW CAUSE

    # 5. S1: Prototype Denoising
    if "s1" in steps:
        log.info("=" * 60)
        log.info("Running S1: DINOv2 Prototype Denoising")
        log.info("=" * 60)

        prototypes, class_counts = compute_global_prototypes(features, current_labels)
        current_labels = apply_prototype_correction(
            features, current_labels, prototypes, class_counts,
            threshold=args.s1_threshold,
        )

        save_patch_labels(
            current_labels, args.cityscapes_root,
            "pseudo_semantic_cause_s1", entries, args.split,
        )

        if not args.skip_eval:
            log.info("Evaluating S1")
            results["S1"] = run_evaluation(
                args.cityscapes_root, args.split,
                "pseudo_semantic_cause_s1", args.instance_subdir,
                args.thing_mode, args.cc_min_area, eval_hw,
            )

    # 6. S2: k-NN Diffusion
    if "s2" in steps:
        log.info("=" * 60)
        log.info("Running S2: DINOv2 k-NN Label Diffusion")
        log.info("=" * 60)

        current_labels = step_s2_knn_diffusion(
            features, current_labels,
            k=args.s2_k, alpha=args.s2_alpha,
            num_iterations=args.s2_iterations,
        )

        save_patch_labels(
            current_labels, args.cityscapes_root,
            "pseudo_semantic_cause_s12", entries, args.split,
        )

        if not args.skip_eval:
            log.info("Evaluating S1+S2")
            results["S1+S2"] = run_evaluation(
                args.cityscapes_root, args.split,
                "pseudo_semantic_cause_s12", args.instance_subdir,
                args.thing_mode, args.cc_min_area, eval_hw,
            )

    # 7. S3: Depth-Aware CRF
    if "s3" in steps:
        log.info("=" * 60)
        log.info("Running S3: Depth-Aware DenseCRF")
        log.info("=" * 60)

        refined_full = step_s3_depth_crf(
            current_labels, args.cityscapes_root, entries, args.split,
            crf_iterations=args.s3_iterations,
            sxy_bilateral=args.s3_sxy_bilateral,
            srgb=args.s3_srgb,
            compat_bilateral=args.s3_compat_bilateral,
            sxy_gaussian=args.s3_sxy_gaussian,
            compat_gaussian=args.s3_compat_gaussian,
            sxy_depth=args.s3_sxy_depth,
            sdepth=args.s3_sdepth,
            compat_depth=args.s3_compat_depth,
            gt_prob=args.s3_gt_prob,
        )

        save_fullres_labels(
            refined_full, args.cityscapes_root,
            "pseudo_semantic_cause_s123", entries, args.split,
        )

        if not args.skip_eval:
            log.info("Evaluating S1+S2+S3")
            results["S1+S2+S3"] = run_evaluation(
                args.cityscapes_root, args.split,
                "pseudo_semantic_cause_s123", args.instance_subdir,
                args.thing_mode, args.cc_min_area, eval_hw,
            )

    # 8. Summary
    if results:
        print_comparison_table(results)

    # 9. Save JSON
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
