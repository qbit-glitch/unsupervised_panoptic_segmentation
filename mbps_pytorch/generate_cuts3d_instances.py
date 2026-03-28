#!/usr/bin/env python3
"""Generate instance pseudo-labels using CutS3D (ICCV 2025).

Wraps the existing CutS3D implementation at models/instance/cuts3d.py
to produce NPZ instance files compatible with evaluate_cascade_pseudolabels.py.

Algorithm: DINOv2 features → Affinity Matrix → Spatial Importance Sharpening
→ NCut bipartition → LocalCut (MinCut on 3D k-NN graph) → CRF → masks

Usage:
    python mbps_pytorch/generate_cuts3d_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --limit 5 --device mps
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add parent to path for model imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.instance.cuts3d import extract_pseudo_masks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Cityscapes thing class trainIDs
DEFAULT_THING_IDS = set(range(11, 19))
CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# DINOv2 ViT-B/14 patch grid for 448x896 input
FEAT_H, FEAT_W, FEAT_DIM = 32, 64, 768
WORK_H, WORK_W = 512, 1024

# Cityscapes camera intrinsics at patch resolution (32x64)
# Original 2048x1024: fx=2262.52, fy=2265.30, cx=1096.98, cy=513.14
# At patch level (each patch covers 14px at 448x896 input):
#   fx_patch = 2262.52 * (64/2048) = 70.7
#   fy_patch = 2265.30 * (32/1024) = 70.8
#   cx_patch = 1096.98 * (64/2048) = 34.3
#   cy_patch = 513.14 * (32/1024) = 16.1
CS_FX, CS_FY = 70.7, 70.8
CS_CX, CS_CY = 34.3, 16.1

# Depth scale: SPIdepth outputs normalized [0,1], convert to ~meters
DEPTH_SCALE = 80.0


def find_image_triples(cityscapes_root, split, feature_subdir="dinov2_features",
                       depth_subdir="depth_spidepth",
                       semantic_subdir="pseudo_semantic_mapped_k80"):
    """Find matching feature + depth + semantic + image file sets."""
    root = Path(cityscapes_root)
    feat_dir = root / feature_subdir / split
    depth_dir = root / depth_subdir / split
    sem_dir = root / semantic_subdir / split
    img_dir = root / "leftImg8bit" / split

    triples = []
    for feat_path in sorted(feat_dir.rglob("*.npy")):
        stem = feat_path.stem
        # Remove _leftImg8bit suffix if present
        base_stem = stem.replace("_leftImg8bit", "")

        # Find matching files
        city = feat_path.parent.name
        depth_path = depth_dir / city / f"{base_stem}.npy"
        sem_path = sem_dir / city / f"{base_stem}.png"
        img_path = img_dir / city / f"{base_stem}_leftImg8bit.png"

        if depth_path.exists() and sem_path.exists():
            triples.append({
                "feat": feat_path,
                "depth": depth_path,
                "semantic": sem_path,
                "image": img_path if img_path.exists() else None,
                "stem": base_stem,
                "city": city,
            })

    return triples


def process_single_image(entry, thing_ids, device, max_instances=20,
                         tau_ncut=0.0, tau_knn=0.115, k=10,
                         sigma_gauss=3.0, beta=0.45, min_mask_frac=0.005,
                         max_mask_frac=0.25,
                         use_crf=False, sc_samples=3):
    """Process one image through CutS3D and return instances.

    Returns list of (mask_512x1024, class_id, score) tuples.

    Key change vs original: initial_active mask restricts CutS3D to thing-class
    patches only (using pseudo-semantic labels at patch level). This prevents the
    NCut from finding large stuff bipartitions (road, sky, buildings) and focuses
    extraction on thing-class regions where instance separation is meaningful.
    """
    # Load features
    features = np.load(str(entry["feat"])).astype(np.float32)  # (2048, 768)
    features_t = torch.from_numpy(features).to(device)

    # Load depth (keep normalized [0,1] — local_cut_3d normalizes 3D coords internally)
    depth = np.load(str(entry["depth"])).astype(np.float32)  # (H, W) in [0,1]
    if depth.shape != (WORK_H, WORK_W):
        depth = np.array(
            Image.fromarray(depth).resize((WORK_W, WORK_H), Image.BILINEAR)
        )
    depth_t = torch.from_numpy(depth).to(device)

    # Load semantic labels
    semantic = np.array(Image.open(str(entry["semantic"])))
    if semantic.shape != (WORK_H, WORK_W):
        semantic = np.array(
            Image.fromarray(semantic).resize((WORK_W, WORK_H), Image.NEAREST)
        )

    # Build thing-class active mask at patch level (32x64 grid)
    # A patch is "active" if its majority semantic label is a thing class.
    # This restricts NCut to thing-class regions, preventing large stuff bipartitions.
    sem_patch = np.array(
        Image.fromarray(semantic).resize((FEAT_W, FEAT_H), Image.NEAREST)
    )  # (32, 64)
    thing_mask_2d = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
    for cls_id in thing_ids:
        thing_mask_2d[sem_patch == cls_id] = 1.0
    initial_active = torch.from_numpy(thing_mask_2d.reshape(-1)).to(device)  # (K,)

    # Load image for CRF (optional)
    if use_crf and entry["image"] is not None and entry["image"].exists():
        img = np.array(Image.open(str(entry["image"])).resize(
            (WORK_W, WORK_H), Image.BILINEAR
        )).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).to(device)
    else:
        # Dummy image (CRF disabled)
        img_t = torch.zeros((WORK_H, WORK_W, 3), dtype=torch.float32, device=device)

    # Run CutS3D — initial_active restricts to thing patches; depth stays [0,1]
    # since local_cut_3d normalizes the 3D point cloud to [0,1]^3 internally.
    result = extract_pseudo_masks(
        features=features_t,
        depth=depth_t,
        image=img_t,
        patch_h=FEAT_H,
        patch_w=FEAT_W,
        max_instances=max_instances,
        tau_ncut=tau_ncut,
        tau_knn=tau_knn,
        k=k,
        sigma_gauss=sigma_gauss,
        beta=beta,
        min_mask_size=min_mask_frac,
        max_mask_size=max_mask_frac,
        sc_samples=sc_samples,
        use_crf=use_crf,
        fx=CS_FX, fy=CS_FY, cx=CS_CX, cy=CS_CY,
        initial_active=initial_active,
    )

    # Extract valid masks and convert to full resolution
    instances = []
    for i in range(result.num_valid):
        mask_patch = result.masks[i].cpu().numpy()  # (K,) at patch resolution
        score = result.scores[i].item()

        # Reshape to patch grid and upsample to full resolution
        mask_2d = mask_patch.reshape(FEAT_H, FEAT_W)
        mask_full = np.array(
            Image.fromarray(mask_2d.astype(np.float32)).resize(
                (WORK_W, WORK_H), Image.BILINEAR
            )
        ) > 0.5  # binarize after upsampling

        if mask_full.sum() < 100:
            continue

        # Determine class from semantic overlap
        sem_vals = semantic[mask_full]
        sem_vals = sem_vals[sem_vals < 19]  # valid trainIDs only
        if len(sem_vals) == 0:
            continue

        counts = np.bincount(sem_vals, minlength=19)
        majority_cls = int(counts.argmax())

        # Keep only thing-class instances
        if majority_cls not in thing_ids:
            continue

        instances.append((mask_full, majority_cls, score))

    return instances


def save_instances(instances, output_path, h=WORK_H, w=WORK_W):
    """Save instances in NPZ format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instances:
        np.savez_compressed(
            str(output_path),
            masks=np.zeros((0, h * w), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            num_valid=0,
            h_patches=h,
            w_patches=w,
        )
        return

    num = len(instances)
    masks = np.zeros((num, h * w), dtype=bool)
    scores = np.zeros(num, dtype=np.float32)

    for i, (mask, cls, score) in enumerate(instances):
        masks[i] = mask.ravel()
        scores[i] = score

    np.savez_compressed(
        str(output_path),
        masks=masks,
        scores=scores,
        num_valid=num,
        h_patches=h,
        w_patches=w,
    )


def main():
    parser = argparse.ArgumentParser(description="CutS3D instance pseudo-labels")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output_subdir", type=str, default="pseudo_instance_cuts3d")
    parser.add_argument("--feature_subdir", type=str, default="dinov2_features")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_mapped_k80")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--max_instances", type=int, default=20)
    parser.add_argument("--tau_ncut", type=float, default=0.0)
    parser.add_argument("--tau_knn", type=float, default=0.115)
    parser.add_argument("--k_nn", type=int, default=10)
    parser.add_argument("--sigma_gauss", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.45)
    parser.add_argument("--min_mask_frac", type=float, default=0.005)
    parser.add_argument("--max_mask_frac", type=float, default=0.25,
                        help="Max mask fraction (skip large stuff bipartitions, default: 0.25)")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--sc_samples", type=int, default=3,
                        help="Spatial Confidence samples (default: 3, paper: 6)")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Find image triples
    triples = find_image_triples(
        args.cityscapes_root, args.split,
        feature_subdir=args.feature_subdir,
        depth_subdir=args.depth_subdir,
        semantic_subdir=args.semantic_subdir,
    )
    logger.info(f"Found {len(triples)} image triples")

    if args.limit:
        triples = triples[:args.limit]
        logger.info(f"Limited to {args.limit} images")

    output_dir = Path(args.cityscapes_root) / args.output_subdir / args.split
    total_instances = 0
    per_class_counts = {cls: 0 for cls in sorted(DEFAULT_THING_IDS)}
    t0 = time.time()

    for entry in tqdm(triples, desc="CutS3D"):
        out_path = output_dir / entry["city"] / f"{entry['stem']}.npz"
        if out_path.exists():
            continue

        with torch.no_grad():
            instances = process_single_image(
                entry, DEFAULT_THING_IDS, device,
                max_instances=args.max_instances,
                tau_ncut=args.tau_ncut,
                tau_knn=args.tau_knn,
                k=args.k_nn,
                sigma_gauss=args.sigma_gauss,
                beta=args.beta,
                min_mask_frac=args.min_mask_frac,
                max_mask_frac=args.max_mask_frac,
                use_crf=args.use_crf,
                sc_samples=args.sc_samples,
            )

        save_instances(instances, out_path)

        n = len(instances)
        total_instances += n
        for _, cls, _ in instances:
            per_class_counts[cls] += 1

    elapsed = time.time() - t0
    n_images = len(triples)
    avg = total_instances / max(n_images, 1)

    stats = {
        "total_images": n_images,
        "total_instances": total_instances,
        "avg_instances_per_image": round(avg, 2),
        "per_class_counts": {
            CS_NAMES.get(cls, f"class_{cls}"): count
            for cls, count in sorted(per_class_counts.items())
        },
        "config": {
            "max_instances": args.max_instances,
            "tau_ncut": args.tau_ncut,
            "tau_knn": args.tau_knn,
            "k_nn": args.k_nn,
            "sigma_gauss": args.sigma_gauss,
            "beta": args.beta,
            "sc_samples": args.sc_samples,
            "use_crf": args.use_crf,
            "device": str(device),
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    stats_path = Path(args.cityscapes_root) / args.output_subdir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Done in {elapsed:.1f}s ({elapsed/max(n_images,1):.1f}s/img)")
    logger.info(f"Total: {total_instances} instances, {avg:.1f}/img")
    logger.info(f"Per-class: {stats['per_class_counts']}")


if __name__ == "__main__":
    main()
