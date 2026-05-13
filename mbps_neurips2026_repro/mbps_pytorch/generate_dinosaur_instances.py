#!/usr/bin/env python3
"""Generate instance pseudo-labels using trained DINOSAUR slot attention.

Loads a trained DINOSAUR model and extracts per-slot masks from DINOv3 features.
Each slot mask is classified via majority vote from CAUSE-TR semantic map.
Only thing-class masks above min_area are kept.

Usage:
    python mbps_pytorch/generate_dinosaur_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --cause27 \
        --checkpoint checkpoints/dinosaur/best.pth \
        --feature_subdir dinov3_features \
        --semantic_subdir pseudo_semantic_cause \
        --output_dir pseudo_instances_dinosaur \
        --min_area 500
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

# Import model from training script
from train_dinosaur import DINOSAUR, GRID_H, GRID_W, N_PATCHES, FEAT_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

WORK_H, WORK_W = 512, 1024
IGNORE_LABEL = 255

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

THING_IDS = set(range(11, 19))

CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# CAUSE 27-class → 19 trainID mapping
_CAUSE27_TO_TRAINID = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


def load_model(checkpoint_path, device="mps"):
    """Load trained DINOSAUR model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = DINOSAUR(
        input_dim=config.get("input_dim", FEAT_DIM),
        slot_dim=config["slot_dim"],
        num_slots=config["num_slots"],
        num_iters=config["num_iters"],
        decoder_hidden=config.get("decoder_hidden", 2048),
        decoder_layers=config.get("decoder_layers", 3),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded DINOSAUR from {checkpoint_path} (epoch {epoch})")
    logger.info(f"  num_slots={config['num_slots']}, slot_dim={config['slot_dim']}")

    return model, config


def crf_refine_slot_masks(soft_masks, image, n_iters=10,
                          sxy_gauss=3, compat_gauss=3,
                          sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    """Refine soft slot masks using dense CRF with image as guidance.

    Args:
        soft_masks: (K, H, W) float32 soft assignment probabilities per slot
        image: (H, W, 3) uint8 RGB image
        n_iters: CRF inference iterations

    Returns:
        refined_hard: (K, H, W) bool refined hard masks
    """
    K, H, W = soft_masks.shape

    d = dcrf.DenseCRF2D(W, H, K)

    # Unary potentials from soft masks
    # Clamp to avoid log(0)
    probs = np.clip(soft_masks, 1e-6, 1.0)
    probs = probs / probs.sum(axis=0, keepdims=True)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Pairwise: appearance-independent (smoothness)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)

    # Pairwise: appearance-dependent (bilateral — uses image color)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image, compat=compat_bilateral,
    )

    # Inference
    Q = d.inference(n_iters)
    Q = np.array(Q).reshape(K, H, W)

    # Hard assignment
    assignment = Q.argmax(axis=0)  # (H, W)
    refined_hard = np.zeros((K, H, W), dtype=bool)
    for k in range(K):
        refined_hard[k] = assignment == k

    return refined_hard


def slot_masks_to_instances(slot_masks, semantic_19, thing_ids=THING_IDS,
                            min_area=500, patch_h=None, patch_w=None):
    """Convert DINOSAUR slot masks to instance masks with semantic classification.

    Args:
        slot_masks: (K, GRID_H, GRID_W) — hard assignment masks per slot
        semantic_19: (WORK_H, WORK_W) — Cityscapes 19-class trainIDs
        thing_ids: set of thing trainIDs
        min_area: minimum instance area in pixels
        patch_h, patch_w: patch dimensions for upsampling

    Returns:
        list of (mask, class_id, score) tuples
    """
    if patch_h is None:
        patch_h = WORK_H // GRID_H
    if patch_w is None:
        patch_w = WORK_W // GRID_W

    K = slot_masks.shape[0]
    instances = []

    for k in range(K):
        slot_mask_grid = slot_masks[k]  # (GRID_H, GRID_W)

        # Skip empty slots
        if slot_mask_grid.sum() == 0:
            continue

        # Upsample to pixel resolution via nearest neighbor
        pixel_mask = np.zeros((WORK_H, WORK_W), dtype=bool)
        for r in range(GRID_H):
            for c in range(GRID_W):
                if slot_mask_grid[r, c]:
                    pixel_mask[r * patch_h:(r + 1) * patch_h,
                               c * patch_w:(c + 1) * patch_w] = True

        # Majority-vote semantic class (ignoring 255/void)
        sem_in_mask = semantic_19[pixel_mask]
        valid = sem_in_mask[sem_in_mask != IGNORE_LABEL]
        if len(valid) == 0:
            continue

        # Get majority class
        classes, counts = np.unique(valid, return_counts=True)
        majority_cls = classes[counts.argmax()]

        # Only keep thing classes
        if majority_cls not in thing_ids:
            continue

        # Refine mask: intersect with semantic class for better boundaries
        refined_mask = pixel_mask & (semantic_19 == majority_cls)

        # Split into connected components (slot may cover disjoint areas)
        labeled, n_cc = ndimage.label(refined_mask)
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = cc_mask.sum()
            if area >= min_area:
                # Score: area-based (larger = more confident)
                score = float(area) / (WORK_H * WORK_W)
                instances.append((cc_mask, int(majority_cls), score))

    return instances


def slot_masks_to_instances_pixel(slot_masks, semantic_19, thing_ids=THING_IDS,
                                   min_area=500):
    """Convert pixel-resolution slot masks (from CRF) to instances.

    Args:
        slot_masks: (K, WORK_H, WORK_W) — boolean masks at full pixel resolution
        semantic_19: (WORK_H, WORK_W) — Cityscapes 19-class trainIDs
        thing_ids: set of thing trainIDs
        min_area: minimum instance area in pixels

    Returns:
        list of (mask, class_id, score) tuples
    """
    K = slot_masks.shape[0]
    instances = []

    for k in range(K):
        pixel_mask = slot_masks[k]

        if pixel_mask.sum() == 0:
            continue

        # Majority-vote semantic class
        sem_in_mask = semantic_19[pixel_mask]
        valid = sem_in_mask[sem_in_mask != IGNORE_LABEL]
        if len(valid) == 0:
            continue

        classes, counts = np.unique(valid, return_counts=True)
        majority_cls = classes[counts.argmax()]

        if majority_cls not in thing_ids:
            continue

        # Refine mask: intersect with semantic class
        refined_mask = pixel_mask & (semantic_19 == majority_cls)

        # Split into connected components
        labeled, n_cc = ndimage.label(refined_mask)
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = cc_mask.sum()
            if area >= min_area:
                score = float(area) / (WORK_H * WORK_W)
                instances.append((cc_mask, int(majority_cls), score))

    return instances


def process_single_image(model, feature_path, semantic_path, device,
                         cause27=False, min_area=500, use_decoder_masks=True,
                         use_crf=False, image_path=None):
    """Process one image through DINOSAUR and extract instances.

    Args:
        use_crf: If True, refine slot masks with dense CRF using the original image.
        image_path: Path to original RGB image (required if use_crf=True).

    Returns:
        masks: (N, WORK_H, WORK_W) bool
        scores: (N,) float32
        boxes: (N, 4) float32
    """
    # Load features
    features = np.load(feature_path).astype(np.float32)
    features_t = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, N, 768)

    # Get slot masks
    with torch.no_grad():
        hard_masks, soft_masks = model.get_masks(features_t, use_decoder_masks=use_decoder_masks)
    # hard_masks: (1, K, GRID_H, GRID_W)

    if use_crf and HAS_CRF and image_path is not None:
        # CRF refinement: upsample soft masks to pixel resolution, then refine
        soft_np = soft_masks[0].cpu().numpy()  # (K, N_patches)
        K = soft_np.shape[0]
        soft_grid = soft_np.reshape(K, GRID_H, GRID_W)  # (K, 32, 64)

        # Bilinear upsample soft masks to pixel resolution
        soft_pixel = np.zeros((K, WORK_H, WORK_W), dtype=np.float32)
        for k in range(K):
            soft_pixel[k] = np.array(
                Image.fromarray(soft_grid[k]).resize((WORK_W, WORK_H), Image.BILINEAR)
            )

        # Load original image for CRF bilateral term
        img = np.array(Image.open(image_path).convert("RGB").resize(
            (WORK_W, WORK_H), Image.BILINEAR
        ), dtype=np.uint8)

        # Apply CRF
        refined = crf_refine_slot_masks(soft_pixel, img)  # (K, H, W) bool
        slot_masks = refined  # Already at pixel resolution
        pixel_level = True
    else:
        slot_masks = hard_masks[0].cpu().numpy().astype(bool)  # (K, GRID_H, GRID_W)
        pixel_level = False

    # Load semantic map
    sem_img = Image.open(semantic_path)
    sem_np = np.array(sem_img.resize((WORK_W, WORK_H), Image.NEAREST), dtype=np.uint8)
    if cause27:
        sem_19 = _CAUSE27_TO_TRAINID[sem_np]
    else:
        sem_19 = sem_np

    # Convert slots to instances
    if pixel_level:
        # Masks are already at pixel resolution from CRF
        instances = slot_masks_to_instances_pixel(
            slot_masks, sem_19, min_area=min_area,
        )
    else:
        instances = slot_masks_to_instances(
            slot_masks, sem_19, min_area=min_area,
        )

    if not instances:
        return (np.zeros((0, WORK_H, WORK_W), dtype=bool),
                np.array([], dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32))

    # Sort by score descending
    instances.sort(key=lambda x: x[2], reverse=True)

    masks = np.stack([inst[0] for inst in instances], axis=0)
    scores = np.array([inst[2] for inst in instances], dtype=np.float32)

    # Normalize scores
    if scores.max() > 0:
        scores = scores / scores.max()

    # Compute boxes
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if len(ys) > 0:
            boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]

    return masks, scores, boxes


def main():
    parser = argparse.ArgumentParser("Generate instances from trained DINOSAUR")
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DINOSAUR checkpoint (best.pth)")
    parser.add_argument("--feature_subdir", default="dinov3_features")
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_cause")
    parser.add_argument("--output_dir", default="pseudo_instances_dinosaur")
    parser.add_argument("--min_area", type=int, default=500)
    parser.add_argument("--cause27", action="store_true")
    parser.add_argument("--use_attention_masks", action="store_true",
                        help="Use slot attention masks instead of decoder masks")
    parser.add_argument("--use_crf", action="store_true",
                        help="Refine slot masks with dense CRF using original images")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.use_crf and not HAS_CRF:
        logger.error("pydensecrf not installed. Install with: pip install pydensecrf")
        return

    root = args.cityscapes_root
    feat_dir = os.path.join(root, args.feature_subdir, args.split)
    sem_dir = os.path.join(root, args.semantic_subdir, args.split)
    img_dir = os.path.join(root, "leftImg8bit", args.split)
    out_dir = os.path.join(root, args.output_dir, args.split)

    device = torch.device(args.device)

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Discover images
    image_list = []
    for city in sorted(os.listdir(sem_dir)):
        city_sem = os.path.join(sem_dir, city)
        if not os.path.isdir(city_sem):
            continue
        for fname in sorted(os.listdir(city_sem)):
            if not fname.endswith(".png"):
                continue
            stem = fname.replace(".png", "")
            sem_path = os.path.join(city_sem, fname)
            feat_path = os.path.join(feat_dir, city, stem + ".npy")
            if not os.path.exists(feat_path):
                # Try with _leftImg8bit suffix
                feat_path = os.path.join(feat_dir, city, stem + "_leftImg8bit.npy")
                if not os.path.exists(feat_path):
                    continue
            # Find original image for CRF
            img_path = None
            if args.use_crf:
                img_path = os.path.join(img_dir, city, stem + "_leftImg8bit.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(img_dir, city, stem + ".png")
                    if not os.path.exists(img_path):
                        img_path = None
            image_list.append((city, stem, feat_path, sem_path, img_path))

    if args.limit:
        image_list = image_list[:args.limit]

    logger.info(f"Processing {len(image_list)} images (CRF={'ON' if args.use_crf else 'OFF'})")

    os.makedirs(out_dir, exist_ok=True)
    total_instances = 0
    t0 = time.time()

    for city, stem, feat_path, sem_path, img_path in tqdm(image_list, desc="DINOSAUR instances"):
        masks, scores, boxes = process_single_image(
            model, feat_path, sem_path, device,
            cause27=args.cause27, min_area=args.min_area,
            use_decoder_masks=not args.use_attention_masks,
            use_crf=args.use_crf, image_path=img_path,
        )

        # Save NPZ
        city_out = os.path.join(out_dir, city)
        os.makedirs(city_out, exist_ok=True)
        np.savez_compressed(
            os.path.join(city_out, stem + ".npz"),
            masks=masks, scores=scores, boxes=boxes,
            num_valid=len(masks),
        )
        total_instances += len(masks)

        # Optional visualization
        if args.visualize and len(masks) > 0:
            vis = np.zeros((WORK_H, WORK_W, 3), dtype=np.uint8)
            rng = np.random.RandomState(42)
            colors = rng.randint(50, 255, size=(len(masks), 3))
            for i in range(len(masks)):
                vis[masks[i]] = colors[i]
            vis_img = Image.fromarray(vis)
            vis_img.save(os.path.join(city_out, stem + "_vis.png"))

    elapsed = time.time() - t0
    n = len(image_list)
    avg = total_instances / max(n, 1)
    logger.info(f"Done. {total_instances} instances from {n} images "
                f"({avg:.1f} avg/img, {elapsed:.1f}s)")

    # Save stats
    stats = {
        "split": args.split,
        "num_images": n,
        "total_instances": total_instances,
        "avg_instances_per_image": round(avg, 2),
        "checkpoint": args.checkpoint,
        "num_slots": config["num_slots"],
        "min_area": args.min_area,
        "use_crf": args.use_crf,
    }
    stats_path = os.path.join(root, args.output_dir, f"stats_{args.split}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
