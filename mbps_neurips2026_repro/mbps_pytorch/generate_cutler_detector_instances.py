#!/usr/bin/env python3
"""Generate class-agnostic instance masks using CutLER or CuVLER pre-trained detectors.

CutLER (CVPR 2023) / CuVLER (CVPR 2024) train Cascade Mask R-CNN + DropLoss on
unsupervised pseudo-labels (MaskCut / VoteCut), then self-train. The resulting
detectors generalize to arbitrary images without any manual annotation.

This script runs the pre-trained detector on Cityscapes images and saves instance
masks as NPZ files compatible with the panoptic evaluation pipeline.

Usage:
    # Quick test (3 images, CutLER):
    python mbps_pytorch/generate_cutler_detector_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split val --limit 3 --visualize

    # Full val set with CuVLER:
    python mbps_pytorch/generate_cutler_detector_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --model cuvler --split val --score_thresh 0.35

    # Full train set:
    python mbps_pytorch/generate_cutler_detector_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --split train --score_thresh 0.35
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUTLER_DIR = os.path.join(SCRIPT_DIR, "..", "refs", "cutler", "cutler")
CUVLER_DIR = os.path.join(SCRIPT_DIR, "..", "refs", "cuvler")
CUVLER_CAD_DIR = os.path.join(CUVLER_DIR, "cad")


def setup_cutler_predictor(weights_path, score_thresh=0.35, max_detections=50, device="cpu", input_size=None):
    """Setup CutLER's Cascade Mask R-CNN predictor."""
    sys.path.insert(0, os.path.abspath(CUTLER_DIR))

    from modeling.roi_heads.custom_cascade_rcnn import CustomCascadeROIHeads
    from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
    try:
        ROI_HEADS_REGISTRY.register(CustomCascadeROIHeads)
    except Exception:
        pass  # Already registered

    from config import add_cutler_config
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    add_cutler_config(cfg)

    config_path = os.path.join(
        CUTLER_DIR, "model_zoo", "configs", "CutLER-ImageNet",
        "cascade_mask_rcnn_R_50_FPN_demo.yaml"
    )
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = os.path.abspath(weights_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.TEST.DETECTIONS_PER_IMAGE = max_detections
    if input_size is not None:
        cfg.INPUT.MIN_SIZE_TEST = input_size
        cfg.INPUT.MAX_SIZE_TEST = int(input_size * 2.5)

    predictor = DefaultPredictor(cfg)
    return predictor


def setup_cuvler_predictor(weights_path, score_thresh=0.35, max_detections=50, device="cpu", input_size=None):
    """Setup CuVLER's Cascade Mask R-CNN predictor."""
    # Add refs/cuvler/ to sys.path so `cad.` package imports resolve
    sys.path.insert(0, os.path.abspath(CUVLER_DIR))

    # Register CuVLER's CustomCascadeROIHeads with detectron2's registry
    from cad.modeling.roi_heads.custom_cascade_rcnn import CustomCascadeROIHeads
    from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
    try:
        ROI_HEADS_REGISTRY.register(CustomCascadeROIHeads)
    except Exception:
        pass  # Already registered

    # Register CustomMaskRCNNConvUpsampleHead (uses detectron2's registry directly)
    import cad.modeling.roi_heads.mask_head  # noqa: F401

    from cad.config import add_cuvler_config
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    add_cuvler_config(cfg)

    config_path = os.path.join(
        CUVLER_CAD_DIR, "model_zoo", "configs", "CutVER-ImageNet",
        "cascade_mask_rcnn_R_50_FPN_self_train.yaml"
    )
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = os.path.abspath(weights_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.TEST.DETECTIONS_PER_IMAGE = max_detections
    if input_size is not None:
        cfg.INPUT.MIN_SIZE_TEST = input_size
        cfg.INPUT.MAX_SIZE_TEST = int(input_size * 2.5)

    predictor = DefaultPredictor(cfg)
    return predictor


def get_cityscapes_images(cityscapes_root, split="val"):
    """Get list of Cityscapes image paths."""
    img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
    images = []
    for city in sorted(os.listdir(img_dir)):
        city_dir = os.path.join(img_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.endswith("_leftImg8bit.png"):
                images.append({
                    "path": os.path.join(city_dir, fname),
                    "city": city,
                    "stem": fname.replace("_leftImg8bit.png", ""),
                })
    return images


def visualize_instances(img_rgb, masks, scores, save_path):
    """Save overlay visualization of detected instances."""
    vis = img_rgb.copy().astype(float)
    n = len(masks)
    if n == 0:
        Image.fromarray(img_rgb).save(save_path)
        return

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(n, 3))

    for i in range(n):
        mask = masks[i]
        vis[mask] = vis[mask] * 0.5 + colors[i] * 0.5

    Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(save_path)


def main():
    parser = argparse.ArgumentParser("CutLER/CuVLER detector instance generation")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="cutler", choices=["cutler", "cuvler"],
                        help="Which detector to use")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto: pseudo_instances_{model})")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to checkpoint (auto-detected if not set)")
    parser.add_argument("--score_thresh", type=float, default=0.35,
                        help="Detection confidence threshold")
    parser.add_argument("--max_detections", type=int, default=50,
                        help="Maximum detections per image")
    parser.add_argument("--min_mask_area", type=int, default=100,
                        help="Minimum mask area in pixels")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu recommended for detectron2 on Mac)")
    parser.add_argument("--input_size", type=int, default=None,
                        help="Override test input size (shorter side). Default: 800")
    parser.add_argument("--visualize", action="store_true",
                        help="Save overlay visualizations")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N images")
    args = parser.parse_args()

    # Auto-detect output dir
    if args.output_dir is None:
        args.output_dir = f"pseudo_instances_{args.model}"

    # Auto-detect weights
    if args.weights is None:
        if args.model == "cutler":
            args.weights = os.path.join(
                SCRIPT_DIR, "..", "refs", "cutler", "weights",
                "cutler_cascade_final.pth"
            )
        else:  # cuvler
            args.weights = os.path.join(
                SCRIPT_DIR, "..", "refs", "cuvler", "weights",
                "cuvler_self_trained.pth"
            )
    if not os.path.exists(args.weights):
        print(f"ERROR: {args.model} weights not found at {args.weights}")
        sys.exit(1)

    # Setup predictor
    model_name = args.model.upper()
    print(f"Loading {model_name} Cascade Mask R-CNN (score_thresh={args.score_thresh})...")
    if args.model == "cutler":
        predictor = setup_cutler_predictor(
            args.weights, args.score_thresh, args.max_detections, args.device, args.input_size
        )
    else:
        predictor = setup_cuvler_predictor(
            args.weights, args.score_thresh, args.max_detections, args.device, args.input_size
        )
    print("Predictor ready.")

    # Get images
    images = get_cityscapes_images(args.cityscapes_root, args.split)
    if args.limit:
        images = images[:args.limit]
    print(f"Processing {len(images)} images from {args.split} split")

    # Create output dirs
    output_base = os.path.join(args.output_dir, args.split)
    os.makedirs(output_base, exist_ok=True)
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, "vis", args.split)
        os.makedirs(vis_dir, exist_ok=True)

    # Process
    stats = {
        "total_images": 0,
        "total_instances": 0,
        "instances_per_image": [],
        "time_per_image": [],
    }

    for img_info in tqdm(images, desc=f"{model_name} detector"):
        t0 = time.time()

        # Load image (detectron2 expects BGR)
        img_rgb = np.array(Image.open(img_info["path"]).convert("RGB"))
        img_bgr = img_rgb[:, :, ::-1]

        # Run detector
        outputs = predictor(img_bgr)
        instances = outputs["instances"]

        # Extract masks and scores
        n = len(instances)
        if n > 0:
            masks = instances.pred_masks.cpu().numpy()  # (N, H, W) bool
            scores = instances.scores.cpu().numpy()      # (N,) float
            boxes = instances.pred_boxes.tensor.cpu().numpy()  # (N, 4)

            # Filter by area
            areas = np.array([m.sum() for m in masks])
            keep = areas >= args.min_mask_area
            masks = masks[keep]
            scores = scores[keep]
            boxes = boxes[keep]
            n = len(masks)
        else:
            H, W = img_rgb.shape[:2]
            masks = np.zeros((0, H, W), dtype=bool)
            scores = np.array([], dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)

        # Save as NPZ (compatible with existing pipeline)
        city_dir = os.path.join(output_base, img_info["city"])
        os.makedirs(city_dir, exist_ok=True)
        npz_path = os.path.join(city_dir, f"{img_info['stem']}.npz")
        np.savez_compressed(
            npz_path,
            masks=masks,        # (N, H, W) bool
            scores=scores,      # (N,) float32
            boxes=boxes,        # (N, 4) float32
            num_valid=n,
        )

        # Visualization
        if args.visualize:
            vis_city_dir = os.path.join(vis_dir, img_info["city"])
            os.makedirs(vis_city_dir, exist_ok=True)
            vis_path = os.path.join(vis_city_dir, f"{img_info['stem']}_overlay.png")
            visualize_instances(img_rgb, masks, scores, vis_path)

        elapsed = time.time() - t0
        stats["total_images"] += 1
        stats["total_instances"] += n
        stats["instances_per_image"].append(n)
        stats["time_per_image"].append(elapsed)

        if stats["total_images"] <= 5 or stats["total_images"] % 100 == 0:
            avg_inst = np.mean(stats["instances_per_image"])
            tqdm.write(
                f"  {img_info['stem']}: {n} instances (scores: "
                f"{scores[:3].round(3).tolist() if n > 0 else []}), "
                f"{elapsed:.1f}s"
            )

    # Summary
    if stats["total_images"] > 0:
        inst_arr = np.array(stats["instances_per_image"])
        time_arr = np.array(stats["time_per_image"])
        print(f"\n{'='*60}")
        print(f"{model_name} Instance Generation Complete")
        print(f"{'='*60}")
        print(f"  Images processed:        {stats['total_images']}")
        print(f"  Total instances:         {stats['total_instances']}")
        print(f"  Avg instances/image:     {inst_arr.mean():.1f}")
        print(f"  Median instances/image:  {np.median(inst_arr):.0f}")
        print(f"  Min/Max instances:       {inst_arr.min()}/{inst_arr.max()}")
        print(f"  Avg time/image:          {time_arr.mean():.1f}s")
        print(f"  Output:                  {output_base}")

        # Save stats
        summary = {
            "config": {
                "model": args.model,
                "score_thresh": args.score_thresh,
                "max_detections": args.max_detections,
                "min_mask_area": args.min_mask_area,
                "device": args.device,
                "split": args.split,
            },
            "total_images": stats["total_images"],
            "total_instances": stats["total_instances"],
            "avg_instances_per_image": float(inst_arr.mean()),
            "median_instances_per_image": float(np.median(inst_arr)),
            "min_instances": int(inst_arr.min()),
            "max_instances": int(inst_arr.max()),
            "avg_time_per_image": float(time_arr.mean()),
        }
        stats_path = os.path.join(args.output_dir, f"stats_{args.split}.json")
        with open(stats_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Stats saved:             {stats_path}")


if __name__ == "__main__":
    main()
