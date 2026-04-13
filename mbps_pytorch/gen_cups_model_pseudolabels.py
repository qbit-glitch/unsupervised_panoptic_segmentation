"""Generate pseudo-labels from the official CUPS model checkpoint.

Runs the trained CUPS model (Stage-3) on all Cityscapes training images
and saves predictions as pseudo-labels in CUPS format. These labels serve
as the control experiment: training DINOv3 on CUPS's own predictions
isolates whether improvement comes from backbone or pseudo-label quality.

Usage:
    python mbps_pytorch/gen_cups_model_pseudolabels.py \
        --checkpoint weights/cups.ckpt \
        --data_root /path/to/cityscapes \
        --output_dir /path/to/cityscapes/cups_pseudo_labels_official \
        --device cpu
"""

import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

# Add CUPS to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_CUPS_ROOT = os.path.join(_PROJECT_ROOT, "refs", "cups")
if _CUPS_ROOT not in sys.path:
    sys.path.insert(0, _CUPS_ROOT)

from cups.model import (
    panoptic_cascade_mask_r_cnn_from_checkpoint,
    prediction_to_standard_format,
)
from cups.thingstuff_split import ThingStuffSplitter

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CUPS model pseudo-labels")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "weights", "cups.ckpt"),
        help="Path to cups.ckpt",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/Users/qbit-glitch/Desktop/datasets/cityscapes",
        help="Cityscapes root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_official",
        help="Output directory for pseudo-labels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="CUPS model confidence threshold",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have pseudo-labels",
    )
    return parser.parse_args()


def get_image_paths(data_root: str, split: str) -> List[str]:
    """Get all image paths from Cityscapes leftImg8bit split."""
    pattern = os.path.join(data_root, "leftImg8bit", split, "*", "*_leftImg8bit.png")
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No images found at {pattern}. Check data_root={data_root}"
        )
    logger.info("Found %d images in %s split", len(paths), split)
    return paths


def image_name_from_path(path: str) -> str:
    """Extract image name: e.g. 'aachen_000000_000019_leftImg8bit'."""
    return os.path.splitext(os.path.basename(path))[0]


@torch.no_grad()
def generate_pseudolabels(args: argparse.Namespace) -> None:
    """Generate pseudo-labels from CUPS model inference."""
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading CUPS model from %s on %s...", args.checkpoint, args.device)
    model, num_things, num_stuffs = panoptic_cascade_mask_r_cnn_from_checkpoint(
        path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    model = model.to(args.device)
    model.eval()
    logger.info(
        "Model loaded: %d thing classes, %d stuff classes", num_things, num_stuffs
    )

    # Stuff and thing class indices (raw pseudo-class IDs)
    stuff_classes = tuple(range(num_stuffs))
    thing_classes = tuple(i + num_stuffs for i in range(num_things))
    total_classes = num_stuffs + num_things
    logger.info(
        "Stuff classes: 0-%d, Thing classes: %d-%d, Total: %d",
        num_stuffs - 1,
        num_stuffs,
        num_stuffs + num_things - 1,
        total_classes,
    )

    # ThingStuffSplitter for class distribution stats
    thingstuff_split = ThingStuffSplitter(num_classes_all=total_classes)

    # Get image paths
    image_paths = get_image_paths(args.data_root, args.split)

    skipped = 0
    failed = 0
    processed = 0

    for img_path in tqdm(image_paths, desc="Generating pseudo-labels"):
        img_name = image_name_from_path(img_path)
        sem_path = os.path.join(args.output_dir, f"{img_name}_semantic.png")
        inst_path = os.path.join(args.output_dir, f"{img_name}_instance.png")

        # Skip if already exists (resume mode)
        if args.resume and os.path.isfile(sem_path) and os.path.isfile(inst_path):
            # Still update thingstuff stats from existing files
            sem_pseudo = torch.from_numpy(
                np.array(Image.open(sem_path), dtype=np.int64)
            )
            inst_pseudo = torch.from_numpy(
                np.array(Image.open(inst_path), dtype=np.int64)
            )
            panoptic = torch.stack([sem_pseudo, inst_pseudo], dim=-1)
            ignore_mask = panoptic[..., 0] == 255
            panoptic[ignore_mask, 0] = 0
            panoptic[ignore_mask, 1] = 0
            thingstuff_split.update(panoptic)
            skipped += 1
            continue

        try:
            # Load image as float [0, 1] (CUPS convention)
            image = torch.from_numpy(
                np.array(Image.open(img_path), dtype=np.float32)
            )
            # HWC -> CHW
            image = image.permute(2, 0, 1) / 255.0
            image = image.to(args.device)

            # Run inference
            prediction = model([{"image": image}])

            # Convert to standard format: [H, W, 2] (semantic, instance)
            panoptic: Tensor = prediction_to_standard_format(
                prediction[0]["panoptic_seg"],
                stuff_classes=stuff_classes,
                thing_classes=thing_classes,
            )

            # Move to CPU for saving
            panoptic = panoptic.cpu()

            # Update thing/stuff statistics (mask out 255 ignore label)
            panoptic_for_stats = panoptic.clone()
            ignore_mask = panoptic_for_stats[..., 0] == 255
            panoptic_for_stats[ignore_mask, 0] = 0
            panoptic_for_stats[ignore_mask, 1] = 0
            thingstuff_split.update(panoptic_for_stats)

            # Save semantic and instance as uint8 PNGs
            semantic = panoptic[..., 0].numpy().astype(np.uint8)
            instance = panoptic[..., 1].numpy().astype(np.uint8)

            Image.fromarray(semantic).save(sem_path)
            Image.fromarray(instance).save(inst_path)
            processed += 1

        except Exception as e:
            logger.warning("Failed for %s: %s", img_name, e)
            failed += 1
            # Save empty labels for failed images
            h, w = 1024, 2048  # Cityscapes default
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(sem_path)
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(inst_path)

        # Log progress every 100 images
        if (processed + skipped + failed) % 100 == 0:
            logger.info(
                "Progress: %d processed, %d skipped, %d failed",
                processed,
                skipped,
                failed,
            )

    # Save thing/stuff split statistics
    instances_pixel, instances_mask, class_dist = thingstuff_split.compute()
    save_data = {
        "distribution all pixels": class_dist,
        "distribution inside object proposals": instances_pixel,
        "distribution per object proposal": instances_mask,
    }
    split_path = os.path.join(args.output_dir, "pseudo_classes_split_1.pt")
    torch.save(save_data, split_path)
    logger.info("Saved thing/stuff split statistics to %s", split_path)

    logger.info(
        "Done: %d processed, %d skipped, %d failed out of %d total",
        processed,
        skipped,
        failed,
        len(image_paths),
    )

    # Print class distribution summary
    logger.info("Class distribution (top 10):")
    sorted_classes = class_dist.argsort(descending=True)
    for i in range(min(10, len(sorted_classes))):
        cls_id = sorted_classes[i].item()
        count = class_dist[cls_id].item()
        pct = 100.0 * count / class_dist.sum().item() if class_dist.sum() > 0 else 0
        is_thing = "thing" if cls_id >= num_stuffs else "stuff"
        logger.info("  Class %2d (%s): %12d pixels (%.1f%%)", cls_id, is_thing, count, pct)


if __name__ == "__main__":
    args = parse_args()
    generate_pseudolabels(args)
