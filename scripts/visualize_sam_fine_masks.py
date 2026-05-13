#!/usr/bin/env python3
"""Quick visualizer for SAM fine-mask .npz outputs.

Usage:
    python scripts/visualize_sam_fine_masks.py \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --masks_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/sam_fine_masks_sam2/val \
        --split val \
        --n_images 15 \
        --out_dir /tmp/sam_vis
"""
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Class name lookup (matches generate_sam_fine_masks.py FINE_CLASS_NAMES order)
_IDX_TO_NAME = {
    0: "person", 1: "bicycle", 2: "motorcycle", 3: "rider",
    4: "traffic sign", 5: "traffic light", 6: "truck", 7: "bus",
    8: "train", 9: "guard rail", 10: "caravan", 11: "trailer",
    12: "car",
    13: "pole",
}


def overlay_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay all masks on the image with random distinct colors."""
    rng = np.random.default_rng(seed=0)
    overlay = image.copy().astype(np.float32)
    for mask in masks:
        color = rng.integers(50, 255, size=3).astype(np.float32)
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--masks_dir", required=True,
                        help="Split-level directory, e.g. sam_fine_masks_sam2/val/")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--n_images", type=int, default=15)
    parser.add_argument("--out_dir", default="/tmp/sam_vis")
    args = parser.parse_args()

    masks_root = Path(args.masks_dir)
    img_root = Path(args.cityscapes_root) / "leftImg8bit" / args.split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(masks_root.rglob("*_fine_masks.npz"))[: args.n_images]
    if not npz_files:
        print(f"No .npz files found under {masks_root}")
        return

    for npz_path in npz_files:
        # Derive original image path
        city = npz_path.parent.name
        stem = npz_path.stem.replace("_fine_masks", "")
        img_path = img_root / city / f"{stem}_leftImg8bit.png"

        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        data = np.load(str(npz_path))
        masks = data["masks"]           # (N, H, W) bool
        areas = data["areas"]           # (N,) int
        ious = data["iou_scores"]       # (N,) float
        cls_labels = (data["class_labels"] if "class_labels" in data
                      else np.full(len(masks), -1, dtype=np.int32))

        n_masks = masks.shape[0]

        # Resize masks to match original image if they differ
        if n_masks > 0 and masks.shape[1:] != image.shape[:2]:
            from PIL import Image as PILImage
            h, w = image.shape[:2]
            resized = np.stack([
                np.array(PILImage.fromarray(m.astype(np.uint8) * 255).resize(
                    (w, h), PILImage.NEAREST)) > 127
                for m in masks
            ])
            masks = resized

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        axes[0].imshow(image)
        axes[0].set_title(f"{stem}\n({n_masks} fine masks)", fontsize=9)
        axes[0].axis("off")

        if n_masks > 0:
            vis = overlay_masks(image, masks)
            axes[1].imshow(vis)
            # Annotate each mask centroid: class name (if SAM3), IoU score, area
            for mask, iou, area, cls_idx in zip(masks, ious, areas, cls_labels):
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    cy, cx = int(ys.mean()), int(xs.mean())
                    cls_name = _IDX_TO_NAME.get(int(cls_idx), "")
                    label = f"{cls_name}\n{iou:.2f}  {area}px" if cls_name else f"{iou:.2f}\n{area}px"
                    axes[1].text(
                        cx, cy, label,
                        color="white", fontsize=6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5),
                    )
        else:
            axes[1].imshow(image)
            axes[1].set_title("(no fine masks found)")
        axes[1].set_title(
            f"SAM fine masks overlay\nN={n_masks}, area range: "
            f"{int(areas.min()) if n_masks else 0}–{int(areas.max()) if n_masks else 0} px",
            fontsize=9,
        )
        axes[1].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"{stem}.png"
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}  ({n_masks} masks, IoU range {ious.min():.2f}–{ious.max():.2f})"
              if n_masks else f"  Saved: {out_path}  (0 masks)")

    print(f"\nDone. View output in: {out_dir}/")


if __name__ == "__main__":
    main()
