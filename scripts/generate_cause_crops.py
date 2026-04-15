"""
Generate pre-cropped training data for CAUSE-TR.

The official CAUSE/STEGO pipeline uses CroppedDataset with crop_type="five"
and crop_ratio=0.5. This script creates 5 crops per image (4 corners + center)
at 50% scale, saving them as indexed files for CroppedDataset to load.

Output structure:
    {data_dir}/cityscapes/cropped/cityscapes_five_crop_0.5/
        img/train/{0,1,...,N}.jpg
        img/val/{0,1,...,N}.jpg
        label/train/{0,1,...,N}.png
        label/val/{0,1,...,N}.png

Label format: 1-based class IDs (CroppedDataset subtracts 1 at load time).
    Cityscapes gtFine IDs 7-33 are remapped to 1-27.
    Void / unlabeled → 0 (becomes -1 after subtraction).

Usage:
    python refs/cause/generate_crops.py \
        --data_dir /Users/qbit-glitch/Desktop/datasets
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets.cityscapes import Cityscapes
from tqdm import tqdm


FIRST_NONVOID = 7  # Cityscapes label IDs start at 7 for valid classes


def get_five_crops(
    img: Image.Image,
    label: Image.Image,
    crop_ratio: float = 0.5,
) -> list:
    """Generate 5 crops: 4 corners + center at given ratio."""
    W, H = img.size
    cw = int(W * crop_ratio)
    ch = int(H * crop_ratio)

    positions = [
        (0, 0),                      # top-left
        (W - cw, 0),                 # top-right
        (0, H - ch),                 # bottom-left
        (W - cw, H - ch),            # bottom-right
        ((W - cw) // 2, (H - ch) // 2),  # center
    ]

    crops = []
    for x, y in positions:
        box = (x, y, x + cw, y + ch)
        img_crop = img.crop(box)
        label_crop = label.crop(box)
        crops.append((img_crop, label_crop))
    return crops


def remap_cityscapes_label(label_img: Image.Image) -> Image.Image:
    """Remap Cityscapes label IDs to 1-based (1-27). Void=0."""
    arr = np.array(label_img, dtype=np.int32)
    remapped = np.zeros_like(arr, dtype=np.uint8)

    # Valid classes: IDs 7-33 → 1-27
    valid_mask = (arr >= FIRST_NONVOID) & (arr < FIRST_NONVOID + 27)
    remapped[valid_mask] = (arr[valid_mask] - FIRST_NONVOID + 1).astype(np.uint8)
    # Everything else stays 0 (void)

    return Image.fromarray(remapped)


def generate_split(
    data_dir: str,
    split: str,
    output_root: str,
    crop_ratio: float = 0.5,
) -> None:
    """Generate crops for one split (train/val)."""
    cityscapes_root = os.path.join(data_dir, "cityscapes")
    dataset = Cityscapes(
        cityscapes_root,
        split=split,
        mode="fine",
        target_type="semantic",
        transform=None,
        target_transform=None,
    )

    img_dir = os.path.join(output_root, "img", split)
    label_dir = os.path.join(output_root, "label", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    idx = 0
    for i in tqdm(range(len(dataset)), desc=f"Cropping {split}"):
        img, label = dataset[i]
        label = remap_cityscapes_label(label)
        crops = get_five_crops(img, label, crop_ratio)
        for img_crop, label_crop in crops:
            img_crop.save(os.path.join(img_dir, f"{idx}.jpg"), quality=95)
            label_crop.save(os.path.join(label_dir, f"{idx}.png"))
            idx += 1

    print(f"  {split}: {idx} crops saved to {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CAUSE/STEGO crops")
    parser.add_argument(
        "--data_dir",
        default="/Users/qbit-glitch/Desktop/datasets",
        type=str,
    )
    parser.add_argument("--crop_ratio", default=0.5, type=float)
    args = parser.parse_args()

    output_root = os.path.join(
        args.data_dir,
        "cityscapes",
        "cropped",
        f"cityscapes_five_crop_{args.crop_ratio}",
    )
    print(f"Output: {output_root}")

    generate_split(args.data_dir, "train", output_root, args.crop_ratio)
    generate_split(args.data_dir, "val", output_root, args.crop_ratio)
    print("Done.")


if __name__ == "__main__":
    main()
