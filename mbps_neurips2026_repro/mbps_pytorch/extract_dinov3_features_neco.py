#!/usr/bin/env python3
"""Extract DINOv3 patch features using a NeCo fine-tuned backbone.

Loads a checkpoint from train_neco_dinov3.py, replaces the backbone in the
standard DINOv3 feature extractor, and re-extracts features for COCO val2017.

The output format is identical to the standard dinov3_features/ files
(N, D) .npy arrays), saved to dinov3_neco_features/val2017/ so that mmgd_cut.py
can use them as a drop-in replacement by adding "dinov3_neco" to FEATURE_SOURCES.

Usage:
    python extract_dinov3_features_neco.py \
        --coco_root /path/to/coco \
        --checkpoint checkpoints/neco_dinov3/neco_best.pth \
        --device cuda

    # Smoke test (50 images)
    python extract_dinov3_features_neco.py \
        --coco_root /path/to/coco \
        --checkpoint checkpoints/neco_dinov3/neco_best.pth \
        --n_images 50 --device mps
"""

import argparse
import gc
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_neco_backbone(checkpoint_path: str, device: str):
    """Load backbone from a NeCo checkpoint.

    Returns (model, embed_dim, patch_size).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Build model (same architecture as training)
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("facebook/dinov2-base", trust_remote_code=True)
        embed_dim = 768
        patch_size = 14
        logger.info("Loaded DINOv2-base architecture (HuggingFace)")
    except Exception:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        embed_dim = 768
        patch_size = 14
        logger.info("Loaded DINOv2-vitb14-reg architecture (torch.hub)")

    # Load fine-tuned weights
    state = ckpt.get("backbone_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))
    logger.info(
        "Loaded NeCo checkpoint from %s (epoch %s, loss %.4f)",
        checkpoint_path,
        ckpt.get("epoch", "?"),
        ckpt.get("loss", float("nan")),
    )

    model = model.to(device).eval()
    return model, embed_dim, patch_size


@torch.no_grad()
def extract_features_for_image(
    model,
    img_path: Path,
    img_size: int,
    patch_size: int,
    device: str,
) -> Optional[np.ndarray]:
    """Extract (N, D) patch features for a single image.

    Returns None if extraction fails.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.warning("Could not open %s: %s", img_path, e)
        return None

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img_t = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    h_patches = img_size // patch_size
    w_patches = img_size // patch_size

    try:
        # HuggingFace API
        out = model(pixel_values=img_t)
        tokens = out.last_hidden_state  # (1, 1+regs+N, D)
        n_patch = h_patches * w_patches
        patch_tokens = tokens[0, -n_patch:, :].cpu().float().numpy()  # (N, D)
    except Exception:
        # torch.hub DINOv2 API
        feats = model.forward_features(img_t)
        patch_tokens = feats["x_norm_patchtokens"][0].cpu().float().numpy()  # (N, D)

    return patch_tokens


def extract_neco_features(
    coco_root: str,
    checkpoint_path: str,
    output_subdir: str = "dinov3_neco_features",
    img_size: int = 512,
    device: str = "cuda",
    n_images: Optional[int] = None,
) -> None:
    """Extract and cache NeCo-enhanced DINOv3 features for COCO val2017.

    Args:
        coco_root:      COCO dataset root.
        checkpoint_path: Path to NeCo checkpoint (from train_neco_dinov3.py).
        output_subdir:  Subdirectory under coco_root for output .npy files.
        img_size:       Input resolution for feature extraction.
        device:         Compute device.
        n_images:       Limit number of images (for testing).
    """
    coco_p = Path(coco_root)
    img_dir = coco_p / "val2017"
    out_dir = coco_p / output_subdir / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    if n_images:
        img_files = img_files[:n_images]

    existing = {f.stem for f in out_dir.glob("*.npy")}
    todo = [f for f in img_files if f.stem not in existing]
    logger.info(
        "NeCo feature extraction: %d total, %d cached, %d to extract",
        len(img_files), len(existing), len(todo),
    )

    if not todo:
        logger.info("All features already extracted.")
        return

    model, embed_dim, patch_size = load_neco_backbone(checkpoint_path, device)

    h_patches = img_size // patch_size
    w_patches = img_size // patch_size
    n_patches = h_patches * w_patches
    logger.info(
        "Grid: %d×%d = %d patches, dim=%d",
        h_patches, w_patches, n_patches, embed_dim,
    )

    for i, img_path in enumerate(tqdm(todo, desc="Extracting NeCo features")):
        feats = extract_features_for_image(model, img_path, img_size, patch_size, device)
        if feats is None:
            continue

        np.save(out_dir / f"{img_path.stem}.npy", feats.astype(np.float16))

        if device == "mps" and (i + 1) % 100 == 0:
            torch.mps.empty_cache()
            gc.collect()

    logger.info("Extraction complete. Saved to %s", out_dir)

    # Save metadata
    meta = {
        "checkpoint": checkpoint_path,
        "img_size": img_size,
        "patch_size": patch_size,
        "h_patches": h_patches,
        "w_patches": w_patches,
        "n_patches": n_patches,
        "embed_dim": embed_dim,
        "output_subdir": output_subdir,
    }
    import json
    with open(out_dir.parent / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract DINOv3 features using NeCo fine-tuned backbone"
    )
    p.add_argument("--coco_root", required=True)
    p.add_argument("--checkpoint", required=True, help="NeCo checkpoint path")
    p.add_argument("--output_subdir", default="dinov3_neco_features",
                   help="Output subdirectory under coco_root")
    p.add_argument("--img_size", type=int, default=512,
                   help="Input image size for extraction")
    p.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n_images", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extract_neco_features(
        coco_root=args.coco_root,
        checkpoint_path=args.checkpoint,
        output_subdir=args.output_subdir,
        img_size=args.img_size,
        device=args.device,
        n_images=args.n_images,
    )


if __name__ == "__main__":
    main()
