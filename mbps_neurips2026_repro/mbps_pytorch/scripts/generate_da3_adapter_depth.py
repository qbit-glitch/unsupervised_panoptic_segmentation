"""Generate depth NPYs using adapted DA3 (or DA2/DepthPro) model.

Usage:
    python scripts/generate_da3_adapter_depth.py \
        --checkpoint ./checkpoints/da3_dora_adapter/best_val.pt \
        --model_type dav3 \
        --image_dir /path/to/cityscapes/leftImg8bit/val \
        --output_dir /path/to/cityscapes/depth_da3_dora_adapter/val \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.models.adapters import (
    inject_lora_into_depth_model,
    freeze_non_adapter_params,
)
from mbps_pytorch.train_depth_adapter_lora import load_dav3_model, load_da2_model, load_depthpro_model

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_adapted_depth_npy(
    checkpoint_path: str,
    model_type: str,
    image_dir: str,
    output_dir: str,
    device: torch.device,
    image_size: tuple = (512, 1024),
    variant_override: str | None = None,
    rank_override: int | None = None,
    alpha_override: float | None = None,
    late_block_start_override: int | None = None,
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg")))
    if not image_files:
        logger.error("No images found in %s", image_dir)
        return
    logger.info("Found %d images", len(image_files))

    logger.info("Loading %s model...", model_type)
    if model_type == "dav3":
        model = load_dav3_model(device=str(device))
    elif model_type == "da2":
        model = load_da2_model(device=str(device))
    elif model_type == "depthpro":
        model = load_depthpro_model(device=str(device))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adapter_config = ckpt.get("adapter_config", {})
    logger.info("Checkpoint adapter_config: %s", adapter_config)

    variant = variant_override if variant_override is not None else adapter_config.get("variant", "dora")
    rank = rank_override if rank_override is not None else adapter_config.get("rank", 4)
    alpha = alpha_override if alpha_override is not None else adapter_config.get("alpha", 4.0)
    late_block_start = late_block_start_override if late_block_start_override is not None else adapter_config.get("late_block_start", 6)
    adapt_decoder = adapter_config.get("adapt_decoder", False)

    if model_type == "depthpro":
        from mbps_pytorch.models.adapters import inject_lora_into_depthpro
        inject_lora_into_depthpro(
            model, variant=variant, rank=rank, alpha=alpha,
            late_block_start=late_block_start,
            adapt_patch_encoder=True, adapt_image_encoder=True, adapt_fov_encoder=False,
        )
    else:
        inject_lora_into_depth_model(
            model, variant=variant, rank=rank, alpha=alpha,
            late_block_start=late_block_start,
            adapt_decoder=adapt_decoder,
        )
    freeze_non_adapter_params(model)

    # Validate adapter keys
    model_adapter_keys = {k for k in model.state_dict().keys() if any(x in k for x in ("lora_A", "lora_B", "lora_magnitude", "dwconv", "conv_gate"))}
    ckpt_adapter_keys = {k for k in ckpt["model"].keys() if any(x in k for x in ("lora_A", "lora_B", "lora_magnitude", "dwconv", "conv_gate"))}
    missing_adapters = model_adapter_keys - ckpt_adapter_keys
    if missing_adapters:
        raise RuntimeError(f"Adapter checkpoint missing keys: {sorted(missing_adapters)[:10]}")

    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()
    logger.info("Model loaded with adapted depth model.")

    for img_path in tqdm(image_files, desc="Generating depth NPYs"):
        rel_path = img_path.relative_to(image_dir)
        out_path = output_dir / rel_path.with_suffix(".npy")
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB").resize((image_size[1], image_size[0]), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.inference_mode():
            if model_type == "dav3":
                pred = model.inference_batch(img_tensor)
                depth = pred.cpu().numpy().squeeze()
            else:
                inputs = model.processor(images=[img], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                pred = model(**inputs).predicted_depth
                depth = pred.cpu().numpy().squeeze()

        # Normalize depth to [0, 1] for consistent gradient thresholding in evaluator
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)
        np.save(str(out_path), depth_norm.astype(np.float32))

    logger.info("Depth NPYs saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate adapted depth NPYs")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["dav3", "da2", "depthpro"])
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--late_block_start", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    generate_adapted_depth_npy(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=device,
        image_size=tuple(args.image_size),
        variant_override=args.variant,
        rank_override=args.rank,
        alpha_override=args.alpha,
        late_block_start_override=args.late_block_start,
    )


if __name__ == "__main__":
    main()
