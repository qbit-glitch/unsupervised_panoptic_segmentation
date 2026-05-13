"""Generate DepthG semantic pseudo-labels with optional DepthPro depth guidance.

Stereo-free alternative to CUPS gen_pseudo_labels.py. Uses precomputed monocular
depth maps (DepthPro) instead of stereo disparity for the depth-guided sliding
window inference. Also supports running without any depth guidance as a baseline.

Two modes:
  --depth_mode none     : Equal-weight blend of sliding window + single-image
  --depth_mode depthpro : DepthPro depth-weighted blend (near=global, far=local)

Output: city-nested semantic PNGs compatible with evaluate_cascade_pseudolabels.py
and evaluate_semantic_pseudolabels.py.

Usage:
    python mbps_pytorch/generate_depthg_semantic_pseudolabels.py \
        --depthg_ckpt weights/depthg.ckpt \
        --cityscapes_root /path/to/cityscapes \
        --split val \
        --output_dir /path/to/cityscapes/depthg_semantic_depthpro \
        --depth_mode depthpro \
        --device cuda:0
"""

import argparse
import logging
import os
import pathlib
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: DepthG/src MUST come before script dir to prevent
# mbps_pytorch/data/ from shadowing external/depthg/src/data/
# (copied from gen_cups_pseudo_labels_remote.py:38-51)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_CUPS_ROOT = os.path.join(_PROJECT_ROOT, "refs", "cups")
_DEPTHG_ROOT = os.path.join(_CUPS_ROOT, "external", "depthg")
_DEPTHG_SRC = os.path.join(_DEPTHG_ROOT, "src")
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
for _p in [_DEPTHG_SRC, _DEPTHG_ROOT, _CUPS_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(_SCRIPT_DIR)

from cups.crf import dense_crf  # noqa: E402
from cups.semantics.model import DepthG  # noqa: E402
from cups.utils import normalize  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ImageNet normalization (same as cups.utils.normalize)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DepthG semantic pseudo-labels (stereo-free)",
    )
    parser.add_argument(
        "--depthg_ckpt", type=str, default="weights/depthg.ckpt",
        help="Path to DepthG checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--cityscapes_root", type=str, required=True,
        help="Cityscapes dataset root (contains leftImg8bit/, gtFine/)",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val"],
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for semantic PNGs (city-nested)",
    )
    parser.add_argument(
        "--depth_mode", type=str, default="depthpro",
        choices=["none", "depthpro"],
        help="Depth source: none (uniform blend) or depthpro (depth-weighted)",
    )
    parser.add_argument(
        "--depth_dir", type=str, default=None,
        help="Path to precomputed depth .npy files (default: {cityscapes_root}/depth_depthpro)",
    )
    parser.add_argument(
        "--depth_scale", type=float, default=50.0,
        help="Scale factor for depth->weight: weight = 1/(depth*scale+1). "
             "50.0 maps [0,1] DepthPro range to near=1.0, far=0.02",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Crops per batch in sliding window (4 for 11GB, 16 for 48GB)",
    )
    parser.add_argument(
        "--no_crf", action="store_true",
        help="Disable DenseCRF post-processing",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing PNGs")
    parser.add_argument(
        "--img_h", type=int, default=640,
        help="DepthG input height (default: 640, CUPS standard)",
    )
    parser.add_argument(
        "--img_w", type=int, default=1280,
        help="DepthG input width (default: 1280, CUPS standard)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
def discover_images(
    cityscapes_root: str, split: str,
) -> List[Tuple[str, str, str]]:
    """Find Cityscapes images and return (img_path, city, stem) tuples.

    stem example: 'frankfurt_000000_000294_leftImg8bit'
    """
    img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    results = []
    for city in sorted(os.listdir(img_dir)):
        city_dir = os.path.join(img_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if not fname.endswith(".png"):
                continue
            stem = fname.replace(".png", "")
            results.append((os.path.join(city_dir, fname), city, stem))

    logger.info("Found %d images in %s/%s", len(results), img_dir, split)
    return results


# ---------------------------------------------------------------------------
# Depth loading
# ---------------------------------------------------------------------------
def load_depth_weight(
    depth_dir: str,
    city: str,
    stem: str,
    target_h: int,
    target_w: int,
    depth_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Load DepthPro depth map and convert to CUPS-compatible depth_weight.

    DepthPro maps: [0,1] float32, 0=near, 1=far.
    CUPS formula: depth_weight = 1/(metric_depth + 1).
    Conversion: depth_weight = 1/(depth_npy * depth_scale + 1).

    Returns:
        depth_weight: (1, 1, H, W) tensor on device.
    """
    # depth .npy files use stem WITHOUT '_leftImg8bit' suffix
    depth_stem = stem.replace("_leftImg8bit", "")
    npy_path = os.path.join(depth_dir, city, f"{depth_stem}.npy")

    if not os.path.isfile(npy_path):
        logger.warning("Depth map not found: %s, using uniform weight", npy_path)
        return torch.full((1, 1, target_h, target_w), 0.5, device=device)

    depth = np.load(npy_path).astype(np.float32)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Upsample to target resolution if needed
    if depth_t.shape[-2] != target_h or depth_t.shape[-1] != target_w:
        depth_t = F.interpolate(
            depth_t, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )

    depth_weight = 1.0 / (depth_t * depth_scale + 1.0)
    return depth_weight.to(device)


# ---------------------------------------------------------------------------
# Memory-efficient DepthG inference
# (adapted from gen_cups_pseudo_labels_remote.py:266-312)
# ---------------------------------------------------------------------------
def _slide_segment_batched(
    depthg: DepthG, img: torch.Tensor, batch_size: int = 4,
) -> torch.Tensor:
    """Memory-efficient slide_segment: process crops in small batches."""
    from einops import rearrange

    unfolded = F.unfold(
        img, depthg.crop, stride=depthg.stride,
        padding=(depthg.bottom_pad, depthg.right_pad),
    )
    unfolded = rearrange(
        unfolded, "B (C H W) N -> (B N) C H W",
        H=depthg.crop[0], W=depthg.crop[1],
    )
    n_crops = unfolded.shape[0]
    all_logits = []
    for i in range(0, n_crops, batch_size):
        chunk = unfolded[i : i + batch_size]
        with torch.amp.autocast(chunk.device.type):
            _, code = depthg.model.net(chunk)
            code = F.interpolate(
                code, depthg.crop, mode="bilinear", align_corners=False,
            )
            logits = depthg.model.cluster_probe(code, 2, log_probs=True)
        all_logits.append(logits.float())

    crop_seg_logits = torch.cat(all_logits, dim=0)
    c = crop_seg_logits.size(1)
    crop_seg_logits = rearrange(
        crop_seg_logits, "(B N) C H W -> B (C H W) N", B=img.size(0),
    )
    preds = F.fold(
        crop_seg_logits, (img.size(-2), img.size(-1)),
        depthg.crop, stride=depthg.stride,
        padding=(depthg.bottom_pad, depthg.right_pad),
    )
    count_mat = F.fold(
        torch.ones(
            crop_seg_logits.size(0),
            crop_seg_logits.size(1) // c,
            crop_seg_logits.size(2),
            device=img.device,
        ),
        (img.size(-2), img.size(-1)),
        depthg.crop,
        stride=depthg.stride,
        padding=(depthg.bottom_pad, depthg.right_pad),
    )
    return preds / count_mat


def _depth_guided_sliding_window_lowmem(
    depthg: DepthG,
    img: torch.Tensor,
    depth_weight: torch.Tensor,
    batch_size: int = 4,
) -> torch.Tensor:
    """Memory-efficient depth-guided sliding window inference.

    Combines sliding-window prediction (good for distant/small objects) with
    single-image prediction (good for nearby/large objects) using depth_weight.

    depth_weight near 1.0 → prefer single-image (near objects)
    depth_weight near 0.0 → prefer sliding window (far objects)
    """
    out_slidingw = _slide_segment_batched(depthg, img, batch_size=batch_size)

    img_small = F.interpolate(
        img, (img.shape[-2] // 2, img.shape[-1] // 2),
        mode="bilinear", align_corners=False,
    ).float()

    with torch.amp.autocast(img_small.device.type):
        code = depthg.model.net(img_small)[-1]
        code2 = depthg.model.net(img_small.flip(dims=[3]))[-1]

    code = ((code + code2.flip(dims=[3])) / 2).float()
    code = F.interpolate(
        code, (img.shape[-2], img.shape[-1]),
        mode="bilinear", align_corners=False,
    )
    out_singleimg = depthg.model.cluster_probe(code, 2, log_probs=True)

    weight = depth_weight.expand_as(out_slidingw)
    return out_singleimg * weight + out_slidingw * (1 - weight)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    img_shape = np.array([args.img_h, args.img_w])
    use_crf = not args.no_crf

    # Auto-infer depth directory
    if args.depth_mode == "depthpro" and args.depth_dir is None:
        args.depth_dir = os.path.join(args.cityscapes_root, "depth_depthpro")

    logger.info("=== DepthG Semantic Pseudo-Label Generation ===")
    logger.info("Checkpoint: %s", args.depthg_ckpt)
    logger.info("Split: %s", args.split)
    logger.info("Depth mode: %s (scale=%.1f)", args.depth_mode, args.depth_scale)
    logger.info("CRF: %s", "enabled" if use_crf else "disabled")
    logger.info("Output: %s", args.output_dir)
    logger.info("Image shape: %dx%d", args.img_h, args.img_w)

    # Discover images
    images = discover_images(args.cityscapes_root, args.split)
    if not images:
        raise RuntimeError("No images found")

    # Load DepthG model
    # PyTorch 2.6+ defaults weights_only=True, but DepthG checkpoint uses
    # OmegaConf + Hydra internals. Monkey-patch torch.load for this load only.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(
        *a, **{**kw, "weights_only": False},
    )
    logger.info("Loading DepthG from %s ...", args.depthg_ckpt)
    model = DepthG(
        device=str(device),
        checkpoint_root=args.depthg_ckpt,
        img_shape=img_shape,
        stride=(int(img_shape[0] // 4), int(img_shape[0] // 4)),
        crop=(int(img_shape[0] // 2), int(img_shape[0] // 2)),
    )
    torch.load = _orig_torch_load  # Restore original
    n_clusters = model.model.cluster_probe.n_classes
    logger.info(
        "DepthG loaded: %d output clusters, device=%s",
        n_clusters, device,
    )
    if device.type == "cuda" and hasattr(torch.cuda, "memory_allocated"):
        logger.info(
            "GPU memory: %.0f MB",
            torch.cuda.memory_allocated(device) / 1e6,
        )

    # Create output directory
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Validate depth directory
    if args.depth_mode == "depthpro":
        depth_split_dir = os.path.join(args.depth_dir, args.split)
        if not os.path.isdir(depth_split_dir):
            # Try without split subdirectory
            if os.path.isdir(args.depth_dir):
                depth_split_dir = args.depth_dir
                logger.warning(
                    "No %s/ subdirectory in depth_dir, using %s directly",
                    args.split, args.depth_dir,
                )
            else:
                raise FileNotFoundError(
                    f"Depth directory not found: {depth_split_dir}"
                )
    else:
        depth_split_dir = None

    # Inference loop
    processed = 0
    skipped = 0
    failed = []
    t_start = time.time()

    for img_path, city, stem in tqdm(images, desc="DepthG inference"):
        # Output path: city-nested
        out_city_dir = os.path.join(args.output_dir, args.split, city)
        out_path = os.path.join(out_city_dir, f"{stem}.png")

        # Resume support
        if args.resume and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue

        try:
            # Load and preprocess image
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(pil_img).unsqueeze(0)  # (1,3,H,W)
            img_tensor = F.interpolate(
                img_tensor, size=(args.img_h, args.img_w),
                mode="bilinear", align_corners=False,
            ).to(device, torch.float32)

            # Normalize with ImageNet stats
            img_norm = normalize(img_tensor)

            # Load depth weight
            if args.depth_mode == "depthpro":
                depth_weight = load_depth_weight(
                    depth_split_dir, city, stem,
                    args.img_h, args.img_w,
                    args.depth_scale, device,
                )
            else:
                # No depth: uniform 0.5 blend
                depth_weight = torch.full(
                    (1, 1, args.img_h, args.img_w), 0.5, device=device,
                )

            # Run DepthG inference
            out = _depth_guided_sliding_window_lowmem(
                model, img_norm, depth_weight, batch_size=args.batch_size,
            )

            # CRF post-processing (CPU)
            if use_crf:
                crf_result = dense_crf(
                    img_norm[0].detach().cpu(),
                    out[0].detach().cpu(),
                )
                cluster_pred = (
                    torch.from_numpy(crf_result).unsqueeze(0).argmax(1).long()
                )
            else:
                cluster_pred = out.argmax(1).long().cpu()

            # Save as uint8 PNG
            label_np = cluster_pred.squeeze().numpy().astype(np.uint8)
            pathlib.Path(out_city_dir).mkdir(parents=True, exist_ok=True)
            Image.fromarray(label_np).save(out_path)
            processed += 1

        except Exception as e:
            failed.append(stem)
            tqdm.write(f"Failed: {stem}: {e}")

        # Free GPU memory periodically
        if processed % 20 == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        if processed > 0 and processed % 50 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed
            logger.info(
                "Progress: %d done, %d skipped, %d failed (%.1f img/s)",
                processed, skipped, len(failed), rate,
            )

    # Summary
    elapsed = time.time() - t_start
    logger.info("=== Done ===")
    logger.info(
        "Processed: %d, Skipped: %d, Failed: %d, Time: %.1fs (%.2f img/s)",
        processed, skipped, len(failed), elapsed,
        processed / max(elapsed, 1),
    )
    logger.info("Output: %s", args.output_dir)
    logger.info("Clusters: %d", n_clusters)

    if failed:
        logger.warning("Failed images (%d): %s", len(failed), failed[:10])

    # Print evaluation commands
    logger.info("\n--- Evaluation Commands ---")
    logger.info(
        "# 27-class CAUSE (CUPS-standard):\n"
        "python mbps_pytorch/evaluate_cascade_pseudolabels.py \\\n"
        "    --cityscapes_root %s --split %s \\\n"
        "    --semantic_subdir %s \\\n"
        "    --num_clusters %d --cause27 --skip_instance \\\n"
        "    --output eval_depthg_%s_%s.json",
        args.cityscapes_root, args.split,
        os.path.basename(args.output_dir),
        n_clusters if n_clusters > 27 else 0,
        args.depth_mode, args.split,
    )
    logger.info(
        "# 19-class standard:\n"
        "python mbps_pytorch/evaluate_semantic_pseudolabels.py \\\n"
        "    --pred_dir %s/%s --gt_dir %s/gtFine/%s \\\n"
        "    --remap_cause27 \\\n"
        "    --output eval_depthg_%s_%s_19cls.json",
        args.output_dir, args.split,
        args.cityscapes_root, args.split,
        args.depth_mode, args.split,
    )


if __name__ == "__main__":
    main()
