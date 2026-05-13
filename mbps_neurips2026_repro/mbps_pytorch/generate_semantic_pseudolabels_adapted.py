#!/usr/bin/env python3
"""Generate semantic pseudo-labels using adapted DINOv2 + CAUSE-TR.

Loads a checkpoint from train_semantic_adapter.py and generates
semantic pseudo-labels via k-means clustering on adapted features.

Usage:
    python mbps_pytorch/generate_semantic_pseudolabels_adapted.py \\
        --checkpoint results/semantic_adapter_dora/best.pt \\
        --data_dir /data/cityscapes/leftImg8bit/train \\
        --output_dir /data/cityscapes/pseudo_semantic_adapted/train \\
        --num_clusters 54
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform

from mbps_pytorch.models.adapters import (
    inject_lora_into_dinov2,
    inject_lora_into_cause_tr,
    freeze_non_adapter_params,
    count_adapter_params,
    set_dinov2_spatial_dims,
)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_cause_args():
    from types import SimpleNamespace
    return SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )


def preprocess_image(image_path, device, resize_to_crop=False, crop_size=322):
    img = Image.open(image_path).convert("RGB")
    orig_size = (img.height, img.width)
    if resize_to_crop:
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    H, W = tensor.shape[-2:]
    # NOTE: Images are resized to the nearest multiple of patch_size (14).
    # For Cityscapes 1024x2048, this becomes 1022x2044 (2 rows + 4 cols discarded).
    # The cropped region is restored via bilinear interpolation during upsampling.
    # Bottom-right pixels may be slightly degraded. Consider padding instead.
    new_H = (H // 14) * 14
    new_W = (W // 14) * 14
    if new_H != H or new_W != W:
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)
    return tensor.to(device), orig_size


@torch.inference_mode()
def extract_features_single(backbone, segment, pixel_values, crop_size=322):
    _, _, H, W = pixel_values.shape
    h_patches = H // 14
    w_patches = W // 14

    # If image is exactly crop_size, process directly
    if H == crop_size and W == crop_size:
        set_dinov2_spatial_dims(backbone, h_patches=23, w_patches=23)
        feat = backbone(pixel_values)[:, 1:, :]
        seg_feat = segment.head(feat)
        code = transform(seg_feat)
        code_pooled = F.adaptive_avg_pool2d(code, (h_patches, w_patches))
        features = code_pooled.squeeze(0).permute(1, 2, 0).reshape(-1, 90)
        return features.float().cpu().numpy(), h_patches, w_patches

    # Sliding window with patch-aligned stride (multiple of 14)
    patch_stride = crop_size // 14 // 2  # half overlap in patch space
    stride_pixels = patch_stride * 14

    y_positions = list(range(0, H - crop_size + 1, stride_pixels))
    if not y_positions or y_positions[-1] + crop_size < H:
        y_positions.append(H - crop_size)
    y_positions = sorted(set(y_positions))

    x_positions = list(range(0, W - crop_size + 1, stride_pixels))
    if not x_positions or x_positions[-1] + crop_size < W:
        x_positions.append(W - crop_size)
    x_positions = sorted(set(x_positions))

    # NOTE: Non-uniform overlap: interior crops have exactly 50% overlap,
    # but boundary crops may overlap up to ~77% with their neighbor.
    # The visit-count accumulator ensures correct averaging regardless.
    code_sum = torch.zeros(90, h_patches, w_patches, device=pixel_values.device)
    count = torch.zeros(h_patches, w_patches, device=pixel_values.device)

    for y_pos in y_positions:
        for x_pos in x_positions:
            crop = pixel_values[:, :, y_pos:y_pos + crop_size, x_pos:x_pos + crop_size]
            ch, cw = crop.shape[2], crop.shape[3]
            if ch < crop_size or cw < crop_size:
                pad_h = crop_size - ch
                pad_w = crop_size - cw
                crop = F.pad(crop, (0, pad_w, 0, pad_h), mode='reflect')

            set_dinov2_spatial_dims(backbone, h_patches=23, w_patches=23)
            feat = backbone(crop)[:, 1:, :]
            seg_feat = segment.head(feat)
            code = transform(seg_feat)  # (1, 90, 23, 23)

            # Map crop to patch grid of full image
            y_patch = y_pos // 14
            x_patch = x_pos // 14
            ph = ch // 14
            pw = cw // 14

            code_resized = F.interpolate(code, size=(ph, pw), mode='bilinear', align_corners=False)[0]
            code_sum[:, y_patch:y_patch + ph, x_patch:x_patch + pw] += code_resized
            count[y_patch:y_patch + ph, x_patch:x_patch + pw] += 1

    code_avg = code_sum / count.clamp(min=1)
    features = code_avg.permute(1, 2, 0).reshape(-1, 90)
    return features.float().cpu().numpy(), h_patches, w_patches


def fit_kmeans(all_features, num_clusters=54, subsample=500_000, seed=42):
    n_total = all_features.shape[0]
    if subsample > 0 and n_total > subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=subsample, replace=False)
        fit_features = all_features[idx]
        logger.info("Subsampled %d/%d features for K-means", subsample, n_total)
    else:
        fit_features = all_features
    logger.info("Fitting MiniBatchKMeans with K=%d...", num_clusters)
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, batch_size=10000, max_iter=300,
        random_state=seed, n_init=3, verbose=1,
    )
    kmeans.fit(fit_features)
    logger.info("K-means inertia: %.2f", kmeans.inertia_)
    return kmeans


def _has_lora_keys(state_dict) -> bool:
    """Detect whether a checkpoint was saved from an adapter-wrapped model."""
    return any(
        k.endswith(("lora_A", "lora_B", "lora_magnitude"))
        or ".lora_" in k
        or "dwconv" in k
        or "conv_gate" in k
        for k in state_dict.keys()
    )


def _load_state_checked(module, state_dict, component_name, require_lora):
    """Load state_dict and fail loudly if LoRA keys were silently dropped."""
    result = module.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    dropped_lora = [
        k for k in unexpected
        if k.endswith(("lora_A", "lora_B", "lora_magnitude"))
        or ".lora_" in k
        or "dwconv" in k
        or "conv_gate" in k
    ]
    if dropped_lora:
        raise RuntimeError(
            f"[{component_name}] {len(dropped_lora)} LoRA/DoRA parameters were dropped because "
            f"the receiver has no adapter wrappers. First few: {dropped_lora[:5]}. "
            f"Call inject_lora_into_* BEFORE load_state_dict."
        )
    if missing:
        # Base weights missing is a hard error; LoRA keys missing in a checkpoint
        # without LoRA training is expected (then require_lora=False).
        base_missing = [k for k in missing if ".lora_" not in k]
        if base_missing:
            logger.warning("[%s] %d base params missing (expected 0). First few: %s",
                           component_name, len(base_missing), base_missing[:5])
    if result.missing_keys:
        logger.warning("[%s] Missing keys: %s", component_name, result.missing_keys[:10])
    if result.unexpected_keys:
        logger.warning("[%s] Unexpected keys: %s", component_name, result.unexpected_keys[:10])
    if require_lora:
        lora_loaded = sum(
            1
            for k in state_dict.keys()
            if k.endswith(("lora_A", "lora_B", "lora_magnitude"))
            or "dwconv" in k
            or "conv_gate" in k
        )
        logger.info("[%s] %d LoRA-style parameters loaded successfully.",
                    component_name, lora_loaded)
        # Verify all adapter keys in model are present in checkpoint
        model_adapter_keys = {k for k in module.state_dict().keys() if any(x in k for x in ("lora_A", "lora_B", "lora_magnitude", "dwconv", "conv_gate"))}
        ckpt_adapter_keys = {k for k in state_dict.keys() if any(x in k for x in ("lora_A", "lora_B", "lora_magnitude", "dwconv", "conv_gate"))}
        missing_adapters = model_adapter_keys - ckpt_adapter_keys
        if missing_adapters:
            raise RuntimeError(f"[{component_name}] Adapter checkpoint missing keys: {sorted(missing_adapters)[:10]}. "
                               f"Checkpoint may be from a non-adapted model. Expected {len(model_adapter_keys)} adapter params, found {len(ckpt_adapter_keys)}.")


def generate_adapted_pseudolabels(
    checkpoint_path, data_dir, output_dir, device,
    num_clusters=54, subsample=500_000, use_crf=False, seed=42,
    adapter_config=None, cause_checkpoint_dir=None,
    resize_to_crop=False,
):
    from types import SimpleNamespace
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    image_files = sorted(list(data_path.rglob("*.png")) + list(data_path.rglob("*.jpg")))
    if not image_files:
        logger.error("No images found in %s", data_dir)
        return
    logger.info("Found %d images", len(image_files))

    # Resolve CAUSE reference checkpoint directory
    if cause_checkpoint_dir is None:
        cause_checkpoint_dir = Path(__file__).resolve().parent.parent / "refs" / "cause"
    cause_checkpoint_dir = Path(cause_checkpoint_dir)

    # Load checkpoint
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Resolve adapter config: CLI > checkpoint metadata > auto-disable
    ckpt_config = ckpt.get("adapter_config", {}) if isinstance(ckpt, dict) else {}
    cfg = dict(ckpt_config)
    if adapter_config:
        cfg.update({k: v for k, v in adapter_config.items() if v is not None})

    ckpt_has_lora = _has_lora_keys(ckpt.get("backbone", {}))
    if ckpt_has_lora and not cfg:
        raise RuntimeError(
            "Checkpoint contains LoRA weights but no adapter_config was provided "
            "(neither in checkpoint metadata nor via CLI). Pass --variant/--rank/--alpha/"
            "--late_block_start matching the training run."
        )
    use_adapter = bool(cfg) and ckpt_has_lora
    logger.info("Adapter config: %s (use_adapter=%s)", cfg, use_adapter)

    # --- Build backbone -----------------------------------------------------
    backbone = dinov2_vit_base_14()
    state = torch.load(
        str(cause_checkpoint_dir / "checkpoint" / "dinov2_vit_base_14.pth"),
        map_location="cpu", weights_only=True,
    )
    result_backbone = backbone.load_state_dict(state, strict=False)
    if result_backbone.missing_keys:
        logger.warning("Backbone missing keys: %s", result_backbone.missing_keys[:10])
    if result_backbone.unexpected_keys:
        logger.warning("Backbone unexpected keys: %s", result_backbone.unexpected_keys[:10])

    if use_adapter:
        inject_lora_into_dinov2(
            backbone,
            variant=cfg.get("variant", "dora"),
            rank=cfg.get("rank", 4),
            alpha=cfg.get("alpha", 4.0),
            dropout=cfg.get("dropout", 0.0),  # no dropout at inference
            late_block_start=cfg.get("late_block_start", 6),
        )
        freeze_non_adapter_params(backbone)

    # --- Build segment + cluster from CAUSE baseline ------------------------
    cause_args = build_cause_args()
    segment = Segment_TR(cause_args)
    cluster = Cluster(cause_args)

    # Load CAUSE baseline weights so the non-adapter params are correct
    seg_path = cause_checkpoint_dir / "CAUSE" / "cityscapes" / \
        "dinov2_vit_base_14" / "2048" / "segment_tr.pth"
    if seg_path.exists():
        result_seg = segment.load_state_dict(
            torch.load(str(seg_path), map_location="cpu", weights_only=True),
            strict=False,
        )
        if result_seg.missing_keys:
            logger.warning("Segment missing keys: %s", result_seg.missing_keys[:10])
        if result_seg.unexpected_keys:
            logger.warning("Segment unexpected keys: %s", result_seg.unexpected_keys[:10])
        logger.info("Loaded CAUSE segment_tr baseline from %s", seg_path)
    else:
        logger.warning("CAUSE segment_tr.pth not found at %s — segment is randomly initialized",
                       seg_path)

    # --- Load modularity codebook (required by TRDecoder and Cluster) -------
    mod_path = cause_checkpoint_dir / "CAUSE" / "cityscapes" / "modularity" / \
        "dinov2_vit_base_14" / "2048" / "modular.npy"
    if not mod_path.exists():
        raise FileNotFoundError(
            f"Modularity codebook not found at {mod_path}. "
            f"TRDecoder requires this — without it, code->cluster projections are undefined."
        )
    cb = torch.from_numpy(np.load(str(mod_path)))
    cluster.codebook.data = cb.clone()
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb.clone()
    segment.head_ema.codebook = cb.clone()
    logger.info("Loaded CAUSE modularity codebook: shape=%s", tuple(cb.shape))

    if use_adapter and cfg.get("adapt_cause", False):
        inject_lora_into_cause_tr(
            segment,
            variant=cfg.get("variant", "dora"),
            rank=cfg.get("rank", 4),
            alpha=cfg.get("alpha", 4.0),
            dropout=cfg.get("dropout", 0.0),
            adapt_head=True, adapt_projection=False, adapt_ema=False,
        )
        freeze_non_adapter_params(segment)

    # --- Now load the trained adapter checkpoint with strict verification ---
    _load_state_checked(backbone, ckpt["backbone"], "backbone", require_lora=use_adapter)
    _load_state_checked(segment, ckpt["segment"], "segment",
                        require_lora=use_adapter and cfg.get("adapt_cause", False))
    _load_state_checked(cluster, ckpt["cluster"], "cluster", require_lora=False)

    # Re-pin codebook: some checkpoints may carry a stale/missing codebook
    segment.head.codebook = cb.clone().to(device)
    segment.head_ema.codebook = cb.clone().to(device)

    backbone = backbone.to(device).eval()
    segment = segment.to(device).eval()
    cluster = cluster.to(device).eval()

    if use_adapter:
        n_adapter = count_adapter_params(backbone) + count_adapter_params(segment)
        logger.info("Adapter params present after load: %d", n_adapter)
        if n_adapter == 0:
            raise RuntimeError(
                "use_adapter=True but 0 adapter params after injection+load. "
                "Something is wrong with the adapter config or checkpoint."
            )

    # Extract features
    all_features_list = []
    per_image_info = []
    for img_path in tqdm(image_files, desc="Extracting features"):
        pixel_values, orig_size = preprocess_image(str(img_path), device, resize_to_crop=resize_to_crop)
        features, h_p, w_p = extract_features_single(backbone, segment, pixel_values)
        all_features_list.append(features)
        per_image_info.append((h_p, w_p, features.shape[0]))

    all_features = np.concatenate(all_features_list, axis=0)
    logger.info("All features shape: %s (%.1f GB)", all_features.shape, all_features.nbytes / 1e9)

    # K-means
    kmeans = fit_kmeans(all_features, num_clusters=num_clusters, subsample=subsample, seed=seed)
    os.makedirs(output_path, exist_ok=True)
    np.savez(
        str(output_path / "kmeans_model.npz"),
        cluster_centers=kmeans.cluster_centers_,
        n_clusters=num_clusters,
        inertia=kmeans.inertia_,
    )

    # Predict and save
    offset = 0
    cluster_pixel_counts = np.zeros(num_clusters, dtype=np.int64)
    for img_path, (h_p, w_p, n_patches) in tqdm(
        zip(image_files, per_image_info), total=len(image_files), desc="Saving labels"
    ):
        feat = all_features[offset:offset + n_patches]
        labels = kmeans.predict(feat)
        offset += n_patches

        label_map = labels.reshape(h_p, w_p).astype(np.uint8)
        img = Image.open(img_path).convert("RGB")
        H, W = img.height, img.width
        label_map_up = np.array(Image.fromarray(label_map).resize((W, H), Image.NEAREST), dtype=np.int32)

        rel_path = img_path.relative_to(data_path)
        out_path = output_path / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_map_up.astype(np.uint8)).save(str(out_path))

        for c in range(num_clusters):
            cluster_pixel_counts[c] += (label_map_up == c).sum()

    stats = {
        "num_images": len(image_files),
        "num_clusters": num_clusters,
        "checkpoint": checkpoint_path,
        "cluster_pixel_counts": {int(c): int(cluster_pixel_counts[c]) for c in range(num_clusters)},
        "empty_clusters": int((cluster_pixel_counts == 0).sum()),
    }
    with open(output_path / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Pseudo-labels saved to %s", output_path)
    logger.info("Empty clusters: %d/%d", stats["empty_clusters"], num_clusters)


def main():
    parser = argparse.ArgumentParser(description="Generate adapted semantic pseudo-labels")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_clusters", type=int, default=54)
    parser.add_argument("--subsample", type=int, default=500_000)
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    # Adapter config (falls back to checkpoint["adapter_config"] if present)
    parser.add_argument("--variant", type=str, default=None,
                        choices=[None, "lora", "dora", "conv_dora"])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--late_block_start", type=int, default=None)
    parser.add_argument("--adapt_cause", action="store_true", default=None)
    parser.add_argument("--cause_checkpoint_dir", type=str, default=None,
                        help="Path to refs/cause (contains checkpoint/ and CAUSE/ subdirs)")
    parser.add_argument("--resize_to_crop", action="store_true",
                        help="Resize images to crop_size (322) for fast single-crop inference")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    adapter_config = {
        "variant": args.variant,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "late_block_start": args.late_block_start,
        "adapt_cause": args.adapt_cause,
    }
    adapter_config = {k: v for k, v in adapter_config.items() if v is not None}

    generate_adapted_pseudolabels(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        num_clusters=args.num_clusters,
        subsample=args.subsample,
        use_crf=args.use_crf,
        seed=args.seed,
        adapter_config=adapter_config if adapter_config else None,
        cause_checkpoint_dir=args.cause_checkpoint_dir,
        resize_to_crop=args.resize_to_crop,
    )


if __name__ == "__main__":
    main()
