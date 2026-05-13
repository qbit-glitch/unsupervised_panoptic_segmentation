#!/usr/bin/env python3
"""Generate fine-grained object masks for Cityscapes using SAM + CLIP filtering.

Supports three backends:
  --backend sam1   Classic SAM v1 — automatic. Use --clip_filter to keep only target classes.
  --backend sam2   SAM2 automatic mask generator. Use --clip_filter to keep only target classes.
  --backend sam3   SAM3 text-prompted (transformers, Meta Nov 2025). Queries target classes
                   by name directly — no CLIP needed. Requires HF access token.

CLIP filtering (--clip_filter):
  Works with sam1/sam2 backends. After mask generation, each mask's bounding-box crop
  is classified by CLIP (openai/clip-vit-base-patch32) against 12 target class names +
  background rejection classes. Only masks where a target class wins with confidence >=
  --clip_confidence are kept, with the matched class label stored in class_labels.

Output per image: .npz with keys
    masks        (N, H, W) bool    — fine-grained binary masks
    areas        (N,)      int32   — pixel area of each mask
    iou_scores   (N,)      float32 — SAM model confidence per mask
    class_labels (N,)      int32   — class idx into FINE_CLASS_NAMES (-1 if no CLIP filter)

Usage — SAM2 + CLIP filter (recommended — no HF token needed):
    python -u scripts/generate_sam_fine_masks.py \\
        --cityscapes_root /path/to/cityscapes \\
        --split train \\
        --backend sam2 \\
        --sam_checkpoint weights/sam2.1_hiera_base_plus.pt \\
        --sam2_config configs/sam2.1/sam2.1_hiera_b+.yaml \\
        --device cpu \\
        --clip_filter \\
        --clip_confidence 0.20 \\
        --resize_height 512 \\
        > logs/sam2_clip_fine_masks_train.log 2>&1

Usage — SAM3 (text-prompted, best quality — requires approved HF token):
    python -u scripts/generate_sam_fine_masks.py \\
        --cityscapes_root /path/to/cityscapes \\
        --split train \\
        --backend sam3 \\
        --hf_token hf_XXXX \\
        --device mps \\
        --resize_height 512 \\
        > logs/sam3_fine_masks_train.log 2>&1
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Target classes ───────────────────────────────────────────────────────────
# Tier 1: dead thing classes depth pseudo-labels miss
# Tier 2: rare stuff/thing classes
FINE_CLASS_NAMES: List[str] = [
    # Tier 1 — thing classes
    "person",          # 0
    "bicycle",         # 1
    "motorcycle",      # 2
    "rider",           # 3
    "traffic sign",    # 4
    "traffic light",   # 5
    # Tier 2 — rare stuff/thing classes
    "truck",           # 6
    "bus",             # 7
    "train",           # 8
    "guard rail",      # 9
    "caravan",         # 10
    "trailer",         # 11
    # Tier 1 addition — car (added after val baseline to preserve val indices)
    "car",             # 12
    # Stuff class — poles are thin verticals depth-based methods always miss
    "pole",            # 13
]

# Background / rejection classes for CLIP filtering
_BG_CLASS_NAMES: List[str] = [
    "road", "sidewalk", "building", "wall", "sky",
    "vegetation", "terrain", "ground", "water", "pavement",
    "parking lot", "fence",
]

# Per-class maximum area in pixels AT resize_height=512.
# Masks larger than this are physically impossible for that class and are rejected.
# Scale linearly if using a different resize_height.
_CLASS_MAX_AREA_512: List[int] = [
    # person, bicycle, motorcycle, rider, traffic sign, traffic light
    60_000, 20_000, 25_000, 30_000, 8_000, 6_000,
    # truck, bus, train, guard rail, caravan, trailer
    150_000, 150_000, 200_000, 15_000, 80_000, 100_000,
    # car
    80_000,
    # pole — thin vertical, area is small but not capped very low (groups of poles)
    10_000,
]

_NAME_TO_IDX = {name: i for i, name in enumerate(FINE_CLASS_NAMES)}


# ── Image discovery ──────────────────────────────────────────────────────────

def get_image_paths(cityscapes_root: str, split: str):
    img_dir = Path(cityscapes_root) / "leftImg8bit" / split
    paths = sorted(img_dir.rglob("*_leftImg8bit.png"))
    if not paths:
        raise FileNotFoundError(f"No images found under {img_dir}")
    cities = [p.parent.name for p in paths]
    stems = [p.stem.replace("_leftImg8bit", "") for p in paths]
    return paths, cities, stems


# ── Backend loaders ──────────────────────────────────────────────────────────

def _load_sam1(args):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam = sam.to(args.device)
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_area,
    )
    logger.info("SAM1 (%s) loaded", args.sam_model_type)
    return gen


def _load_sam2(args):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    model = build_sam2(args.sam2_config, args.sam_checkpoint, device=args.device)
    gen = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_area,
    )
    logger.info("SAM2 loaded (config=%s)", args.sam2_config)
    return gen


def _load_sam3(args) -> Tuple:
    """Load SAM3 via HuggingFace transformers (v5.1+) with token authentication."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError(
            "SAM3 requires a HuggingFace access token. "
            "Pass --hf_token or set the HF_TOKEN environment variable."
        )

    model_id = args.hf_model_id
    logger.info("Loading SAM3 from %s ...", model_id)
    processor = Sam3Processor.from_pretrained(model_id, token=token)
    model = Sam3Model.from_pretrained(model_id, token=token, torch_dtype=torch.float32)
    model = model.to(args.device).eval()
    logger.info("SAM3 loaded on %s", args.device)
    return model, processor


def _load_clip(model_id: str, device: str):
    """Load CLIP for mask crop classification."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP (%s) for mask filtering ...", model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32)
    # CLIP runs on CPU regardless to avoid MPS float16 issues
    clip_device = "cpu"
    model = model.to(clip_device).eval()
    logger.info("CLIP loaded on %s", clip_device)
    return model, processor, clip_device


# ── CLIP mask filtering ──────────────────────────────────────────────────────

def _classify_masks_clip(
    image_np: np.ndarray,
    masks: List[np.ndarray],
    areas: List[int],
    clip_model,
    clip_processor,
    clip_device: str,
    confidence_thresh: float,
    resize_height: int = 512,
    pad_frac: float = 0.15,
    bg_veto_ratio: float = 0.80,
) -> List[int]:
    """Classify each mask's bounding-box crop with CLIP.

    Accepts a mask only when ALL three conditions hold:
      1. A target class wins with prob >= confidence_thresh.
      2. The winning background class prob < bg_veto_ratio * target_prob
         (background veto — prevents large ambiguous regions from passing).
      3. The mask area is below the per-class maximum area cap
         (area sanity — prevents buildings from being labeled traffic lights).

    Returns class index into FINE_CLASS_NAMES, or -1 to reject.
    """
    import torch

    if not masks:
        return []

    all_classes = FINE_CLASS_NAMES + _BG_CLASS_NAMES
    texts = [f"a photo of a {c}" for c in all_classes]
    n_target = len(FINE_CLASS_NAMES)
    n_bg = len(_BG_CLASS_NAMES)
    h, w = image_np.shape[:2]

    # Scale area caps to current resize_height
    scale = (resize_height / 512.0) ** 2
    area_caps = [int(cap * scale) for cap in _CLASS_MAX_AREA_512]

    # Crop each mask's bounding box with padding
    crops = []
    for mask in masks:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            crops.append(Image.fromarray(image_np))
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        pad_y = max(int((y1 - y0) * pad_frac), 4)
        pad_x = max(int((x1 - x0) * pad_frac), 4)
        crop = image_np[
            max(0, y0 - pad_y): min(h, y1 + pad_y),
            max(0, x0 - pad_x): min(w, x1 + pad_x),
        ]
        crops.append(Image.fromarray(crop))

    # CLIP inference in batches
    batch_size = 32
    all_probs = []
    for i in range(0, len(crops), batch_size):
        inputs = clip_processor(
            text=texts,
            images=crops[i: i + batch_size],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = clip_model(**inputs).logits_per_image  # (B, n_all)
        all_probs.append(logits.softmax(dim=-1).cpu().float().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (N, n_all)

    results = []
    for probs, area in zip(all_probs, areas):
        target_probs = probs[:n_target]
        bg_probs = probs[n_target:]

        best_target_idx = int(target_probs.argmax())
        best_target_prob = float(target_probs[best_target_idx])
        best_bg_prob = float(bg_probs.max())

        # Gate 1: target confidence
        if best_target_prob < confidence_thresh:
            results.append(-1)
            continue

            # Gate 2: background veto — if bg is nearly as strong, reject
        if best_bg_prob >= bg_veto_ratio * best_target_prob:
            results.append(-1)
            continue

        # Gate 3: area sanity per class
        if area > area_caps[best_target_idx]:
            results.append(-1)
            continue

        results.append(best_target_idx)

    return results


# ── Per-image processing ─────────────────────────────────────────────────────

def _process_auto(
    generator,
    image_np: np.ndarray,
    max_area_frac: float,
    min_area: int,
    clip_model=None,
    clip_processor=None,
    clip_device: str = "cpu",
    clip_confidence: float = 0.20,
) -> Optional[dict]:
    """SAM1/SAM2 automatic mask generation with optional CLIP class filtering."""
    h, w = image_np.shape[:2]
    max_area_px = int(max_area_frac * h * w)

    raw = generator.generate(image_np)
    candidates = [m for m in raw if min_area <= m["area"] <= max_area_px]
    if not candidates:
        return None

    masks_raw = [m["segmentation"].astype(np.bool_) for m in candidates]
    areas_raw = [m["area"] for m in candidates]
    ious_raw = [m["predicted_iou"] for m in candidates]

    if clip_model is not None:
        # Classify each mask crop and keep only target-class matches
        cls_labels = _classify_masks_clip(
            image_np, masks_raw, areas_raw, clip_model, clip_processor, clip_device,
            confidence_thresh=clip_confidence,
            resize_height=image_np.shape[0],
        )
        kept = [(mask, area, iou, cls) for mask, area, iou, cls
                in zip(masks_raw, areas_raw, ious_raw, cls_labels) if cls >= 0]
        if not kept:
            return None
        masks_raw, areas_raw, ious_raw, cls_raw = zip(*kept)
    else:
        cls_raw = [-1] * len(masks_raw)

    # Sort by area ascending (fine → coarse)
    order = np.argsort(areas_raw)
    masks = np.stack([masks_raw[i] for i in order])
    areas = np.array([areas_raw[i] for i in order], dtype=np.int32)
    iou_scores = np.array([ious_raw[i] for i in order], dtype=np.float32)
    class_labels = np.array([cls_raw[i] for i in order], dtype=np.int32)

    return dict(masks=masks, areas=areas, iou_scores=iou_scores, class_labels=class_labels)


def _process_sam3(
    model,
    processor,
    image_np: np.ndarray,
    class_names: List[str],
    device: str,
    min_area: int,
    iou_thresh: float,
) -> Optional[dict]:
    """SAM3 text-prompted segmentation — vision embeddings cached across class queries."""
    import torch
    import torch.nn.functional as F

    orig_h, orig_w = image_np.shape[:2]
    pil_image = Image.fromarray(image_np)

    img_inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=pixel_values)

    all_masks, all_areas, all_iou, all_cls = [], [], [], []

    for cls_name in class_names:
        cls_idx = _NAME_TO_IDX.get(cls_name, -1)

        text_inputs = processor(text=cls_name, return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                vision_embeds=vision_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        pred_masks = outputs.pred_masks[0]
        pred_logits = outputs.pred_logits
        presence = outputs.presence_logits

        if pred_logits is not None and presence is not None:
            scores = (pred_logits[0].sigmoid() * presence[0].sigmoid()).cpu().float().numpy()
        elif pred_logits is not None:
            scores = pred_logits[0].sigmoid().cpu().float().numpy()
        else:
            scores = np.ones(pred_masks.shape[0], dtype=np.float32)

        masks_full = F.interpolate(
            pred_masks.unsqueeze(0).float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        masks_bin = (masks_full.sigmoid() > 0.5).cpu().numpy()

        for mask, score in zip(masks_bin, scores):
            if score < iou_thresh:
                continue
            area = int(mask.sum())
            if area < min_area:
                continue
            all_masks.append(mask.astype(bool))
            all_areas.append(area)
            all_iou.append(float(score))
            all_cls.append(cls_idx)

    if not all_masks:
        return None

    order = np.argsort(all_areas)
    return dict(
        masks=np.stack([all_masks[i] for i in order]),
        areas=np.array([all_areas[i] for i in order], dtype=np.int32),
        iou_scores=np.array([all_iou[i] for i in order], dtype=np.float32),
        class_labels=np.array([all_cls[i] for i in order], dtype=np.int32),
    )


def _empty_result(h: int, w: int) -> dict:
    return dict(
        masks=np.zeros((0, h, w), dtype=bool),
        areas=np.array([], dtype=np.int32),
        iou_scores=np.array([], dtype=np.float32),
        class_labels=np.array([], dtype=np.int32),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SAM fine-grained masks for Cityscapes")

    # Data
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output_dir", default=None,
                        help="Output root (default: <cityscapes_root>/sam_fine_masks_<backend>)")

    # Backend
    parser.add_argument("--backend", default="sam2", choices=["sam1", "sam2", "sam3"])
    parser.add_argument("--device", default="cpu", help="Torch device: cuda / mps / cpu")

    # SAM3 HuggingFace settings
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_model_id", default="facebook/sam3")

    # SAM1/SAM2 checkpoint
    parser.add_argument("--sam_checkpoint", default=None)
    parser.add_argument("--sam_model_type", default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")

    # SAM generation params
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.78)
    parser.add_argument("--stability_score_thresh", type=float, default=0.82)
    # SAM3 uses presence×quality product scores (max ~0.7), not SAM2 IoU estimates.
    # A lower threshold is required to capture small/hard classes like bicycle.
    # 0.35 captures small distant objects while avoiding near-zero noise.
    parser.add_argument("--sam3_score_thresh", type=float, default=0.35,
                        help="Min presence×quality score to keep a SAM3 mask (default 0.35)")
    parser.add_argument("--max_area_frac", type=float, default=0.15,
                        help="Keep masks with area <= frac * H * W (raised from 0.05 to catch trucks/buses)")
    parser.add_argument("--min_area", type=int, default=200,
                        help="Discard masks smaller than this (px). 200 captures small distant objects.")

    # CLIP filtering (for sam1/sam2 backends)
    parser.add_argument("--clip_filter", action="store_true",
                        help="Use CLIP to keep only tier-1/tier-2 class masks")
    parser.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_confidence", type=float, default=0.60,
                        help="Min CLIP softmax prob for a target class to keep the mask (>=0.60 recommended)")

    # SAM3 target classes
    parser.add_argument("--sam3_classes", nargs="+", default=None,
                        help=f"(sam3) Target class names. Default: all {len(FINE_CLASS_NAMES)}")

    # Image options
    parser.add_argument("--resize_height", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    target_classes = args.sam3_classes or FINE_CLASS_NAMES
    logger.info("Backend: %s | CLIP filter: %s", args.backend, args.clip_filter)
    logger.info("Target classes: %s", target_classes)

    suffix = f"{args.backend}_clip" if (args.clip_filter and args.backend != "sam3") else args.backend
    out_root = (
        Path(args.output_dir) if args.output_dir
        else Path(args.cityscapes_root) / f"sam_fine_masks_{suffix}"
    )
    out_split = out_root / args.split
    out_split.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_split)

    image_paths, cities, stems = get_image_paths(args.cityscapes_root, args.split)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
        cities = cities[: args.max_images]
        stems = stems[: args.max_images]
    logger.info("Processing %d images", len(image_paths))

    # Load models
    clip_model = clip_processor = clip_device = None
    if args.backend == "sam1":
        generator = _load_sam1(args)
    elif args.backend == "sam2":
        generator = _load_sam2(args)
    else:
        generator = None

    if args.clip_filter and args.backend in ("sam1", "sam2"):
        clip_model, clip_processor, clip_device = _load_clip(args.clip_model_id, args.device)

    if args.backend == "sam3":
        sam3_model, sam3_proc = _load_sam3(args)

    n_skipped = n_empty = n_saved = 0
    total_kept = total_raw = 0

    for img_path, city, stem in tqdm(zip(image_paths, cities, stems),
                                      total=len(image_paths), desc=f"SAM-{args.backend}"):
        city_dir = out_split / city
        city_dir.mkdir(exist_ok=True)
        out_path = city_dir / f"{stem}_fine_masks.npz"

        if out_path.exists():
            n_skipped += 1
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))

        if args.resize_height is not None and image_np.shape[0] != args.resize_height:
            h, w = image_np.shape[:2]
            new_w = int(w * args.resize_height / h)
            image_np = np.array(
                Image.fromarray(image_np).resize((new_w, args.resize_height), Image.BILINEAR)
            )

        if args.backend == "sam3":
            result = _process_sam3(
                sam3_model, sam3_proc, image_np,
                target_classes, args.device, args.min_area, args.sam3_score_thresh,
            )
        else:
            result = _process_auto(
                generator, image_np, args.max_area_frac, args.min_area,
                clip_model=clip_model,
                clip_processor=clip_processor,
                clip_device=clip_device,
                clip_confidence=args.clip_confidence,
            )

        if result is None:
            result = _empty_result(*image_np.shape[:2])
            n_empty += 1
        else:
            n_saved += 1
            total_kept += len(result["masks"])

        np.savez_compressed(str(out_path), **result)

    logger.info("Done. Saved=%d  Empty=%d  Skipped=%d  Avg masks/img=%.1f",
                n_saved, n_empty, n_skipped,
                total_kept / max(1, n_saved))
    logger.info("Output: %s", out_split)

    if args.backend == "sam3" or args.clip_filter:
        logger.info("Class label mapping: %s", {i: n for i, n in enumerate(FINE_CLASS_NAMES)})


if __name__ == "__main__":
    main()
