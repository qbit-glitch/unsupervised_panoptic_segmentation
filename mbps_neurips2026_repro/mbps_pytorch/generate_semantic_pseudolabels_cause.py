#!/usr/bin/env python3
"""
Generate unsupervised semantic pseudo-labels using CAUSE-TR (Pattern Recognition 2024).

Uses DINOv2 ViT-B/14 backbone + CAUSE-TR pretrained heads to produce 27-class
Cityscapes semantic predictions without any ground truth supervision.

Hungarian matching (on val set with GT) is used only to assign semantic names
to the 27 discovered clusters — standard protocol in unsupervised segmentation.

Outputs 27-class pseudo-labels (CAUSE class IDs 0-26, corresponding to
Cityscapes labelIDs 7-33 offset by first_nonvoid=7).

Usage:
    # Step 1: Compute Hungarian matching on val set
    python generate_semantic_pseudolabels_cause.py --mode hungarian \
        --cityscapes_root /path/to/cityscapes --gpu 0

    # Step 2: Generate pseudo-labels
    python generate_semantic_pseudolabels_cause.py --mode generate \
        --cityscapes_root /path/to/cityscapes \
        --output_dir /path/to/cityscapes/pseudo_semantic_cause \
        --split both --gpu 0

    # Or both in one go:
    python generate_semantic_pseudolabels_cause.py --mode all \
        --cityscapes_root /path/to/cityscapes \
        --output_dir /path/to/cityscapes/pseudo_semantic_cause \
        --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Add CAUSE repo to path for imports
CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from models.dinov2vit import dinov2_vit_base_14
from modules.segment import Segment_TR
from modules.segment_module import Cluster, transform, untransform

# Try importing CRF; make it optional
try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("Warning: pydensecrf not installed. CRF post-processing disabled.")

# Cityscapes 27-class names (labelIDs 7-33, offset by first_nonvoid=7)
CAUSE_27_CLASSES = [
    "road", "sidewalk", "parking", "rail_track",       # 0-3
    "building", "wall", "fence", "guard_rail",          # 4-7
    "bridge", "tunnel", "pole", "polegroup",            # 8-11
    "traffic_light", "traffic_sign", "vegetation",      # 12-14
    "terrain", "sky", "person", "rider",                # 15-18
    "car", "truck", "bus", "caravan",                   # 19-22
    "trailer", "train", "motorcycle", "bicycle",        # 23-26
]

# For evaluation: map CAUSE 27-class → standard 19 trainIDs
CAUSE27_TO_TRAINID_19 = {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}

# ImageNet normalization (used by DINOv2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in IMAGENET_STD]),
    transforms.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1., 1., 1.]),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate unsupervised semantic pseudo-labels with CAUSE-TR"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["hungarian", "generate", "all"],
                        help="Mode: hungarian (compute mapping), generate (pseudo-labels), all (both)")
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for pseudo-labels (default: {cityscapes_root}/pseudo_semantic_cause)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to CAUSE repo with checkpoints (default: refs/cause/)")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "val", "both"],
                        help="Which split(s) to generate pseudo-labels for")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for sliding window)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_crf", action="store_true",
                        help="Skip CRF post-processing")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device: auto (detect best), cuda, mps, or cpu")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (for CUDA)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of images (0 = all, for debugging)")
    parser.add_argument("--save_logits", action="store_true",
                        help="Save soft logits as .pt files (27, 32, 64) for M2PR refiner")
    parser.add_argument("--logits_h", type=int, default=32,
                        help="Target height for logits downsampling (default: 32)")
    parser.add_argument("--logits_w", type=int, default=64,
                        help="Target width for logits downsampling (default: 64)")
    parser.add_argument("--scales", type=str, default="1.0",
                        help="Comma-separated inference scales (e.g., '0.75,1.0,1.5'). "
                             "Default '1.0' = single-scale (current behavior).")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Override output subdirectory name (e.g., 'pseudo_semantic_cause_ms')")
    parser.add_argument("--sinder_checkpoint", type=str, default=None,
                        help="Path to SINDER-repaired DINOv2 ViT-B/14 checkpoint (optional)")
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vitb14", "dinov3_vitl16"],
                        help="Backbone: dinov2_vitb14 (original CAUSE) or dinov3_vitl16 (retrained)")
    parser.add_argument("--dinov3_checkpoint_dir", type=str, default=None,
                        help="Path to retrained DINOv3 CAUSE heads (for --backbone dinov3_vitl16)")
    return parser.parse_args()


def get_device(args):
    """Select best available device."""
    if args.device == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{args.gpu}")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif args.device == "cuda":
        return torch.device(f"cuda:{args.gpu}")
    elif args.device == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_cause_models(checkpoint_dir, device, sinder_checkpoint=None,
                      backbone="dinov2_vitb14", dinov3_checkpoint_dir=None):
    """Load backbone + CAUSE-TR segment head + cluster with pretrained weights.

    Args:
        backbone: "dinov2_vitb14" (original) or "dinov3_vitl16" (retrained)
        dinov3_checkpoint_dir: path to retrained DINOv3 heads (for dinov3_vitl16)
    """
    if backbone == "dinov3_vitl16":
        return _load_dinov3_models(device, dinov3_checkpoint_dir)
    else:
        return _load_dinov2_models(checkpoint_dir, device, sinder_checkpoint)


def _load_dinov3_models(device, dinov3_checkpoint_dir):
    """Load DINOv3 ViT-L/16 backbone + retrained CAUSE-TR heads."""
    from train_cause_dinov3 import DINOv3Backbone

    cause_args = SimpleNamespace(
        dim=1024,
        reduced_dim=90,
        projection_dim=2048,
        num_codebook=2048,
        n_classes=27,
        num_queries=20 * 20,  # 320/16 = 20 patches per side
        crop_size=320,
        patch_size=16,
    )

    # 1. Load DINOv3 ViT-L/16 backbone
    net = DINOv3Backbone(device=device).to(device)
    net.eval()

    # 2. Resolve checkpoint dir
    if dinov3_checkpoint_dir is None:
        dinov3_checkpoint_dir = os.path.join(
            Path(__file__).resolve().parent.parent, "refs", "cause", "CAUSE_dinov3", "final"
        )

    # 3. Load Segment_TR head
    seg_path = os.path.join(dinov3_checkpoint_dir, "segment_tr.pth")
    print(f"Loading DINOv3 Segment_TR from {seg_path}")
    segment = Segment_TR(cause_args).to(device)
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    segment.load_state_dict(seg_state, strict=False)
    segment.eval()

    # 4. Load Cluster
    cluster_path = os.path.join(dinov3_checkpoint_dir, "cluster_tr.pth")
    print(f"Loading DINOv3 Cluster from {cluster_path}")
    cluster = Cluster(cause_args).to(device)
    cluster_state = torch.load(cluster_path, map_location="cpu", weights_only=True)
    cluster.load_state_dict(cluster_state, strict=False)

    # 5. Load modularity codebook
    mod_path = os.path.join(dinov3_checkpoint_dir, "modular.npy")
    print(f"Loading DINOv3 codebook from {mod_path}")
    cb = torch.from_numpy(np.load(mod_path)).to(device)
    cluster.codebook.data = cb
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    print(f"DINOv3 models loaded. Backbone: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net, segment, cluster, cause_args


def _load_dinov2_models(checkpoint_dir, device, sinder_checkpoint=None):
    """Load DINOv2 ViT-B/14 backbone + original CAUSE-TR heads."""

    cause_args = SimpleNamespace(
        dim=768,
        reduced_dim=90,
        projection_dim=2048,
        num_codebook=2048,
        n_classes=27,
        num_queries=23 * 23,  # 322/14 = 23 patches per side
        crop_size=322,
        patch_size=14,
    )

    # 1. Load DINOv2 ViT-B/14 backbone
    if sinder_checkpoint is not None:
        print(f"Loading SINDER-repaired DINOv2 ViT-B/14 from {sinder_checkpoint}")
        net = dinov2_vit_base_14()
        state = torch.load(sinder_checkpoint, map_location="cpu", weights_only=True)
        msg = net.load_state_dict(state, strict=False)
        print(f"  SINDER backbone loaded: {msg}")
    else:
        backbone_path = os.path.join(checkpoint_dir, "checkpoint", "dinov2_vit_base_14.pth")
        print(f"Loading DINOv2 ViT-B/14 from {backbone_path}")
        net = dinov2_vit_base_14()
        state = torch.load(backbone_path, map_location="cpu", weights_only=True)
        msg = net.load_state_dict(state, strict=False)
        print(f"  Backbone loaded: {msg}")
    net = net.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    # 2. Load Segment_TR head
    seg_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "dinov2_vit_base_14", "2048", "segment_tr.pth")
    print(f"Loading Segment_TR from {seg_path}")
    segment = Segment_TR(cause_args).to(device)
    seg_state = torch.load(seg_path, map_location="cpu", weights_only=True)
    segment.load_state_dict(seg_state, strict=False)
    segment.eval()

    # 3. Load Cluster
    cluster_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "dinov2_vit_base_14", "2048", "cluster_tr.pth")
    print(f"Loading Cluster from {cluster_path}")
    cluster = Cluster(cause_args).to(device)
    cluster_state = torch.load(cluster_path, map_location="cpu", weights_only=True)
    cluster.load_state_dict(cluster_state, strict=False)

    # 4. Load modularity codebook and inject into models
    mod_path = os.path.join(checkpoint_dir, "CAUSE", "cityscapes", "modularity",
                            "dinov2_vit_base_14", "2048", "modular.npy")
    print(f"Loading modularity codebook from {mod_path}")
    codebook = np.load(mod_path)
    cb = torch.from_numpy(codebook).to(device)
    cluster.codebook.data = cb
    cluster.codebook.requires_grad = False
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    print(f"All models loaded. Backbone: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net, segment, cluster, cause_args


def dense_crf(image_np, logits_np, max_iter=10):
    """Apply DenseCRF post-processing. image_np: (H,W,3) uint8, logits_np: (C,H,W) float."""
    if not HAS_CRF:
        return logits_np

    C, H, W = logits_np.shape
    image = np.ascontiguousarray(image_np[:, :, ::-1])  # RGB → BGR

    U = crf_utils.unary_from_softmax(logits_np)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=image, compat=4)

    Q = d.inference(max_iter)
    return np.array(Q).reshape((C, H, W))


def cause_forward_single_crop(net, segment, cluster, img_tensor):
    """
    Run CAUSE-TR inference on a single crop (322×322 for DINOv2, 320×320 for DINOv3).
    Returns: (1, 27, H_patch, W_patch) log-softmax logits.
    """
    with torch.no_grad():
        feat = net(img_tensor)[:, 1:, :]  # remove CLS token
        feat_flip = net(img_tensor.flip(dims=[3]))[:, 1:, :]

        seg_feat = transform(segment.head_ema(feat))
        seg_feat_flip = transform(segment.head_ema(feat_flip))
        seg_feat = untransform((seg_feat + seg_feat_flip.flip(dims=[3])) / 2)

        # Cluster assignment with log-softmax for CRF
        logits = cluster.forward_centroid(seg_feat, crf=True)  # (1, 27, H, W)

    return logits


def sliding_window_inference(net, segment, cluster, img_resized, crop_size=322):
    """
    Run CAUSE-TR with 2D sliding window over a resized image.
    img_resized: (1, 3, H, W) tensor where H >= crop_size, W >= crop_size.
    Returns: (27, H, W) merged log-softmax logits.
    """
    _, _, H, W = img_resized.shape

    if H == crop_size and W == crop_size:
        # Square image exactly matching crop size — single crop
        logits = cause_forward_single_crop(net, segment, cluster, img_resized)
        return logits[0]  # (27, 23, 23)

    # 2D grid of crop positions with ~50% overlap
    stride = crop_size // 2  # 161 pixels overlap

    if H <= crop_size:
        y_positions = [0]
    else:
        y_positions = list(range(0, H - crop_size, stride))
        if y_positions[-1] + crop_size < H:
            y_positions.append(H - crop_size)

    if W <= crop_size:
        x_positions = [0]
    else:
        x_positions = list(range(0, W - crop_size, stride))
        if x_positions[-1] + crop_size < W:
            x_positions.append(W - crop_size)

    # Deduplicate and sort
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    # Accumulate logits at pixel resolution
    logit_sum = torch.zeros(27, H, W, device=img_resized.device)
    count = torch.zeros(1, H, W, device=img_resized.device)

    for y_pos in y_positions:
        for x_pos in x_positions:
            # Extract crop (pad if needed for edge cases)
            y_end = min(y_pos + crop_size, H)
            x_end = min(x_pos + crop_size, W)
            crop = img_resized[:, :, y_pos:y_end, x_pos:x_end]

            # Pad to crop_size if the crop is smaller (edge cases)
            ch, cw = crop.shape[2], crop.shape[3]
            if ch < crop_size or cw < crop_size:
                pad_h = crop_size - ch
                pad_w = crop_size - cw
                crop = F.pad(crop, (0, pad_w, 0, pad_h), mode='reflect')

            crop_logits = cause_forward_single_crop(net, segment, cluster, crop)
            # Upsample crop logits to pixel space
            crop_logits_up = F.interpolate(
                crop_logits, size=(crop_size, crop_size),
                mode="bilinear", align_corners=False
            )[0]  # (27, crop_size, crop_size)

            # Only accumulate the non-padded region
            logit_sum[:, y_pos:y_end, x_pos:x_end] += crop_logits_up[:, :ch, :cw]
            count[:, y_pos:y_end, x_pos:x_end] += 1

    # Average overlapping regions
    merged = logit_sum / count.clamp(min=1)
    return merged  # (27, H, W)


def multiscale_inference(net, segment, cluster, img_pil, crop_size, patch_size,
                         scales, normalize_fn, device):
    """
    Run CAUSE-TR at multiple image scales and average logits.

    For each scale factor s:
    1. Resize image so shortest side = crop_size * s (aligned to patch_size)
    2. Run sliding_window_inference with 322×322 crops
    3. Resize logits to a common resolution (1.0× scale)
    4. Average all scale logits

    Args:
        img_pil: PIL Image (original resolution, e.g., 2048×1024)
        scales: list of floats (e.g., [0.75, 1.0, 1.5])
    Returns:
        (27, common_h, common_w) averaged logits
    """
    orig_w, orig_h = img_pil.size
    base_scale = crop_size / min(orig_h, orig_w)  # 322/1024 ≈ 0.314

    # Common resolution = 1.0× scale dimensions
    common_h = (int(round(orig_h * base_scale)) // patch_size) * patch_size
    common_w = (int(round(orig_w * base_scale)) // patch_size) * patch_size

    all_logits = []
    for s in scales:
        eff = base_scale * s
        new_h = int(round(orig_h * eff))
        new_w = int(round(orig_w * eff))
        # Align to patch_size and ensure >= crop_size
        new_h = max(crop_size, (new_h // patch_size) * patch_size)
        new_w = max(crop_size, (new_w // patch_size) * patch_size)

        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = normalize_fn(transforms.ToTensor()(img_resized)).unsqueeze(0).to(device)

        logits = sliding_window_inference(net, segment, cluster, img_tensor, crop_size)
        # (27, new_h, new_w)

        # Resize logits to common resolution
        logits_common = F.interpolate(
            logits.unsqueeze(0), size=(common_h, common_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (27, common_h, common_w)

        all_logits.append(logits_common)

    # Average across scales
    merged = torch.stack(all_logits, dim=0).mean(dim=0)  # (27, common_h, common_w)
    return merged


def get_cityscapes_images(cityscapes_root, split):
    """Enumerate all Cityscapes images for a split."""
    img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
    images = []
    for city in sorted(os.listdir(img_dir)):
        city_dir = os.path.join(img_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.endswith("_leftImg8bit.png"):
                stem = fname.replace("_leftImg8bit.png", "")
                images.append({
                    "city": city,
                    "stem": stem,
                    "img_path": os.path.join(city_dir, fname),
                })
    return images


def get_gt_label_path(cityscapes_root, split, city, stem):
    """Get GT label path for Hungarian matching."""
    return os.path.join(
        cityscapes_root, "gtFine", split, city,
        f"{stem}_gtFine_labelIds.png"
    )


def compute_hungarian(args, net, segment, cluster, crop_size=322):
    """
    Run CAUSE-TR on val set with center crops to compute Hungarian matching.
    This maps the 27 unsupervised cluster IDs to the 27 GT class indices.
    """
    print(f"\n=== Computing Hungarian Matching on Val Set (crop_size={crop_size}) ===")

    device = next(net.parameters()).device
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    FIRST_NONVOID = 7

    images = get_cityscapes_images(args.cityscapes_root, "val")
    if args.limit > 0:
        images = images[:args.limit]
    print(f"Processing {len(images)} val images")

    # Confusion matrix for Hungarian matching (27 × 27)
    histogram = torch.zeros(27, 27, device=device)

    for entry in tqdm(images, desc="Hungarian matching"):
        # Load and preprocess image
        img_pil = Image.open(entry["img_path"]).convert("RGB")
        img_tensor = transforms.Compose([
            transforms.Resize(crop_size, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])(img_pil)
        img_tensor = normalize(img_tensor).unsqueeze(0).to(device)

        # Load GT label
        gt_path = get_gt_label_path(args.cityscapes_root, "val", entry["city"], entry["stem"])
        gt_pil = Image.open(gt_path)
        gt_tensor = transforms.Compose([
            transforms.Resize(crop_size, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(crop_size),
        ])(gt_pil)
        gt = torch.tensor(np.array(gt_tensor), dtype=torch.long).to(device)

        # Map GT labelIDs to CAUSE 27-class
        gt = gt - FIRST_NONVOID
        gt[gt < 0] = -1
        gt[gt >= 27] = -1

        # Run CAUSE inference
        logits = cause_forward_single_crop(net, segment, cluster, img_tensor)
        logits_up = F.interpolate(logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        preds = logits_up.argmax(dim=1).squeeze(0)

        # Accumulate confusion matrix
        mask = (gt >= 0) & (gt < 27) & (preds >= 0) & (preds < 27)
        if mask.sum() > 0:
            hist = torch.bincount(
                27 * gt[mask] + preds[mask], minlength=27 * 27
            ).reshape(27, 27).t()
            histogram += hist

    # Solve Hungarian assignment
    from scipy.optimize import linear_sum_assignment
    assignments = linear_sum_assignment(histogram.cpu().numpy(), maximize=True)

    # Compute mIoU with Hungarian assignment
    hist_matched = histogram[np.argsort(assignments[1]), :]
    tp = torch.diag(hist_matched)
    fp = torch.sum(hist_matched, dim=0) - tp
    fn = torch.sum(hist_matched, dim=1) - tp
    iou = tp / (tp + fp + fn)
    miou = iou[~torch.isnan(iou)].mean().item() * 100

    print(f"\n=== Hungarian Matching Results (27-class) ===")
    print(f"mIoU: {miou:.1f}%")
    print(f"Cluster→Class mapping: {list(assignments[1])}")

    # Per-class IoU
    for i, name in enumerate(CAUSE_27_CLASSES):
        if not torch.isnan(iou[i]):
            print(f"  {name:20s}: {iou[i].item()*100:.1f}%")

    # Save mapping
    output_dir = args.output_dir or os.path.join(args.cityscapes_root, "pseudo_semantic_cause")
    os.makedirs(output_dir, exist_ok=True)
    mapping = {
        "cluster_to_class": [int(x) for x in assignments[1]],
        "class_to_cluster": [int(x) for x in assignments[0]],
        "miou_27class": miou,
        "per_class_iou": {name: float(iou[i].item() * 100) if not torch.isnan(iou[i]) else None
                          for i, name in enumerate(CAUSE_27_CLASSES)},
    }
    mapping_path = os.path.join(output_dir, "hungarian_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Mapping saved to {mapping_path}")

    return assignments[1]  # cluster_to_class array


def generate_pseudolabels(args, net, segment, cluster, hungarian_mapping,
                          crop_size=322, patch_size=14):
    """Generate 27-class pseudo-labels for the specified split(s).

    If --save_logits is set, also saves soft logits as .pt files at
    (27, logits_h, logits_w) resolution for M2PR refiner training.
    """
    device = next(net.parameters()).device
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Parse scales
    scales = [float(s) for s in args.scales.split(",")]
    use_multiscale = not (len(scales) == 1 and scales[0] == 1.0)

    output_dir = args.output_dir or os.path.join(args.cityscapes_root, "pseudo_semantic_cause")
    if args.output_subdir:
        output_dir = os.path.join(args.cityscapes_root, args.output_subdir)
    save_logits = getattr(args, "save_logits", False)
    logits_h = getattr(args, "logits_h", 32)
    logits_w = getattr(args, "logits_w", 64)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        scale_str = ",".join(f"{s:.2f}" for s in scales)
        print(f"\n=== Generating Pseudo-Labels for {split} (crop={crop_size}, patch={patch_size}, scales=[{scale_str}]) ===")
        images = get_cityscapes_images(args.cityscapes_root, split)
        if args.limit > 0:
            images = images[:args.limit]
        print(f"Processing {len(images)} images")

        stats = {"total": 0, "per_class": {i: 0 for i in range(27)}}

        for entry in tqdm(images, desc=f"Generating {split}"):
            # Load image
            img_pil = Image.open(entry["img_path"]).convert("RGB")
            orig_w, orig_h = img_pil.size  # typically 2048, 1024

            if use_multiscale:
                # Multi-scale inference
                merged_logits = multiscale_inference(
                    net, segment, cluster, img_pil, crop_size, patch_size,
                    scales, normalize, device
                )  # (27, common_h, common_w)
                # Get the common resolution for CRF
                base_scale = crop_size / min(orig_h, orig_w)
                new_h = (int(round(orig_h * base_scale)) // patch_size) * patch_size
                new_w = (int(round(orig_w * base_scale)) // patch_size) * patch_size
                img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
            else:
                # Single-scale inference (original behavior)
                scale = crop_size / min(orig_h, orig_w)
                new_h = int(round(orig_h * scale))
                new_w = int(round(orig_w * scale))
                new_h = (new_h // patch_size) * patch_size
                new_w = (new_w // patch_size) * patch_size

                img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
                img_tensor = transforms.ToTensor()(img_resized)
                img_normalized = normalize(img_tensor).unsqueeze(0).to(device)

                merged_logits = sliding_window_inference(
                    net, segment, cluster, img_normalized, crop_size=crop_size
                )  # (27, new_h, new_w)

            # Ensure output directory exists (needed for both logits and PNGs)
            city_dir = os.path.join(output_dir, split, entry["city"])
            os.makedirs(city_dir, exist_ok=True)

            # Save soft logits for M2PR refiner (before CRF/argmax)
            if save_logits:
                # Reorder channels according to Hungarian mapping
                logits_mapped = merged_logits[hungarian_mapping]  # (27, new_h, new_w)
                # Apply softmax for probability distribution
                probs = F.softmax(logits_mapped * 3, dim=0)  # (27, new_h, new_w), alpha=3 from CAUSE
                # Downsample to patch level
                probs_patch = F.interpolate(
                    probs.unsqueeze(0),
                    size=(logits_h, logits_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (27, logits_h, logits_w)
                # Save as float16 to save disk space
                logits_path = os.path.join(city_dir, f"{entry['stem']}_logits.pt")
                torch.save(probs_patch.half().cpu(), logits_path)

            # Optional CRF post-processing
            if not args.skip_crf and HAS_CRF:
                # Convert image to numpy for CRF
                img_np = np.array(img_resized)  # (H, W, 3) uint8
                softmax_logits = F.softmax(merged_logits * 3, dim=0).cpu().numpy()  # alpha=3 from CAUSE
                crf_result = dense_crf(img_np, softmax_logits)
                pred = np.argmax(crf_result, axis=0)  # (new_h, new_w)
            else:
                pred = merged_logits.argmax(dim=0).cpu().numpy()  # (new_h, new_w)

            # Apply Hungarian mapping: cluster IDs → CAUSE 27-class IDs
            mapped_pred = hungarian_mapping[pred]

            # Resize to original resolution
            mapped_pred_pil = Image.fromarray(mapped_pred.astype(np.uint8))
            mapped_pred_full = mapped_pred_pil.resize((orig_w, orig_h), Image.NEAREST)
            mapped_pred_full = np.array(mapped_pred_full)

            # Save argmax PNG
            out_path = os.path.join(city_dir, f"{entry['stem']}.png")
            Image.fromarray(mapped_pred_full.astype(np.uint8)).save(out_path)

            # Stats
            stats["total"] += 1
            for cls_id in range(27):
                stats["per_class"][cls_id] += int((mapped_pred_full == cls_id).sum())

        # Save stats
        stats_path = os.path.join(output_dir, split, "stats.json")
        stats["per_class_names"] = {
            str(i): {"name": CAUSE_27_CLASSES[i], "pixels": stats["per_class"][i]}
            for i in range(27)
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        total_pixels = sum(stats["per_class"].values())
        print(f"\n{split}: Generated {stats['total']} pseudo-labels")
        print(f"Class distribution (top 10):")
        sorted_classes = sorted(stats["per_class"].items(), key=lambda x: -x[1])
        for cls_id, count in sorted_classes[:10]:
            pct = count / total_pixels * 100 if total_pixels > 0 else 0
            print(f"  {CAUSE_27_CLASSES[int(cls_id)]:20s}: {pct:.1f}%")


def main():
    args = parse_args()

    os.environ["XFORMERS_DISABLED"] = "1"

    # Select device
    device = get_device(args)
    print(f"Using device: {device}")

    # Resolve checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(__file__).resolve().parent.parent / "refs" / "cause")

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.cityscapes_root, "pseudo_semantic_cause")

    # Load models
    net, segment, cluster, cause_args = load_cause_models(
        args.checkpoint_dir, device, sinder_checkpoint=args.sinder_checkpoint,
        backbone=args.backbone, dinov3_checkpoint_dir=args.dinov3_checkpoint_dir,
    )
    # Get backbone-specific crop size and patch size
    crop_size = getattr(cause_args, "crop_size", 322)
    patch_size = getattr(cause_args, "patch_size", 14)
    print(f"Backbone: {args.backbone}, crop_size={crop_size}, patch_size={patch_size}")

    # Hungarian matching
    hungarian_mapping = None
    if args.mode in ("hungarian", "all"):
        cluster_to_class = compute_hungarian(args, net, segment, cluster, crop_size=crop_size)
        hungarian_mapping = np.array(cluster_to_class, dtype=np.int64)

    # Print scale config
    scales = [float(s) for s in args.scales.split(",")]
    print(f"Inference scales: {scales}")

    # Generate pseudo-labels
    if args.mode in ("generate", "all"):
        if hungarian_mapping is None:
            # Load saved mapping (always from base output_dir, not output_subdir)
            base_output = args.output_dir
            mapping_path = os.path.join(base_output, "hungarian_mapping.json")
            if not os.path.exists(mapping_path):
                print(f"Error: Hungarian mapping not found at {mapping_path}")
                print("Run with --mode hungarian first.")
                sys.exit(1)
            with open(mapping_path) as f:
                mapping = json.load(f)
            hungarian_mapping = np.array(mapping["cluster_to_class"], dtype=np.int64)
            print(f"Loaded Hungarian mapping from {mapping_path}")

        generate_pseudolabels(args, net, segment, cluster, hungarian_mapping,
                              crop_size=crop_size, patch_size=patch_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
