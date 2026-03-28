#!/usr/bin/env python3
"""
MaskCut instance pseudo-label generation for Cityscapes using DINOv3 backbone.
Adapted from CutLER (Meta, CVPR 2023) with DINOv3 ViT-B/16 backbone.

Usage:
    # Quick test on 5 images:
    python mbps_pytorch/generate_cutler_instances.py \
        --cityscapes_root datasets/cityscapes \
        --output_dir pseudo_instances_cutler \
        --split val --limit 5 --visualize

    # Full val set:
    python mbps_pytorch/generate_cutler_instances.py \
        --cityscapes_root datasets/cityscapes \
        --output_dir pseudo_instances_cutler \
        --split val --N 6 --tau 0.15
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import PIL
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.linalg import eigh
from scipy import ndimage
import json
import time


# ─── Inline utilities (from TokenCut / CutLER) ───────────────────────────────

def resize_pil(I, patch_size=16):
    """Resize PIL image so dimensions are divisible by patch_size."""
    w, h = I.size
    new_w = int(round(w / patch_size)) * patch_size
    new_h = int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size
    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h


def IoU(mask1, mask2):
    """Compute IoU between two binary masks."""
    mask1, mask2 = (mask1 > 0.5).to(torch.bool), (mask2 > 0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    if union == 0:
        return 0.0
    return (intersection.to(torch.float) / union).mean().item()


def detect_box(bipartition, seed, dims, scales=None, initial_im_size=None):
    """Extract connected component containing the seed patch."""
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]
    mask = np.where(objects == cc)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])
    return pred, [ymin, xmin, ymax, xmax], objects, mask


def densecrf(image, mask):
    """Apply DenseCRF to refine a binary mask."""
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils

    h, w = mask.shape
    fg = mask.reshape(1, h, w).astype(float)
    bg = 1 - fg
    output_logits = torch.from_numpy(np.concatenate((bg, fg), axis=0))

    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(
        output_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c, h, w = output_probs.shape
    U = crf_utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=7)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=image, compat=10)

    Q = d.inference(10)
    Q = np.array(Q).reshape((c, h, w))
    MAP = np.argmax(Q, axis=0).reshape((h, w)).astype(np.float32)
    return MAP


# ─── DINOv3 Feature Extractor ────────────────────────────────────────────────

class DINOv3Feat(nn.Module):
    """Extract Key features from DINOv3's last attention layer."""

    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", vit_feat="k"):
        super().__init__()
        from transformers import AutoModel

        print(f"Loading DINOv3 model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.vit_feat = vit_feat

        config = self.model.config
        self.patch_size = config.patch_size  # 16
        self.num_register_tokens = getattr(config, "num_register_tokens", 4)
        self.embed_dim = config.hidden_size  # 768 for ViT-B
        self.num_heads = config.num_attention_heads  # 12

        # DINOv3 structure: model.layer (not model.encoder.layer)
        self._hook_output = {}
        last_attn = self.model.layer[-1].attention
        last_attn.k_proj.register_forward_hook(
            lambda m, i, o: self._hook_output.__setitem__("k", o)
        )
        last_attn.q_proj.register_forward_hook(
            lambda m, i, o: self._hook_output.__setitem__("q", o)
        )
        last_attn.v_proj.register_forward_hook(
            lambda m, i, o: self._hook_output.__setitem__("v", o)
        )

    @torch.no_grad()
    def forward(self, img):
        """
        Args:
            img: (1, 3, H, W) tensor, normalized
        Returns:
            feats: (1, feat_dim, num_patches) features
        """
        h, w = img.shape[2], img.shape[3]
        feat_h, feat_w = h // self.patch_size, w // self.patch_size

        # Forward pass
        _ = self.model(img)

        # Get Key features: (B, num_tokens, hidden_dim)
        k = self._hook_output["k"]
        q = self._hook_output["q"]
        v = self._hook_output["v"]

        bs = k.shape[0]
        skip = 1 + self.num_register_tokens  # 1 CLS + 4 registers = 5

        if self.vit_feat == "k":
            feats = k[:, skip:].transpose(1, 2).reshape(bs, self.embed_dim, feat_h * feat_w)
        elif self.vit_feat == "q":
            feats = q[:, skip:].transpose(1, 2).reshape(bs, self.embed_dim, feat_h * feat_w)
        elif self.vit_feat == "v":
            feats = v[:, skip:].transpose(1, 2).reshape(bs, self.embed_dim, feat_h * feat_w)
        else:
            raise ValueError(f"Unknown vit_feat: {self.vit_feat}")

        # Ensure float32 (MPS doesn't support float64)
        return feats.float()


# ─── MaskCut Algorithm (adapted for non-square grids) ────────────────────────

def get_affinity_matrix(feats, tau, eps=1e-5):
    """Compute binary affinity matrix from normalized features."""
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0, 1) @ feats).cpu().numpy()
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D


def second_smallest_eigenvector(A, D):
    """Get the second smallest eigenvector for NCut."""
    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])
    return eigenvec, eigenvectors[:, 0]


def get_salient_areas(second_smallest_vec):
    """Threshold eigenvector to get foreground/background partition."""
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition


def check_num_fg_corners(bipartition, dims):
    """Check how many image corners belong to foreground."""
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r = bipartition_[0][0], bipartition_[0][-1]
    bottom_l, bottom_r = bipartition_[-1][0], bipartition_[-1][-1]
    return int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)


def get_masked_affinity_matrix(painting, feats, mask, dims):
    """Mask out already-discovered regions. Supports non-square grids."""
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, dims[0], dims[1])
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting


def maskcut_forward(feats, dims, scales, init_image_size, tau=0, N=3, device="cpu"):
    """
    MaskCut: iterative NCut for multi-object discovery.

    Args:
        feats: (feat_dim, num_patches) features
        dims: (feat_h, feat_w) patch grid dimensions
        scales: (scale_h, scale_w) from patch grid to image
        init_image_size: (H, W) original image size
        tau: affinity threshold
        N: max number of masks
        device: torch device
    """
    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = torch.zeros(dims, dtype=torch.float32, device=device)
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, dims)

        # Check if there's enough unmasked area left
        if i > 0:
            unmasked_ratio = 1.0 - painting.mean().item()
            if unmasked_ratio < 0.05:
                break

        # Construct affinity matrix and solve NCut
        A, D = get_affinity_matrix(feats, tau)
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        bipartition = get_salient_areas(second_smallest_vec)

        # Determine if partition should be reversed
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # Extract connected component containing the seed
        bipartition = bipartition.reshape(dims).astype(float)
        _, _, _, cc = detect_box(
            bipartition, seed, dims, scales=scales, initial_im_size=init_image_size
        )
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0], cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask.astype(np.float32)).to(device)

        # Filter: skip if too similar to previous mask or too small
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.numel()
            if IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = torch.zeros(dims, dtype=torch.float32, device=device)
        current_mask = pseudo_mask

        # Upsample mask to image resolution and subtract previous masks
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition_up = F.interpolate(
            pseudo_mask.unsqueeze(0).unsqueeze(0),
            size=init_image_size,
            mode="nearest",
        ).squeeze()
        bipartition_masked = bipartition_up.cpu().numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # Upsample eigenvector
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = torch.from_numpy(eigvec.astype(np.float32)).to(device)
        eigvec = F.interpolate(
            eigvec.unsqueeze(0).unsqueeze(0),
            size=init_image_size,
            mode="nearest",
        ).squeeze()
        eigvecs.append(eigvec.cpu().numpy())

    return bipartitions, eigvecs


def maskcut_single_image(img_path, backbone, patch_size, tau, N=6,
                         fixed_size=480, device="cpu", use_crf=True):
    """
    Run MaskCut on a single image.

    Args:
        img_path: path to image
        backbone: feature extractor
        patch_size: ViT patch size
        tau: affinity threshold
        N: max masks per image
        fixed_size: resize input to this size (square)
        device: torch device
        use_crf: whether to apply CRF post-processing

    Returns:
        masks: list of binary masks at original image resolution (H, W)
        I_orig: original PIL image
    """
    I_orig = Image.open(img_path).convert("RGB")
    orig_w, orig_h = I_orig.size

    # Resize to fixed size (square)
    I_resized = I_orig.resize((fixed_size, fixed_size), PIL.Image.LANCZOS)

    # Make dimensions divisible by patch_size
    I_ready, w, h, feat_w, feat_h = resize_pil(I_resized, patch_size)

    # Image normalization
    ToTensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tensor = ToTensor(I_ready).unsqueeze(0).to(device)

    # Extract features
    feat = backbone(tensor)[0]  # (feat_dim, num_patches)

    # Run MaskCut
    bipartitions, eigvecs = maskcut_forward(
        feat, [feat_h, feat_w], [patch_size, patch_size], [h, w],
        tau=tau, N=N, device=device,
    )

    # Post-process each mask
    final_masks = []
    I_resized_np = np.array(I_resized)

    for bipartition in bipartitions:
        if np.sum(bipartition) < 1:
            continue

        if use_crf:
            pseudo_mask = densecrf(I_resized_np, bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

            # Filter if CRF changed the mask too much
            mask1 = torch.from_numpy(bipartition.astype(float))
            mask2 = torch.from_numpy(pseudo_mask.astype(float))
            if IoU(mask1, mask2) < 0.5:
                pseudo_mask = bipartition
        else:
            pseudo_mask = ndimage.binary_fill_holes(bipartition >= 0.5)

        # Resize to original resolution
        mask_pil = Image.fromarray(np.uint8(pseudo_mask * 255))
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        mask_np = np.asarray(mask_pil) > 127

        if np.sum(mask_np) > 0:
            final_masks.append(mask_np)

    return final_masks, I_orig


# ─── Cityscapes Processing ───────────────────────────────────────────────────

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


def masks_to_instance_map(masks, min_area=100):
    """Convert list of binary masks to an instance ID map.

    Args:
        masks: list of (H, W) binary arrays
        min_area: minimum mask area in pixels

    Returns:
        instance_map: (H, W) uint16 array, 0=background, 1,2,...=instances
    """
    if not masks:
        h, w = 1024, 2048  # Cityscapes default
        return np.zeros((h, w), dtype=np.uint16)

    h, w = masks[0].shape
    instance_map = np.zeros((h, w), dtype=np.uint16)
    instance_id = 1

    # Sort masks by area (largest first) to handle overlaps
    sorted_masks = sorted(masks, key=lambda m: np.sum(m), reverse=True)

    for mask in sorted_masks:
        if np.sum(mask) < min_area:
            continue
        # Only assign to pixels not yet claimed
        free_pixels = (instance_map == 0) & mask
        if np.sum(free_pixels) < min_area:
            continue
        instance_map[free_pixels] = instance_id
        instance_id += 1

    return instance_map


def visualize_instances(instance_map, save_path):
    """Save a colored visualization of instance map."""
    np.random.seed(42)
    max_id = instance_map.max()
    colors = np.random.randint(50, 255, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background is black

    vis = colors[instance_map]
    Image.fromarray(vis).save(save_path)


def main():
    parser = argparse.ArgumentParser("MaskCut instance generation with DINOv3")
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--output_dir", type=str, default="pseudo_instances_cutler",
                        help="Output directory for instance masks")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Dataset split")
    parser.add_argument("--backbone", type=str, default="dinov3",
                        choices=["dinov3", "dinov3-large"],
                        help="Backbone model")
    parser.add_argument("--vit_feat", type=str, default="k", choices=["k", "q", "v"],
                        help="Which attention feature to use")
    parser.add_argument("--tau", type=float, default=0.15,
                        help="Affinity threshold for NCut")
    parser.add_argument("--N", type=int, default=6,
                        help="Max number of masks per image")
    parser.add_argument("--fixed_size", type=int, default=480,
                        help="Resize input images to this square size")
    parser.add_argument("--min_area", type=int, default=200,
                        help="Minimum instance area in pixels (at original resolution)")
    parser.add_argument("--no_crf", action="store_true",
                        help="Disable CRF post-processing")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N images (for testing)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    args = parser.parse_args()

    # Device setup
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load backbone
    model_map = {
        "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    model_name = model_map[args.backbone]
    backbone = DINOv3Feat(model_name=model_name, vit_feat=args.vit_feat)
    backbone.eval()
    backbone.to(device)
    patch_size = backbone.patch_size
    print(f"Backbone: {args.backbone} (patch_size={patch_size}, dim={backbone.embed_dim})")

    # Get image list
    images = get_cityscapes_images(args.cityscapes_root, args.split)
    if args.limit:
        images = images[:args.limit]
    print(f"Processing {len(images)} images from {args.split} split")

    # Create output directories
    output_base = os.path.join(args.output_dir, args.split)
    os.makedirs(output_base, exist_ok=True)
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, "vis", args.split)
        os.makedirs(vis_dir, exist_ok=True)

    # Process images
    stats = {"total_images": 0, "total_instances": 0, "instances_per_image": [],
             "time_per_image": [], "config": vars(args)}

    for img_info in tqdm(images, desc="MaskCut"):
        t0 = time.time()

        try:
            masks, I_orig = maskcut_single_image(
                img_info["path"], backbone, patch_size,
                tau=args.tau, N=args.N, fixed_size=args.fixed_size,
                device=device, use_crf=not args.no_crf,
            )
        except Exception as e:
            print(f"Error processing {img_info['stem']}: {e}")
            continue

        # Convert to instance map
        instance_map = masks_to_instance_map(masks, min_area=args.min_area)
        num_instances = int(instance_map.max())

        # Save instance map
        city_dir = os.path.join(output_base, img_info["city"])
        os.makedirs(city_dir, exist_ok=True)
        out_path = os.path.join(city_dir, f"{img_info['stem']}_instance.png")
        Image.fromarray(instance_map).save(out_path)

        # Save visualization
        if args.visualize:
            vis_city_dir = os.path.join(vis_dir, img_info["city"])
            os.makedirs(vis_city_dir, exist_ok=True)
            vis_path = os.path.join(vis_city_dir, f"{img_info['stem']}_vis.png")
            visualize_instances(instance_map, vis_path)

        # Update stats
        elapsed = time.time() - t0
        stats["total_images"] += 1
        stats["total_instances"] += num_instances
        stats["instances_per_image"].append(num_instances)
        stats["time_per_image"].append(elapsed)

        if stats["total_images"] <= 3 or stats["total_images"] % 50 == 0:
            avg_inst = np.mean(stats["instances_per_image"])
            avg_time = np.mean(stats["time_per_image"])
            tqdm.write(
                f"  {img_info['stem']}: {num_instances} instances, "
                f"{elapsed:.1f}s (avg: {avg_inst:.1f} inst/img, {avg_time:.1f}s/img)"
            )

    # Summary
    if stats["total_images"] > 0:
        avg_inst = np.mean(stats["instances_per_image"])
        avg_time = np.mean(stats["time_per_image"])
        print(f"\n{'='*60}")
        print(f"Done: {stats['total_images']} images processed")
        print(f"Average instances per image: {avg_inst:.1f}")
        print(f"Average time per image: {avg_time:.1f}s")
        print(f"Total instances: {stats['total_instances']}")
        print(f"Output saved to: {output_base}")

        # Save stats
        stats_out = {k: v for k, v in stats.items()
                     if k not in ("time_per_image", "instances_per_image")}
        stats_out["avg_instances_per_image"] = float(avg_inst)
        stats_out["avg_time_per_image"] = float(avg_time)
        stats_out["instances_distribution"] = {
            "min": int(np.min(stats["instances_per_image"])),
            "max": int(np.max(stats["instances_per_image"])),
            "median": float(np.median(stats["instances_per_image"])),
        }
        stats_path = os.path.join(args.output_dir, f"stats_{args.split}.json")
        with open(stats_path, "w") as f:
            json.dump(stats_out, f, indent=2)
        print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
