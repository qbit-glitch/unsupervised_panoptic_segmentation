#!/usr/bin/env python3
"""Diagnostic script for center/offset instance inference pipeline.

Instruments each stage of the inference pipeline to find where instance
detection fails. Loads model, runs on a single image, and prints detailed
stats at each stage.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mbps_pytorch.refine_net import DepthGuidedUNet
from mbps_pytorch.panoptic_deeplab import panoptic_inference_center_offset

CITYSCAPES_ROOT = "/Users/qbit-glitch/Desktop/datasets/cityscapes"
CHECKPOINT = "checkpoints/unet_p2b_instance_heads/best.pth"
DEVICE = "mps"

_THING_IDS = set(range(11, 19))

def load_model():
    model = DepthGuidedUNet(
        num_classes=19, bridge_dim=192,
        num_decoder_stages=2, rich_skip=True,
        num_final_blocks=1, block_type="attention",
        window_size=8, num_heads=4,
        use_instance_heads=True,
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    if "epoch" in ckpt:
        print(f"  Checkpoint epoch: {ckpt['epoch']}")
    if "metrics" in ckpt:
        print(f"  Checkpoint metrics: {ckpt['metrics']}")
    model.eval()
    model.to(DEVICE)
    print(f"Loaded model from {CHECKPOINT}")
    return model


def sobel_gradients(depth_2d):
    """Compute Sobel dx, dy on a 2D depth map."""
    from scipy.ndimage import sobel
    dx = sobel(depth_2d.astype(np.float64), axis=1).astype(np.float32)
    dy = sobel(depth_2d.astype(np.float64), axis=0).astype(np.float32)
    return np.stack([dx, dy], axis=0)  # (2, H, W)


def load_sample_data():
    """Load a single sample from val set."""
    val_dir = Path(CITYSCAPES_ROOT) / "leftImg8bit" / "val"
    imgs = sorted(val_dir.glob("*/*_leftImg8bit.png"))
    img_path = imgs[0]  # First val image
    city = img_path.parent.name
    stem = img_path.name.replace("_leftImg8bit.png", "")
    print(f"Image: {city}/{stem}")

    # Load DINOv2 features: (2048, 768) float16 → (768, 32, 64) float32
    feat_path = Path(CITYSCAPES_ROOT) / "dinov2_features" / "val" / city / f"{stem}_leftImg8bit.npy"
    features_raw = np.load(feat_path).astype(np.float32)  # (2048, 768)
    features = features_raw.reshape(32, 64, -1).transpose(2, 0, 1)  # (768, 32, 64)
    features = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
    print(f"  DINOv2 features: {features.shape} ({features.dtype})")

    # Load depth: (512, 1024) → patch (1, 32, 64) for model input
    depth_path = Path(CITYSCAPES_ROOT) / "depth_spidepth" / "val" / city / f"{stem}.npy"
    depth_full = np.load(depth_path)  # (512, 1024)
    depth_patch = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
    depth_patch = F.interpolate(depth_patch, size=(32, 64), mode="bilinear", align_corners=False)
    depth_patch = depth_patch.squeeze(0)  # (1, 32, 64)
    depth_np = depth_patch.numpy()

    # Sobel gradients at patch resolution
    depth_grads = sobel_gradients(depth_np[0])  # (2, 32, 64)
    depth_grads_t = torch.from_numpy(depth_grads).unsqueeze(0).float().to(DEVICE)

    depth_t = depth_patch.unsqueeze(0).float().to(DEVICE)  # (1, 1, 32, 64)
    depth_full_t = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    print(f"  Depth patch: {depth_t.shape}, Depth grads: {depth_grads_t.shape}")
    print(f"  Depth full: {depth_full_t.shape}")

    # Load instance targets (for comparison)
    center_path = Path(CITYSCAPES_ROOT) / "instance_targets_128x256" / "val" / city / f"{stem}_center.npy"
    offset_path = Path(CITYSCAPES_ROOT) / "instance_targets_128x256" / "val" / city / f"{stem}_offset.npy"
    target_center = np.load(center_path)
    target_offset = np.load(offset_path)
    print(f"  Target center: shape={target_center.shape}, max={target_center.max():.4f}, "
          f"n>0.01={np.sum(target_center > 0.01)}, n>0.5={np.sum(target_center > 0.5)}")
    print(f"  Target offset: shape={target_offset.shape}, range=[{target_offset.min():.1f}, {target_offset.max():.1f}]")

    return features, depth_t, depth_grads_t, depth_full_t, city, stem, target_center, target_offset


def diagnose_inference(model, features, depth_t, depth_grads_t, depth_full_t, target_center, target_offset):
    """Run inference and instrument every stage."""
    print("\n=== STAGE 1: Model Forward Pass ===")
    with torch.no_grad():
        out = model(features, depth_t, depth_grads_t, depth_full=depth_full_t)

    logits = out["semantic"]
    pred_center = out["center"]
    pred_offset = out["offset"]
    print(f"  Semantic logits: {logits.shape}")
    print(f"  Pred center: {pred_center.shape}, min={pred_center.min():.6f}, max={pred_center.max():.6f}, mean={pred_center.mean():.6f}")
    print(f"  Pred offset: {pred_offset.shape}, range=[{pred_offset.min():.2f}, {pred_offset.max():.2f}]")

    # Check center peaks at native 128x256 resolution
    center_128 = pred_center[0, 0].cpu().numpy()  # (128, 256)
    print(f"\n  === Center Heatmap at 128x256 ===")
    for thresh in [0.5, 0.3, 0.2, 0.1, 0.05, 0.03]:
        n = np.sum(center_128 > thresh)
        print(f"    Pixels > {thresh}: {n}")

    # NMS at 128x256
    center_t_128 = pred_center[0:1]  # (1, 1, 128, 256)
    for nms_k in [3, 5, 7]:
        pooled = F.max_pool2d(center_t_128, nms_k, stride=1, padding=nms_k // 2)
        nms_mask = (center_t_128 == pooled).squeeze().cpu().numpy()
        for thresh in [0.1, 0.05, 0.03]:
            peaks = (center_128 > thresh) & nms_mask
            n_peaks = peaks.sum()
            if n_peaks > 0:
                peak_vals = center_128[peaks]
                print(f"    NMS kernel={nms_k}, thresh={thresh}: {n_peaks} peaks (max={peak_vals.max():.3f}, mean={peak_vals.mean():.3f})")
            else:
                print(f"    NMS kernel={nms_k}, thresh={thresh}: 0 peaks")

    print(f"\n=== STAGE 2: Upsampling to 512x1024 ===")
    H, W = 512, 1024

    # Semantic prediction at 512x1024
    pred_27 = logits.argmax(dim=1)[0].cpu().numpy()  # 128x256
    pred_sem = np.array(Image.fromarray(pred_27.astype(np.uint8)).resize((W, H), Image.NEAREST))
    thing_mask_sem = np.isin(pred_sem, list(_THING_IDS))
    print(f"  Semantic pred: {pred_sem.shape}, unique classes: {np.unique(pred_sem)}")
    print(f"  Thing pixels (from semantic): {thing_mask_sem.sum()} / {H*W} ({100*thing_mask_sem.sum()/(H*W):.1f}%)")
    for cls in sorted(_THING_IDS):
        n = np.sum(pred_sem == cls)
        if n > 0:
            from mbps_pytorch.generate_depth_guided_instances import CS_NAMES
            print(f"    Class {cls} ({CS_NAMES.get(cls, '?')}): {n} pixels")

    # Upsample center/offset
    center_up = F.interpolate(pred_center, size=(H, W), mode="bilinear", align_corners=False)
    offset_up = F.interpolate(pred_offset, size=(H, W), mode="bilinear", align_corners=False)
    scale_h = H / logits.shape[2]  # 512/128 = 4
    scale_w = W / logits.shape[3]  # 1024/256 = 4
    offset_up[0, 0] *= scale_h
    offset_up[0, 1] *= scale_w
    print(f"  Scale factors: h={scale_h}, w={scale_w}")
    print(f"  Upsampled center: {center_up.shape}, max={center_up.max():.6f}")
    print(f"  Upsampled offset: {offset_up.shape}, range=[{offset_up.min():.2f}, {offset_up.max():.2f}]")

    # Check center peaks at 512x1024 WITH thing_mask filter
    center_512 = center_up[0, 0].cpu().numpy()  # (512, 1024)
    center_t_512 = center_up[0:1]  # (1, 1, 512, 1024)
    pooled_512 = F.max_pool2d(center_t_512, 7, stride=1, padding=3)
    nms_mask_512 = (center_t_512 == pooled_512).squeeze().cpu().numpy()

    print(f"\n  === Center Peaks at 512x1024 (before thing_mask filter) ===")
    for thresh in [0.1, 0.05, 0.03]:
        peaks_no_filter = (center_512 > thresh) & nms_mask_512
        n = peaks_no_filter.sum()
        if n > 0:
            peak_vals = center_512[peaks_no_filter]
            print(f"    thresh={thresh}: {n} peaks (max={peak_vals.max():.3f})")
        else:
            print(f"    thresh={thresh}: 0 peaks")

    print(f"\n  === Center Peaks at 512x1024 (WITH thing_mask filter) ===")
    for thresh in [0.1, 0.05, 0.03]:
        peaks_with_filter = (center_512 > thresh) & nms_mask_512 & thing_mask_sem
        n = peaks_with_filter.sum()
        if n > 0:
            peak_vals = center_512[peaks_with_filter]
            print(f"    thresh={thresh}: {n} peaks (max={peak_vals.max():.3f})")
            # Show location of peaks
            py, px = np.where(peaks_with_filter)
            for j in range(min(5, len(py))):
                cls_at_peak = pred_sem[py[j], px[j]]
                print(f"      Peak {j}: ({py[j]}, {px[j]}) val={center_512[py[j], px[j]]:.3f} class={cls_at_peak} ({CS_NAMES.get(cls_at_peak, '?')})")
        else:
            print(f"    thresh={thresh}: 0 peaks")

    # How many peaks get FILTERED OUT by thing_mask?
    for thresh in [0.1, 0.05, 0.03]:
        peaks_all = (center_512 > thresh) & nms_mask_512
        peaks_thing = peaks_all & thing_mask_sem
        n_lost = peaks_all.sum() - peaks_thing.sum()
        print(f"    thresh={thresh}: {peaks_all.sum()} total → {peaks_thing.sum()} in thing_mask (lost {n_lost})")

    print(f"\n=== STAGE 3: Full Panoptic Inference ===")
    # Build sem_19 (same as eval code)
    sem_19 = torch.zeros(19, H, W)
    for c in range(19):
        sem_19[c] = torch.from_numpy((pred_sem == c).astype(np.float32))

    center_i = center_up[0].cpu()  # (1, H, W)
    offset_i = offset_up[0].cpu()  # (2, H, W)

    # Call the actual inference function
    pred_pan, pred_segments = panoptic_inference_center_offset(
        sem_19, center_i, offset_i, _THING_IDS,
        center_threshold=0.1, nms_kernel=7, min_area=50,
    )
    if isinstance(pred_pan, torch.Tensor):
        pred_pan = pred_pan.numpy()

    n_stuff = sum(1 for sid, cls in pred_segments.items() if cls not in _THING_IDS)
    n_things = sum(1 for sid, cls in pred_segments.items() if cls in _THING_IDS)
    print(f"  Total segments: {len(pred_segments)} (stuff={n_stuff}, things={n_things})")
    for sid, cls in sorted(pred_segments.items()):
        area = (pred_pan == sid).sum()
        kind = "THING" if cls in _THING_IDS else "stuff"
        print(f"    Seg {sid}: class={cls} ({CS_NAMES.get(cls, '?')}), area={area}, type={kind}")

    # Also try with lower thresholds
    for thresh in [0.05, 0.03]:
        pred_pan2, pred_segs2 = panoptic_inference_center_offset(
            sem_19, center_i, offset_i, _THING_IDS,
            center_threshold=thresh, nms_kernel=7, min_area=50,
        )
        n_things2 = sum(1 for sid, cls in pred_segs2.items() if cls in _THING_IDS)
        print(f"\n  With thresh={thresh}: {n_things2} thing segments")
        for sid, cls in sorted(pred_segs2.items()):
            if cls in _THING_IDS:
                area = (pred_pan2 == sid).sum() if isinstance(pred_pan2, np.ndarray) else (pred_pan2.numpy() == sid).sum()
                print(f"    Seg {sid}: class={cls} ({CS_NAMES.get(cls, '?')}), area={area}")

    # Also try with NMS at 128x256 BEFORE upsampling (alternative approach)
    print(f"\n=== STAGE 4: Offset Voting Analysis ===")
    # Check offset accuracy for thing pixels
    offset_512 = offset_up[0].cpu().numpy()  # (2, 512, 1024)
    target_offset_512 = np.zeros((2, H, W), dtype=np.float32)

    # Compare predicted vs target offset at 128x256
    pred_off_128 = pred_offset[0].cpu().numpy()  # (2, 128, 256)
    tc = target_center  # (128, 256)
    to = target_offset  # (2, 128, 256)
    valid_mask = tc > 0.01
    if valid_mask.sum() > 0:
        pred_dy = pred_off_128[0][valid_mask]
        pred_dx = pred_off_128[1][valid_mask]
        target_dy = to[0][valid_mask]
        target_dx = to[1][valid_mask]
        err_dy = np.abs(pred_dy - target_dy)
        err_dx = np.abs(pred_dx - target_dx)
        print(f"  Offset error at valid pixels ({valid_mask.sum()} pixels):")
        print(f"    dy: mean={err_dy.mean():.2f}, median={np.median(err_dy):.2f}, max={err_dy.max():.2f}")
        print(f"    dx: mean={err_dx.mean():.2f}, median={np.median(err_dx):.2f}, max={err_dx.max():.2f}")
        print(f"    L2 error: mean={np.sqrt(err_dy**2 + err_dx**2).mean():.2f}")

    # Check where center peaks are relative to target centers
    print(f"\n=== STAGE 5: Peak Location Accuracy ===")
    tc_peaks_y, tc_peaks_x = np.where(tc > 0.8)  # Target center peaks
    print(f"  Target center peaks (>0.8): {len(tc_peaks_y)}")
    for i in range(min(5, len(tc_peaks_y))):
        y, x = tc_peaks_y[i], tc_peaks_x[i]
        pred_val = center_128[y, x]
        print(f"    Target peak at ({y}, {x}): target_val={tc[y,x]:.3f}, pred_val={pred_val:.3f}")

    # Find predicted peaks and their nearest target peaks
    nms_128 = F.max_pool2d(pred_center[0:1], 7, stride=1, padding=3)
    nms_mask_128 = (pred_center[0:1] == nms_128).squeeze().cpu().numpy()
    pred_peaks = (center_128 > 0.05) & nms_mask_128
    pp_y, pp_x = np.where(pred_peaks)
    print(f"  Predicted peaks (>0.05, NMS=7): {len(pp_y)}")
    for i in range(min(10, len(pp_y))):
        y, x = pp_y[i], pp_x[i]
        # Find nearest target peak
        if len(tc_peaks_y) > 0:
            dists = np.sqrt((tc_peaks_y - y)**2 + (tc_peaks_x - x)**2)
            nearest_idx = dists.argmin()
            nearest_dist = dists[nearest_idx]
            print(f"    Pred peak at ({y}, {x}): val={center_128[y,x]:.3f}, "
                  f"nearest target peak at ({tc_peaks_y[nearest_idx]}, {tc_peaks_x[nearest_idx]}) dist={nearest_dist:.1f}")
        else:
            print(f"    Pred peak at ({y}, {x}): val={center_128[y,x]:.3f}, NO target peaks")


if __name__ == "__main__":
    model = load_model()
    features, depth_t, depth_grads_t, depth_full_t, city, stem, target_center, target_offset = load_sample_data()
    diagnose_inference(model, features, depth_t, depth_grads_t, depth_full_t, target_center, target_offset)
