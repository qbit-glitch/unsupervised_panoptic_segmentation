#!/usr/bin/env python3
"""
Complete Option 1B evaluation: UNet P2-B semantics + depth-guided instances → PQ

Uses:
- Pre-computed DINOv2 features (dinov2_features/)
- Pre-computed SPIdepth depth maps (depth_spidepth/)
- Trained UNet P2-B attention checkpoint (unet_p2b_attention/best.pth)
- Pre-computed depth-guided instances (pseudo_instance_spidepth/)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from collections import defaultdict

REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT))

PATCH_H, PATCH_W = 32, 64  # DINOv2 ViT-B/14 at 448x896 input

# Cityscapes: label ID → train ID mapping
_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}
_THING_TRAIN_IDS = {11, 12, 13, 14, 15, 16, 17, 18}  # person, rider, car, truck, bus, train, motorcycle, bicycle
_STUFF_TRAIN_IDS = set(range(19)) - _THING_TRAIN_IDS

# LUT for k=80 cluster → trainID (uses kmeans_centroids.npz)
def load_cluster_lut():
    centroids = REPO_ROOT / "datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz"
    if centroids.exists():
        data = np.load(centroids)
        c2c = data["cluster_to_class"]
        lut = np.full(256, 255, dtype=np.uint8)
        for cid in range(len(c2c)):
            lut[cid] = int(c2c[cid])
        return lut
    return None


def sobel_gradients(depth_2d):
    """depth_2d: (H, W) → (2, H, W)"""
    d = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0).float()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(d, kx, padding=1).squeeze()
    grad_y = F.conv2d(d, ky, padding=1).squeeze()
    return torch.stack([grad_x, grad_y], dim=0).numpy()


def load_unet(device):
    from mbps_pytorch.refine_net import DepthGuidedUNet
    ckpt_path = REPO_ROOT / "checkpoints/unet_p2b_attention/best.pth"
    model = DepthGuidedUNet(
        num_classes=19,
        feature_dim=768,
        bridge_dim=192,
        num_bottleneck_blocks=2,
        skip_dim=32,
        coupling_strength=0.1,
        gradient_checkpointing=False,
        rich_skip=True,
        num_final_blocks=1,
        num_decoder_stages=2,
        block_type="attention",
        window_size=8,
        num_heads=4,
    )
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} — {missing[:3]}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} — {unexpected[:3]}")
    model = model.to(device).eval()
    print(f"✓ UNet P2-B loaded from {ckpt_path}")
    return model


def run_unet_inference(model, device, output_dir):
    """Run UNet on all 500 val images using pre-computed features."""
    from mbps_pytorch.refine_net import DepthGuidedUNet  # noqa

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_root = REPO_ROOT / "datasets/cityscapes/dinov2_features/val"
    depth_root = REPO_ROOT / "datasets/cityscapes/depth_spidepth/val"
    img_root = REPO_ROOT / "datasets/cityscapes/leftImg8bit/val"

    # Collect all val entries
    entries = []
    for city in sorted(img_root.iterdir()):
        if not city.is_dir():
            continue
        for img_path in sorted(city.glob("*_leftImg8bit.png")):
            stem = img_path.stem.replace("_leftImg8bit", "")
            entries.append((city.name, stem))

    print(f"  Running UNet on {len(entries)} val images...")

    with torch.no_grad():
        for city, stem in tqdm(entries, desc="UNet inference"):
            out_path = output_dir / f"{stem}_sem.png"
            if out_path.exists():
                continue

            # Load DINOv2 features: (2048, 768) → (768, 32, 64)
            feat_path = feat_root / city / f"{stem}_leftImg8bit.npy"
            if not feat_path.exists():
                print(f"  WARNING: features missing for {stem}")
                np.save(str(output_dir / f"{stem}_sem_MISSING"), np.zeros(1))
                continue
            features = np.load(str(feat_path)).astype(np.float32)
            features = features.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)  # (768, 32, 64)

            # Load depth: → (1, 32, 64)
            depth_path = depth_root / city / f"{stem}.npy"
            if not depth_path.exists():
                depth_np = np.zeros((1, 512, 1024), dtype=np.float32)
            else:
                depth_np = np.load(str(depth_path)).astype(np.float32)  # (512, 1024)

            depth_full = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 1024)
            depth_patch = F.interpolate(depth_full, size=(PATCH_H, PATCH_W),
                                        mode="bilinear", align_corners=False).squeeze(0)  # (1, 32, 64)
            depth_grads_np = sobel_gradients(depth_patch.squeeze(0).numpy())  # (2, 32, 64)

            # Tensors on device
            feat_t = torch.from_numpy(features).unsqueeze(0).to(device)   # (1, 768, 32, 64)
            depth_t = depth_patch.unsqueeze(0).to(device)                 # (1, 1, 32, 64)
            grads_t = torch.from_numpy(depth_grads_np).unsqueeze(0).to(device)  # (1, 2, 32, 64)
            depth_full_t = depth_full.to(device)                          # (1, 1, 512, 1024)

            logits = model(feat_t, depth_t, grads_t, depth_full=depth_full_t)  # (1, 19, 128, 256)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)      # (128, 256)

            # Save as PNG (trainID space, 0-18)
            Image.fromarray(pred).save(str(out_path))

    count = len(list(output_dir.glob("*_sem.png")))
    print(f"  ✓ Semantic predictions saved: {count} images")
    return output_dir


def evaluate_full(sem_dir, inst_dir, gt_dir, eval_hw=(512, 1024)):
    """Evaluate panoptic predictions: PQ, PQ_stuff, PQ_things, SQ, RQ, mIoU."""
    H, W = eval_hw
    num_cls = 19

    confusion = np.zeros((num_cls, num_cls), dtype=np.int64)
    tp = np.zeros(num_cls)
    fp = np.zeros(num_cls)
    fn = np.zeros(num_cls)
    iou_sum = np.zeros(num_cls)

    sem_dir = Path(sem_dir)
    inst_dir = Path(inst_dir)
    gt_dir = Path(gt_dir)

    sem_files = sorted(sem_dir.glob("*_sem.png"))
    if not sem_files:
        print(f"  No semantic predictions found in {sem_dir}!")
        return None

    print(f"  Evaluating {len(sem_files)} images...")

    for sem_path in tqdm(sem_files, desc="Evaluating"):
        stem = sem_path.stem.replace("_sem", "")

        # Extract city from stem (e.g., frankfurt_000000_000294 → frankfurt)
        city = "_".join(stem.split("_")[:1]) if stem.startswith("frankfurt") \
            else "_".join(stem.split("_")[:1])
        # More robust: check all possible cities
        for c in ["frankfurt", "lindau", "munster"]:
            if stem.startswith(c):
                city = c
                break

        # Load semantic prediction (128×256 trainID) → upsample to 512×1024
        sem_patch = np.array(Image.open(str(sem_path)))  # (128, 256) uint8
        pred_sem = np.array(
            Image.fromarray(sem_patch).resize((W, H), Image.NEAREST)
        )  # (512, 1024)

        # Load depth-guided instances
        # File naming: {stem}_instance.png in city subdir
        inst_path = inst_dir / city / f"{stem}_instance.png"
        if not inst_path.exists():
            # Try flat
            inst_path = inst_dir / f"{stem}_instance.png"
            if not inst_path.exists():
                # Try without city prefix
                inst_files = list(inst_dir.glob(f"**/{stem}_instance.png"))
                if inst_files:
                    inst_path = inst_files[0]
                else:
                    inst_path = None

        if inst_path is not None and inst_path.exists():
            instance_map = np.array(Image.open(str(inst_path)))  # (512, 1024) uint16
        else:
            # Fallback: connected components
            instance_map = None

        # Load GT
        gt_path = gt_dir / city / f"{stem}_gtFine_labelIds.png"
        if not gt_path.exists():
            continue
        gt_raw = np.array(Image.open(str(gt_path)))
        gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
        for raw_id, tid in _CS_ID_TO_TRAIN.items():
            gt_sem[gt_raw == raw_id] = tid

        # Resize GT if needed
        if gt_sem.shape != (H, W):
            gt_sem = np.array(Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

        # mIoU accumulation
        valid = (pred_sem < num_cls) & (gt_sem < num_cls)
        np.add.at(confusion,
                  (gt_sem[valid].astype(int), pred_sem[valid].astype(int)),
                  1)

        # Instance-aware PQ evaluation
        # Build predicted panoptic: stuff=class_id, things=class_id*1000+inst_id
        pan_pred = pred_sem.astype(np.uint32)
        if instance_map is not None:
            inst_resized = np.array(
                Image.fromarray(instance_map.astype(np.uint16)).resize((W, H), Image.NEAREST)
            ).astype(np.uint32)
        else:
            inst_resized = None

        # For things: assign instance IDs
        for cls_id in _THING_TRAIN_IDS:
            mask = pred_sem == cls_id
            if not mask.any():
                continue
            if inst_resized is not None and inst_resized.any():
                # Use pre-computed instances within this class mask
                unique_insts = np.unique(inst_resized[mask])
                for inst_id in unique_insts:
                    if inst_id == 0:
                        continue
                    inst_mask = mask & (inst_resized == inst_id)
                    pan_pred[inst_mask] = cls_id * 1000 + inst_id
            else:
                # Connected components fallback
                labeled, n_comp = ndimage.label(mask)
                for inst_id in range(1, n_comp + 1):
                    inst_mask = labeled == inst_id
                    if inst_mask.sum() >= 50:  # min area
                        pan_pred[inst_mask] = cls_id * 1000 + inst_id

        # Build GT panoptic
        gt_inst_path = gt_dir / city / f"{stem}_gtFine_instanceIds.png"
        if not gt_inst_path.exists():
            continue
        gt_inst_raw = np.array(Image.open(str(gt_inst_path))).astype(np.uint32)
        if gt_inst_raw.shape != (H, W):
            gt_inst_raw = np.array(
                Image.fromarray(gt_inst_raw.astype(np.uint16)).resize((W, H), Image.NEAREST)
            ).astype(np.uint32)

        pan_gt = gt_sem.astype(np.uint32)
        for cls_id in _THING_TRAIN_IDS:
            mask = gt_sem == cls_id
            if not mask.any():
                continue
            # GT instance IDs from instanceIds.png: floor(id/1000)=labelId, id%1000=inst
            raw_insts = gt_inst_raw[mask]
            unique_raw = np.unique(raw_insts)
            for raw_inst_id in unique_raw:
                inst_id_within = raw_inst_id % 1000
                if inst_id_within == 0:
                    continue
                inst_mask = mask & (gt_inst_raw == raw_inst_id)
                pan_gt[inst_mask] = cls_id * 1000 + inst_id_within

        # PQ matching (IoU > 0.5)
        pred_segs = np.unique(pan_pred)
        gt_segs = np.unique(pan_gt)

        gt_seg_sizes = {g: (pan_gt == g).sum() for g in gt_segs}
        pred_seg_sizes = {p: (pan_pred == p).sum() for p in pred_segs}

        for cls_id in range(num_cls):
            # Get all GT and pred segments for this class
            if cls_id in _STUFF_TRAIN_IDS:
                gt_ids = [cls_id] if cls_id in gt_segs else []
                pred_ids = [cls_id] if cls_id in pred_segs else []
            else:
                gt_ids = [g for g in gt_segs if g // 1000 == cls_id and g % 1000 > 0]
                pred_ids = [p for p in pred_segs if p // 1000 == cls_id and p % 1000 > 0]

            matched_gt = set()
            matched_pred = set()

            for p_id in pred_ids:
                p_mask = pan_pred == p_id
                best_iou = 0.0
                best_g_id = None
                for g_id in gt_ids:
                    g_mask = pan_gt == g_id
                    intersection = (p_mask & g_mask).sum()
                    if intersection == 0:
                        continue
                    union = p_mask.sum() + g_mask.sum() - intersection
                    iou = intersection / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_g_id = g_id

                if best_iou > 0.5 and best_g_id not in matched_gt:
                    tp[cls_id] += 1
                    iou_sum[cls_id] += best_iou
                    matched_gt.add(best_g_id)
                    matched_pred.add(p_id)
                else:
                    fp[cls_id] += 1

            for g_id in gt_ids:
                if g_id not in matched_gt:
                    fn[cls_id] += 1

    # Compute metrics
    # mIoU
    per_class_iou = np.zeros(num_cls)
    for c in range(num_cls):
        tp_c = confusion[c, c]
        fp_c = confusion[:, c].sum() - tp_c
        fn_c = confusion[c, :].sum() - tp_c
        denom = tp_c + fp_c + fn_c
        if denom > 0:
            per_class_iou[c] = tp_c / denom
    valid_classes = [c for c in range(num_cls) if confusion[c, :].sum() > 0]
    miou = per_class_iou[valid_classes].mean() * 100

    # PQ, SQ, RQ per class
    pq_per_cls = np.zeros(num_cls)
    sq_per_cls = np.zeros(num_cls)
    rq_per_cls = np.zeros(num_cls)
    for c in range(num_cls):
        denom = tp[c] + 0.5 * fp[c] + 0.5 * fn[c]
        if tp[c] > 0:
            sq_per_cls[c] = iou_sum[c] / tp[c]
            rq_per_cls[c] = tp[c] / denom if denom > 0 else 0
            pq_per_cls[c] = sq_per_cls[c] * rq_per_cls[c]

    present = [c for c in range(num_cls) if tp[c] + fn[c] > 0]
    stuff_present = [c for c in present if c in _STUFF_TRAIN_IDS]
    things_present = [c for c in present if c in _THING_TRAIN_IDS]

    pq = pq_per_cls[present].mean() * 100 if present else 0
    pq_stuff = pq_per_cls[stuff_present].mean() * 100 if stuff_present else 0
    pq_things = pq_per_cls[things_present].mean() * 100 if things_present else 0
    sq = sq_per_cls[present].mean() * 100 if present else 0
    rq = rq_per_cls[present].mean() * 100 if present else 0

    return {
        "PQ": pq, "PQ_stuff": pq_stuff, "PQ_things": pq_things,
        "SQ": sq, "RQ": rq, "mIoU": miou,
        "pq_per_cls": pq_per_cls,
    }


CS_TRAINID_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_base = Path("/tmp/unet_cups_final_eval")
    output_base.mkdir(parents=True, exist_ok=True)

    sem_dir = output_base / "semantics"
    inst_dir = REPO_ROOT / "datasets/cityscapes/pseudo_instance_spidepth/val"
    gt_dir = REPO_ROOT / "datasets/cityscapes/gtFine/val"

    print("\n" + "=" * 70)
    print("Option 1B: UNet P2-B Semantics + Depth-Guided Instances → PQ")
    print("=" * 70)

    # Step 1: UNet inference
    print("\n[1/2] UNet P2-B semantic inference...")
    model = load_unet(device)
    run_unet_inference(model, device, sem_dir)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 2: Evaluate
    print("\n[2/2] Panoptic evaluation...")
    results = evaluate_full(sem_dir, inst_dir, gt_dir, eval_hw=(512, 1024))

    if results is None:
        print("✗ Evaluation failed — no predictions found")
        return 1

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  PQ        = {results['PQ']:.2f}")
    print(f"  PQ_stuff  = {results['PQ_stuff']:.2f}")
    print(f"  PQ_things = {results['PQ_things']:.2f}")
    print(f"  SQ        = {results['SQ']:.2f}")
    print(f"  RQ        = {results['RQ']:.2f}")
    print(f"  mIoU      = {results['mIoU']:.2f}%")
    print(f"\n  CUPS CVPR 2025 baseline: PQ=27.8")
    diff = results['PQ'] - 27.8
    print(f"  Our result vs CUPS: {diff:+.2f} PQ")
    print("=" * 70)

    # Per-class breakdown
    print("\nPer-class PQ:")
    for c, name in enumerate(CS_TRAINID_NAMES):
        tag = "(things)" if c in _THING_TRAIN_IDS else "(stuff) "
        print(f"  {name:<20} {tag}  PQ={results['pq_per_cls'][c]*100:.2f}")

    # Save results
    import json
    results_save = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in results.items()}
    out_json = output_base / "results.json"
    with open(str(out_json), "w") as f:
        json.dump(results_save, f, indent=2)
    print(f"\nResults saved to {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
