#!/usr/bin/env python3
"""Merged evaluation: UNet P2-B semantics + CUPS Stage-3 instances.

Uses CUPS's own PanopticQualitySemanticMatching metric with Hungarian matching.

Strategy:
1. Pre-compute UNet predictions for all 500 val images (trainID-19 → CAUSE-27)
2. Run CUPS model validation, replacing stuff semantics with UNet predictions
3. Use CUPS's metric for PQ computation

Expected: PQ_stuff ~35, PQ_things ~25-27, combined PQ ~29+

Usage (on remote):
    export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:$PYTHONPATH
    cd /media/santosh/Kuldeep/panoptic_segmentation
    python -u scripts/eval_merged_unet_cups.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "refs" / "cups"))

import cups
from cups.data import (
    CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES, CITYSCAPES_CLASSNAMES,
    CityscapesPanopticValidation, collate_function_validation,
)
from cups.model.model import prediction_to_standard_format
from cups.metrics.panoptic_quality import PanopticQualitySemanticMatching
from torch.utils.data import DataLoader

# Disable wandb
os.environ["WANDB_MODE"] = "disabled"

# UNet model
from mbps_pytorch.refine_net import DepthGuidedUNet

# Constants
PATCH_H, PATCH_W = 32, 64

# trainID-19 → CAUSE-27 mapping
TRAINID_TO_CAUSE27_LUT = np.full(256, 255, dtype=np.uint8)
for tid, c27 in {
    0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 10, 6: 12, 7: 13,
    8: 14, 9: 15, 10: 16, 11: 17, 12: 18, 13: 19, 14: 20,
    15: 21, 16: 24, 17: 25, 18: 26,
}.items():
    TRAINID_TO_CAUSE27_LUT[tid] = c27

# Sobel kernels for depth gradients
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        dtype=torch.float32).reshape(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        dtype=torch.float32).reshape(1, 1, 3, 3)


def compute_sobel(depth_2d_np):
    """Compute Sobel gradients. depth_2d: (H, W) numpy → (2, H, W) numpy."""
    d = torch.from_numpy(depth_2d_np).unsqueeze(0).unsqueeze(0).float()
    grad_x = F.conv2d(d, SOBEL_X, padding=1).squeeze()
    grad_y = F.conv2d(d, SOBEL_Y, padding=1).squeeze()
    return torch.stack([grad_x, grad_y], dim=0).numpy()


def load_unet_predictions(device):
    """Load UNet P2-B model and generate predictions for all 500 val images.

    Returns dict: {"city/stem" → CAUSE-27 prediction numpy (128, 256)}
    """
    cityscapes_root = str(REPO_ROOT / "datasets" / "cityscapes")
    checkpoint_path = str(REPO_ROOT / "checkpoints" / "unet_p2b_attention" / "best.pth")

    # Build UNet model (matching config.json)
    unet = DepthGuidedUNet(
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

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    unet.load_state_dict(state_dict, strict=False)
    unet = unet.to(device)
    unet.eval()
    print(f"UNet P2-B loaded: {sum(p.numel() for p in unet.parameters())} params")

    # Find all val images
    img_dir = os.path.join(cityscapes_root, "leftImg8bit", "val")
    entries = []
    for city in sorted(os.listdir(img_dir)):
        city_path = os.path.join(img_dir, city)
        if not os.path.isdir(city_path):
            continue
        for fname in sorted(os.listdir(city_path)):
            if not fname.endswith("_leftImg8bit.png"):
                continue
            stem = fname.replace("_leftImg8bit.png", "")
            entries.append({"stem": stem, "city": city})

    print(f"Processing {len(entries)} val images with UNet...")
    predictions = {}

    with torch.no_grad():
        for entry in tqdm(entries, desc="UNet inference"):
            stem, city = entry["stem"], entry["city"]
            name = f"{city}/{stem}"

            # Load DINOv2 features: (2048, 768) → (1, 768, 32, 64)
            feat_path = os.path.join(
                cityscapes_root, "dinov2_features", "val", city,
                f"{stem}_leftImg8bit.npy",
            )
            features = np.load(feat_path).astype(np.float32)
            features = features.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)
            features = torch.from_numpy(features).unsqueeze(0).to(device)

            # Load depth: (512, 1024) → patch (1, 1, 32, 64) + full (1, 1, 512, 1024)
            depth_path = os.path.join(
                cityscapes_root, "depth_spidepth", "val", city, f"{stem}.npy",
            )
            depth_full_np = np.load(depth_path)
            depth_full = torch.from_numpy(depth_full_np).float().unsqueeze(0).unsqueeze(0).to(device)
            depth_patch = F.interpolate(
                depth_full, size=(PATCH_H, PATCH_W), mode="bilinear", align_corners=False,
            )

            # Sobel gradients at patch resolution
            depth_patch_np = depth_patch.squeeze().cpu().numpy()
            depth_grads = compute_sobel(depth_patch_np)
            depth_grads = torch.from_numpy(depth_grads).unsqueeze(0).to(device)

            # Forward pass
            logits = unet(features, depth_patch, depth_grads, depth_full)
            pred_trainid = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_cause27 = TRAINID_TO_CAUSE27_LUT[pred_trainid]
            predictions[name] = pred_cause27

    print(f"UNet predictions: {len(predictions)} images")

    # Free GPU memory
    del unet
    torch.cuda.empty_cache()

    return predictions


def main():
    device = torch.device("cuda:0")

    # Step 1: Generate UNet predictions
    unet_preds = load_unet_predictions(device)

    # Step 2: Load CUPS config & model
    config = cups.get_default_config(
        experiment_config_file=str(REPO_ROOT / "refs/cups/configs/train_cityscapes_vitb.yaml")
    )
    config.defrost()
    config.MODEL.CHECKPOINT = str(
        REPO_ROOT / "experiments/experiments/cups_vitb_k80_stage3"
        / "Unsupervised Panoptic Segmentation/sh17g3sl/checkpoints"
        / "ups_checkpoint_step=000100.ckpt"
    )
    config.freeze()

    # Extract pseudo classes from checkpoint
    ckpt = torch.load(config.MODEL.CHECKPOINT, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    thing_pseudo = tuple(hp.get("thing_pseudo_classes", tuple(range(65, 80))))
    stuff_pseudo = tuple(hp.get("stuff_pseudo_classes", tuple(range(0, 65))))
    print(f"thing_pseudo: {thing_pseudo}")
    print(f"stuff_pseudo: {stuff_pseudo[:3]}...{stuff_pseudo[-2:]}")
    del ckpt

    # Build CUPS model
    model = cups.build_model_pseudo(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=thing_pseudo,
        stuff_pseudo_classes=stuff_pseudo,
        class_weights=None,
        copy_paste_augmentation=torch.nn.Identity(),
        resolution_jitter_augmentation=torch.nn.Identity(),
        photometric_augmentation=torch.nn.Identity(),
        use_tta=False,
        class_names=CITYSCAPES_CLASSNAMES,
        classes_mask=None,
    )
    model = model.to(device)
    model.eval()
    print("CUPS model loaded")

    # Step 3: Create PQ metric
    # Merged prediction space:
    # - Stuff: CAUSE-27 values (0-16) from UNet
    # - Things: k80 cluster IDs (65-79) from CUPS
    metric = PanopticQualitySemanticMatching(
        things=CITYSCAPES_THING_CLASSES,
        stuffs=CITYSCAPES_STUFF_CLASSES,
        num_clusters=80,
        things_prototype=set(thing_pseudo),
        stuffs_prototype=set(range(17)),
        cache_device="cpu",
        sync_on_compute=False,
        dist_sync_on_step=False,
    )

    # Also create a CUPS-only metric for comparison
    metric_cups_only = PanopticQualitySemanticMatching(
        things=CITYSCAPES_THING_CLASSES,
        stuffs=CITYSCAPES_STUFF_CLASSES,
        num_clusters=80,
        things_prototype=set(thing_pseudo),
        stuffs_prototype=set(stuff_pseudo),
        cache_device="cpu",
        sync_on_compute=False,
        dist_sync_on_step=False,
    )

    # Step 4: Validation dataset
    val_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT,
        crop_resolution=(512, 1024),
        num_classes=27,
        resize_scale=0.5,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )
    print(f"Val dataset: {len(val_dataset)} images")

    # Step 5: Run merged validation
    print("\nRunning merged evaluation (UNet stuff + CUPS things)...")
    missing_count = 0

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Merged eval")):
            images, panoptic_labels, image_names = batch

            # Move images to device
            for img_dict in images:
                for k, v in img_dict.items():
                    if isinstance(v, torch.Tensor):
                        img_dict[k] = v.to(device)

            # CUPS forward pass
            predictions = model(images)

            # Convert to standard format [B, H, W, 2]
            panoptic_preds = torch.stack(
                [
                    prediction_to_standard_format(
                        predictions[i]["panoptic_seg"],
                        stuff_classes=stuff_pseudo,
                        thing_classes=thing_pseudo,
                    )
                    for i in range(len(predictions))
                ],
                dim=0,
            )

            # Move to CPU for metric and manipulation
            panoptic_preds = panoptic_preds.cpu()

            # Save a copy for CUPS-only metric
            panoptic_preds_cups = panoptic_preds.clone()

            # Replace stuff semantics with UNet predictions
            H, W = panoptic_preds.shape[1], panoptic_preds.shape[2]

            for i in range(len(predictions)):
                # CUPS names: "frankfurt_000000_000294_leftImg8bit"
                # UNet keys:  "frankfurt/frankfurt_000000_000294"
                raw_name = image_names[i]
                stem = raw_name.replace("_leftImg8bit", "")
                city = stem.split("_")[0]
                name = f"{city}/{stem}"

                if name in unet_preds:
                    unet_pred = unet_preds[name]  # (128, 256) CAUSE-27 uint8

                    # Resize UNet prediction to match CUPS resolution
                    unet_tensor = torch.from_numpy(unet_pred.astype(np.int64))
                    unet_resized = F.interpolate(
                        unet_tensor.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W),
                        mode="nearest",
                    ).squeeze().long()

                    # Replace stuff pixels (where instance_id == 0)
                    stuff_mask = panoptic_preds[i, :, :, 1] == 0
                    panoptic_preds[i, :, :, 0][stuff_mask] = unet_resized[stuff_mask]
                else:
                    missing_count += 1

            # Update both metrics (already on CPU)
            panoptic_labels_cpu = panoptic_labels.cpu()
            metric.update(panoptic_preds, panoptic_labels_cpu)
            metric_cups_only.update(panoptic_preds_cups, panoptic_labels_cpu)

    if missing_count > 0:
        print(f"Warning: {missing_count} images missing UNet predictions")

    # Step 6: Compute results
    print("\nComputing PQ (merged)...")
    (
        pq, sq, rq, pq_t, sq_t, rq_t, pq_s, sq_s, rq_s,
        pq_c, sq_c, rq_c, miou, acc, assignments
    ) = metric.compute()

    print("\nComputing PQ (CUPS only)...")
    (
        pq2, sq2, rq2, pq_t2, sq_t2, rq_t2, pq_s2, sq_s2, rq_s2,
        pq_c2, sq_c2, rq_c2, miou2, acc2, assignments2
    ) = metric_cups_only.compute()

    # Print results
    print(f"\n{'='*70}")
    print(f"CUPS ONLY (baseline, should match ~27.10)")
    print(f"{'='*70}")
    print(f"PQ={pq2.item()*100:.2f} | PQ_stuff={pq_s2.item()*100:.2f} | "
          f"PQ_things={pq_t2.item()*100:.2f} | mIoU={miou2.item()*100:.2f}%")

    print(f"\n{'='*70}")
    print(f"MERGED: UNet P2-B Stuff + CUPS Stage-3 Things")
    print(f"{'='*70}")
    print(f"PQ      = {pq.item()*100:.2f}")
    print(f"SQ      = {sq.item()*100:.2f}")
    print(f"RQ      = {rq.item()*100:.2f}")
    print(f"PQ_things = {pq_t.item()*100:.2f}")
    print(f"SQ_things = {sq_t.item()*100:.2f}")
    print(f"RQ_things = {rq_t.item()*100:.2f}")
    print(f"PQ_stuff  = {pq_s.item()*100:.2f}")
    print(f"SQ_stuff  = {sq_s.item()*100:.2f}")
    print(f"RQ_stuff  = {rq_s.item()*100:.2f}")
    print(f"mIoU    = {miou.item()*100:.2f}%")
    print(f"Acc     = {acc.item()*100:.2f}%")

    # Per-class PQ comparison
    print(f"\n{'='*70}")
    print(f"Per-class PQ comparison:")
    print(f"{'Class':20s}  {'CUPS':>8s}  {'Merged':>8s}  {'Diff':>8s}")
    print(f"{'-'*50}")
    for idx in range(pq_c.shape[0]):
        cname = CITYSCAPES_CLASSNAMES[idx] if idx < len(CITYSCAPES_CLASSNAMES) else f"class_{idx}"
        cups_pq = pq_c2[idx].item() * 100
        merged_pq = pq_c[idx].item() * 100
        diff = merged_pq - cups_pq
        marker = " <<<" if abs(diff) > 5 else ""
        print(f"  {cname:20s}  {cups_pq:7.2f}  {merged_pq:7.2f}  {diff:+7.2f}{marker}")

    print(f"\n{'='*70}")
    print(f"FINAL: PQ={pq.item()*100:.2f} | PQ_stuff={pq_s.item()*100:.2f} | "
          f"PQ_things={pq_t.item()*100:.2f} | mIoU={miou.item()*100:.2f}%")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
