#!/usr/bin/env python3
"""Two-pass merged: UNet P2-B semantics + CUPS Stage-3 instances (LOCAL CPU).

Pass 1: Run CUPS on 500 val images, cache predictions, get cluster→GT assignments.
Pass 2: Remap CUPS thing clusters to GT space, replace stuff with UNet predictions,
        evaluate using torchmetrics PanopticQuality (no matching needed).

Usage:
    export WANDB_MODE=disabled
    export PYTHONPATH=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups:$PYTHONPATH
    cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
    nohup python -u scripts/run_merged_local.py > logs/run_merged_local.log 2>&1 &
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

os.environ["WANDB_MODE"] = "disabled"

# Disable MPS (Apple Metal GPU) — CUPS + DINOv2 on MPS hangs indefinitely.
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

REPO_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
CITYSCAPES_ROOT = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
CHECKPOINT = REPO_ROOT / "checkpoints/cups_vitb_k80_stage3/ups_checkpoint_step=000100.ckpt"
CUPS_CONFIG = REPO_ROOT / "refs/cups/configs/val_cups_stage3_vitb_k80_local.yaml"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "refs" / "cups"))

import cups
from cups.data import (
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_CLASSNAMES,
    CityscapesPanopticValidation,
    collate_function_validation,
)
from cups.model import prediction_to_standard_format
from cups.metrics.panoptic_quality import PanopticQualitySemanticMatching
from cups.augmentation import PhotometricAugmentations, ResolutionJitter
from torchmetrics.detection import PanopticQuality as PanopticQualityTM
from torch.utils.data import DataLoader

from mbps_pytorch.refine_net import DepthGuidedUNet

# CUPS Stage-3 pseudo class split (from checkpoint shapes)
THING_PSEUDO = tuple(range(65, 80))   # 15 thing clusters
STUFF_PSEUDO = tuple(range(0, 65))    # 65 stuff clusters

PATCH_H, PATCH_W = 32, 64

# trainID-19 to CAUSE-27 LUT
TRAINID_TO_CAUSE27_LUT = np.full(256, 255, dtype=np.uint8)
for _tid, _c27 in {
    0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 10, 6: 12, 7: 13,
    8: 14, 9: 15, 10: 16, 11: 17, 12: 18, 13: 19, 14: 20,
    15: 21, 16: 24, 17: 25, 18: 26,
}.items():
    TRAINID_TO_CAUSE27_LUT[_tid] = _c27

# Standard trainID-19 classes in CAUSE-27 space (excludes parking, rail_track,
# guard_rail, bridge, tunnel, polegroup, caravan, trailer)
STANDARD_STUFF_C27 = {0, 1, 4, 5, 6, 10, 12, 13, 14, 15, 16}   # 11 standard stuff
STANDARD_THINGS_C27 = {17, 18, 19, 20, 21, 24, 25, 26}           # 8 standard things

SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        dtype=torch.float32).reshape(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        dtype=torch.float32).reshape(1, 1, 3, 3)


def compute_sobel(depth_2d_np):
    d = torch.from_numpy(depth_2d_np).unsqueeze(0).unsqueeze(0).float()
    gx = F.conv2d(d, SOBEL_X, padding=1).squeeze()
    gy = F.conv2d(d, SOBEL_Y, padding=1).squeeze()
    return torch.stack([gx, gy], dim=0).numpy()


def load_unet_predictions():
    """Run UNet P2-B on all 500 val images.
    Returns dict: {"city/stem" -> CAUSE-27 prediction (128, 256) uint8}
    """
    checkpoint_path = REPO_ROOT / "checkpoints/unet_p2b_attention/best.pth"

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

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    unet.load_state_dict(sd, strict=False)
    unet.train(False)
    print(f"UNet P2-B loaded: {sum(p.numel() for p in unet.parameters()):,} params")

    img_dir = CITYSCAPES_ROOT / "leftImg8bit" / "val"
    entries = []
    for city in sorted(os.listdir(img_dir)):
        city_path = img_dir / city
        if city_path.is_dir():
            for fname in sorted(os.listdir(city_path)):
                if fname.endswith("_leftImg8bit.png"):
                    stem = fname.replace("_leftImg8bit.png", "")
                    entries.append({"stem": stem, "city": city})

    print(f"Processing {len(entries)} val images with UNet...")
    predictions = {}

    with torch.no_grad():
        for entry in tqdm(entries, desc="UNet inference"):
            stem, city = entry["stem"], entry["city"]
            key = f"{city}/{stem}"

            feat_path = (
                CITYSCAPES_ROOT / "dinov2_features" / "val" / city
                / f"{stem}_leftImg8bit.npy"
            )
            features = np.load(str(feat_path)).astype(np.float32)
            features = features.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)
            features = torch.from_numpy(features).unsqueeze(0)

            depth_path = (
                CITYSCAPES_ROOT / "depth_spidepth" / "val" / city / f"{stem}.npy"
            )
            depth_full_np = np.load(str(depth_path))
            depth_full = torch.from_numpy(depth_full_np).float().unsqueeze(0).unsqueeze(0)
            depth_patch = F.interpolate(
                depth_full, size=(PATCH_H, PATCH_W), mode="bilinear", align_corners=False
            )
            depth_grads = torch.from_numpy(
                compute_sobel(depth_patch.squeeze().cpu().numpy())
            ).unsqueeze(0)

            logits = unet(features, depth_patch, depth_grads, depth_full)
            pred_trainid = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            predictions[key] = TRAINID_TO_CAUSE27_LUT[pred_trainid]

    print(f"UNet predictions ready: {len(predictions)} images")
    del unet
    return predictions


def build_cups_model():
    config = cups.get_default_config(
        experiment_config_file=str(CUPS_CONFIG),
        command_line_arguments=[],
    )
    model = cups.build_model_self(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=list(THING_PSEUDO),
        stuff_pseudo_classes=list(STUFF_PSEUDO),
        class_weights=None,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
        mask_refiner=None,
    )
    print(f"Loading CUPS checkpoint: {CHECKPOINT.name}")
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"  Missing  ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)}): {unexpected[:3]}")
    model.to("cpu")
    model.train(False)
    return model, config


def make_cups_metric_27cls():
    """Standard CUPS 27-class CAUSE metric (with Hungarian matching)."""
    return PanopticQualitySemanticMatching(
        things=CITYSCAPES_THING_CLASSES,
        stuffs=CITYSCAPES_STUFF_CLASSES,
        num_clusters=80,
        things_prototype=set(THING_PSEUDO),
        stuffs_prototype=set(STUFF_PSEUDO),
        cache_device="cpu",
        sync_on_compute=False,
        dist_sync_on_step=False,
    )


def unpack16(result):
    """Unpack 16-element tuple from PanopticQualitySemanticMatching.compute()."""
    (pq, sq, rq, pq_c, sq_c, rq_c,
     pq_t, sq_t, rq_t, pq_s, sq_s, rq_s,
     miou, acc, assignments, _pred) = result
    return pq, sq, rq, pq_c, sq_c, rq_c, pq_t, sq_t, rq_t, pq_s, sq_s, rq_s, miou, acc, assignments


def extract_merged_metrics(metric, per_class_result):
    """Extract PQ metrics from PanopticQualityTM.compute() result.

    per_class_result: (27, 3) tensor — rows indexed by continuous_id,
    cols = [pq, sq, rq]. Remaps to CAUSE-27 class ordering.
    """
    c2cause = {cont: cause27 for cause27, cont in metric.cat_id_to_continuous_id.items()}
    pq_c = torch.zeros(27, dtype=per_class_result.dtype)
    for cont_id, cause27_cls in c2cause.items():
        pq_c[cause27_cls] = per_class_result[cont_id, 0]
    pq  = pq_c.mean()
    pq_t = pq_c[sorted(CITYSCAPES_THING_CLASSES)].mean()
    pq_s = pq_c[sorted(CITYSCAPES_STUFF_CLASSES)].mean()
    return pq, pq_t, pq_s, pq_c


def main():
    print("=" * 70)
    print("Two-Pass Merged: UNet P2-B stuff + CUPS Stage-3 things  (LOCAL CPU)")
    print("=" * 70)

    # Step 0: Pre-compute UNet predictions
    unet_preds = load_unet_predictions()

    # Step 1: Build CUPS model + val loader
    cups_model, config = build_cups_model()

    val_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        resize_scale=config.DATA.VAL_SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        num_classes=config.DATA.NUM_CLASSES,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )
    print(f"Val dataset: {len(val_dataset)} images\n")

    # ── PASS 1: Run CUPS, cache predictions + GT labels ──────────────────────
    metric_cups27 = make_cups_metric_27cls()

    all_cups_preds = []   # List of (H, W, 2) int16 tensors (cluster space 0-79)
    all_gt_labels = []    # List of (H, W, 2) int16 tensors
    all_image_keys = []   # For UNet lookup

    print("PASS 1: Running CUPS inference on 500 val images...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="CUPS Pass 1")):
            images, panoptic_labels, image_names = batch

            for img_dict in images:
                for k, v in img_dict.items():
                    if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                        img_dict[k] = v.to("cpu")

            raw_preds = cups_model(images)

            panoptic_preds = torch.stack(
                [
                    prediction_to_standard_format(
                        raw_preds[i]["panoptic_seg"],
                        stuff_classes=STUFF_PSEUDO,
                        thing_classes=THING_PSEUDO,
                    )
                    for i in range(len(raw_preds))
                ],
                dim=0,
            ).cpu()

            lbl = panoptic_labels.cpu()
            metric_cups27.update(panoptic_preds, lbl)

            # Cache as int16 to save memory (values 0-79 fit in int16)
            for i in range(len(raw_preds)):
                all_cups_preds.append(panoptic_preds[i].to(torch.int16))
                all_gt_labels.append(lbl[i].to(torch.int16))
                name = image_names[i]
                stem = name.replace("_leftImg8bit", "")
                city = stem.split("_")[0]
                all_image_keys.append(f"{city}/{stem}")

            if (batch_idx + 1) % 50 == 0:
                print(f"  [{batch_idx + 1:3d}/500]")

    # Free CUPS model memory
    del cups_model

    # Compute CUPS-only metrics + extract assignments
    print("\nComputing CUPS-only metrics (may take a few minutes)...")
    result_cups27 = unpack16(metric_cups27.compute())

    # assignments_cups27[k] = GT CAUSE-27 class for cluster k (0-79)
    assignments_cups27 = result_cups27[-1].long()   # shape [80]

    pq27, _, _, pq_c27, _, _, pq_t27, _, _, pq_s27, _, _, miou27, _, _ = result_cups27

    # 19-class numbers derived post-hoc: pq_c27[i] = PQ for CAUSE-27 class i
    # (continuous ID == CAUSE-27 ID since stuffs={0..16}, things={17..26} are consecutive)
    _s19 = sorted(STANDARD_STUFF_C27)
    _t19 = sorted(STANDARD_THINGS_C27)
    _19  = _s19 + _t19
    pq19_cups   = pq_c27[_19].mean().item() * 100
    pq_s19_cups = pq_c27[_s19].mean().item() * 100
    pq_t19_cups = pq_c27[_t19].mean().item() * 100

    print(f"\n{'='*70}")
    print("CUPS Stage-3 baseline:")
    print(f"  27-class CAUSE : PQ={pq27.item()*100:.2f}  "
          f"PQ_stuff={pq_s27.item()*100:.2f}  PQ_things={pq_t27.item()*100:.2f}  "
          f"mIoU={miou27.item()*100:.2f}%")
    print(f"  19-class std   : PQ={pq19_cups:.2f}  "
          f"PQ_stuff={pq_s19_cups:.2f}  PQ_things={pq_t19_cups:.2f}  (derived from 27-cls)")

    # Free CUPS metric cache (large)
    del metric_cups27

    # ── PASS 2: Build merged predictions, update merged metrics ───────────────
    # Use PanopticQualityTM directly — no matching needed (GT-space predictions),
    # incremental updates (no caching), low memory.
    metric_merged27 = PanopticQualityTM(
        things=CITYSCAPES_THING_CLASSES,
        stuffs=CITYSCAPES_STUFF_CLASSES,
        allow_unknown_preds_category=True,
        return_per_class=True,
        return_sq_and_rq=True,
    )

    print("\nPASS 2: Building merged predictions + evaluating...")
    missing_count = 0

    for idx, (cups_pred_i16, gt_label_i16, key) in enumerate(
        tqdm(
            zip(all_cups_preds, all_gt_labels, all_image_keys),
            total=len(all_cups_preds),
            desc="Merged Pass 2",
        )
    ):
        cups_pred = cups_pred_i16.long()   # (H, W, 2)
        gt_label = gt_label_i16.long()     # (H, W, 2)
        H, W = cups_pred.shape[:2]

        # Remap thing pixels: CUPS cluster (65-79) → GT thing class (17-26)
        # Only modify non-void thing pixels (instance_id > 0).
        merged = cups_pred.clone()
        thing_mask = (merged[..., 1] > 0) & (merged[..., 0] != 255)
        merged[..., 0][thing_mask] = assignments_cups27[merged[..., 0][thing_mask]]

        # Replace stuff pixels (instance_id == 0) with UNet CAUSE-27 prediction
        stuff_mask = cups_pred[..., 1] == 0
        if key in unet_preds:
            unet_pred = unet_preds[key]   # (128, 256) uint8 CAUSE-27
            unet_resized = F.interpolate(
                torch.from_numpy(unet_pred.astype(np.int64))
                .unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode="nearest",
            ).squeeze().long()

            merged[..., 0][stuff_mask] = unet_resized[stuff_mask]

            # Void UNet thing predictions in stuff regions (class > 16 in CAUSE-27)
            thing_in_stuff = stuff_mask & (merged[..., 0] > 16)
            merged[..., 0][thing_in_stuff] = 255
        else:
            # Fallback: keep CUPS stuff remapped to GT space via assignments
            stuff_nv = stuff_mask & (cups_pred[..., 0] != 255)
            merged[..., 0][stuff_nv] = assignments_cups27[cups_pred[..., 0][stuff_nv]]
            missing_count += 1

        merged_b = merged.unsqueeze(0)    # (1, H, W, 2)
        gt_b = gt_label.unsqueeze(0)      # (1, H, W, 2)
        metric_merged27.update(merged_b, gt_b)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1:3d}/500]")

    if missing_count:
        print(f"Warning: {missing_count} images had no UNet prediction (used CUPS fallback)")

    # Compute merged metrics
    print("\nComputing merged metrics...")
    # per_class_result: (27, 3) — rows = continuous_id, cols = [pq, sq, rq]
    per_class_result = metric_merged27.compute()
    pqm27, pq_tm27, pq_sm27, pq_cm27 = extract_merged_metrics(metric_merged27, per_class_result)

    # 19-class merged: derived from per-class tensor indexed by CAUSE-27 class
    pq19_merged   = pq_cm27[_19].mean().item() * 100
    pq_s19_merged = pq_cm27[_s19].mean().item() * 100
    pq_t19_merged = pq_cm27[_t19].mean().item() * 100

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL RESULTS:")
    print(f"\n  27-class CAUSE metric (same as reported CUPS 27.78):")
    print(f"    CUPS Stage-3 only : PQ={pq27.item()*100:.2f}  "
          f"PQ_stuff={pq_s27.item()*100:.2f}  PQ_things={pq_t27.item()*100:.2f}")
    print(f"    Merged (UNet+CUPS): PQ={pqm27.item()*100:.2f}  "
          f"PQ_stuff={pq_sm27.item()*100:.2f}  PQ_things={pq_tm27.item()*100:.2f}")
    delta27 = (pqm27.item() - pq27.item()) * 100
    print(f"    Delta: {delta27:+.2f} ({'BETTER' if delta27 > 0 else 'WORSE'})")

    print(f"\n  19-class standard metric (derived from per-class 27-cls PQ):")
    print(f"    CUPS Stage-3 only : PQ={pq19_cups:.2f}  "
          f"PQ_stuff={pq_s19_cups:.2f}  PQ_things={pq_t19_cups:.2f}")
    print(f"    Merged (UNet+CUPS): PQ={pq19_merged:.2f}  "
          f"PQ_stuff={pq_s19_merged:.2f}  PQ_things={pq_t19_merged:.2f}")
    delta19 = pq19_merged - pq19_cups
    print(f"    Delta: {delta19:+.2f} ({'BETTER' if delta19 > 0 else 'WORSE'})")

    # Per-class breakdown on 27-class metric
    classnames = list(CITYSCAPES_CLASSNAMES)
    print(f"\n  Per-class PQ (27-class CAUSE metric):")
    print(f"  {'Class':22s}  {'CUPS':>8s}  {'Merged':>8s}  {'Diff':>8s}")
    print("  " + "-" * 57)
    for i in range(pq_c27.shape[0]):
        cname = classnames[i] if i < len(classnames) else f"class_{i}"
        c_pq = pq_c27[i].item() * 100
        m_pq = pq_cm27[i].item() * 100
        diff = m_pq - c_pq
        marker = " <<" if abs(diff) > 5 else ""
        print(f"  {cname:22s}  {c_pq:8.2f}  {m_pq:8.2f}  {diff:+8.2f}{marker}")

    print(f"\n{'='*70}")
    print(f"UNet P2-B reported: PQ=28.00 (19-class, own instances)")
    print(f"CUPS Stage-3 reprt: PQ=27.78 (27-class CAUSE metric)")
    print(f"Merged 27-class   : PQ={pqm27.item()*100:.2f}")
    print(f"Merged 19-class   : PQ={pq19_merged:.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
