#!/usr/bin/env python3
"""Verify CUPS Stage-3 PQ using CUPS's own evaluation pipeline.

Uses trainer.validate() → guaranteed to reproduce training PQ (should be ~27.67).
Then runs a simple merged eval with UNet stuff + CUPS things.

Usage (on remote):
    export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:$PYTHONPATH
    cd /media/santosh/Kuldeep/panoptic_segmentation
    python -u scripts/verify_cups_pq.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "refs" / "cups"))

# ---- Config ----
STAGE3_CKPT = str(
    REPO_ROOT / "experiments/experiments/cups_vitb_k80_stage3"
    / "Unsupervised Panoptic Segmentation/sh17g3sl/checkpoints"
    / "ups_checkpoint_step=000100.ckpt"
)
STAGE3_CONFIG = str(REPO_ROOT / "refs/cups/configs/train_self_cityscapes.yaml")
UNET_CKPT = str(REPO_ROOT / "checkpoints/unet_p2b_attention/best.pth")

PATCH_H, PATCH_W = 32, 64

# CAUSE-27 class info
CAUSE27_NAMES = [
    'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
    'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
    'train', 'motorcycle', 'bicycle'
]
CAUSE27_THINGS = set(range(17, 27))
CAUSE27_STUFFS = set(range(0, 17))

# trainID-19 stuff → CAUSE-27
TRAINID_TO_CAUSE27 = {
    0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 10, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
}


def sobel_gradients(depth_2d):
    d = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0).float()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(d, kx, padding=1).squeeze()
    grad_y = F.conv2d(d, ky, padding=1).squeeze()
    return torch.stack([grad_x, grad_y], dim=0).numpy()


def part1_verify_cups_pq():
    """Use CUPS's own trainer.validate() to verify PQ ≈ 27.67."""
    print("=" * 70)
    print("PART 1: Verify CUPS Stage-3 PQ via trainer.validate()")
    print("=" * 70)

    import cups
    from cups.data import (
        CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES, CITYSCAPES_CLASSNAMES,
        CityscapesPanopticValidation, collate_function_validation,
    )
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer

    # Load config (self-training config for Stage-3)
    config = cups.get_default_config(experiment_config_file=STAGE3_CONFIG)
    config.defrost()
    config.MODEL.CHECKPOINT = STAGE3_CKPT
    config.freeze()

    # Extract pseudo classes from checkpoint
    ckpt = torch.load(STAGE3_CKPT, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    thing_pseudo = tuple(hp.get("thing_pseudo_classes", tuple(range(65, 80))))
    stuff_pseudo = tuple(hp.get("stuff_pseudo_classes", tuple(range(0, 65))))
    print(f"  thing_pseudo_classes: {thing_pseudo}")
    print(f"  stuff_pseudo_classes: {stuff_pseudo[:5]}...{stuff_pseudo[-3:]}")
    del ckpt

    # Build model using build_model_self (same as training)
    model = cups.build_model_self(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=thing_pseudo,
        stuff_pseudo_classes=stuff_pseudo,
        class_weights=None,
        class_names=CITYSCAPES_CLASSNAMES,
        classes_mask=None,
        freeze_bn=True,
    )
    model.eval()
    print("  Model loaded via build_model_self")

    # Validation dataset (same params as train_self.py hardcodes)
    val_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
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
    print(f"  Val dataset: {len(val_dataset)} images")

    # Run validation using CUPS's exact pipeline
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    print("\n  Running trainer.validate()...")
    results = trainer.validate(model=model, dataloaders=val_loader)
    print(f"\n  CUPS validation results: {results}")

    # Return the model and pseudo classes for Part 2
    return model, config, thing_pseudo, stuff_pseudo


def part2_merged_eval(cups_model, config, thing_pseudo, stuff_pseudo):
    """Compute merged UNet stuff + CUPS things PQ."""
    print("\n" + "=" * 70)
    print("PART 2: Merged UNet stuff + CUPS things evaluation")
    print("=" * 70)

    from cups.data import (
        CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES,
        CityscapesPanopticValidation, collate_function_validation,
    )
    from cups.model.model import prediction_to_standard_format
    from torch.utils.data import DataLoader

    device = next(cups_model.parameters()).device

    # Val dataset
    val_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=(512, 1024),
        num_classes=27,
        resize_scale=0.5,
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_function_validation, drop_last=False, pin_memory=False,
    )

    # UNet inference
    print("\n  [2a] UNet inference...")
    unet_sem_dir = Path("/tmp/verify_cups_pq/unet_sem")
    unet_sem_dir.mkdir(parents=True, exist_ok=True)
    _run_unet(device, unet_sem_dir)

    # CUPS inference + build cost matrix
    print("\n  [2b] CUPS inference + cost matrix...")
    num_k80 = len(thing_pseudo) + len(stuff_pseudo)
    cost_matrix = np.zeros((num_k80, 27), dtype=np.int64)
    all_data = []

    cups_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="CUPS inference")):
            images, panoptic_labels, image_names = batch
            for img_dict in images:
                for k, v in img_dict.items():
                    if isinstance(v, torch.Tensor):
                        img_dict[k] = v.to(device)

            predictions = cups_model(images)

            for i in range(len(predictions)):
                name = image_names[i]
                std_pred = prediction_to_standard_format(
                    predictions[i]["panoptic_seg"],
                    stuff_classes=stuff_pseudo,
                    thing_classes=thing_pseudo,
                ).cpu().numpy()

                gt = panoptic_labels[i].cpu().numpy()
                gt_sem = gt[:, :, 0]
                gt_inst = gt[:, :, 1]

                pred_sem_k80 = std_pred[:, :, 0]
                valid = (pred_sem_k80 >= 0) & (pred_sem_k80 < num_k80) & (gt_sem >= 0) & (gt_sem < 27)
                if valid.any():
                    np.add.at(cost_matrix, (pred_sem_k80[valid].astype(np.int64), gt_sem[valid].astype(np.int64)), 1)

                # Strip to bare stem (no city prefix, no _leftImg8bit)
                stem = name.split("/")[-1].replace("_leftImg8bit", "")
                all_data.append((stem, std_pred, gt_sem, gt_inst))

                if len(all_data) <= 2:
                    H, W = std_pred.shape[:2]
                    print(f"\n  [DEBUG img {len(all_data)}] name={name}, stem={stem}, res={H}x{W}")
                    k80_unique = np.unique(pred_sem_k80)
                    print(f"    pred k80 unique: {k80_unique[:10]}...{k80_unique[-5:]} (total {len(k80_unique)})")
                    print(f"    pred k80 thing IDs (>=65): {k80_unique[k80_unique >= 65]}")
                    print(f"    GT sem unique: {np.unique(gt_sem)[:15]}")
                    print(f"    GT inst range: [{gt_inst.min()}, {gt_inst.max()}], n_unique={len(np.unique(gt_inst))}")
                    inst_unique = np.unique(std_pred[:, :, 1])
                    print(f"    pred inst unique: {inst_unique[:15]} (total {len(inst_unique)})")

    # Hungarian matching
    print("\n  [2c] Hungarian matching (k80 → CAUSE-27)...")
    from scipy.optimize import linear_sum_assignment

    thing_pseudo_arr = np.array(sorted(thing_pseudo))
    stuff_pseudo_arr = np.array(sorted(stuff_pseudo))
    things_arr = np.array(sorted(CAUSE27_THINGS))
    stuffs_arr = np.array(sorted(CAUSE27_STUFFS))

    cost_things = cost_matrix[np.ix_(thing_pseudo_arr, things_arr)]
    cost_stuffs = cost_matrix[np.ix_(stuff_pseudo_arr, stuffs_arr)]

    def match(cost):
        n_rows, n_cols = cost.shape
        assignments = -np.ones(n_rows, dtype=np.int64)
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
        assignments[row_ind] = col_ind
        for i in range(n_rows):
            if assignments[i] == -1:
                assignments[i] = cost[i].argmax()
        return assignments

    thing_assign = match(cost_things)
    stuff_assign = match(cost_stuffs)

    k80_to_c27 = np.full(num_k80, 255, dtype=np.int64)
    for i, k80 in enumerate(thing_pseudo_arr):
        k80_to_c27[k80] = things_arr[thing_assign[i]]
    for i, k80 in enumerate(stuff_pseudo_arr):
        k80_to_c27[k80] = stuffs_arr[stuff_assign[i]]

    print("  Thing cluster assignments:")
    for k80 in thing_pseudo_arr:
        c27 = k80_to_c27[k80]
        name = CAUSE27_NAMES[c27] if c27 < 27 else "?"
        # Show cost matrix row for this cluster
        row = cost_matrix[k80, things_arr]
        top3_idx = np.argsort(row)[::-1][:3]
        top3 = [(CAUSE27_NAMES[things_arr[j]], int(row[j])) for j in top3_idx]
        print(f"    k80={k80} → {name}(c27={c27})  top3: {top3}")

    # Build panoptic maps + PQ
    print("\n  [2d] Building panoptic maps + computing PQ...")

    # Per-class PQ accumulators
    tp = np.zeros(27)
    fp = np.zeros(27)
    fn = np.zeros(27)
    iou_sum = np.zeros(27)
    tp_m = np.zeros(27)
    fp_m = np.zeros(27)
    fn_m = np.zeros(27)
    iou_sum_m = np.zeros(27)

    for idx, (stem, std_pred, gt_sem, gt_inst) in enumerate(tqdm(all_data, desc="PQ eval")):
        H, W = std_pred.shape[:2]
        pred_k80 = std_pred[:, :, 0]
        pred_inst = std_pred[:, :, 1]

        # Map k80 → CAUSE-27
        cups_sem = np.full((H, W), 255, dtype=np.int64)
        valid = (pred_k80 >= 0) & (pred_k80 < num_k80)
        cups_sem[valid] = k80_to_c27[pred_k80[valid].astype(int)]

        # Build panoptic maps: cls * OFFSET + inst_id
        OFFSET = 10000
        cups_pan = cups_sem * OFFSET
        gt_pan = gt_sem.astype(np.int64) * OFFSET

        # Set thing instance IDs in pred and GT
        for c in CAUSE27_THINGS:
            for arr_sem, arr_inst, arr_pan in [(cups_sem, pred_inst, cups_pan), (gt_sem, gt_inst, gt_pan)]:
                mask = arr_sem == c
                if not mask.any():
                    continue
                for iid in np.unique(arr_inst[mask]):
                    if iid == 0:
                        continue
                    arr_pan[mask & (arr_inst == iid)] = c * OFFSET + int(iid)

        # --- CUPS-only PQ ---
        _eval_pq_image(cups_pan, gt_pan, tp, fp, fn, iou_sum)

        # --- Merged PQ (UNet stuff + CUPS things) ---
        unet_path = unet_sem_dir / f"{stem}_sem.png"
        if unet_path.exists():
            unet_pred = np.array(Image.open(str(unet_path)))
            unet_up = np.array(Image.fromarray(unet_pred).resize((W, H), Image.NEAREST))
            unet_c27 = np.full((H, W), 255, dtype=np.int64)
            for tid, c27 in TRAINID_TO_CAUSE27.items():
                unet_c27[unet_up == tid] = c27

            merged_sem = cups_sem.copy()
            merged_inst = pred_inst.astype(np.int64).copy()
            is_stuff = pred_inst == 0
            unet_valid = unet_c27 < 27
            replace = is_stuff & unet_valid
            merged_sem[replace] = unet_c27[replace]
            merged_inst[replace] = 0

            merged_pan = merged_sem.astype(np.int64) * OFFSET
            for c in CAUSE27_THINGS:
                mask = merged_sem == c
                if not mask.any():
                    continue
                for iid in np.unique(merged_inst[mask]):
                    if iid == 0:
                        continue
                    merged_pan[mask & (merged_inst == iid)] = c * OFFSET + int(iid)
        else:
            merged_pan = cups_pan.copy()

        _eval_pq_image(merged_pan, gt_pan, tp_m, fp_m, fn_m, iou_sum_m)

    # Print results
    _print_results("CUPS standalone", tp, fp, fn, iou_sum)
    _print_results("UNet+CUPS merged", tp_m, fp_m, fn_m, iou_sum_m)


def _eval_pq_image(pan_pred, pan_gt, tp, fp, fn, iou_sum):
    """Per-image PQ accumulation."""
    pred_ids = np.unique(pan_pred)
    gt_ids = np.unique(pan_gt)

    for c in range(27):
        if c in CAUSE27_STUFFS:
            p_segs = [c * 10000] if (c * 10000) in pred_ids else []
            g_segs = [c * 10000] if (c * 10000) in gt_ids else []
        else:
            p_segs = [p for p in pred_ids if p // 10000 == c and p % 10000 > 0]
            g_segs = [g for g in gt_ids if g // 10000 == c and g % 10000 > 0]

        matched_gt = set()
        for p_id in p_segs:
            p_mask = pan_pred == p_id
            best_iou, best_g = 0.0, None
            for g_id in g_segs:
                g_mask = pan_gt == g_id
                inter = int((p_mask & g_mask).sum())
                if inter == 0:
                    continue
                union = int(p_mask.sum()) + int(g_mask.sum()) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_g = iou, g_id
            if best_iou > 0.5 and best_g not in matched_gt:
                tp[c] += 1
                iou_sum[c] += best_iou
                matched_gt.add(best_g)
            else:
                fp[c] += 1
        for g_id in g_segs:
            if g_id not in matched_gt:
                fn[c] += 1


def _print_results(label, tp, fp, fn, iou_sum):
    pq_per = np.zeros(27)
    for c in range(27):
        d = tp[c] + 0.5 * fp[c] + 0.5 * fn[c]
        if tp[c] > 0:
            sq = iou_sum[c] / tp[c]
            rq = tp[c] / d if d > 0 else 0
            pq_per[c] = sq * rq

    present = [c for c in range(27) if tp[c] + fp[c] + fn[c] > 0]
    stuff_p = [c for c in present if c in CAUSE27_STUFFS]
    thing_p = [c for c in present if c in CAUSE27_THINGS]

    pq = pq_per[present].mean() * 100 if present else 0
    pq_s = pq_per[stuff_p].mean() * 100 if stuff_p else 0
    pq_t = pq_per[thing_p].mean() * 100 if thing_p else 0

    print(f"\n  {label}:")
    print(f"    PQ={pq:.2f}  PQ_stuff={pq_s:.2f}  PQ_things={pq_t:.2f}")
    print(f"    {'Class':<16} {'Type':>5}  {'PQ':>6}  {'TP':>5} {'FP':>5} {'FN':>5}")
    for c in range(27):
        if tp[c] + fp[c] + fn[c] > 0:
            tag = "thing" if c in CAUSE27_THINGS else "stuff"
            print(f"    {CAUSE27_NAMES[c]:<16} {tag:>5}  {pq_per[c]*100:6.2f}  {int(tp[c]):5d} {int(fp[c]):5d} {int(fn[c]):5d}")


def _run_unet(device, output_dir):
    """Run UNet P2-B inference."""
    from mbps_pytorch.refine_net import DepthGuidedUNet

    output_dir = Path(output_dir)
    # Check if already done
    existing = list(output_dir.glob("*_sem.png"))
    if len(existing) >= 500:
        print(f"    UNet predictions already cached ({len(existing)} files)")
        return

    model = DepthGuidedUNet(
        num_classes=19, feature_dim=768, bridge_dim=192, num_bottleneck_blocks=2,
        skip_dim=32, coupling_strength=0.1, gradient_checkpointing=False,
        rich_skip=True, num_final_blocks=1, num_decoder_stages=2,
        block_type="attention", window_size=8, num_heads=4,
    )
    ckpt = torch.load(UNET_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"    UNet loaded from {UNET_CKPT}")

    feat_root = REPO_ROOT / "datasets/cityscapes/dinov2_features/val"
    depth_root = REPO_ROOT / "datasets/cityscapes/depth_spidepth/val"
    img_root = REPO_ROOT / "datasets/cityscapes/leftImg8bit/val"

    entries = []
    for city in sorted(img_root.iterdir()):
        if not city.is_dir():
            continue
        for img_path in sorted(city.glob("*_leftImg8bit.png")):
            stem = img_path.stem.replace("_leftImg8bit", "")
            entries.append((city.name, stem))

    with torch.no_grad():
        for city, stem in tqdm(entries, desc="UNet inference"):
            out_path = output_dir / f"{stem}_sem.png"
            if out_path.exists():
                continue
            feat_path = feat_root / city / f"{stem}_leftImg8bit.npy"
            if not feat_path.exists():
                continue
            features = np.load(str(feat_path)).astype(np.float32)
            features = features.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)
            depth_path = depth_root / city / f"{stem}.npy"
            depth_np = np.load(str(depth_path)).astype(np.float32) if depth_path.exists() else np.zeros((512, 1024), dtype=np.float32)
            depth_full = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
            depth_patch = F.interpolate(depth_full, size=(PATCH_H, PATCH_W), mode="bilinear", align_corners=False).squeeze(0)
            depth_grads_np = sobel_gradients(depth_patch.squeeze(0).numpy())
            feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
            depth_t = depth_patch.unsqueeze(0).to(device)
            grads_t = torch.from_numpy(depth_grads_np).unsqueeze(0).to(device)
            depth_full_t = depth_full.to(device)
            logits = model(feat_t, depth_t, grads_t, depth_full=depth_full_t)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(pred).save(str(out_path))

    del model
    torch.cuda.empty_cache()
    count = len(list(output_dir.glob("*_sem.png")))
    print(f"    Saved {count} UNet predictions")


def main():
    print("=" * 70)
    print("CUPS Stage-3 PQ Verification + UNet Merged Evaluation")
    print("=" * 70)

    # Part 1: Verify CUPS PQ using trainer.validate()
    model, config, thing_pseudo, stuff_pseudo = part1_verify_cups_pq()

    # Part 2: Merged eval
    part2_merged_eval(model, config, thing_pseudo, stuff_pseudo)

    return 0


if __name__ == "__main__":
    sys.exit(main())
