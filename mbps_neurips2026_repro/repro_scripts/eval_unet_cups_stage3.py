#!/usr/bin/env python3
"""
UNet P2-B semantics + CUPS Stage-3 instances → panoptic evaluation

Evaluates in CAUSE-27 space (same as CUPS).

Pipeline:
  1. Run CUPS model → panoptic_seg → standard format [H,W,2] in k80 space
  2. Build cost matrix (k80 × 27) from predictions vs GT
  3. Hungarian matching → k80 → CAUSE-27 assignment
  4. Verify CUPS standalone PQ ≈ 27.67 (our own fast PQ code)
  5. Replace stuff with UNet predictions (mapped to CAUSE-27)
  6. Compute merged PQ

Usage (on remote):
    export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:$PYTHONPATH
    cd /media/santosh/Kuldeep/panoptic_segmentation
    python scripts/eval_unet_cups_stage3.py
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
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path("/media/santosh/Kuldeep/panoptic_segmentation")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "refs" / "cups"))

PATCH_H, PATCH_W = 32, 64

# CAUSE-27 classnames
CAUSE27_NAMES = [
    'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
    'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
    'train', 'motorcycle', 'bicycle'
]

# CAUSE-27 thing/stuff class IDs
CAUSE27_THINGS = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26}
CAUSE27_STUFFS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
NUM_CAUSE27 = 27

# trainID-19 → CAUSE-27 reverse mapping (for stuff classes only)
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


def load_unet(device):
    from mbps_pytorch.refine_net import DepthGuidedUNet
    ckpt_path = REPO_ROOT / "checkpoints/unet_p2b_attention/best.pth"
    model = DepthGuidedUNet(
        num_classes=19, feature_dim=768, bridge_dim=192, num_bottleneck_blocks=2,
        skip_dim=32, coupling_strength=0.1, gradient_checkpointing=False,
        rich_skip=True, num_final_blocks=1, num_decoder_stages=2,
        block_type="attention", window_size=8, num_heads=4,
    )
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"UNet P2-B loaded from {ckpt_path}")
    return model


def run_unet_inference(model, device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"  Running UNet on {len(entries)} val images...")
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
    count = len(list(output_dir.glob("*_sem.png")))
    print(f"  Semantic predictions saved: {count} images")


def load_cups_model(device):
    import cups
    from cups.data import CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES, CITYSCAPES_CLASSNAMES
    config = cups.get_default_config(
        experiment_config_file=str(REPO_ROOT / "refs/cups/configs/train_cityscapes_vitb.yaml")
    )
    config.defrost()
    config.MODEL.CHECKPOINT = str(
        REPO_ROOT / "experiments/experiments/cups_vitb_k80_stage3/Unsupervised Panoptic Segmentation/sh17g3sl/checkpoints/ups_checkpoint_step=000100.ckpt"
    )
    config.freeze()
    model = cups.build_model_pseudo(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        copy_paste_augmentation=torch.nn.Identity(),
        resolution_jitter_augmentation=torch.nn.Identity(),
        photometric_augmentation=torch.nn.Identity(),
        use_tta=False,
        class_names=CITYSCAPES_CLASSNAMES,
        classes_mask=None,
    )
    model = model.to(device).eval()
    print("CUPS Stage-3 model loaded")
    return model, config


def hungarian_match_cups(cost_matrix, thing_pseudo, stuff_pseudo, thing_classes, stuff_classes):
    """Replicate CUPS's _matching_with_separation."""
    num_k80 = len(thing_pseudo) + len(stuff_pseudo)
    thing_pseudo_t = np.array(sorted(thing_pseudo))
    stuff_pseudo_t = np.array(sorted(stuff_pseudo))
    things_t = np.array(sorted(thing_classes))
    stuffs_t = np.array(sorted(stuff_classes))

    # Split cost matrix
    cost_things = cost_matrix[np.ix_(thing_pseudo_t, things_t)]
    cost_stuffs = cost_matrix[np.ix_(stuff_pseudo_t, stuffs_t)]

    def match_core(cost, n_clusters, n_classes):
        if n_clusters == n_classes:
            _, col = linear_sum_assignment(cost, maximize=True)
            return col
        assignments = -np.ones(n_clusters, dtype=np.int64)
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
        assignments[row_ind] = col_ind
        missing = [i for i in range(n_clusters) if i not in row_ind]
        for m in missing:
            assignments[m] = cost[m].argmax()
        return assignments

    thing_assignments = match_core(cost_things, len(thing_pseudo_t), len(things_t))
    stuff_assignments = match_core(cost_stuffs, len(stuff_pseudo_t), len(stuffs_t))

    # Build global assignment: k80 → CAUSE-27
    k80_to_cause27 = np.full(num_k80, 255, dtype=np.int64)
    for i, k80 in enumerate(thing_pseudo_t):
        k80_to_cause27[k80] = things_t[thing_assignments[i]]
    for i, k80 in enumerate(stuff_pseudo_t):
        k80_to_cause27[k80] = stuffs_t[stuff_assignments[i]]

    return k80_to_cause27


def compute_pq(all_pan_pred, all_pan_gt, thing_classes, stuff_classes, num_classes):
    """Fast PQ computation in CAUSE-27 space.

    pan_pred/pan_gt: per-image arrays where value = cls*10000 + inst_id
    For stuff: inst_id = 0. For things: inst_id > 0.
    """
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    iou_sum = np.zeros(num_classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pan_pred, pan_gt, sem_pred, sem_gt in tqdm(
        zip(all_pan_pred["pan"], all_pan_gt["pan"], all_pan_pred["sem_raw"], all_pan_gt["sem"]),
        total=len(all_pan_pred["pan"]), desc="Computing PQ"
    ):
        # mIoU
        valid = (sem_pred < num_classes) & (sem_gt < num_classes)
        if valid.any():
            np.add.at(confusion, (sem_gt[valid].astype(int), sem_pred[valid].astype(int)), 1)

        pred_ids = np.unique(pan_pred)
        gt_ids = np.unique(pan_gt)

        for c in range(num_classes):
            if c in stuff_classes:
                g_ids = [c * 10000] if (c * 10000) in gt_ids else []
                p_ids = [c * 10000] if (c * 10000) in pred_ids else []
            else:
                g_ids = [g for g in gt_ids if g // 10000 == c and g % 10000 > 0]
                p_ids = [p for p in pred_ids if p // 10000 == c and p % 10000 > 0]

            matched_gt = set()
            for p_id in p_ids:
                p_mask = pan_pred == p_id
                best_iou, best_g = 0.0, None
                for g_id in g_ids:
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
            for g_id in g_ids:
                if g_id not in matched_gt:
                    fn[c] += 1

    # Metrics
    pq_per = np.zeros(num_classes)
    sq_per = np.zeros(num_classes)
    rq_per = np.zeros(num_classes)
    for c in range(num_classes):
        d = tp[c] + 0.5 * fp[c] + 0.5 * fn[c]
        if tp[c] > 0:
            sq_per[c] = iou_sum[c] / tp[c]
            rq_per[c] = tp[c] / d if d > 0 else 0
            pq_per[c] = sq_per[c] * rq_per[c]

    present = [c for c in range(num_classes) if (tp[c] + fp[c] + fn[c]) > 0]
    stuff_p = [c for c in present if c in stuff_classes]
    thing_p = [c for c in present if c in thing_classes]

    per_class_iou = np.zeros(num_classes)
    for c in range(num_classes):
        tp_c = confusion[c, c]
        denom = tp_c + (confusion[:, c].sum() - tp_c) + (confusion[c, :].sum() - tp_c)
        if denom > 0:
            per_class_iou[c] = tp_c / denom

    return {
        "PQ": pq_per[present].mean() * 100 if present else 0,
        "PQ_stuff": pq_per[stuff_p].mean() * 100 if stuff_p else 0,
        "PQ_things": pq_per[thing_p].mean() * 100 if thing_p else 0,
        "SQ": sq_per[present].mean() * 100 if present else 0,
        "RQ": rq_per[present].mean() * 100 if present else 0,
        "mIoU": per_class_iou[per_class_iou > 0].mean() * 100,
        "pq_per": pq_per, "tp": tp, "fp": fp, "fn": fn,
    }


def build_panoptic_map(sem_c27, inst_ids, thing_classes):
    """Build panoptic map: cls*10000 + inst_id."""
    H, W = sem_c27.shape
    pan = sem_c27.astype(np.int64) * 10000  # stuff: cls*10000+0
    for c in thing_classes:
        cls_mask = sem_c27 == c
        if not cls_mask.any():
            continue
        inst_vals = inst_ids[cls_mask]
        for iid in np.unique(inst_vals):
            if iid == 0:
                continue
            pan[cls_mask & (inst_ids == iid)] = c * 10000 + int(iid)
    return pan


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_base = Path("/tmp/unet_cups_stage3_eval_v7")
    output_base.mkdir(parents=True, exist_ok=True)
    sem_dir = output_base / "semantics"

    print("\n" + "=" * 70)
    print("UNet P2-B (stuff) + CUPS Stage-3 (things) → Panoptic PQ")
    print("Evaluation in CAUSE-27 space (same as CUPS)")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: UNet inference
    # ---------------------------------------------------------------
    print("\n[1/4] UNet P2-B semantic inference...")
    unet = load_unet(device)
    run_unet_inference(unet, device, sem_dir)
    del unet
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 2: CUPS inference → standard format + cost matrix
    # ---------------------------------------------------------------
    print("\n[2/4] CUPS Stage-3 inference...")
    cups_model, config = load_cups_model(device)

    from cups.data import (
        CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES,
        CityscapesPanopticValidation, collate_function_validation,
    )
    from cups.model.model import prediction_to_standard_format
    from torch.utils.data import DataLoader

    # Hardcode val params to match CUPS train_self.py validation exactly
    val_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT,
        resize_scale=0.5,
        crop_resolution=(512, 1024),
        num_classes=27,
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_function_validation, drop_last=False, pin_memory=False,
    )

    # Get pseudo classes from checkpoint
    ckpt = torch.load(config.MODEL.CHECKPOINT, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    thing_pseudo = tuple(hp.get("thing_pseudo_classes", tuple(range(65, 80))))
    stuff_pseudo = tuple(hp.get("stuff_pseudo_classes", tuple(range(0, 65))))
    num_k80 = len(thing_pseudo) + len(stuff_pseudo)
    print(f"  thing_pseudo: {thing_pseudo}")
    print(f"  stuff_pseudo: {stuff_pseudo[:5]}...{stuff_pseudo[-5:]}")
    del ckpt

    # Build cost matrix and store predictions
    cost_matrix = np.zeros((num_k80, NUM_CAUSE27), dtype=np.int64)
    all_data = []  # list of (stem, std_pred, gt_sem, gt_inst)

    print(f"  Running CUPS on {len(val_dataset)} val images...")
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
                # Convert to standard format [H, W, 2] (k80 space)
                std_pred = prediction_to_standard_format(
                    predictions[i]["panoptic_seg"],
                    stuff_classes=stuff_pseudo,
                    thing_classes=thing_pseudo,
                ).cpu().numpy()  # [H, W, 2]: sem in k80, inst

                # GT from dataset
                gt = panoptic_labels[i].cpu().numpy()  # [H, W, 2]: sem in CAUSE-27, inst
                gt_sem = gt[:, :, 0]
                gt_inst = gt[:, :, 1]

                # Update cost matrix: k80_pred × CAUSE-27_gt
                pred_sem_k80 = std_pred[:, :, 0]
                valid = (pred_sem_k80 >= 0) & (pred_sem_k80 < num_k80) & (gt_sem >= 0) & (gt_sem < NUM_CAUSE27)
                if valid.any():
                    np.add.at(cost_matrix, (pred_sem_k80[valid].astype(np.int64), gt_sem[valid].astype(np.int64)), 1)

                # name is like "frankfurt/frankfurt_000000_000294" — strip city prefix
                parts = name.split("/")
                stem = parts[-1] if len(parts) > 1 else name
                stem = stem.replace("_leftImg8bit", "")
                all_data.append((stem, std_pred, gt_sem, gt_inst))

                if len(all_data) <= 2:
                    H, W = std_pred.shape[:2]
                    print(f"\n  [DEBUG img {len(all_data)}] name={name}, stem={stem}")
                    print(f"    GT sem unique (first 15): {np.unique(gt_sem)[:15]}")
                    print(f"    GT inst range: min={gt_inst.min()}, max={gt_inst.max()}, n_unique={len(np.unique(gt_inst))}")
                    print(f"    Pred k80 unique (first 15): {np.unique(pred_sem_k80)[:15]}")
                    print(f"    Resolution: {H}x{W}")

    del cups_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 3: Hungarian matching + CUPS-only PQ + Merged PQ
    # ---------------------------------------------------------------
    print("\n[3/4] Hungarian matching...")
    k80_to_c27 = hungarian_match_cups(
        cost_matrix, set(thing_pseudo), set(stuff_pseudo),
        CAUSE27_THINGS, CAUSE27_STUFFS,
    )

    print("  Thing assignments (k80 → CAUSE-27):")
    for k80 in sorted(thing_pseudo):
        c27 = k80_to_c27[k80]
        name = CAUSE27_NAMES[c27] if c27 < NUM_CAUSE27 else "?"
        print(f"    k80={k80} → c27={c27} ({name})")

    # Build CUPS-only and merged panoptic maps
    print("\n[4/4] Building panoptic maps and computing PQ...")

    cups_only = {"pan": [], "sem": [], "sem_raw": []}
    merged = {"pan": [], "sem": [], "sem_raw": []}
    gt_maps = {"pan": [], "sem": []}
    sem_dir_path = Path(sem_dir)

    for stem, std_pred, gt_sem, gt_inst in tqdm(all_data, desc="Building maps"):
        H, W = std_pred.shape[:2]
        pred_sem_k80 = std_pred[:, :, 0]
        pred_inst = std_pred[:, :, 1]

        # Remap k80 → CAUSE-27
        cups_sem_c27 = np.full((H, W), 255, dtype=np.int64)
        valid = (pred_sem_k80 >= 0) & (pred_sem_k80 < num_k80)
        cups_sem_c27[valid] = k80_to_c27[pred_sem_k80[valid].astype(int)]

        # Build CUPS-only panoptic map
        cups_pan = build_panoptic_map(cups_sem_c27, pred_inst.astype(np.int64), CAUSE27_THINGS)
        cups_only["pan"].append(cups_pan)
        cups_only["sem"].append(cups_sem_c27)
        cups_only["sem_raw"].append(cups_sem_c27.copy())

        # Build GT panoptic map
        gt_pan = build_panoptic_map(gt_sem, gt_inst.astype(np.int64), CAUSE27_THINGS)
        gt_maps["pan"].append(gt_pan)
        gt_maps["sem"].append(gt_sem)

        if len(gt_maps["pan"]) <= 2:
            gt_u = sorted(np.unique(gt_pan))
            cups_u = sorted(np.unique(cups_pan))
            print(f"  [DEBUG panoptic img {len(gt_maps['pan'])}]")
            print(f"    GT pan unique (first 15): {gt_u[:15]}")
            print(f"    CUPS pan unique (first 15): {cups_u[:15]}")
            sem_check = sem_dir_path / f"{stem}_sem.png"
            print(f"    UNet path: {sem_check} exists={sem_check.exists()}")

        # Load UNet prediction → map to CAUSE-27
        sem_path = sem_dir_path / f"{stem}_sem.png"
        if not sem_path.exists():
            # No UNet prediction — use CUPS for everything
            merged["pan"].append(cups_pan.copy())
            merged["sem"].append(cups_sem_c27.copy())
            merged["sem_raw"].append(cups_sem_c27.copy())
            continue

        unet_pred = np.array(Image.open(str(sem_path)))  # [128, 256] trainID-19
        unet_up = np.array(Image.fromarray(unet_pred).resize((W, H), Image.NEAREST))

        # Convert UNet trainID → CAUSE-27
        unet_c27 = np.full((H, W), 255, dtype=np.int64)
        for tid, c27 in TRAINID_TO_CAUSE27.items():
            unet_c27[unet_up == tid] = c27

        # Build merged: UNet stuff + CUPS things
        merged_sem = cups_sem_c27.copy()
        merged_inst = pred_inst.astype(np.int64).copy()

        # Identify stuff pixels (not thing instances)
        is_thing_pixel = pred_inst > 0  # instance ID > 0 means thing
        is_stuff_pixel = ~is_thing_pixel

        # Replace stuff semantics with UNet where valid
        unet_valid = unet_c27 < NUM_CAUSE27
        replace = is_stuff_pixel & unet_valid
        merged_sem[replace] = unet_c27[replace]
        merged_inst[replace] = 0

        merged_pan = build_panoptic_map(merged_sem, merged_inst, CAUSE27_THINGS)
        merged["pan"].append(merged_pan)
        merged["sem"].append(merged_sem)
        merged["sem_raw"].append(merged_sem.copy())

    # Compute CUPS-only PQ
    print("\n  Computing CUPS standalone PQ...")
    cups_results = compute_pq(cups_only, gt_maps, CAUSE27_THINGS, CAUSE27_STUFFS, NUM_CAUSE27)

    # Compute merged PQ
    print("  Computing merged PQ...")
    merged_results = compute_pq(merged, gt_maps, CAUSE27_THINGS, CAUSE27_STUFFS, NUM_CAUSE27)

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS (CAUSE-27 space)")
    print("=" * 70)
    print(f"  CUPS standalone:    PQ={cups_results['PQ']:6.2f}  PQ_stuff={cups_results['PQ_stuff']:6.2f}  PQ_things={cups_results['PQ_things']:6.2f}  mIoU={cups_results['mIoU']:5.2f}%")
    print(f"  UNet+CUPS merged:   PQ={merged_results['PQ']:6.2f}  PQ_stuff={merged_results['PQ_stuff']:6.2f}  PQ_things={merged_results['PQ_things']:6.2f}  mIoU={merged_results['mIoU']:5.2f}%")
    print(f"  CUPS CVPR 2025:     PQ= 27.80")
    print(f"  Option 1B baseline: PQ= 28.72")
    diff = merged_results['PQ'] - cups_results['PQ']
    print(f"  Merged vs CUPS standalone: {diff:+.2f} PQ")
    print("=" * 70)

    # Per-class comparison
    print("\n  Per-class PQ (CAUSE-27):")
    print(f"  {'Class':<16} {'Type':>5}  {'CUPS PQ':>7}  {'Merged PQ':>9}  {'Diff':>5}  {'TP_m':>5} {'FP_m':>5} {'FN_m':>5}")
    for c27 in range(NUM_CAUSE27):
        tag = "thing" if c27 in CAUSE27_THINGS else "stuff"
        name = CAUSE27_NAMES[c27]
        cpq = cups_results['pq_per'][c27] * 100
        mpq = merged_results['pq_per'][c27] * 100
        d = mpq - cpq
        tp_m = int(merged_results['tp'][c27])
        fp_m = int(merged_results['fp'][c27])
        fn_m = int(merged_results['fn'][c27])
        if cpq > 0 or mpq > 0 or tp_m + fp_m + fn_m > 0:
            print(f"  {name:<16} {tag:>5}  {cpq:7.2f}  {mpq:9.2f}  {d:+5.1f}  {tp_m:5d} {fp_m:5d} {fn_m:5d}")

    # Save
    results = {
        "cups_standalone": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in cups_results.items() if k in ("PQ", "PQ_stuff", "PQ_things", "mIoU", "SQ", "RQ")},
        "merged": {k: float(v) if isinstance(v, (float, np.floating)) else v
                  for k, v in merged_results.items() if k in ("PQ", "PQ_stuff", "PQ_things", "mIoU", "SQ", "RQ")},
    }
    out_json = output_base / "results.json"
    with open(str(out_json), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
