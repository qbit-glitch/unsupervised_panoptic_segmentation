#!/usr/bin/env python3
"""
Semantic-Instance Alignment Ablation Study

Post-hoc alignment methods (A-1 through A-4) that operate on the existing
best checkpoint at inference time. Zero training required.

Methods:
  baseline: Connected-component instances from semantic preds (current)
  A1: Majority voting within pre-computed instances
  A2: Stuff-things selective merge (trained stuff + stage-1 things)
  A3: Confidence-weighted voting within instances
  A4: Majority voting + stuff preservation (hybrid)
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from mbps_pytorch.train_mobile_panoptic import (
    MobilePanopticModel,
    infer_instances_connected_components,
    _CS_ID_TO_TRAIN, _STUFF_IDS, _THING_IDS, _CS_CLASS_NAMES,
)

NUM_CLASSES = 19


# ═══════════════════════════════════════════════════════════════════════
# Pre-computed Instance Loading
# ═══════════════════════════════════════════════════════════════════════

def load_precomputed_instances(cityscapes_root, image_id, target_hw=(512, 1024)):
    """Load pre-computed depth-guided instance masks.

    Returns:
        inst_map: (H, W) uint16, 0=background, 1..N=instance IDs
        inst_classes_s1: dict {inst_id -> trainID} from stage-1 k=80 semantics
    """
    city = image_id.split("_")[0]
    H, W = target_hw

    # Load instance map (512x1024 uint16)
    inst_path = os.path.join(
        cityscapes_root, "pseudo_instance_spidepth", "val", city,
        f"{image_id}_instance.png"
    )
    if not os.path.exists(inst_path):
        return None, None

    inst_map = np.array(Image.open(inst_path))
    if inst_map.shape != (H, W):
        inst_map = np.array(
            Image.fromarray(inst_map).resize((W, H), Image.NEAREST)
        )

    # Load k=80 cluster labels (1024x2048 uint8)
    sem_path = os.path.join(
        cityscapes_root, "pseudo_semantic_raw_k80", "val", city,
        f"{image_id}.png"
    )
    if not os.path.exists(sem_path):
        return inst_map, None

    sem_k80 = np.array(Image.open(sem_path))
    if sem_k80.shape != (H, W):
        sem_k80 = np.array(
            Image.fromarray(sem_k80).resize((W, H), Image.NEAREST)
        )

    # Load cluster -> trainID LUT
    centroids_path = os.path.join(
        cityscapes_root, "pseudo_semantic_raw_k80", "kmeans_centroids.npz"
    )
    if not os.path.exists(centroids_path):
        return inst_map, None

    c2c = np.load(centroids_path)["cluster_to_class"]
    lut = np.full(256, 255, dtype=np.uint8)
    for cid in range(len(c2c)):
        lut[cid] = int(c2c[cid])

    sem_trainid = lut[sem_k80]

    # Majority trainID vote per instance
    inst_classes = {}
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        mask = inst_map == iid
        tids = sem_trainid[mask]
        valid = tids[tids != 255]
        if len(valid) == 0:
            continue
        inst_classes[int(iid)] = int(Counter(valid.tolist()).most_common(1)[0][0])

    return inst_map, inst_classes


# ═══════════════════════════════════════════════════════════════════════
# Panoptic Map from Aligned Semantics + Pre-computed Instances
# ═══════════════════════════════════════════════════════════════════════

def build_panoptic_with_precomputed_instances(
    aligned_sem, inst_map, cc_min_area=50
):
    """Build panoptic map using pre-computed instances for things.

    - Stuff: each class is one segment (merged, no CC splitting)
    - Things: each pre-computed instance with thing-class majority = one segment

    Returns:
        pan_map: (H, W) int32, segment IDs
        segments: dict {segment_id -> trainID}
    """
    H, W = aligned_sem.shape
    pan_map = np.zeros((H, W), dtype=np.int32)
    segments = {}
    nxt = 1

    # 1. Stuff segments from aligned semantics
    for cls in _STUFF_IDS:
        mask = aligned_sem == cls
        if mask.sum() < 64:
            continue
        pan_map[mask] = nxt
        segments[nxt] = cls
        nxt += 1

    # 2. Thing segments from pre-computed instances
    if inst_map is not None:
        for iid in np.unique(inst_map):
            if iid == 0:
                continue
            mask = inst_map == iid
            classes = aligned_sem[mask]
            if len(classes) == 0:
                continue
            majority = Counter(classes.tolist()).most_common(1)[0][0]
            if majority not in _THING_IDS:
                continue
            if mask.sum() < cc_min_area:
                continue
            pan_map[mask] = nxt
            segments[nxt] = majority
            nxt += 1
    else:
        # Fallback to CC if no instances
        for cls in _THING_IDS:
            cls_mask = aligned_sem == cls
            if cls_mask.sum() < cc_min_area:
                continue
            labeled, n_cc = ndimage.label(cls_mask)
            for comp in range(1, n_cc + 1):
                cmask = labeled == comp
                if cmask.sum() < cc_min_area:
                    continue
                pan_map[cmask] = nxt
                segments[nxt] = cls
                nxt += 1

    return pan_map, segments


# ═══════════════════════════════════════════════════════════════════════
# Alignment Methods
# ═══════════════════════════════════════════════════════════════════════

def align_a1_majority(sem_pred, inst_map, softmax_probs=None, **kw):
    """A-1: Majority voting within pre-computed instances."""
    aligned = sem_pred.copy()
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        mask = inst_map == iid
        votes = sem_pred[mask]
        if len(votes) == 0:
            continue
        aligned[mask] = Counter(votes.tolist()).most_common(1)[0][0]
    return aligned


def align_a2_selective(sem_pred, inst_map, softmax_probs=None,
                       inst_classes_s1=None, **kw):
    """A-2: Trained model for stuff, stage-1 class for things in instances."""
    aligned = sem_pred.copy()
    if inst_classes_s1 is None:
        return aligned
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        if iid not in inst_classes_s1:
            continue
        s1_cls = inst_classes_s1[iid]
        if s1_cls in _THING_IDS:
            mask = inst_map == iid
            aligned[mask] = s1_cls
    return aligned


def align_a3_confidence(sem_pred, inst_map, softmax_probs=None, **kw):
    """A-3: Confidence-weighted voting within instances."""
    aligned = sem_pred.copy()
    if softmax_probs is None:
        return align_a1_majority(sem_pred, inst_map)
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        mask = inst_map == iid
        probs = softmax_probs[mask]  # (N, 19)
        weighted = probs.sum(axis=0)  # (19,)
        aligned[mask] = int(np.argmax(weighted))
    return aligned


def align_a4_majority_stuff(sem_pred, inst_map, softmax_probs=None, **kw):
    """A-4: Majority voting for thing instances only, keep trained stuff."""
    aligned = sem_pred.copy()
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        mask = inst_map == iid
        votes = sem_pred[mask]
        if len(votes) == 0:
            continue
        majority = Counter(votes.tolist()).most_common(1)[0][0]
        if majority in _THING_IDS:
            aligned[mask] = majority
    return aligned


METHODS = {
    "baseline": None,
    "A1": align_a1_majority,
    "A2": align_a2_selective,
    "A3": align_a3_confidence,
    "A4": align_a4_majority_stuff,
}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run_study(args):
    device = torch.device(args.device)
    H, W = 512, 1024

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = MobilePanopticModel(
        backbone_name="repvit_m0_9",
        num_classes=NUM_CLASSES,
        fpn_type="bifpn",
        instance_head="none",
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    model.train(False)  # inference mode

    # Gather val images
    cityscapes_root = args.cityscapes_root
    val_img_dir = os.path.join(cityscapes_root, "leftImg8bit", "val")
    gt_dir = os.path.join(cityscapes_root, "gtFine", "val")

    image_ids = []
    for city in sorted(os.listdir(val_img_dir)):
        cdir = os.path.join(val_img_dir, city)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith("_leftImg8bit.png"):
                image_ids.append(fn.replace("_leftImg8bit.png", ""))
    print(f"Validation images: {len(image_ids)}")

    # Select methods
    method_names = args.methods.split(",")
    if "all" in method_names:
        method_names = ["baseline", "A1", "A2", "A3", "A4"]
    print(f"Methods: {method_names}")

    # Per-method accumulators
    accum = {}
    for m in method_names:
        accum[m] = {
            "tp": np.zeros(NUM_CLASSES),
            "fp": np.zeros(NUM_CLASSES),
            "fn": np.zeros(NUM_CLASSES),
            "iou_sum": np.zeros(NUM_CLASSES),
            "confusion": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
        }

    # Image normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    t0 = time.time()
    for idx, img_id in enumerate(tqdm(image_ids, desc="Processing")):
        city = img_id.split("_")[0]

        # Load & preprocess image
        img_path = os.path.join(val_img_dir, city, f"{img_id}_leftImg8bit.png")
        img = Image.open(img_path).convert("RGB")
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)
        img_t = (img_t - mean) / std
        img_t = F.interpolate(img_t, size=(H, W), mode="bilinear", align_corners=False)

        # Forward pass
        with torch.no_grad():
            out = model(img_t)
            logits = out["logits"]  # (1, 19, h, w)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)
            sem_pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()
            softmax_np = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Load pre-computed instances
        inst_map, inst_classes_s1 = load_precomputed_instances(
            cityscapes_root, img_id, target_hw=(H, W)
        )

        # Load GT semantic
        gt_sem_path = os.path.join(gt_dir, city, f"{img_id}_gtFine_labelIds.png")
        if not os.path.exists(gt_sem_path):
            continue
        gt_raw = np.array(Image.open(gt_sem_path))
        gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
        for raw_id, tid in _CS_ID_TO_TRAIN.items():
            gt_sem[gt_raw == raw_id] = tid
        if gt_sem.shape != (H, W):
            gt_sem = np.array(Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

        # Build GT panoptic map
        gt_pan = np.zeros((H, W), dtype=np.int32)
        gt_segments = {}
        gt_nxt = 1

        for cls in _STUFF_IDS:
            mask = gt_sem == cls
            if mask.sum() < 64:
                continue
            gt_pan[mask] = gt_nxt
            gt_segments[gt_nxt] = cls
            gt_nxt += 1

        gt_inst_path = os.path.join(gt_dir, city, f"{img_id}_gtFine_instanceIds.png")
        if os.path.exists(gt_inst_path):
            gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
            if gt_inst.shape != (H, W):
                gt_inst = np.array(
                    Image.fromarray(gt_inst.astype(np.int32)).resize((W, H), Image.NEAREST),
                    dtype=np.int32
                )
            for uid in np.unique(gt_inst):
                if uid < 1000:
                    continue
                raw_cls = uid // 1000
                if raw_cls not in _CS_ID_TO_TRAIN:
                    continue
                tid = _CS_ID_TO_TRAIN[raw_cls]
                if tid not in _THING_IDS:
                    continue
                mask = gt_inst == uid
                if mask.sum() < 10:
                    continue
                gt_pan[mask] = gt_nxt
                gt_segments[gt_nxt] = tid
                gt_nxt += 1

        gt_by_cat = defaultdict(list)
        for sid, cat in gt_segments.items():
            gt_by_cat[cat].append(sid)

        # Run each method
        for mname in method_names:
            if mname == "baseline":
                pred_pan, pred_segments = infer_instances_connected_components(
                    sem_pred, cc_min_area=50
                )
                pred_for_miou = sem_pred
            else:
                align_fn = METHODS[mname]
                empty_inst = np.zeros((H, W), dtype=np.uint16)
                aligned = align_fn(
                    sem_pred,
                    inst_map if inst_map is not None else empty_inst,
                    softmax_probs=softmax_np,
                    inst_classes_s1=inst_classes_s1,
                )
                pred_pan, pred_segments = build_panoptic_with_precomputed_instances(
                    aligned, inst_map, cc_min_area=50
                )
                pred_for_miou = aligned

            # Confusion matrix for mIoU
            valid_gt = (gt_sem < NUM_CLASSES) & (pred_for_miou < NUM_CLASSES)
            if valid_gt.sum() > 0:
                np.add.at(accum[mname]["confusion"],
                          (gt_sem[valid_gt], pred_for_miou[valid_gt]), 1)

            # PQ matching
            pred_by_cat = defaultdict(list)
            for sid, cat in pred_segments.items():
                pred_by_cat[cat].append(sid)

            matched_pred = set()
            a = accum[mname]
            for cat in range(NUM_CLASSES):
                for gt_id in gt_by_cat.get(cat, []):
                    gt_mask = gt_pan == gt_id
                    best_iou, best_pid = 0.0, None
                    for pid in pred_by_cat.get(cat, []):
                        if pid in matched_pred:
                            continue
                        inter = np.sum(gt_mask & (pred_pan == pid))
                        union = np.sum(gt_mask | (pred_pan == pid))
                        if union == 0:
                            continue
                        iou_val = inter / union
                        if iou_val > best_iou:
                            best_iou, best_pid = iou_val, pid
                    if best_iou > 0.5 and best_pid is not None:
                        a["tp"][cat] += 1
                        a["iou_sum"][cat] += best_iou
                        matched_pred.add(best_pid)
                    else:
                        a["fn"][cat] += 1
                for pid in pred_by_cat.get(cat, []):
                    if pid not in matched_pred:
                        a["fp"][cat] += 1

    # ═══════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    print("=" * 78)
    print(f"{'Method':<12} {'PQ':>7} {'PQ_st':>7} {'PQ_th':>7} {'mIoU':>7}")
    print("-" * 78)

    for mname in method_names:
        a = accum[mname]
        tp, fp_arr, fn_arr, iou_s = a["tp"], a["fp"], a["fn"], a["iou_sum"]
        conf = a["confusion"]

        # mIoU
        intersection = np.diag(conf)
        union = conf.sum(1) + conf.sum(0) - intersection
        iou_per = np.where(union > 0, intersection / union, 0.0)
        miou = iou_per[union > 0].mean() * 100 if (union > 0).any() else 0.0

        # PQ
        all_pq, stuff_pq, thing_pq = [], [], []
        per_cls = {}
        for c in range(NUM_CLASSES):
            t, f, n, s = tp[c], fp_arr[c], fn_arr[c], iou_s[c]
            if t + f + n > 0:
                sq = s / (t + 1e-8)
                rq = t / (t + 0.5 * f + 0.5 * n)
                pq = sq * rq
            else:
                pq = 0.0
            per_cls[_CS_CLASS_NAMES[c]] = round(pq * 100, 2)
            if tp[c] + fp_arr[c] + fn_arr[c] > 0:
                all_pq.append(pq)
                (stuff_pq if c in _STUFF_IDS else thing_pq).append(pq)

        pq_all = float(np.mean(all_pq)) * 100 if all_pq else 0.0
        pq_st = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
        pq_th = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0

        print(f"{mname:<12} {pq_all:>7.2f} {pq_st:>7.2f} {pq_th:>7.2f} {miou:>7.2f}")

        if args.verbose:
            print(f"  Per-class PQ: {per_cls}")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic-Instance Alignment Ablation"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--cityscapes_root", type=str,
        default="/Users/qbit-glitch/Desktop/datasets/cityscapes"
    )
    parser.add_argument("--methods", type=str, default="all",
                        help="baseline,A1,A2,A3,A4 or 'all'")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_study(args)


if __name__ == "__main__":
    main()
