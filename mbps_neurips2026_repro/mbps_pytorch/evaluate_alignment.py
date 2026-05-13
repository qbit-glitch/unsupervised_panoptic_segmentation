#!/usr/bin/env python3
"""Phase 0: Post-hoc semantic-instance alignment evaluation.

Combines DepthGuidedUNet semantics (PQ_stuff=35.04) with depth-guided
instance splitting (PQ_things=19.41) to achieve the theoretical oracle PQ.

Methods:
  - baseline: UNet semantics + connected component instances (current eval)
  - depth_split: UNet semantics + depth-guided instance splitting on thing regions
  - depth_split_vote: depth_split + majority vote class correction within instances

Usage:
    python -u mbps_pytorch/evaluate_alignment.py \
        --checkpoint checkpoints/hires_unet_depth_guided/best.pth \
        --method all
"""

import argparse
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm

from mbps_pytorch.refine_net import DepthGuidedUNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Cityscapes class definitions
PATCH_H, PATCH_W = 32, 64
_STUFF_IDS = set(range(11))      # road..sky (trainIDs 0-10)
_THING_IDS = set(range(11, 19))  # person..bicycle (trainIDs 11-18)
_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# Cityscapes raw labelID → trainID
_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}


def load_model(checkpoint_path, device):
    """Load DepthGuidedUNet from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    num_classes = cfg.get("num_classes", 19)

    model = DepthGuidedUNet(
        in_dim=768,
        depth_dim=1,
        bridge_dim=cfg.get("bridge_dim", 192),
        num_classes=num_classes,
        num_blocks=cfg.get("num_blocks", 4),
        num_bottleneck_blocks=cfg.get("num_bottleneck_blocks", 2),
        skip_dim=cfg.get("skip_dim", 32),
        block_type=cfg.get("block_type", "conv"),
        target_h=cfg.get("target_h", 128),
        target_w=cfg.get("target_w", 256),
        gradient_checkpointing=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    log.info(f"Loaded UNet: epoch {ckpt['epoch']}, "
             f"PQ={ckpt['metrics'].get('PQ', 'N/A')}, "
             f"block_type={cfg.get('block_type')}")
    return model, cfg


def depth_guided_instances(semantic, depth, grad_threshold=0.20,
                           min_area=500, dilation_iters=3,
                           depth_blur_sigma=1.0):
    """Split thing regions using depth gradient edges.

    Returns list of (mask, class_id, score) tuples.
    """
    # Smooth depth
    if depth_blur_sigma > 0:
        depth_smooth = gaussian_filter(depth.astype(np.float64),
                                       sigma=depth_blur_sigma)
    else:
        depth_smooth = depth.astype(np.float64)

    # Depth gradient magnitude (Sobel)
    gx = sobel(depth_smooth, axis=1)
    gy = sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    depth_edges = grad_mag > grad_threshold

    assigned = np.zeros(semantic.shape, dtype=bool)
    instances = []

    for cls in sorted(_THING_IDS):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        # Remove depth edges from class mask
        split_mask = cls_mask & ~depth_edges

        # Connected components on split mask
        labeled, n_cc = ndimage.label(split_mask)

        cc_list = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                cc_list.append((cc_id, cc_mask, area))
        cc_list.sort(key=lambda x: -x[2])

        for _, cc_mask, area in cc_list:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask,
                                                  iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final_mask = cc_mask | reclaimed
            else:
                final_mask = cc_mask

            if final_mask.sum() < min_area:
                continue

            assigned |= final_mask
            instances.append((final_mask, cls, float(final_mask.sum())))

    # Sort by area, normalize scores
    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]

    return instances


def build_panoptic_cc(pred_sem, cc_min_area=50):
    """Build panoptic map using connected components (baseline)."""
    H, W = pred_sem.shape
    pred_pan = np.zeros((H, W), dtype=np.int32)
    pred_segments = {}
    nxt = 1

    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pred_pan[mask] = nxt
        pred_segments[nxt] = cls
        nxt += 1

    for cls in _THING_IDS:
        cls_mask = pred_sem == cls
        if cls_mask.sum() < cc_min_area:
            continue
        labeled, n_cc = ndimage.label(cls_mask)
        for comp in range(1, n_cc + 1):
            cmask = labeled == comp
            if cmask.sum() < cc_min_area:
                continue
            pred_pan[cmask] = nxt
            pred_segments[nxt] = cls
            nxt += 1

    return pred_pan, pred_segments


def build_panoptic_depth_split(pred_sem, depth, cc_min_area=50,
                                grad_threshold=0.20, min_area=500):
    """Build panoptic map using depth-guided instance splitting."""
    H, W = pred_sem.shape
    pred_pan = np.zeros((H, W), dtype=np.int32)
    pred_segments = {}
    nxt = 1

    # Stuff: same as baseline
    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pred_pan[mask] = nxt
        pred_segments[nxt] = cls
        nxt += 1

    # Things: depth-guided splitting
    instances = depth_guided_instances(
        pred_sem, depth,
        grad_threshold=grad_threshold,
        min_area=min_area,
    )
    for mask, cls, score in instances:
        pred_pan[mask] = nxt
        pred_segments[nxt] = cls
        nxt += 1

    return pred_pan, pred_segments


def build_panoptic_depth_split_vote(pred_sem, pred_probs, depth,
                                     cc_min_area=50, grad_threshold=0.20,
                                     min_area=500):
    """Depth-guided splitting + confidence-weighted majority vote.

    For each thing instance, reassign class using weighted vote from
    semantic softmax probabilities (A-3 + A-4 combined).
    """
    H, W = pred_sem.shape
    pred_pan = np.zeros((H, W), dtype=np.int32)
    pred_segments = {}
    nxt = 1

    # Stuff: UNet predictions directly
    for cls in _STUFF_IDS:
        mask = pred_sem == cls
        if mask.sum() < 64:
            continue
        pred_pan[mask] = nxt
        pred_segments[nxt] = cls
        nxt += 1

    # Things: depth-guided splitting with confidence-weighted class vote
    instances = depth_guided_instances(
        pred_sem, depth,
        grad_threshold=grad_threshold,
        min_area=min_area,
    )
    for mask, orig_cls, score in instances:
        if pred_probs is not None:
            # Confidence-weighted vote across all pixels in this instance
            probs_in_mask = pred_probs[:, mask]  # (19, N)
            weighted_sum = probs_in_mask.sum(axis=1)  # (19,)
            voted_cls = int(weighted_sum.argmax())
            # Only use vote if it's a thing class; otherwise keep original
            if voted_cls in _THING_IDS:
                cls = voted_cls
            else:
                cls = orig_cls
        else:
            cls = orig_cls

        pred_pan[mask] = nxt
        pred_segments[nxt] = cls
        nxt += 1

    return pred_pan, pred_segments


def compute_pq(pred_pan, pred_segments, gt_pan, gt_segments, num_cls=19):
    """Compute per-category PQ between predicted and GT panoptic maps."""
    tp = np.zeros(num_cls)
    fp = np.zeros(num_cls)
    fn = np.zeros(num_cls)
    iou_sum = np.zeros(num_cls)

    gt_by_cat = defaultdict(list)
    for sid, cat in gt_segments.items():
        gt_by_cat[cat].append(sid)
    pred_by_cat = defaultdict(list)
    for sid, cat in pred_segments.items():
        pred_by_cat[cat].append(sid)

    matched_pred = set()
    for cat in range(num_cls):
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
                tp[cat] += 1
                iou_sum[cat] += best_iou
                matched_pred.add(best_pid)
            else:
                fn[cat] += 1

        # Unmatched predictions = FP
        for pid in pred_by_cat.get(cat, []):
            if pid not in matched_pred:
                fp[cat] += 1

    return tp, fp, fn, iou_sum


def summarize_pq(tp, fp, fn, iou_sum, label=""):
    """Print PQ summary."""
    num_cls = len(tp)
    pq_per = np.zeros(num_cls)
    sq_per = np.zeros(num_cls)
    rq_per = np.zeros(num_cls)

    for c in range(num_cls):
        if tp[c] == 0:
            continue
        sq_per[c] = iou_sum[c] / tp[c]
        rq_per[c] = tp[c] / (tp[c] + 0.5 * fp[c] + 0.5 * fn[c])
        pq_per[c] = sq_per[c] * rq_per[c]

    # Compute averages
    active = (tp + fn > 0) | (fp > 0)
    stuff_mask = np.array([i in _STUFF_IDS for i in range(num_cls)])
    thing_mask = np.array([i in _THING_IDS for i in range(num_cls)])

    pq_all = pq_per[active].mean() * 100 if active.any() else 0
    pq_stuff = pq_per[active & stuff_mask].mean() * 100 if (active & stuff_mask).any() else 0
    pq_things = pq_per[active & thing_mask].mean() * 100 if (active & thing_mask).any() else 0

    log.info(f"\n{'='*60}")
    log.info(f"Panoptic Quality — {label}")
    log.info(f"{'='*60}")
    log.info(f"  PQ:       {pq_all:.2f}")
    log.info(f"  PQ_stuff: {pq_stuff:.2f}")
    log.info(f"  PQ_things:{pq_things:.2f}")
    log.info(f"  TP={int(tp.sum())}, FP={int(fp.sum())}, FN={int(fn.sum())}")
    log.info(f"\nPer-class:")
    for c in range(num_cls):
        kind = "stuff" if c in _STUFF_IDS else "thing"
        log.info(f"  {_CS_CLASS_NAMES[c]:20s} ({kind}): "
                 f"PQ={pq_per[c]*100:5.1f}  SQ={sq_per[c]*100:5.1f}  "
                 f"RQ={rq_per[c]*100:5.1f}  "
                 f"TP={int(tp[c]):5d}  FP={int(fp[c]):5d}  FN={int(fn[c]):5d}")

    return pq_all, pq_stuff, pq_things


def build_gt_panoptic(gt_sem, gt_inst, H, W):
    """Build GT panoptic map from semantic + instance GT."""
    gt_pan = np.zeros((H, W), dtype=np.int32)
    gt_segments = {}
    nxt = 1

    for cls in _STUFF_IDS:
        mask = gt_sem == cls
        if mask.sum() < 64:
            continue
        gt_pan[mask] = nxt
        gt_segments[nxt] = cls
        nxt += 1

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
        gt_pan[mask] = nxt
        gt_segments[nxt] = tid
        nxt += 1

    return gt_pan, gt_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="checkpoints/hires_unet_depth_guided/best.pth")
    parser.add_argument("--cityscapes_root",
                        default="/Users/qbit-glitch/Desktop/datasets/cityscapes")
    parser.add_argument("--method", default="all",
                        choices=["baseline", "depth_split", "depth_split_vote", "all"])
    parser.add_argument("--eval_hw", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--cc_min_area", type=int, default=50)
    parser.add_argument("--grad_threshold", type=float, default=0.20)
    parser.add_argument("--depth_min_area", type=int, default=500)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    # Load model
    model, cfg = load_model(args.checkpoint, device)
    depth_subdir = cfg.get("depth_subdir", "depth_dav3")
    feature_subdir = cfg.get("feature_subdir", "dinov3_features")
    num_classes = cfg.get("num_classes", 19)

    # Determine cluster→trainID LUT if overclustered
    cluster_to_trainid_lut = None
    semantic_subdir = cfg.get("semantic_subdir", "pseudo_semantic_mapped_k80")
    if num_classes > 19:
        centroids_path = cfg.get("centroids_path")
        if centroids_path and os.path.exists(centroids_path):
            data = np.load(centroids_path, allow_pickle=True)
            cluster_to_trainid_lut = data["cluster_to_trainid"]
            log.info(f"Loaded cluster→trainID LUT ({num_classes}→19)")

    # Find val images
    H, W = args.eval_hw
    img_dir = os.path.join(args.cityscapes_root, "leftImg8bit", "val")
    gt_dir = os.path.join(args.cityscapes_root, "gtFine", "val")

    entries = []
    for city in sorted(os.listdir(img_dir)):
        city_path = os.path.join(img_dir, city)
        if not os.path.isdir(city_path):
            continue
        for fname in sorted(os.listdir(city_path)):
            if not fname.endswith("_leftImg8bit.png"):
                continue
            stem = fname.replace("_leftImg8bit.png", "")
            entries.append((city, stem))
    log.info(f"Found {len(entries)} val images")

    methods = (["baseline", "depth_split", "depth_split_vote"]
               if args.method == "all"
               else [args.method])

    # Per-method accumulators
    results = {}
    for method in methods:
        results[method] = {
            "tp": np.zeros(19),
            "fp": np.zeros(19),
            "fn": np.zeros(19),
            "iou_sum": np.zeros(19),
            "confusion": np.zeros((19, 19), dtype=np.int64),
        }

    # Run evaluation
    t0 = time.time()
    with torch.no_grad():
        for idx, (city, stem) in enumerate(tqdm(entries, desc="Evaluating")):
            # Load DINOv2 features
            feat_path = os.path.join(
                args.cityscapes_root, feature_subdir, "val", city,
                f"{stem}_leftImg8bit.npy")
            features = np.load(feat_path).astype(np.float32)
            features = features.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)
            feat_t = torch.from_numpy(features).unsqueeze(0).to(device)

            # Load depth
            depth_path = os.path.join(
                args.cityscapes_root, depth_subdir, "val", city,
                f"{stem}_leftImg8bit.npy")
            depth_full = np.load(depth_path)  # (512, 1024)

            depth_patch = torch.from_numpy(depth_full).float().unsqueeze(0).unsqueeze(0)
            depth_patch = F.interpolate(
                depth_patch, size=(PATCH_H, PATCH_W),
                mode="bilinear", align_corners=False)
            depth_t = depth_patch.to(device)

            # Compute Sobel gradients on depth
            d_np = depth_patch.squeeze().numpy()
            gx = np.gradient(d_np, axis=1)
            gy = np.gradient(d_np, axis=0)
            depth_grads = np.stack([gx, gy], axis=0).astype(np.float32)
            grads_t = torch.from_numpy(depth_grads).unsqueeze(0).to(device)

            # Forward pass
            logits = model(feat_t, depth_t, grads_t)  # (1, C, target_h, target_w)

            # Get predictions at eval resolution
            logits_up = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False)
            pred_probs_t = F.softmax(logits_up, dim=1)

            if num_classes > 19 and cluster_to_trainid_lut is not None:
                pred_cls = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()
                pred_sem = cluster_to_trainid_lut[pred_cls].astype(np.uint8)
                # Re-derive probs in 19-class space
                probs_np = pred_probs_t.squeeze(0).cpu().numpy()
                pred_probs_19 = np.zeros((19, H, W), dtype=np.float32)
                for k in range(num_classes):
                    tid = int(cluster_to_trainid_lut[k])
                    if tid < 19:
                        pred_probs_19[tid] += probs_np[k]
            else:
                pred_sem = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_probs_19 = pred_probs_t.squeeze(0).cpu().numpy()

            # Load GT
            gt_sem_path = os.path.join(
                gt_dir, city, f"{stem}_gtFine_labelIds.png")
            gt_inst_path = os.path.join(
                gt_dir, city, f"{stem}_gtFine_instanceIds.png")
            if not os.path.exists(gt_sem_path):
                continue
            gt_raw = np.array(Image.open(gt_sem_path))
            gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
            for raw_id, tid in _CS_ID_TO_TRAIN.items():
                gt_sem[gt_raw == raw_id] = tid
            if gt_sem.shape != (H, W):
                gt_sem = np.array(
                    Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

            gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
            if gt_inst.shape != (H, W):
                gt_inst = np.array(
                    Image.fromarray(gt_inst).resize((W, H), Image.NEAREST))

            gt_pan, gt_segments = build_gt_panoptic(gt_sem, gt_inst, H, W)

            # Resize depth to eval resolution for depth-guided splitting
            depth_eval = np.array(
                Image.fromarray(depth_full).resize((W, H), Image.BILINEAR))

            # mIoU confusion (same for all methods since semantic pred is same)
            valid = (gt_sem < 19) & (pred_sem < 19)
            if valid.sum() > 0:
                for method in methods:
                    np.add.at(results[method]["confusion"],
                              (gt_sem[valid], pred_sem[valid]), 1)
                    break  # confusion same for all methods

            # Evaluate each method
            for method in methods:
                if method == "baseline":
                    pan, segs = build_panoptic_cc(pred_sem, args.cc_min_area)
                elif method == "depth_split":
                    pan, segs = build_panoptic_depth_split(
                        pred_sem, depth_eval, args.cc_min_area,
                        args.grad_threshold, args.depth_min_area)
                elif method == "depth_split_vote":
                    pan, segs = build_panoptic_depth_split_vote(
                        pred_sem, pred_probs_19, depth_eval,
                        args.cc_min_area, args.grad_threshold,
                        args.depth_min_area)

                tp, fp, fn, iou_s = compute_pq(pan, segs, gt_pan, gt_segments)
                results[method]["tp"] += tp
                results[method]["fp"] += fp
                results[method]["fn"] += fn
                results[method]["iou_sum"] += iou_s

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed:.1f}s")

    # Print mIoU (same for all methods)
    conf = results[methods[0]]["confusion"]
    intersection = np.diag(conf)
    union = conf.sum(axis=1) + conf.sum(axis=0) - intersection
    iou_per = np.where(union > 0, intersection / union, 0)
    active_cls = union > 0
    miou = iou_per[active_cls].mean() * 100
    log.info(f"\nSemantic mIoU: {miou:.2f}%")
    for c in range(19):
        if active_cls[c]:
            log.info(f"  {_CS_CLASS_NAMES[c]:20s}: IoU = {iou_per[c]*100:.1f}%")

    # Print PQ for each method
    summary = {}
    for method in methods:
        r = results[method]
        pq, pqs, pqt = summarize_pq(
            r["tp"], r["fp"], r["fn"], r["iou_sum"], label=method)
        summary[method] = {"PQ": pq, "PQ_stuff": pqs, "PQ_things": pqt}

    # Final comparison
    log.info(f"\n{'='*60}")
    log.info("ALIGNMENT COMPARISON")
    log.info(f"{'='*60}")
    log.info(f"{'Method':<25s} {'PQ':>8s} {'PQ_stuff':>10s} {'PQ_things':>10s}")
    log.info("-" * 55)
    for method in methods:
        s = summary[method]
        log.info(f"{method:<25s} {s['PQ']:8.2f} {s['PQ_stuff']:10.2f} "
                 f"{s['PQ_things']:10.2f}")
    log.info(f"{'CUPS (reference)':<25s} {'27.80':>8s} {'~34':>10s} {'~18':>10s}")


if __name__ == "__main__":
    main()
