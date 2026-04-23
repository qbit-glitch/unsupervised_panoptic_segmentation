#!/usr/bin/env python3
"""
Evaluate CAUSE-TR checkpoints (DINOv2 ViT-B/14 vs DINOv3 ViT-L/16) with K=80
overclustering on Cityscapes val.

Extracts 90-dim Segment_TR features for all val images, runs k-means K=80,
uses many-to-one Hungarian matching → mIoU.

Usage:
    # DINOv2 ViT-B/14 (original CAUSE):
    python mbps_pytorch/eval_cause_k80.py \
        --backbone dinov2 \
        --cityscapes_root /path/to/cityscapes

    # DINOv3 ViT-L/16 (retrained):
    python mbps_pytorch/eval_cause_k80.py \
        --backbone dinov3 \
        --cityscapes_root /path/to/cityscapes
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from tqdm import tqdm

# Add CAUSE repo to path
CAUSE_DIR = str(Path(__file__).resolve().parent.parent / "refs" / "cause")
if CAUSE_DIR not in sys.path:
    sys.path.insert(0, CAUSE_DIR)

from modules.segment import Segment_TR
from modules.segment_module import transform, untransform
from models.dinov2vit import dinov2_vit_base_14  # must import before MBPS_DIR is prepended

from mbps_pytorch.models.adapters import (
    inject_lora_into_dinov2,
    inject_lora_into_cause_tr,
    freeze_non_adapter_params,
    count_adapter_params,
    set_dinov2_spatial_dims,
)

# ── Cityscapes panoptic helpers (inlined to avoid FalconKwayCut import chain) ─

_THING_IDS = set(range(11, 19))   # person … bicycle
_STUFF_IDS  = set(range(0, 11))   # road … sky
_THING_RAW_IDS = {24, 25, 26, 27, 28, 31, 32, 33}

_CS_ID_TO_TRAIN_FULL = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}


def remap_labelids(gt: np.ndarray) -> np.ndarray:
    out = np.full_like(gt, 255, dtype=np.uint8)
    for raw_id, tid in _CS_ID_TO_TRAIN_FULL.items():
        out[gt == raw_id] = tid
    return out


def stem_to_gt_inst(cs_root: Path, stem: str):
    base = stem.replace("_leftImg8bit", "")
    city = base.split("_")[0]
    cand = cs_root / "gtFine" / "val" / city / f"{base}_gtFine_instanceIds.png"
    return cand if cand.exists() else None


def stem_to_instance(cs_root: Path, stem: str):
    base = stem.replace("_leftImg8bit", "")
    city = base.split("_")[0]
    for name in [base, stem]:
        cand = cs_root / "pseudo_instance_spidepth" / "val" / city / f"{name}.npz"
        if cand.exists():
            return cand
    return None


def load_pred_instances(inst_path, image_h: int = 1024, image_w: int = 2048):
    data = np.load(str(inst_path))
    masks = data["masks"]
    if "num_valid" in data:
        masks = masks[:int(data["num_valid"])]
    if masks.shape[0] == 0:
        return None
    h_p = int(data["h_patches"]) if "h_patches" in data else image_h // 16
    w_p = int(data["w_patches"]) if "w_patches" in data else image_w // 16
    M, N = masks.shape
    if N == h_p * w_p:
        masks = masks.reshape(M, h_p, w_p)
    resized = []
    for m in masks:
        m_img = Image.fromarray(m.astype(np.uint8) * 255)
        m_img = m_img.resize((image_w, image_h), Image.NEAREST)
        resized.append(np.array(m_img) > 127)
    return np.stack(resized)


def compute_panoptic_quality(pred_sem, pred_inst_masks, gt_label_path, gt_inst_path,
                              iou_threshold=0.5):
    H, W = pred_sem.shape
    gt_label = remap_labelids(
        np.array(Image.open(gt_label_path).resize((W, H), Image.NEAREST))
    )
    gt_inst_map = np.array(
        Image.open(gt_inst_path).resize((W, H), Image.NEAREST), dtype=np.int32
    )
    NUM_CLS = 19

    # GT segments
    gt_segs = {}
    sid = 1
    for tid in _STUFF_IDS:
        mask = gt_label == tid
        if mask.sum() == 0:
            continue
        gt_segs[sid] = {"mask": mask, "class_id": tid, "is_thing": False}
        sid += 1
    for uid in np.unique(gt_inst_map):
        if uid < 1000:
            continue
        raw_class = uid // 1000
        if raw_class not in _CS_ID_TO_TRAIN_FULL:
            continue
        tid = _CS_ID_TO_TRAIN_FULL[raw_class]
        mask = gt_inst_map == uid
        if mask.sum() < 10:
            continue
        gt_segs[sid] = {"mask": mask, "class_id": tid, "is_thing": True}
        sid += 1

    # Predicted segments
    pred_segs = {}
    sid = 1
    for tid in _STUFF_IDS:
        mask = pred_sem == tid
        if mask.sum() == 0:
            continue
        pred_segs[sid] = {"mask": mask, "class_id": tid, "is_thing": False}
        sid += 1
    if pred_inst_masks is not None:
        for m in pred_inst_masks:
            if m.sum() < 10:
                continue
            votes = pred_sem[m]
            votes = votes[votes < NUM_CLS]
            if len(votes) == 0:
                continue
            counts = np.bincount(votes.astype(np.int32), minlength=NUM_CLS)
            thing_counts = {c: counts[c] for c in _THING_IDS if counts[c] > 0}
            if not thing_counts:
                continue
            class_id = max(thing_counts, key=thing_counts.get)
            pred_segs[sid] = {"mask": m.astype(bool), "class_id": class_id, "is_thing": True}
            sid += 1

    # Match
    tp  = np.zeros(NUM_CLS)
    fp  = np.zeros(NUM_CLS)
    fn  = np.zeros(NUM_CLS)
    iou_sum = np.zeros(NUM_CLS)

    matched_gt = set()
    matched_pred = set()

    for pid, ps in pred_segs.items():
        best_iou, best_gid = 0.0, None
        for gid, gs in gt_segs.items():
            if gs["class_id"] != ps["class_id"]:
                continue
            if gid in matched_gt:
                continue
            inter = float((ps["mask"] & gs["mask"]).sum())
            union = float((ps["mask"] | gs["mask"]).sum())
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou, best_gid = iou, gid
        if best_iou >= iou_threshold and best_gid is not None:
            c = ps["class_id"]
            tp[c]      += 1
            iou_sum[c] += best_iou
            matched_gt.add(best_gid)
            matched_pred.add(pid)

    for pid, ps in pred_segs.items():
        if pid not in matched_pred:
            fp[ps["class_id"]] += 1
    for gid, gs in gt_segs.items():
        if gid not in matched_gt:
            fn[gs["class_id"]] += 1

    return {"tp": tp, "fp": fp, "fn": fn, "iou_sum": iou_sum}

# ─── Cityscapes constants ────────────────────────────────────────────────────

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
NUM_CLASSES = 19
IGNORE_LABEL = 255
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _remap(gt: np.ndarray) -> np.ndarray:
    out = np.full_like(gt, IGNORE_LABEL, dtype=np.uint8)
    for raw, tid in _CS_ID_TO_TRAIN.items():
        out[gt == raw] = tid
    return out


# ─── Model loading ───────────────────────────────────────────────────────────

def _has_lora_keys(state_dict) -> bool:
    """Detect whether a checkpoint was saved from an adapter-wrapped model."""
    return any(
        k.endswith(("lora_A", "lora_B", "lora_magnitude"))
        or ".lora_" in k
        or "dwconv" in k
        or "conv_gate" in k
        for k in state_dict.keys()
    )


def _load_state_checked(module, state_dict, component_name, require_lora):
    """Load state_dict and fail loudly if LoRA/DoRA keys were silently dropped."""
    result = module.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    dropped_lora = [
        k for k in unexpected
        if k.endswith(("lora_A", "lora_B", "lora_magnitude"))
        or ".lora_" in k
        or "dwconv" in k
        or "conv_gate" in k
    ]
    if dropped_lora:
        raise RuntimeError(
            f"[{component_name}] {len(dropped_lora)} LoRA/DoRA parameters were dropped because "
            f"the receiver has no adapter wrappers. First few: {dropped_lora[:5]}. "
            f"Call inject_lora_into_* BEFORE load_state_dict."
        )
    if missing:
        base_missing = [k for k in missing if ".lora_" not in k]
        if base_missing:
            print(f"WARNING [{component_name}] {len(base_missing)} base params missing (expected 0). "
                  f"First few: {base_missing[:5]}")
    if require_lora:
        lora_loaded = sum(
            1
            for k in state_dict.keys()
            if k.endswith(("lora_A", "lora_B", "lora_magnitude"))
            or "dwconv" in k
            or "conv_gate" in k
        )
        print(f"[{component_name}] {lora_loaded} LoRA-style parameters loaded successfully.")


def load_dinov2(
    device: torch.device,
    adapter_checkpoint: str = None,
    variant: str = "dora",
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.0,
    late_block_start: int = 6,
    adapt_cause: bool = False,
):
    """CAUSE DINOv2 ViT-B/14 + Segment_TR. Optionally load DoRA adapter checkpoint."""
    from models.dinov2vit import dinov2_vit_base_14

    cause_args = SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )
    ckpt_root = Path(__file__).resolve().parent.parent / "refs" / "cause"

    # Backbone (dinov2_vit_base_14 imported at module top)
    net = dinov2_vit_base_14()
    state = torch.load(ckpt_root / "checkpoint" / "dinov2_vit_base_14.pth",
                       map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=False)

    # Segment_TR
    seg_path = ckpt_root / "CAUSE" / "cityscapes" / "dinov2_vit_base_14" / "2048" / "segment_tr.pth"
    segment = Segment_TR(cause_args).to(device)
    segment.load_state_dict(torch.load(seg_path, map_location="cpu", weights_only=True), strict=False)

    # Codebook
    mod_path = ckpt_root / "CAUSE" / "cityscapes" / "modularity" / "dinov2_vit_base_14" / "2048" / "modular.npy"
    if mod_path.exists():
        cb = torch.from_numpy(np.load(str(mod_path))).to(device)
        segment.head.codebook    = cb
        segment.head_ema.codebook = cb

    # ── Inject DoRA adapters if requested ────────────────────────────────────
    use_adapter = adapter_checkpoint is not None
    if use_adapter:
        print(f"Loading adapter checkpoint: {adapter_checkpoint}")
        ckpt = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)

        # Resolve adapter config: checkpoint metadata > CLI defaults
        ckpt_config = ckpt.get("adapter_config", {}) if isinstance(ckpt, dict) else {}
        cfg_variant = ckpt_config.get("variant", variant)
        cfg_rank = ckpt_config.get("rank", rank)
        cfg_alpha = ckpt_config.get("alpha", alpha)
        cfg_dropout = ckpt_config.get("dropout", dropout)
        cfg_late_block_start = ckpt_config.get("late_block_start", late_block_start)
        cfg_adapt_cause = ckpt_config.get("adapt_cause", adapt_cause)

        ckpt_has_lora = _has_lora_keys(ckpt.get("backbone", {}))
        if ckpt_has_lora and not ckpt_config:
            raise RuntimeError(
                "Checkpoint contains LoRA weights but no adapter_config metadata. "
                "Pass --variant/--rank/--alpha/--late_block_start matching training."
            )

        # Inject into backbone
        inject_lora_into_dinov2(
            net,
            variant=cfg_variant,
            rank=cfg_rank,
            alpha=cfg_alpha,
            dropout=cfg_dropout,
            late_block_start=cfg_late_block_start,
        )
        freeze_non_adapter_params(net)

        # Inject into CAUSE-TR head if configured
        if cfg_adapt_cause:
            inject_lora_into_cause_tr(
                segment,
                variant=cfg_variant,
                rank=cfg_rank,
                alpha=cfg_alpha,
                dropout=cfg_dropout,
                adapt_head=True,
                adapt_projection=False,
                adapt_ema=False,
            )
        freeze_non_adapter_params(segment)

        # Load trained adapter weights
        _load_state_checked(net, ckpt["backbone"], "backbone", require_lora=True)
        _load_state_checked(segment, ckpt["segment"], "segment",
                            require_lora=cfg_adapt_cause)

        # Re-pin codebook (checkpoint may carry stale copy)
        segment.head.codebook = cb.clone().to(device)
        segment.head_ema.codebook = cb.clone().to(device)

        n_adapter = count_adapter_params(net) + count_adapter_params(segment)
        print(f"Adapter params loaded: {n_adapter:,}")
        if n_adapter == 0:
            raise RuntimeError("use_adapter=True but 0 adapter params after injection+load.")

    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    segment = segment.to(device).eval()
    for p in segment.parameters():
        p.requires_grad = False

    print(f"DINOv2 ViT-B/14 backbone: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net, segment, cause_args, use_adapter


def load_dinov3(device: torch.device, ckpt_dir: str = None):
    """CAUSE DINOv3 ViT-L/16 + Segment_TR (retrained heads in CAUSE_dinov3/)."""
    mbps_dir = Path(__file__).resolve().parent
    if str(mbps_dir) not in sys.path:
        sys.path.insert(0, str(mbps_dir))
    from train_cause_dinov3 import DINOv3Backbone

    cause_args = SimpleNamespace(
        dim=1024, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=20 * 20, crop_size=320, patch_size=16,
    )
    if ckpt_dir is None:
        ckpt_dir = Path(__file__).resolve().parent.parent / "refs" / "cause" / "CAUSE_dinov3" / "final"
    else:
        ckpt_dir = Path(ckpt_dir)

    # Backbone (frozen DINOv3 ViT-L/16)
    net = DINOv3Backbone(device=device).to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    # Segment_TR
    seg_path = ckpt_dir / "segment_tr.pth"
    segment = Segment_TR(cause_args).to(device)
    segment.load_state_dict(torch.load(seg_path, map_location="cpu", weights_only=True), strict=False)
    segment.eval()

    # Codebook
    mod_path = ckpt_dir / "modular.npy"
    if mod_path.exists():
        cb = torch.from_numpy(np.load(str(mod_path))).to(device)
        segment.head.codebook    = cb
        segment.head_ema.codebook = cb

    print(f"DINOv3 ViT-L/16 backbone: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net, segment, cause_args


# ─── Feature extraction ──────────────────────────────────────────────────────

def _extract_single_crop(net, segment, crop_tensor: torch.Tensor, use_adapter: bool = False) -> torch.Tensor:
    """
    Run backbone + Segment_TR on exactly one crop_size×crop_size image tensor.
    Returns (90, ph, pw) CAUSE features.

    When adapters are active, use segment.head (student) instead of head_ema (frozen teacher),
    since adapters were injected into the student head only.
    """
    head = segment.head if use_adapter else segment.head_ema
    with torch.no_grad():
        img4 = crop_tensor.unsqueeze(0)
        set_dinov2_spatial_dims(net, h_patches=23, w_patches=23)
        feat      = net(img4)[:, 1:, :]
        set_dinov2_spatial_dims(net, h_patches=23, w_patches=23)
        feat_flip = net(img4.flip(dims=[3]))[:, 1:, :]
        seg      = transform(head(feat))
        seg_flip = transform(head(feat_flip))
        seg = (seg + seg_flip.flip(dims=[3])) / 2  # (1, 90, ph, pw)
    return seg.squeeze(0)  # (90, ph, pw)


def extract_all_features(net, segment, cause_args, val_imgs, val_gts, device, use_adapter: bool = False):
    """
    Sliding-window feature extraction. Cityscapes 1024×2048 is processed in
    non-overlapping crop_size×crop_size crops. CAUSE Segment_TR needs exactly
    num_queries = (crop_size/patch_size)^2 patches.

    Returns lists of (90, nH*ph, nW*pw) feature maps and matching GT maps.
    """
    crop_size  = cause_args.crop_size
    patch_size = cause_args.patch_size
    ph = crop_size // patch_size  # patches per crop side

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    to_tensor = transforms.ToTensor()

    feat_maps = []
    gt_maps   = []

    for img_path, gt_path in tqdm(zip(val_imgs, val_gts), total=len(val_imgs),
                                  desc="Extracting features"):
        img    = Image.open(img_path).convert("RGB")
        gt_raw = np.array(Image.open(gt_path))
        gt     = _remap(gt_raw)

        # Resize so dimensions are divisible by crop_size (keep ~2:1 ratio)
        orig_w, orig_h = img.size
        scale = crop_size / min(orig_h, orig_w)  # at least one crop fits
        tgt_h = int(math.ceil(orig_h * scale / crop_size) * crop_size)
        tgt_w = int(math.ceil(orig_w * scale / crop_size) * crop_size)

        img_r = img.resize((tgt_w, tgt_h), Image.BILINEAR)
        img_t = normalize(to_tensor(img_r)).to(device)  # (3, tgt_h, tgt_w)

        n_crops_h = tgt_h // crop_size
        n_crops_w = tgt_w // crop_size
        feat_h = n_crops_h * ph
        feat_w = n_crops_w * ph

        # Process each crop and stitch
        feat_grid = torch.zeros(90, feat_h, feat_w, device=device)
        for iy in range(n_crops_h):
            for ix in range(n_crops_w):
                y0, y1 = iy * crop_size, (iy + 1) * crop_size
                x0, x1 = ix * crop_size, (ix + 1) * crop_size
                crop = img_t[:, y0:y1, x0:x1]
                f = _extract_single_crop(net, segment, crop, use_adapter=use_adapter)  # (90, ph, ph)
                fy0, fy1 = iy * ph, (iy + 1) * ph
                fx0, fx1 = ix * ph, (ix + 1) * ph
                feat_grid[:, fy0:fy1, fx0:fx1] = f

        # Resize GT to feature grid resolution
        gt_small = np.array(
            Image.fromarray(gt).resize((feat_w, feat_h), Image.NEAREST)
        )

        feat_maps.append(feat_grid.cpu().float().numpy())
        gt_maps.append(gt_small)

    return feat_maps, gt_maps


# ─── K-means evaluation ──────────────────────────────────────────────────────

def evaluate_kmeans(feat_maps, gt_maps, k: int, backbone_name: str,
                    val_imgs=None, cs_root=None):
    print(f"\n{'='*60}")
    print(f"K-means K={k} on {backbone_name} CAUSE features (90-dim)")
    print(f"{'='*60}")

    # Collect all valid (feat, gt) pairs
    all_feats, all_gts = [], []
    for fm, gm in zip(feat_maps, gt_maps):
        D, H, W = fm.shape
        feats = fm.reshape(D, -1).T  # (H*W, 90)
        labels = gm.flatten()
        valid = labels < NUM_CLASSES
        all_feats.append(feats[valid])
        all_gts.append(labels[valid])

    feats_all = np.concatenate(all_feats).astype(np.float32)
    gts_all   = np.concatenate(all_gts)
    feats_all = np.nan_to_num(feats_all, nan=0., posinf=0., neginf=0.)

    # L2-normalise — sanitize again after division to catch edge cases
    norms = np.linalg.norm(feats_all, axis=1, keepdims=True)
    feats_norm = feats_all / np.maximum(norms, 1e-8)
    feats_norm = np.nan_to_num(feats_norm, nan=0., posinf=0., neginf=0.)

    # Fit k-means
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, batch_size=10000, max_iter=300,
                         random_state=42, n_init=3, verbose=0)
    km.fit(feats_norm)
    print(f"K-means fit in {time.time()-t0:.1f}s on {len(feats_norm):,} vectors")

    centroids = np.nan_to_num(km.cluster_centers_, nan=0., posinf=0., neginf=0.)
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    centroids_norm = np.nan_to_num(centroids_norm, nan=0., posinf=0., neginf=0.)

    # Build confusion matrix (k, C) using all pixels
    t1 = time.time()
    cluster_preds = (feats_norm @ centroids_norm.T).argmax(axis=1)  # (N,)
    conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    np.add.at(conf, (cluster_preds, gts_all), 1)

    # Hungarian matching: maximize total correct pixels (CUPS-standard)
    row_ind, col_ind = linear_sum_assignment(-conf)
    cluster_to_class = np.full(k, -1, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        cluster_to_class[r] = c
    # Argmax fallback for unmatched clusters
    for kk in range(k):
        if cluster_to_class[kk] == -1 and conf[kk].sum() > 0:
            cluster_to_class[kk] = int(conf[kk].argmax())

    print(f"Confusion + Hungarian in {time.time()-t1:.1f}s")

    # Per-class IoU
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn = np.zeros(NUM_CLASSES, dtype=np.int64)

    for fm, gm in zip(feat_maps, gt_maps):
        D, H, W = fm.shape
        feats = fm.reshape(D, -1).T.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0., posinf=0., neginf=0.)
        n = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_n = feats / np.maximum(n, 1e-8)
        feats_n = np.nan_to_num(feats_n, nan=0., posinf=0., neginf=0.)

        cluster_ids = (feats_n @ centroids_norm.T).argmax(axis=1)
        pred_cls    = cluster_to_class[cluster_ids]
        gt_flat     = gm.flatten()
        valid       = gt_flat < NUM_CLASSES

        for c in range(NUM_CLASSES):
            gc = (gt_flat == c) & valid
            pc = (pred_cls == c) & valid
            tp[c] += int((gc & pc).sum())
            fp[c] += int((~gc & pc & valid).sum())
            fn[c] += int((gc & ~pc).sum())

    denom    = tp + fp + fn
    iou      = np.where(denom > 0, tp / denom, 0.0)
    active   = denom > 0
    miou     = float(iou[active].mean() * 100)

    print(f"\nmIoU = {miou:.2f}%  ({active.sum()}/{NUM_CLASSES} active classes)")
    print("\nPer-class IoU:")
    for c in range(NUM_CLASSES):
        marker = "  " if active[c] else "✗ "
        print(f"  {marker}{_CS_CLASS_NAMES[c]:15s}: {iou[c]*100:5.1f}%")

    # ── Panoptic Quality ─────────────────────────────────────────────────────
    pq_result = None
    if val_imgs is not None and cs_root is not None:
        print("\nComputing Panoptic Quality (PQ)...")
        cs_p = Path(cs_root)
        IMAGE_H, IMAGE_W = 1024, 2048

        pq_tp  = np.zeros(NUM_CLASSES)
        pq_fp  = np.zeros(NUM_CLASSES)
        pq_fn  = np.zeros(NUM_CLASSES)
        pq_iou = np.zeros(NUM_CLASSES)

        for img_path, fm, gm in tqdm(zip(val_imgs, feat_maps, gt_maps),
                                     total=len(val_imgs), desc="PQ eval"):
            # Derive stem from img_path
            stem = Path(img_path).stem  # e.g. frankfurt_000000_000019_leftImg8bit
            gt_inst_path = stem_to_gt_inst(cs_p, stem)
            if gt_inst_path is None:
                continue

            # Build full-res semantic prediction
            D, feat_h, feat_w = fm.shape
            feats = fm.reshape(D, -1).T.astype(np.float32)
            feats = np.nan_to_num(feats, nan=0., posinf=0., neginf=0.)
            n = np.linalg.norm(feats, axis=1, keepdims=True)
            feats_n = feats / np.maximum(n, 1e-8)

            cluster_ids = (feats_n @ centroids_norm.T).argmax(axis=1)
            pred_cls_small = cluster_to_class[cluster_ids].reshape(feat_h, feat_w)

            # Upsample to IMAGE_H × IMAGE_W
            pred_sem = np.array(
                Image.fromarray(pred_cls_small.astype(np.int32), mode="I")
                     .resize((IMAGE_W, IMAGE_H), Image.NEAREST),
                dtype=np.int32,
            )

            # Load depth-guided instances
            inst_path = stem_to_instance(cs_p, stem)
            pred_inst_masks = None
            if inst_path is not None:
                pred_inst_masks = load_pred_instances(inst_path, IMAGE_H, IMAGE_W)

            gt_label_path = cs_p / "gtFine" / "val" / stem.split("_")[0] / \
                            f"{stem.replace('_leftImg8bit','')}_gtFine_labelIds.png"
            if not gt_label_path.exists():
                continue

            pq_img = compute_panoptic_quality(
                pred_sem.astype(np.uint8), pred_inst_masks,
                gt_label_path, gt_inst_path,
            )
            pq_tp  += pq_img["tp"]
            pq_fp  += pq_img["fp"]
            pq_fn  += pq_img["fn"]
            pq_iou += pq_img["iou_sum"]

        # Aggregate PQ
        sq_per = np.where(pq_tp > 0, pq_iou / pq_tp, 0.0)
        rq_per = np.where((pq_tp + 0.5*pq_fp + 0.5*pq_fn) > 0,
                          pq_tp / (pq_tp + 0.5*pq_fp + 0.5*pq_fn), 0.0)
        pq_per = sq_per * rq_per

        def _agg(ids):
            active_ids = [c for c in ids if (pq_tp[c] + pq_fn[c]) > 0]
            if not active_ids:
                return 0.0, 0.0, 0.0
            return (float(np.mean([pq_per[c] for c in active_ids]))*100,
                    float(np.mean([sq_per[c] for c in active_ids]))*100,
                    float(np.mean([rq_per[c] for c in active_ids]))*100)

        pq_all, sq_all, rq_all = _agg(list(range(NUM_CLASSES)))
        pq_th,  sq_th,  rq_th  = _agg(list(_THING_IDS))
        pq_st,  sq_st,  rq_st  = _agg(list(_STUFF_IDS))

        print(f"\nPQ={pq_all:.2f}%  SQ={sq_all:.2f}%  RQ={rq_all:.2f}%")
        print(f"PQ_things={pq_th:.2f}%  PQ_stuff={pq_st:.2f}%")

        pq_result = {"pq": round(pq_all,2), "sq": round(sq_all,2), "rq": round(rq_all,2),
                     "pq_things": round(pq_th,2), "pq_stuff": round(pq_st,2)}

    return miou, iou, pq_result


# ─── Dataset helpers ─────────────────────────────────────────────────────────

def get_val_files(cs_root: str):
    """Return sorted lists of (img_path, gt_path) for Cityscapes val."""
    img_dir = Path(cs_root) / "leftImg8bit" / "val"
    gt_dir  = Path(cs_root) / "gtFine" / "val"

    img_paths, gt_paths = [], []
    for city in sorted(img_dir.iterdir()):
        for img_path in sorted(city.glob("*_leftImg8bit.png")):
            stem = img_path.stem.replace("_leftImg8bit", "")
            gt_path = gt_dir / city.name / f"{stem}_gtFine_labelIds.png"
            if gt_path.exists():
                img_paths.append(str(img_path))
                gt_paths.append(str(gt_path))

    print(f"Found {len(img_paths)} val images")
    return img_paths, gt_paths


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CAUSE K=80 eval: DINOv2 vs DINOv3 (+ DoRA adapters)")
    p.add_argument("--backbone", choices=["dinov2", "dinov3"], required=True)
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint dir for DINOv3 (default: refs/cause/CAUSE_dinov3/final)",
    )
    p.add_argument("--cityscapes_root", type=str,
                   default="/Users/qbit-glitch/Desktop/datasets/cityscapes")
    p.add_argument("--k_values", type=str, default="80",
                   help="Comma-separated K values (default: 80)")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit val images for quick test (0 = all)")

    # Adapter checkpoint arguments
    p.add_argument("--adapter_checkpoint", type=str, default=None,
                   help="Path to adapter checkpoint (.pt) from train_semantic_adapter.py")
    p.add_argument("--variant", type=str, default="dora",
                   choices=["lora", "dora", "conv_dora"],
                   help="Adapter variant (fallback if not in checkpoint metadata)")
    p.add_argument("--rank", type=int, default=4,
                   help="LoRA rank (fallback if not in checkpoint metadata)")
    p.add_argument("--alpha", type=float, default=4.0,
                   help="LoRA alpha scaling (fallback if not in checkpoint metadata)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout on adapter path (inference usually 0)")
    p.add_argument("--late_block_start", type=int, default=6,
                   help="First block for full adapter injection (fallback)")
    p.add_argument("--adapt_cause", action="store_true",
                   help="CAUSE-TR head was adapted during training (fallback)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    k_values = [int(k) for k in args.k_values.split(",")]

    print(f"Backbone: {args.backbone.upper()}  |  K values: {k_values}  |  Device: {device}")

    # Load model
    if args.backbone == "dinov2":
        net, segment, cause_args, use_adapter = load_dinov2(
            device,
            adapter_checkpoint=args.adapter_checkpoint,
            variant=args.variant,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            late_block_start=args.late_block_start,
            adapt_cause=args.adapt_cause,
        )
        backbone_name = "DINOv2 ViT-B/14 + DoRA" if use_adapter else "DINOv2 ViT-B/14"
    else:
        if args.adapter_checkpoint is not None:
            raise NotImplementedError("DoRA adapter evaluation is only supported for DINOv2 backbone.")
        net, segment, cause_args = load_dinov3(device, ckpt_dir=args.checkpoint_dir)
        backbone_name = "DINOv3 ViT-L/16"

    # Get val files
    val_imgs, val_gts = get_val_files(args.cityscapes_root)
    if args.limit > 0:
        val_imgs = val_imgs[:args.limit]
        val_gts  = val_gts[:args.limit]

    # Extract features
    print(f"\nExtracting CAUSE features (crop={cause_args.crop_size}, "
          f"patch={cause_args.patch_size}, reduced_dim=90)...")
    feat_maps, gt_maps = extract_all_features(
        net, segment, cause_args, val_imgs, val_gts, device, use_adapter=use_adapter
    )

    # Evaluate at each K
    results = {}
    for k in k_values:
        miou, iou_per_cls, pq_result = evaluate_kmeans(
            feat_maps, gt_maps, k, backbone_name,
            val_imgs=val_imgs, cs_root=args.cityscapes_root,
        )
        results[k] = {
            "miou": round(miou, 2),
            "per_class": {_CS_CLASS_NAMES[c]: round(float(iou_per_cls[c])*100, 2)
                          for c in range(NUM_CLASSES)},
        }
        if pq_result:
            results[k].update(pq_result)

    print(f"\n{'='*60}")
    print(f"SUMMARY — {backbone_name} CAUSE features")
    print(f"{'='*60}")
    for k, r in results.items():
        pq_str = f"  PQ={r['pq']:.2f}%  PQ_th={r['pq_things']:.2f}%  PQ_st={r['pq_stuff']:.2f}%" \
                 if "pq" in r else ""
        print(f"  K={k:<5d}  mIoU={r['miou']:.2f}%{pq_str}")


if __name__ == "__main__":
    main()
