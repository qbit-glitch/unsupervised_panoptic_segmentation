#!/usr/bin/env python3
"""MMGD-Cut evaluation on Cityscapes for panoptic segmentation metrics.

Runs Falcon K-way NCut on Cityscapes DINOv3 features (single-modal) with
optional R6 multi-scale refinement. Evaluates:
  - Semantic: mIoU (27-class CUPS metric, global Hungarian matching)
  - Panoptic: PQ / SQ / RQ (things + stuff)

Feature layout:
  - Coarse: cityscapes/dinov3_features/val/{city}/*.npy  → (2048, 768) = 32×64
  - Hires:  cityscapes/dinov3_features_hires/val/{city}/*.npy → (8192, 768) = 64×128

Usage:
    # Baseline (coarse NCut only)
    python mmgd_cut_cityscapes.py \
        --cs_root /path/to/cityscapes \
        --device mps

    # R6 multi-scale
    python mmgd_cut_cityscapes.py \
        --cs_root /path/to/cityscapes \
        --device mps --multiscale

    # R4 NAMR post-processing
    python mmgd_cut_cityscapes.py \
        --cs_root /path/to/cityscapes \
        --device mps --namr

    # R6 + R4 combined
    python mmgd_cut_cityscapes.py \
        --cs_root /path/to/cityscapes \
        --device mps --multiscale --namr
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# Re-use solver and post-processors from mmgd_cut.py
sys.path.insert(0, str(Path(__file__).parent))
from falcon_pseudo_semantics import FalconKwayCut
from mmgd_cut import NAMRPostProcessor, merge_multiscale_segments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Cityscapes label tables ──────────────────────────────────────────────────

# 27-class CUPS metric: all Cityscapes classes with raw id 7-33, mapped as c.id - 7
_CS_ID_TO_27CLASS: Dict[int, int] = {i: i - 7 for i in range(7, 34)}

_CS_CLASS_NAMES = [
    "road", "sidewalk", "parking", "rail_track", "building", "wall", "fence",
    "guard_rail", "bridge", "tunnel", "pole", "polegroup",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "caravan", "trailer",
    "train", "motorcycle", "bicycle",
]

_NUM_CLASSES = 27
_STUFF_IDS = set(range(0, 17))    # road…sky (27-class indices 0-16)
_THING_IDS = set(range(17, 27))   # person…bicycle (27-class indices 17-26)

# Raw label IDs that are things (used for GT instance loading)
_THING_RAW_IDS = set(range(24, 34))  # raw ids 24-33


# ─── Cityscapes data helpers ──────────────────────────────────────────────────

def collect_val_files(cs_root: Path) -> List[Tuple[str, Path]]:
    """Return (stem, feat_path) for all val images with DINOv3 features."""
    feat_dir = cs_root / "dinov3_features" / "val"
    entries = []
    for city_dir in sorted(feat_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for npy in sorted(city_dir.glob("*.npy")):
            entries.append((npy.stem, npy))
    return entries


def stem_to_gt_label(cs_root: Path, stem: str) -> Optional[Path]:
    """Map feature stem to gtFine labelIds PNG."""
    # stem: {city}_{ts1}_{ts2}_leftImg8bit
    base = stem.replace("_leftImg8bit", "")
    city = base.split("_")[0]
    cand = cs_root / "gtFine" / "val" / city / f"{base}_gtFine_labelIds.png"
    return cand if cand.exists() else None


def stem_to_gt_inst(cs_root: Path, stem: str) -> Optional[Path]:
    """Map feature stem to gtFine instanceIds PNG."""
    base = stem.replace("_leftImg8bit", "")
    city = base.split("_")[0]
    cand = cs_root / "gtFine" / "val" / city / f"{base}_gtFine_instanceIds.png"
    return cand if cand.exists() else None


def stem_to_hires_feat(cs_root: Path, stem: str) -> Optional[Path]:
    """Map feature stem to hires DINOv3 feature."""
    city = stem.split("_")[0]
    cand = cs_root / "dinov3_features_hires" / "val" / city / f"{stem}.npy"
    return cand if cand.exists() else None


def stem_to_image(cs_root: Path, stem: str) -> Optional[Path]:
    """Map feature stem to leftImg8bit RGB image."""
    city = stem.split("_")[0]
    cand = cs_root / "leftImg8bit" / "val" / city / f"{stem}.png"
    return cand if cand.exists() else None


def stem_to_instance(cs_root: Path, stem: str) -> Optional[Path]:
    """Map feature stem to depth-guided instance NPZ."""
    base = stem.replace("_leftImg8bit", "")
    city = base.split("_")[0]
    # Try with and without _leftImg8bit in filename
    for name in [base, stem]:
        cand = cs_root / "pseudo_instance_spidepth" / "val" / city / f"{name}.npz"
        if cand.exists():
            return cand
    return None


def remap_labelids(gt: np.ndarray) -> np.ndarray:
    """Cityscapes raw labelId → 27-class CUPS index (c.id - 7). 255 = void."""
    remap = np.full(256, 255, dtype=np.uint8)
    for raw_id in range(7, 34):
        remap[raw_id] = raw_id - 7
    return remap[gt]


def load_pred_instances(
    inst_path: Path,
    image_h: int = 512,
    image_w: int = 1024,
) -> Optional[np.ndarray]:
    """Load depth-guided instance masks → (M, H, W) bool at full resolution."""
    data = np.load(str(inst_path))
    masks = data["masks"]           # (M, N_patches) bool
    if "num_valid" in data:
        masks = masks[:int(data["num_valid"])]
    if masks.shape[0] == 0:
        return None

    h_p = int(data["h_patches"]) if "h_patches" in data else image_h // 16
    w_p = int(data["w_patches"]) if "w_patches" in data else image_w // 16

    # Flatten → (M, h_p, w_p) → upsample to image resolution
    M, N = masks.shape
    if N == h_p * w_p:
        masks = masks.reshape(M, h_p, w_p)

    resized = []
    for m in masks:
        m_img = Image.fromarray(m.astype(np.uint8) * 255)
        m_img = m_img.resize((image_w, image_h), Image.NEAREST)
        resized.append(np.array(m_img) > 127)
    return np.stack(resized)


# ─── Semantic evaluation ──────────────────────────────────────────────────────

def hungarian_miou_cityscapes(
    pred: np.ndarray,   # (H, W) values in [0, K_global-1]
    gt: np.ndarray,     # (H, W) trainIds (255=ignore)
    K_global: int = _NUM_CLASSES,
) -> Tuple[float, np.ndarray]:
    """Per-image Hungarian mIoU on 27 Cityscapes classes (CUPS metric)."""
    iou_matrix = np.zeros((K_global, _NUM_CLASSES))
    for k in range(K_global):
        pred_mask = pred == k
        for c in range(_NUM_CLASSES):
            gt_mask = gt == c
            inter = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            iou_matrix[k, c] = inter / (union + 1e-8) if union > 0 else 0.0

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    iou_per_class = np.zeros(_NUM_CLASSES)
    for r, c in zip(row_ind, col_ind):
        iou_per_class[c] = iou_matrix[r, c]
    miou = float(np.mean([iou_per_class[c] for c in range(_NUM_CLASSES) if (gt == c).any()]))
    return miou, iou_per_class


# ─── Panoptic evaluation ──────────────────────────────────────────────────────

def compute_panoptic_quality(
    pred_sem: np.ndarray,       # (H, W) 27-class CUPS IDs 0-26
    pred_inst_masks: Optional[np.ndarray],  # (M, H, W) bool or None
    gt_label_path: Path,
    gt_inst_path: Path,
    iou_threshold: float = 0.5,
) -> Dict:
    """Compute per-image PQ/SQ/RQ for a single image."""
    H, W = pred_sem.shape

    gt_label = remap_labelids(np.array(Image.open(gt_label_path).resize((W, H), Image.NEAREST)))
    gt_inst_map = np.array(Image.open(gt_inst_path).resize((W, H), Image.NEAREST), dtype=np.int32)

    # Build GT segments
    gt_segs: Dict[int, Dict] = {}
    seg_id = 1
    # Stuff: one segment per class
    for tid in _STUFF_IDS:
        mask = gt_label == tid
        if mask.sum() == 0:
            continue
        gt_segs[seg_id] = {"mask": mask, "class_id": tid, "is_thing": False}
        seg_id += 1
    # Things: one segment per instance
    for uid in np.unique(gt_inst_map):
        if uid < 1000:
            continue
        raw_class = uid // 1000
        if raw_class not in _CS_ID_TO_27CLASS:
            continue
        tid = _CS_ID_TO_27CLASS[raw_class]
        mask = gt_inst_map == uid
        if mask.sum() < 10:
            continue
        gt_segs[seg_id] = {"mask": mask, "class_id": tid, "is_thing": True}
        seg_id += 1

    # Build predicted segments
    pred_segs: Dict[int, Dict] = {}
    seg_id = 1
    # Stuff: one segment per predicted class in stuff
    for tid in _STUFF_IDS:
        mask = pred_sem == tid
        if mask.sum() == 0:
            continue
        pred_segs[seg_id] = {"mask": mask, "class_id": tid, "is_thing": False}
        seg_id += 1
    # Things: use instance masks, assign class by majority vote from pred_sem
    if pred_inst_masks is not None:
        for m in pred_inst_masks:
            if m.sum() < 10:
                continue
            votes = pred_sem[m]
            votes_thing = [v for v in votes if v in _THING_IDS]
            if not votes_thing:
                continue
            majority_class = int(np.bincount(votes_thing).argmax())
            pred_segs[seg_id] = {"mask": m, "class_id": majority_class, "is_thing": True}
            seg_id += 1
    else:
        # Fallback: CC on predicted things
        for tid in _THING_IDS:
            mask = pred_sem == tid
            if mask.sum() == 0:
                continue
            pred_segs[seg_id] = {"mask": mask, "class_id": tid, "is_thing": False}
            seg_id += 1

    # Accumulate PQ per class
    tp_per_class = np.zeros(_NUM_CLASSES)
    fp_per_class = np.zeros(_NUM_CLASSES)
    fn_per_class = np.zeros(_NUM_CLASSES)
    iou_sum_per_class = np.zeros(_NUM_CLASSES)

    matched_gt = set()
    matched_pred = set()

    for gid, gseg in gt_segs.items():
        gc = gseg["class_id"]
        for pid, pseg in pred_segs.items():
            if pseg["class_id"] != gc:
                continue
            inter = np.sum(gseg["mask"] & pseg["mask"])
            union = np.sum(gseg["mask"] | pseg["mask"])
            iou = inter / (union + 1e-8) if union > 0 else 0.0
            if iou >= iou_threshold:
                tp_per_class[gc] += 1
                iou_sum_per_class[gc] += iou
                matched_gt.add(gid)
                matched_pred.add(pid)
                break

    for gid, gseg in gt_segs.items():
        if gid not in matched_gt:
            fn_per_class[gseg["class_id"]] += 1

    for pid, pseg in pred_segs.items():
        if pid not in matched_pred:
            fp_per_class[pseg["class_id"]] += 1

    return {
        "tp": tp_per_class,
        "fp": fp_per_class,
        "fn": fn_per_class,
        "iou_sum": iou_sum_per_class,
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_cityscapes_eval(
    cs_root: str,
    device: str = "mps",
    K: int = 80,
    K_global: int = _NUM_CLASSES * 2,  # CUPS-standard: 2× overclustering
    alpha: float = 5.5,
    reg_lambda: float = 0.7,
    n_iter: int = 15,
    beta: float = 0.5,
    multiscale: bool = False,
    namr: bool = False,
    namr_window: int = 2,
    n_images: Optional[int] = None,
    out_json: Optional[str] = None,
) -> Dict:
    """Run MMGD-Cut Cityscapes panoptic eval pipeline."""
    cs_p = Path(cs_root)
    solver = FalconKwayCut(device=device)
    dev = torch.device(device)

    namr_proc = None
    if namr:
        namr_proc = NAMRPostProcessor(window=namr_window)

    entries = collect_val_files(cs_p)
    if n_images:
        entries = entries[:n_images]
    logger.info("Images: %d", len(entries))

    # ── Phase 1: NCut segmentation (coarse + optional fine) ──────────────────
    seg_cache: Dict[str, np.ndarray] = {}          # stem → (H_m, W_m) seg map
    fine_seg_cache: Dict[str, np.ndarray] = {}     # stem → (H_fine, W_fine)
    all_embeddings: List[np.ndarray] = []           # for global k-means
    seg_to_embedding: Dict[Tuple[str, int], np.ndarray] = {}

    MASK_H, MASK_W = 64, 128   # NCut output resolution (= feat grid)
    FEAT_H, FEAT_W = 32, 64    # coarse feature grid
    FINE_H, FINE_W = 64, 128   # hires feature grid

    logger.info("Phase 1: Coarse NCut segmentation")
    for stem, feat_path in tqdm(entries, desc="NCut (coarse)"):
        feats_raw = np.load(str(feat_path)).astype(np.float32)  # (2048, 768)
        feats = torch.from_numpy(feats_raw).unsqueeze(0).to(dev)  # (1, 2048, 768)

        seg_map, _ = solver.segment(
            feats,
            K=K,
            alpha=alpha,
            beta=beta,
            n_iter=n_iter,
            reg_lambda=reg_lambda,
            mask_size=(MASK_H, MASK_W),
            feat_hw=(FEAT_H, FEAT_W),
        )
        seg = seg_map[0, 0]  # (64, 128)
        seg_cache[stem] = seg

        # Pool DINOv3 features per segment for global k-means
        feats_np = feats_raw  # (2048, 768)
        # Resize seg to feat grid for pooling
        seg_at_feat = np.array(
            Image.fromarray(seg.astype(np.uint8)).resize((FEAT_W, FEAT_H), Image.NEAREST)
        )
        for sid in np.unique(seg_at_feat):
            mask = seg_at_feat == sid
            emb = feats_np[mask.ravel()].mean(axis=0)
            key = (stem, int(sid))
            seg_to_embedding[key] = emb
            all_embeddings.append(emb)

    # Optional fine-scale NCut for R6
    if multiscale:
        logger.info("Phase 1b: Fine NCut (hires 64×128)")
        for stem, _ in tqdm(entries, desc="NCut (fine)"):
            hires_path = stem_to_hires_feat(cs_p, stem)
            if hires_path is None:
                continue
            feats_raw = np.load(str(hires_path)).astype(np.float32)  # (8192, 768)
            feats = torch.from_numpy(feats_raw).unsqueeze(0).to(dev)
            fine_K = K * 2
            seg_map, _ = solver.segment(
                feats,
                K=fine_K,
                alpha=alpha,
                beta=beta,
                n_iter=n_iter,
                reg_lambda=reg_lambda,
                mask_size=(FINE_H * 2, FINE_W * 2),  # 128×256
                feat_hw=(FINE_H, FINE_W),
            )
            fine_seg_cache[stem] = seg_map[0, 0]  # (128, 256)

    # ── Phase 2: Global k-means → 19 cluster labels ──────────────────────────
    logger.info("Phase 2: Global k-means (K=%d)", K_global)
    emb_matrix = np.stack(all_embeddings).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=K_global, random_state=42, n_init=5)
    cluster_labels = kmeans.fit_predict(emb_matrix)

    # Build (stem, seg_id) → cluster_id lookup
    seg_to_cluster: Dict[Tuple[str, int], int] = {}
    for idx, (key, _) in enumerate(seg_to_embedding.items()):
        seg_to_cluster[key] = int(cluster_labels[idx])

    IMAGE_H, IMAGE_W = 512, 1024  # evaluation resolution

    # ── Phase 2.5: Global confusion matrix → fixed cluster→class mapping ─────
    # CUPS-standard: one global confusion matrix across ALL images,
    # then Hungarian + argmax fallback for unmatched clusters.
    # Matches CUPS panoptic_quality.py _matching_core (lines 278-288).
    logger.info(
        "Phase 2.5: Global confusion matrix (K_global=%d → C=%d)", K_global, _NUM_CLASSES
    )
    conf_mat = np.zeros((K_global, _NUM_CLASSES), dtype=np.float64)

    for stem, _ in tqdm(entries, desc="Conf-matrix"):
        gt_label_path = stem_to_gt_label(cs_p, stem)
        if gt_label_path is None or stem not in seg_cache:
            continue

        seg = seg_cache[stem]
        pred_cluster = np.full_like(seg, -1, dtype=np.int32)
        for sid in np.unique(seg):
            key = (stem, int(sid))
            if key in seg_to_cluster:
                pred_cluster[seg == sid] = seg_to_cluster[key]

        if multiscale and stem in fine_seg_cache:
            pred_cluster = merge_multiscale_segments(
                pred_cluster.astype(np.uint8), fine_seg_cache[stem],
            ).astype(np.int32)

        if namr_proc is not None:
            img_path = stem_to_image(cs_p, stem)
            if img_path is not None:
                image = np.array(Image.open(img_path).convert("RGB"))
                pred_cluster = namr_proc.refine(pred_cluster, image)

        pred_cluster_full = np.array(
            Image.fromarray(pred_cluster.astype(np.int32), mode="I").resize(
                (IMAGE_W, IMAGE_H), Image.NEAREST
            ),
            dtype=np.int32,
        )
        gt_label = remap_labelids(
            np.array(Image.open(gt_label_path).resize((IMAGE_W, IMAGE_H), Image.NEAREST))
        )

        valid = (gt_label < _NUM_CLASSES) & (pred_cluster_full >= 0) & (pred_cluster_full < K_global)
        gp_v  = pred_cluster_full[valid].astype(np.int64)
        cls_v = gt_label[valid].astype(np.int64)
        np.add.at(conf_mat, (gp_v, cls_v), 1)

    # Global Hungarian + argmax for unmatched (CUPS _matching_core)
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    cluster_to_trainid_global: Dict[int, int] = {
        int(r): int(c) for r, c in zip(row_ind, col_ind)
    }
    matched_clusters = set(cluster_to_trainid_global.keys())
    for k in range(K_global):
        if k not in matched_clusters and conf_mat[k].sum() > 0:
            cluster_to_trainid_global[k] = int(conf_mat[k].argmax())
    logger.info(
        "Fixed mapping: %d Hungarian + %d argmax = %d / %d clusters",
        len(row_ind),
        len(cluster_to_trainid_global) - len(row_ind),
        len(cluster_to_trainid_global),
        K_global,
    )

    # ── Phase 3: Evaluate with fixed mapping ─────────────────────────────────
    logger.info("Phase 3: Evaluation with fixed mapping")

    sem_tp  = np.zeros(_NUM_CLASSES, dtype=np.float64)
    sem_fp  = np.zeros(_NUM_CLASSES, dtype=np.float64)
    sem_fn  = np.zeros(_NUM_CLASSES, dtype=np.float64)
    pq_tp  = np.zeros(_NUM_CLASSES)
    pq_fp  = np.zeros(_NUM_CLASSES)
    pq_fn  = np.zeros(_NUM_CLASSES)
    pq_iou = np.zeros(_NUM_CLASSES)
    n_evaluated = 0

    for stem, _ in tqdm(entries, desc="Evaluating"):
        gt_label_path = stem_to_gt_label(cs_p, stem)
        gt_inst_path  = stem_to_gt_inst(cs_p, stem)
        if gt_label_path is None or gt_inst_path is None:
            continue

        seg = seg_cache[stem]  # (64, 128)

        # Map seg IDs → cluster labels
        pred_cluster = np.full_like(seg, -1, dtype=np.int32)
        for sid in np.unique(seg):
            key = (stem, int(sid))
            if key in seg_to_cluster:
                pred_cluster[seg == sid] = seg_to_cluster[key]

        # Multiscale NCut (coarse-to-fine majority-vote merging)
        if multiscale and stem in fine_seg_cache:
            pred_cluster = merge_multiscale_segments(
                pred_cluster.astype(np.uint8),
                fine_seg_cache[stem],
            ).astype(np.int32)

        # NAMR post-processing (bilateral label refinement)
        if namr_proc is not None:
            img_path = stem_to_image(cs_p, stem)
            if img_path is not None:
                image = np.array(Image.open(img_path).convert("RGB"))
                pred_cluster = namr_proc.refine(pred_cluster, image)

        # Upsample to evaluation resolution
        pred_cluster_full = np.array(
            Image.fromarray(pred_cluster.astype(np.int32), mode="I").resize(
                (IMAGE_W, IMAGE_H), Image.NEAREST
            ),
            dtype=np.int32,
        )

        # Load GT semantic
        gt_label = remap_labelids(
            np.array(Image.open(gt_label_path).resize((IMAGE_W, IMAGE_H), Image.NEAREST))
        )

        # Apply fixed global cluster→class mapping
        pred_sem = np.full((IMAGE_H, IMAGE_W), 255, dtype=np.uint8)
        for k, tid in cluster_to_trainid_global.items():
            pred_sem[pred_cluster_full == k] = tid

        # ── Semantic mIoU (pixel-level, fixed mapping) ────────────────────────
        valid_gt = gt_label < _NUM_CLASSES
        for c in range(_NUM_CLASSES):
            gt_c   = gt_label == c
            pred_c = pred_sem == c
            sem_tp[c] += np.sum(gt_c & pred_c)
            sem_fp[c] += np.sum(valid_gt & ~gt_c & pred_c)
            sem_fn[c] += np.sum(gt_c & ~pred_c)

        # Load depth-guided instances
        pred_inst_masks = None
        inst_path = stem_to_instance(cs_p, stem)
        if inst_path is not None:
            pred_inst_masks = load_pred_instances(inst_path, IMAGE_H, IMAGE_W)

        pq_img = compute_panoptic_quality(
            pred_sem, pred_inst_masks, gt_label_path, gt_inst_path,
        )
        pq_tp  += pq_img["tp"]
        pq_fp  += pq_img["fp"]
        pq_fn  += pq_img["fn"]
        pq_iou += pq_img["iou_sum"]
        n_evaluated += 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    denom_sem      = sem_tp + sem_fp + sem_fn
    per_class_miou = np.where(denom_sem > 0, sem_tp / denom_sem, 0.0)
    active         = denom_sem > 0
    miou           = float(per_class_miou[active].mean() * 100)

    # PQ per class
    sq_per_class = np.where(pq_tp > 0, pq_iou / pq_tp, 0.0)
    rq_per_class = np.where(
        (pq_tp + 0.5 * pq_fp + 0.5 * pq_fn) > 0,
        pq_tp / (pq_tp + 0.5 * pq_fp + 0.5 * pq_fn),
        0.0,
    )
    pq_per_class = sq_per_class * rq_per_class

    def _agg(ids):
        active_ids = [c for c in ids if (pq_tp[c] + pq_fn[c]) > 0]
        if not active_ids:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean([pq_per_class[c] for c in active_ids])) * 100,
            float(np.mean([sq_per_class[c] for c in active_ids])) * 100,
            float(np.mean([rq_per_class[c] for c in active_ids])) * 100,
        )

    all_ids = list(range(_NUM_CLASSES))
    pq_all, sq_all, rq_all = _agg(all_ids)
    pq_th, sq_th, rq_th   = _agg(list(_THING_IDS))
    pq_st, sq_st, rq_st   = _agg(list(_STUFF_IDS))

    logger.info("mIoU=%.2f%%", miou)
    logger.info("PQ=%.2f%%  SQ=%.2f%%  RQ=%.2f%%", pq_all, sq_all, rq_all)
    logger.info("PQ_things=%.2f%%  PQ_stuff=%.2f%%", pq_th, pq_st)

    result = {
        "config": {
            "K": K, "K_global": K_global, "alpha": alpha,
            "reg_lambda": reg_lambda, "n_images": n_evaluated,
            "multiscale": multiscale, "namr": namr,
        },
        "miou": round(miou, 2),
        "per_class_miou": {
            _CS_CLASS_NAMES[c]: round(float(per_class_miou[c]) * 100, 2)
            for c in range(_NUM_CLASSES)
        },
        "pq": round(pq_all, 2),
        "sq": round(sq_all, 2),
        "rq": round(rq_all, 2),
        "pq_things": round(pq_th, 2),
        "sq_things": round(sq_th, 2),
        "rq_things": round(rq_th, 2),
        "pq_stuff":  round(pq_st, 2),
        "sq_stuff":  round(sq_st, 2),
        "rq_stuff":  round(rq_st, 2),
        "per_class_pq": {
            _CS_CLASS_NAMES[c]: round(float(pq_per_class[c]) * 100, 2)
            for c in range(_NUM_CLASSES)
        },
    }

    if out_json:
        # Append/merge into existing results JSON
        existing = {}
        p = Path(out_json)
        if p.exists():
            with open(p) as f:
                existing = json.load(f)
        key = f"K{K}_ms{int(multiscale)}_namr{int(namr)}"
        existing[key] = result
        with open(p, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Results saved to %s", out_json)

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMGD-Cut panoptic segmentation eval on Cityscapes"
    )
    p.add_argument("--cs_root", required=True,
                   help="Path to Cityscapes root (contains leftImg8bit/, gtFine/, etc.)")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--K", type=int, default=80,
                   help="NCut overclustering K (default: 80)")
    p.add_argument("--K_global", type=int, default=54,
                   help="Global k-means clusters (CUPS-standard: 2× num_classes = 38)")
    p.add_argument("--alpha", type=float, default=5.5)
    p.add_argument("--reg_lambda", type=float, default=0.7)
    p.add_argument("--n_iter", type=int, default=15)
    p.add_argument("--multiscale", action="store_true",
                   help="R6: coarse 32×64 + fine 64×128 NCut, boundary-aware merge")
    p.add_argument("--namr", action="store_true",
                   help="R4: NAMR bilateral label refinement post-processing")
    p.add_argument("--namr_window", type=int, default=2)
    p.add_argument("--n_images", type=int, default=None,
                   help="Limit evaluation to first N images (for quick testing)")
    p.add_argument("--out_json", type=str,
                   default="/Users/qbit-glitch/Desktop/datasets/cityscapes/mmgd_cityscapes_results.json",
                   help="JSON file to append results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cityscapes_eval(
        cs_root=args.cs_root,
        device=args.device,
        K=args.K,
        K_global=args.K_global,
        alpha=args.alpha,
        reg_lambda=args.reg_lambda,
        n_iter=args.n_iter,
        multiscale=args.multiscale,
        namr=args.namr,
        namr_window=args.namr_window,
        n_images=args.n_images,
        out_json=args.out_json,
    )

    print("\n" + "═" * 55)
    print("MMGD-Cut Cityscapes Results")
    print("═" * 55)
    print(f"  mIoU:       {result['miou']:.2f}%")
    print(f"  PQ:         {result['pq']:.2f}%")
    print(f"  SQ:         {result['sq']:.2f}%")
    print(f"  RQ:         {result['rq']:.2f}%")
    print(f"  PQ_things:  {result['pq_things']:.2f}%")
    print(f"  PQ_stuff:   {result['pq_stuff']:.2f}%")
    print("─" * 55)
    print("Per-class mIoU:")
    for cls, iou in result["per_class_miou"].items():
        print(f"  {cls:<16s} {iou:.1f}%")


if __name__ == "__main__":
    main()
