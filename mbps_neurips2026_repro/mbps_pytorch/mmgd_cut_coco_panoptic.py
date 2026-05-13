#!/usr/bin/env python3
"""MMGD-Cut COCO-Stuff-27 Panoptic Segmentation Evaluation.

Evaluation protocol: CUPS-standard (CUPS CVPR 2025).
  Phase 1 — Pool DINOv3 embeddings per NCut segment → global k-means (K=27)
             → seg_to_cluster mapping  (locally-consistent per-image segment IDs
               are NOT globally consistent; k-means makes them so)
  Phase 2 — Build global confusion matrix [cluster × GT-class] across all images
  Phase 3 — Global Hungarian → fixed cluster→class assignment
  Phase 4 — Apply fixed assignment to every image → mIoU + PQ

Segment caches:
  Coarse: mmgd_K54_a5.5_reg0.7_dinov3+ssd1b_nodiff_r32/val2017/   (best R3)
  Fine:   mmgd_K108_a5.5_reg0.7_dinov3_hires_nodiff_r64_fine/val2017/ (R6)

Usage:
    python mmgd_cut_coco_panoptic.py --coco_root /path/to/coco
    python mmgd_cut_coco_panoptic.py --coco_root /path/to/coco --multiscale
    python mmgd_cut_coco_panoptic.py --coco_root /path/to/coco --namr
    python mmgd_cut_coco_panoptic.py --coco_root /path/to/coco --multiscale --namr
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from diffcut_pseudo_semantics import (
    COCOSTUFF27_CLASSNAMES,
    NUM_CLASSES,
    STUFF_IDS,
    THING_IDS,
    SUPERCATEGORY_TO_COARSE,
    load_coco_panoptic_gt,
)
from mmgd_cut import NAMRPostProcessor, merge_multiscale_segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─── GT loading ──────────────────────────────────────────────────────────────

def load_coco_panoptic_instances(
    coco_root: str,
    image_id: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load COCO panoptic GT as (instance_id_map, class_map).

    Returns:
        inst_map: (H, W) int32 — unique segment ID per pixel (0 = void)
        cls_map:  (H, W) uint8 — CAUSE-27 class per pixel (255 = void)
    """
    if not hasattr(load_coco_panoptic_instances, "_cache"):
        pan_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
        pan_dir  = Path(coco_root) / "annotations" / "panoptic_val2017"
        with open(pan_json) as f:
            data = json.load(f)
        cat_map = {c["id"]: c["supercategory"] for c in data["categories"]}
        ann_map = {a["image_id"]: a for a in data["annotations"]}
        load_coco_panoptic_instances._cache = (cat_map, ann_map, str(pan_dir))

    cat_map, ann_map, pdir = load_coco_panoptic_instances._cache
    if image_id not in ann_map:
        return None, None

    ann  = ann_map[image_id]
    pan  = np.array(Image.open(Path(pdir) / ann["file_name"]))
    pan_id = (
        pan[:, :, 0].astype(np.int32)
        + pan[:, :, 1].astype(np.int32) * 256
        + pan[:, :, 2].astype(np.int32) * 65536
    )

    inst_map = np.zeros_like(pan_id, dtype=np.int32)
    cls_map  = np.full(pan_id.shape, 255, dtype=np.uint8)

    for seg in ann["segments_info"]:
        supercat = cat_map.get(seg["category_id"])
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            coarse_cls = SUPERCATEGORY_TO_COARSE[supercat]
            mask = pan_id == seg["id"]
            inst_map[mask] = seg["id"]
            cls_map[mask]  = coarse_cls

    return inst_map, cls_map


# ─── Panoptic map builder ─────────────────────────────────────────────────────

def build_panoptic_map(
    pred_cls: np.ndarray,          # (H, W) CAUSE-27 class per pixel (255=void)
    global_pred: np.ndarray,       # (H, W) global cluster IDs — used for thing instances
    instance_counter: List[int],   # mutable [int] — global counter for unique IDs
) -> np.ndarray:
    """Build panoptic ID map.

    Encoding:
      - Stuff:  class_id * 1_000_000  (one merged region per class)
      - Things: class_id * 1_000_000 + instance_idx  (one k-means cluster = one instance)
    """
    H, W = pred_cls.shape
    pan = np.zeros((H, W), dtype=np.int32)

    for c in STUFF_IDS:
        mask = pred_cls == c
        if mask.any():
            pan[mask] = int(c) * 1_000_000

    for c in THING_IDS:
        cls_mask = pred_cls == c
        if not cls_mask.any():
            continue
        for cl_id in np.unique(global_pred[cls_mask]):
            seg_mask = cls_mask & (global_pred == cl_id)
            if seg_mask.sum() < 16:
                continue
            instance_counter[0] += 1
            pan[seg_mask] = int(c) * 1_000_000 + instance_counter[0]

    return pan


# ─── PQ accumulator ───────────────────────────────────────────────────────────

def accumulate_pq(
    pred_pan: np.ndarray,
    gt_inst_map: np.ndarray,
    gt_cls_map: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    iou_sum: np.ndarray,
    iou_thresh: float = 0.5,
) -> None:
    """Update PQ accumulators for one image (in-place)."""
    gt_segs: Dict[int, Tuple[int, np.ndarray]] = {}
    for uid in np.unique(gt_inst_map):
        if uid == 0:
            continue
        mask = gt_inst_map == uid
        cls_votes = gt_cls_map[mask]
        valid = cls_votes[cls_votes != 255]
        if len(valid) == 0:
            continue
        cls = int(np.bincount(valid.astype(np.int64)).argmax())
        gt_segs[uid] = (cls, mask)

    pred_segs: Dict[int, Tuple[int, np.ndarray]] = {}
    for uid in np.unique(pred_pan):
        if uid == 0:
            continue
        mask = pred_pan == uid
        cls = uid // 1_000_000
        if cls >= NUM_CLASSES:
            continue
        pred_segs[uid] = (cls, mask)

    matched_pred = set()

    for gt_uid, (gc, gm) in gt_segs.items():
        best_iou  = 0.0
        best_puid = -1
        for p_uid, (pc, pm) in pred_segs.items():
            if pc != gc:
                continue
            inter = np.sum(gm & pm)
            if inter == 0:
                continue
            iou = inter / (np.sum(gm | pm) + 1e-8)
            if iou > best_iou:
                best_iou, best_puid = iou, p_uid

        if best_iou >= iou_thresh:
            tp[gc]      += 1
            iou_sum[gc] += best_iou
            matched_pred.add(best_puid)
        else:
            fn[gc] += 1

    for p_uid, (pc, _) in pred_segs.items():
        if p_uid not in matched_pred:
            fp[pc] += 1


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_coco_panoptic_eval(
    coco_root: str,
    seg_key: str = "mmgd_K54_a5.5_reg0.7_dinov3+ssd1b_nodiff_r32",
    fine_key: str = "mmgd_K108_a5.5_reg0.7_dinov3_hires_nodiff_r64_fine",
    multiscale: bool = False,
    namr: bool = False,
    namr_window: int = 2,
    K_global: int = 54,            # k-means clusters; 54 = NCut K (overclustering)
    n_images: Optional[int] = None,
    out_json: Optional[str] = None,
) -> Dict:
    """Evaluate MMGD-Cut panoptic on COCO-Stuff-27 with the CUPS eval protocol.

    The CUPS-standard protocol for clustering-based methods:
      1. Pool DINOv3 features per NCut segment.
      2. Global k-means (K=27) → globally-consistent cluster IDs.
      3. Build global confusion matrix across all val images.
      4. Global Hungarian → fixed cluster→class mapping.
      5. Apply fixed mapping → mIoU + PQ.

    NOTE: NCut segment IDs are per-image (not globally consistent). Steps 1-2
    convert them to globally consistent cluster IDs before any matching.
    """
    coco_p       = Path(coco_root)
    seg_dir      = coco_p / "mmgd_segments" / seg_key / "val2017"
    fine_seg_dir = coco_p / "mmgd_segments" / fine_key / "val2017" if multiscale else None

    # DINOv3 feature dir (prefer 64×64, fall back to default)
    dino_dir = coco_p / "dinov3_features_64x64" / "val2017"
    if not dino_dir.exists():
        dino_dir = coco_p / "dinov3_features" / "val2017"

    img_ids = sorted(f.stem for f in seg_dir.glob("*.npy"))
    if n_images:
        img_ids = img_ids[:n_images]
    logger.info(
        "Images: %d | seg_key: %s | multiscale=%s | namr=%s | K_global=%d",
        len(img_ids), seg_key, multiscale, namr, K_global,
    )

    if multiscale and (fine_seg_dir is None or not fine_seg_dir.exists()):
        logger.warning("Fine seg cache not found — disabling multiscale")
        multiscale   = False
        fine_seg_dir = None

    namr_proc = NAMRPostProcessor(window=namr_window) if namr else None

    # ── Phase 1: Pool DINOv3 embeddings → global k-means ─────────────────────
    # NCut segment IDs are LOCAL (per-image). We must map (img_id, seg_id) to a
    # globally-consistent cluster ID before building the confusion matrix.
    logger.info("Phase 1: Pooling DINOv3 embeddings per segment")
    all_embeddings: List[np.ndarray] = []
    seg_info:       List[Tuple[str, int]] = []  # (img_id, local_seg_id)

    for img_id in tqdm(img_ids, desc="Pooling"):
        seg_path  = seg_dir / f"{img_id}.npy"
        dino_path = dino_dir / f"{img_id}.npy"
        if not seg_path.exists() or not dino_path.exists():
            continue

        seg_map = np.load(seg_path)
        seg     = (seg_map[0, 0] if seg_map.ndim == 4 else seg_map).astype(np.int32)

        dino_feats = np.load(dino_path).astype(np.float32)  # (N_tokens, C)
        n_tokens   = dino_feats.shape[0]
        grid       = int(math.sqrt(n_tokens))
        feat_2d    = dino_feats.reshape(grid, grid, -1)

        seg_grid = np.array(
            Image.fromarray(seg.astype(np.uint8)).resize((grid, grid), Image.NEAREST)
        )
        for sid in np.unique(seg_grid):
            mask = seg_grid == sid
            if mask.sum() < 2:
                continue
            embed = feat_2d[mask].mean(axis=0)
            if np.isfinite(embed).all():
                all_embeddings.append(embed)
                seg_info.append((img_id, int(sid)))

    if len(all_embeddings) == 0:
        raise RuntimeError(
            "No embeddings collected. Check that dinov3_features/ exists at "
            f"{dino_dir}"
        )

    all_emb_arr = np.stack(all_embeddings)
    logger.info("Collected %d segment embeddings %s", len(all_emb_arr), all_emb_arr.shape)

    logger.info("Phase 1b: Global k-means K=%d", K_global)
    kmeans = MiniBatchKMeans(
        n_clusters=K_global,
        batch_size=min(4096, len(all_emb_arr)),
        n_init=5,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(all_emb_arr)
    seg_to_cluster: Dict[Tuple[str, int], int] = {
        (img_id, sid): int(cl)
        for (img_id, sid), cl in zip(seg_info, cluster_labels)
    }
    logger.info("seg_to_cluster built: %d entries", len(seg_to_cluster))

    # ── Phase 2: Global confusion matrix ─────────────────────────────────────
    # Apply R6/R4 AFTER converting to global cluster IDs — same order as mmgd_cut.py.
    logger.info("Phase 2: Building global confusion matrix (K=%d × C=%d)", K_global, NUM_CLASSES)
    conf_mat = np.zeros((K_global, NUM_CLASSES), dtype=np.float64)

    for img_id in tqdm(img_ids, desc="Conf-matrix"):
        seg_path = seg_dir / f"{img_id}.npy"
        if not seg_path.exists():
            continue
        seg_map = np.load(seg_path)
        seg     = (seg_map[0, 0] if seg_map.ndim == 4 else seg_map).astype(np.int32)

        # Convert local seg IDs → global cluster IDs
        global_pred = np.full_like(seg, -1)
        for sid in np.unique(seg):
            key = (img_id, int(sid))
            if key in seg_to_cluster:
                global_pred[seg == sid] = seg_to_cluster[key]

        # R6: merge fine segments (on global cluster IDs, same as mmgd_cut.py)
        if multiscale and fine_seg_dir is not None:
            fine_path = fine_seg_dir / f"{img_id}.npy"
            if fine_path.exists():
                fine_raw = np.load(fine_path)
                fine_raw = fine_raw[0, 0] if fine_raw.ndim == 4 else fine_raw
                global_pred = merge_multiscale_segments(
                    global_pred.astype(np.uint8), fine_raw,
                ).astype(np.int32)

        # R4: NAMR post-processing
        if namr_proc is not None:
            img_path = coco_p / "val2017" / f"{img_id}.jpg"
            if img_path.exists():
                image = np.array(Image.open(img_path).convert("RGB"))
                global_pred = namr_proc.refine(global_pred, image)

        gt_sem = load_coco_panoptic_gt(coco_root, int(img_id))
        if gt_sem is None:
            continue

        H_p, W_p = global_pred.shape
        gt_small  = np.array(Image.fromarray(gt_sem).resize((W_p, H_p), Image.NEAREST))

        valid = (gt_small < NUM_CLASSES) & (global_pred >= 0) & (global_pred < K_global)
        gp_v  = global_pred[valid].astype(np.int64)
        cls_v = gt_small[valid].astype(np.int64)
        np.add.at(conf_mat, (gp_v, cls_v), 1)

    # ── Phase 3: Global Hungarian + argmax fallback (CUPS _matching_core) ───────
    # CUPS matching_core (panoptic_quality.py lines 278-288):
    #   1. Hungarian for the best C matched pairs (many-to-one allowed for K>C).
    #   2. Argmax for unmatched clusters → they map to their best class.
    # This ensures ALL K clusters have a class → no pixels are voided due to K>C.
    logger.info("Phase 3: Global Hungarian + argmax fallback (K=%d → C=%d)", K_global, NUM_CLASSES)
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    cluster_to_cls: Dict[int, int] = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    matched = set(cluster_to_cls.keys())
    # Argmax for clusters not selected by Hungarian
    for k in range(K_global):
        if k not in matched and conf_mat[k].sum() > 0:
            cluster_to_cls[k] = int(conf_mat[k].argmax())
    logger.info(
        "Mapping: %d Hungarian + %d argmax = %d / %d clusters assigned",
        len(row_ind), len(cluster_to_cls) - len(row_ind), len(cluster_to_cls), K_global,
    )

    # ── Phase 4: mIoU + PQ with fixed mapping ────────────────────────────────
    logger.info("Phase 4: Evaluating mIoU + PQ with fixed mapping")

    tp_sem  = np.zeros(NUM_CLASSES, dtype=np.float64)
    fp_sem  = np.zeros(NUM_CLASSES, dtype=np.float64)
    fn_sem  = np.zeros(NUM_CLASSES, dtype=np.float64)

    tp_pq   = np.zeros(NUM_CLASSES, dtype=np.float64)
    fp_pq   = np.zeros(NUM_CLASSES, dtype=np.float64)
    fn_pq   = np.zeros(NUM_CLASSES, dtype=np.float64)
    iou_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    instance_ctr = [0]

    for img_id in tqdm(img_ids, desc="Evaluating"):
        seg_path = seg_dir / f"{img_id}.npy"
        if not seg_path.exists():
            continue
        seg_map = np.load(seg_path)
        seg     = (seg_map[0, 0] if seg_map.ndim == 4 else seg_map).astype(np.int32)

        # Convert local → global cluster IDs
        global_pred = np.full_like(seg, -1)
        for sid in np.unique(seg):
            key = (img_id, int(sid))
            if key in seg_to_cluster:
                global_pred[seg == sid] = seg_to_cluster[key]

        # R6
        if multiscale and fine_seg_dir is not None:
            fine_path = fine_seg_dir / f"{img_id}.npy"
            if fine_path.exists():
                fine_raw = np.load(fine_path)
                fine_raw = fine_raw[0, 0] if fine_raw.ndim == 4 else fine_raw
                global_pred = merge_multiscale_segments(
                    global_pred.astype(np.uint8), fine_raw,
                ).astype(np.int32)

        # R4
        if namr_proc is not None:
            img_path = coco_p / "val2017" / f"{img_id}.jpg"
            if img_path.exists():
                image = np.array(Image.open(img_path).convert("RGB"))
                global_pred = namr_proc.refine(global_pred, image)

        gt_sem = load_coco_panoptic_gt(coco_root, int(img_id))
        gt_inst_map, gt_cls_map = load_coco_panoptic_instances(coco_root, int(img_id))
        if gt_sem is None or gt_inst_map is None:
            continue

        H_p, W_p = global_pred.shape
        gt_small  = np.array(Image.fromarray(gt_sem).resize((W_p, H_p), Image.NEAREST))

        # Apply fixed cluster→class mapping
        pred_cls = np.full((H_p, W_p), 255, dtype=np.uint8)
        for cl, c in cluster_to_cls.items():
            pred_cls[global_pred == cl] = c

        # Semantic mIoU
        valid_gt = gt_small < NUM_CLASSES
        for c in range(NUM_CLASSES):
            gt_c   = gt_small == c
            pred_c = pred_cls == c
            tp_sem[c] += np.sum(gt_c & pred_c)
            fp_sem[c] += np.sum(valid_gt & ~gt_c & pred_c)
            fn_sem[c] += np.sum(gt_c & ~pred_c)   # includes void pred → correct FN

        # Panoptic PQ — upsample to GT resolution
        H_gt, W_gt = gt_inst_map.shape
        pred_cls_full = np.array(
            Image.fromarray(pred_cls).resize((W_gt, H_gt), Image.NEAREST)
        )
        gp_pil   = Image.fromarray(global_pred.astype(np.int32), mode="I")
        gp_full  = np.array(gp_pil.resize((W_gt, H_gt), Image.NEAREST), dtype=np.int32)

        pred_pan = build_panoptic_map(pred_cls_full, gp_full, instance_ctr)
        accumulate_pq(pred_pan, gt_inst_map, gt_cls_map, tp_pq, fp_pq, fn_pq, iou_sum)

    # ── Aggregate results ─────────────────────────────────────────────────────
    denom_sem   = tp_sem + fp_sem + fn_sem
    iou_per_cls = np.where(denom_sem > 0, tp_sem / denom_sem, 0.0)
    active_sem  = denom_sem > 0
    miou        = float(iou_per_cls[active_sem].mean() * 100)

    sq  = np.where(tp_pq > 0, iou_sum / tp_pq, 0.0)
    rq  = tp_pq / (tp_pq + 0.5 * fp_pq + 0.5 * fn_pq + 1e-8)
    pq  = sq * rq

    def _agg(ids: set) -> Tuple[float, float, float]:
        active = [c for c in ids if (tp_pq[c] + fn_pq[c]) > 0]
        if not active:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean([pq[c] for c in active])) * 100,
            float(np.mean([sq[c] for c in active])) * 100,
            float(np.mean([rq[c] for c in active])) * 100,
        )

    pq_all, sq_all, rq_all = _agg(set(range(NUM_CLASSES)))
    pq_th,  sq_th,  rq_th  = _agg(THING_IDS)
    pq_st,  sq_st,  rq_st  = _agg(STUFF_IDS)

    logger.info("mIoU=%.2f%%", miou)
    logger.info("PQ=%.2f%%  SQ=%.2f%%  RQ=%.2f%%", pq_all, sq_all, rq_all)
    logger.info("PQ_things=%.2f%%  PQ_stuff=%.2f%%", pq_th, pq_st)

    result = {
        "eval_protocol": "CUPS-standard (global k-means + global Hungarian)",
        "config": {
            "seg_key":    seg_key,
            "K_global":   K_global,
            "multiscale": multiscale,
            "namr":       namr,
            "n_images":   len(img_ids),
        },
        "miou":      round(miou,   2),
        "pq":        round(pq_all, 2),
        "sq":        round(sq_all, 2),
        "rq":        round(rq_all, 2),
        "pq_things": round(pq_th,  2),
        "sq_things": round(sq_th,  2),
        "rq_things": round(rq_th,  2),
        "pq_stuff":  round(pq_st,  2),
        "sq_stuff":  round(sq_st,  2),
        "rq_stuff":  round(rq_st,  2),
        "per_class_pq":   {
            COCOSTUFF27_CLASSNAMES[c]: round(float(pq[c]) * 100, 2)
            for c in range(NUM_CLASSES)
        },
        "per_class_miou": {
            COCOSTUFF27_CLASSNAMES[c]: round(float(iou_per_cls[c]) * 100, 2)
            for c in range(NUM_CLASSES)
        },
    }

    if out_json:
        p = Path(out_json)
        existing = json.loads(p.read_text()) if p.exists() else {}
        key = f"ms{int(multiscale)}_namr{int(namr)}"
        existing[key] = result
        p.write_text(json.dumps(existing, indent=2))
        logger.info("Results saved to %s", out_json)

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMGD-Cut COCO-Stuff-27 panoptic eval (CUPS-standard protocol)"
    )
    p.add_argument("--coco_root",   required=True)
    p.add_argument("--multiscale",  action="store_true", help="R6: merge fine 64×64 segments")
    p.add_argument("--namr",        action="store_true", help="R4: NAMR post-processing")
    p.add_argument("--namr_window", type=int, default=2)
    p.add_argument("--K_global",    type=int, default=54,
                   help="k-means clusters for global assignment (default=54=NCut K)")
    p.add_argument("--n_images",    type=int, default=None)
    p.add_argument("--out_json",    type=str,
                   default="/Users/qbit-glitch/Desktop/datasets/coco/mmgd_panoptic_results.json")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    result = run_coco_panoptic_eval(
        coco_root=args.coco_root,
        multiscale=args.multiscale,
        namr=args.namr,
        namr_window=args.namr_window,
        K_global=args.K_global,
        n_images=args.n_images,
        out_json=args.out_json,
    )
    print("\n" + "═" * 58)
    print("MMGD-Cut COCO-Stuff-27 Panoptic  [CUPS-standard protocol]")
    print("═" * 58)
    print(f"  mIoU:       {result['miou']:.2f}%")
    print(f"  PQ:         {result['pq']:.2f}%")
    print(f"  SQ:         {result['sq']:.2f}%")
    print(f"  RQ:         {result['rq']:.2f}%")
    print(f"  PQ_things:  {result['pq_things']:.2f}%")
    print(f"  PQ_stuff:   {result['pq_stuff']:.2f}%")
    print("─" * 58)
    print("Per-class PQ:")
    for cls, val in result["per_class_pq"].items():
        print(f"  {cls:<18s} {val:.1f}%")


if __name__ == "__main__":
    main()
