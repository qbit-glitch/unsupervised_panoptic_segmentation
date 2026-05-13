"""T0 FRACAL: full-Cityscapes-val evaluation with FRACAL post-hoc calibration.

Runs the same two-pass pipeline as evaluate_cityscapes.py, then:
  1. Computes per-cluster fractal dimension D from baseline argmax predictions.
  2. Shifts sem_seg_head.predictor.bias by lam * (mean_D - D_c).
  3. Re-runs the full eval with the calibrated model.
  4. Reports baseline vs calibrated PQ side by side.

Usage:
    python evaluate_cityscapes_fracal.py \
        --experiment_config_file configs/val_stage3_dcfa_simcf_abc_local.yaml \
        --checkpoint /path/to/best_pq_step=003000.ckpt \
        --device cpu \
        --fracal_lambda 1.0
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchmetrics.detection import PanopticQuality as PanopticQualityTM
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cups  # noqa: E402
from cups.augmentation import PhotometricAugmentations, ResolutionJitter  # noqa: E402
from cups.data import (  # noqa: E402
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_CLASSNAMES,
    CityscapesPanopticValidation,
    collate_function_validation,
)
from cups.metrics.panoptic_quality import (  # noqa: E402
    PanopticQualitySemanticMatching,
    _miou_compute,
    _panoptic_quality_compute,
)
from cups.model.model import prediction_to_standard_format  # noqa: E402
from mbps_pytorch.stage4 import box_counting_dimension  # noqa: E402

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")
NUM_TARGET_CLASSES: int = 27


def _mps_empty_cache() -> None:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _per_cluster_fractal_dim(
    accumulated_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute fractal dim per cluster from the union mask across all val images."""
    max_id = max(accumulated_masks.keys()) if accumulated_masks else 0
    out = np.zeros(max_id + 1, dtype=np.float32)
    for cid, mask in accumulated_masks.items():
        if not mask.any():
            continue
        out[cid] = box_counting_dimension(mask)
    return out


def _eval_pass(
    model: torch.nn.Module,
    dataloader: DataLoader,
    pq_helper,
    stuff_pseudo_classes: list[int],
    thing_pseudo_classes: list[int],
    thing_classes: list[int],
    stuff_classes: list[int],
    num_clusters: int,
    device: torch.device,
    desc: str = "eval",
    collect_masks: bool = False,
) -> dict:
    """Single full evaluation: Pass 1 cost matrix + Hungarian + Pass 2 PQ + per-class breakdown.

    Optionally collects per-cluster union masks for FRACAL D computation.
    """
    cost_matrix = torch.zeros(
        num_clusters, NUM_TARGET_CLASSES, dtype=torch.float32
    )

    # For FRACAL D: union mask per cluster (binary OR across all val imgs)
    accumulated_masks: dict[int, np.ndarray] = {}

    log.info(f"\n[{desc}] PASS 1: cost matrix accumulation")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{desc}: Pass1")):
        images, panoptic_labels, _names = batch
        images_dev = [{"image": img["image"].to(device)} for img in images]
        with torch.no_grad():
            prediction = model(images_dev)

        panoptic_pred = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=stuff_pseudo_classes,
            thing_classes=thing_pseudo_classes,
        )
        sem_pred = panoptic_pred[..., 0]  # (H, W) cluster IDs
        sem_pred_flat = sem_pred.reshape(-1).cpu()
        sem_target_flat = panoptic_labels[..., 0].reshape(-1).cpu()
        cost_matrix += pq_helper._cost_matrix_update(
            sem_pred_flat, sem_target_flat, NUM_TARGET_CLASSES, num_clusters
        ).cpu()

        if collect_masks:
            sem_pred_np = sem_pred.cpu().numpy()
            for cid in np.unique(sem_pred_np):
                cid = int(cid)
                cmask = sem_pred_np == cid
                if cid not in accumulated_masks:
                    accumulated_masks[cid] = cmask.copy()
                else:
                    # OR-aggregate spatial occupancy across images
                    if accumulated_masks[cid].shape == cmask.shape:
                        accumulated_masks[cid] |= cmask
                    else:
                        # Different scales — pad/crop to match
                        accumulated_masks[cid] = accumulated_masks[cid] | (
                            cmask if cmask.shape == accumulated_masks[cid].shape
                            else np.zeros_like(accumulated_masks[cid])
                        )

        del prediction, panoptic_pred, images_dev
        if batch_idx % 50 == 0:
            _mps_empty_cache()
            gc.collect()

    pq_helper.cost_matrix = cost_matrix.to(pq_helper.cost_matrix.device)
    pq_assignments = pq_helper.matching().cpu()
    miou_assignments = pq_helper._matching_no_separation().cpu()

    log.info(f"[{desc}] PASS 2: streaming PQ + mIoU")
    fresh_metric = PanopticQualitySemanticMatching(
        things=thing_classes,
        stuffs=stuff_classes,
        num_clusters=NUM_TARGET_CLASSES,
        cache_device="cpu",
        disable_matching=True,
        sync_on_compute=False,
        dist_sync_on_step=False,
    )
    cost_matrix_matched_miou = torch.zeros(
        NUM_TARGET_CLASSES, NUM_TARGET_CLASSES, dtype=torch.float32
    )

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{desc}: Pass2")):
        images, panoptic_labels, _names = batch
        images_dev = [{"image": img["image"].to(device)} for img in images]
        with torch.no_grad():
            prediction = model(images_dev)
        panoptic_pred = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=stuff_pseudo_classes,
            thing_classes=thing_pseudo_classes,
        )
        panoptic_pred_batch = panoptic_pred.unsqueeze(0).cpu()
        panoptic_labels_cpu = panoptic_labels.cpu()
        pred_pq = PanopticQualitySemanticMatching.map_to_target(
            panoptic_pred_batch, pq_assignments
        )
        if pred_pq.ndim == 3:
            pred_pq = pred_pq.unsqueeze(0)
        if panoptic_labels_cpu.ndim == 3:
            panoptic_labels_cpu = panoptic_labels_cpu.unsqueeze(0)
        PanopticQualityTM.update(fresh_metric, pred_pq, panoptic_labels_cpu)
        pred_miou = PanopticQualitySemanticMatching.map_to_target(
            panoptic_pred_batch, miou_assignments
        )
        cost_matrix_matched_miou += pq_helper._cost_matrix_update(
            pred_miou[..., 0].reshape(-1),
            panoptic_labels[..., 0].reshape(-1),
            NUM_TARGET_CLASSES,
            NUM_TARGET_CLASSES,
        ).cpu()
        del prediction, panoptic_pred, panoptic_pred_batch, pred_pq, pred_miou, images_dev
        if batch_idx % 50 == 0:
            _mps_empty_cache()
            gc.collect()

    (
        pq, sq, rq,
        pq_c, sq_c, rq_c,
        pq_t, sq_t, rq_t,
        pq_s, sq_s, rq_s,
    ) = _panoptic_quality_compute(
        fresh_metric.iou_sum,
        fresh_metric.true_positives,
        fresh_metric.false_positives,
        fresh_metric.false_negatives,
        fresh_metric.cat_id_to_continuous_id,
        fresh_metric.things,
        fresh_metric.stuffs,
        None,
    )
    miou, acc = _miou_compute(cost_matrix_matched_miou)

    return {
        "pq": pq.item(),
        "sq": sq.item(),
        "rq": rq.item(),
        "pq_t": pq_t.item(),
        "pq_s": pq_s.item(),
        "miou": miou.item(),
        "acc": acc.item(),
        "pq_c": pq_c.detach().cpu().numpy().tolist(),
        "sq_c": sq_c.detach().cpu().numpy().tolist(),
        "rq_c": rq_c.detach().cpu().numpy().tolist(),
        "accumulated_masks": accumulated_masks,
    }


def _find_sem_seg_predictor(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the final semantic-head predictor (final 1x1 Conv producing logits)."""
    # Common attribute patterns in Detectron2 PanopticFPN + CUPS:
    candidates: list[torch.nn.Module] = []
    for name, module in model.named_modules():
        if name.endswith("sem_seg_head.predictor"):
            candidates.append(module)
    if not candidates:
        raise RuntimeError("No sem_seg_head.predictor module found in model.")
    if len(candidates) > 1:
        log.warning(f"Multiple sem_seg_head.predictor candidates found ({len(candidates)})")
    return candidates[0]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--experiment_config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fracal_lambda", type=float, default=1.0)
    parser.add_argument(
        "--max_images", type=int, default=0,
        help="0 = full val (500). Use a small N for smoke tests.",
    )
    parser.add_argument(
        "--output_json", type=str,
        default="results/t0_fracal_dcfa_simcf_abc_step3000.json",
    )
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"

    config = cups.get_default_config(experiment_config_file=args.experiment_config_file)
    config.defrost()
    config.MODEL.CHECKPOINT = args.checkpoint
    config.SYSTEM.ACCELERATOR = args.device
    config.freeze()
    seed_everything(config.SYSTEM.SEED)

    validation_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        num_classes=NUM_TARGET_CLASSES,
        resize_scale=config.DATA.VAL_SCALE,
    )
    if args.max_images > 0:
        validation_dataset.images = validation_dataset.images[: args.max_images]

    dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
    )
    log.info(f"{len(validation_dataset)} validation samples")

    model = cups.build_model_self(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=CITYSCAPES_CLASSNAMES,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None, resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )
    device = torch.device(args.device)
    model = model.to(device)
    model.train(False)

    stuff_pseudo_classes = model.hparams.stuff_pseudo_classes
    thing_pseudo_classes = model.hparams.thing_pseudo_classes
    num_clusters = len(stuff_pseudo_classes) + len(thing_pseudo_classes)
    log.info(
        f"Pseudo classes: {len(stuff_pseudo_classes)} stuff + "
        f"{len(thing_pseudo_classes)} things = {num_clusters} clusters"
    )

    # ── BASELINE PASS ────────────────────────────────────────────────────
    log.info("\n" + "=" * 64)
    log.info("BASELINE EVAL (no FRACAL)")
    log.info("=" * 64)

    baseline = _eval_pass(
        model,
        dataloader,
        model.panoptic_quality,
        stuff_pseudo_classes,
        thing_pseudo_classes,
        CITYSCAPES_THING_CLASSES,
        CITYSCAPES_STUFF_CLASSES,
        num_clusters,
        device,
        desc="baseline",
        collect_masks=True,
    )

    log.info(
        f"\nBaseline: PQ={baseline['pq']*100:.2f}% "
        f"PQ_t={baseline['pq_t']*100:.2f}% PQ_s={baseline['pq_s']*100:.2f}% "
        f"mIoU={baseline['miou']*100:.2f}%"
    )

    # ── FRACAL CALIBRATION ───────────────────────────────────────────────
    log.info("\n" + "=" * 64)
    log.info(f"FRACAL CALIBRATION (lambda={args.fracal_lambda})")
    log.info("=" * 64)

    cluster_d = _per_cluster_fractal_dim(baseline["accumulated_masks"])
    log.info(f"Per-cluster D: shape={cluster_d.shape}, "
             f"min={cluster_d.min():.3f}, mean={cluster_d.mean():.3f}, "
             f"max={cluster_d.max():.3f}")
    n_zero_d = int((cluster_d == 0).sum())
    log.info(f"  {n_zero_d} clusters with D=0 (likely vacant in val predictions)")

    mean_d = cluster_d.mean()
    shift = args.fracal_lambda * (mean_d - cluster_d)
    shift_t = torch.from_numpy(shift).to(device, dtype=torch.float32)

    sem_predictor = _find_sem_seg_predictor(model)
    if not hasattr(sem_predictor, "bias") or sem_predictor.bias is None:
        raise RuntimeError("sem_seg_head.predictor has no bias param to shift.")

    bias_size = sem_predictor.bias.shape[0]
    log.info(f"Detected sem_seg_head.predictor.bias: shape={tuple(sem_predictor.bias.shape)}")

    # Pad/truncate shift to match bias dim (bias is num_clusters+1 typically: K+bg)
    if shift_t.shape[0] != bias_size:
        new_shift = torch.zeros(bias_size, device=device)
        n = min(shift_t.shape[0], bias_size)
        new_shift[:n] = shift_t[:n]
        shift_t = new_shift
        log.info(f"  shift padded/truncated to bias size {bias_size}")

    original_bias = sem_predictor.bias.detach().clone()
    with torch.no_grad():
        sem_predictor.bias.add_(shift_t)

    # ── CALIBRATED PASS ──────────────────────────────────────────────────
    calibrated = _eval_pass(
        model,
        dataloader,
        model.panoptic_quality,
        stuff_pseudo_classes,
        thing_pseudo_classes,
        CITYSCAPES_THING_CLASSES,
        CITYSCAPES_STUFF_CLASSES,
        num_clusters,
        device,
        desc="calibrated",
        collect_masks=False,
    )
    # Restore bias for cleanliness
    with torch.no_grad():
        sem_predictor.bias.copy_(original_bias)

    log.info(
        f"\nCalibrated: PQ={calibrated['pq']*100:.2f}% "
        f"PQ_t={calibrated['pq_t']*100:.2f}% PQ_s={calibrated['pq_s']*100:.2f}% "
        f"mIoU={calibrated['miou']*100:.2f}%"
    )

    # ── COMPARISON ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 64)
    log.info("BASELINE vs FRACAL — per-class PQ deltas")
    log.info("=" * 64)
    log.info(f"{'Class':<20} {'Baseline':>10} {'Calibrated':>12} {'Δ':>10}")
    log.info("-" * 56)
    for i, name in enumerate(CITYSCAPES_CLASSNAMES):
        if i < len(baseline["pq_c"]):
            b = baseline["pq_c"][i]
            c = calibrated["pq_c"][i]
            log.info(f"{name:<20} {b:10.4f} {c:12.4f} {c-b:+10.4f}")

    log.info("-" * 56)
    log.info(
        f"OVERALL          PQ {baseline['pq']:>8.4f} → {calibrated['pq']:>10.4f} "
        f"({(calibrated['pq']-baseline['pq'])*100:+.4f} pp)"
    )

    # Save JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lambda": args.fracal_lambda,
        "baseline": {k: v for k, v in baseline.items() if k != "accumulated_masks"},
        "calibrated": calibrated,
        "per_cluster_d": cluster_d.tolist(),
        "shift": shift.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
