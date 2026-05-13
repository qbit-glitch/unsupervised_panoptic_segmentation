"""Cross-dataset assessment: DINOv3 Stage-3 on MOTSChallenge (OOD evaluation).

Memory-safe two-pass approach:
  Pass 1: Accumulate cost matrix only (no prediction caching) -> Hungarian matching
  Pass 2: Stream PQ + mIoU computation with known assignments

MOTS has 2 classes: background (stuff=0) and person (thing=1).
"""
import gc
import json
import os
import sys
import logging
from argparse import ArgumentParser
from typing import List, Set

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchmetrics.detection import PanopticQuality as PanopticQualityTM
from tqdm import tqdm

import cups
from cups.augmentation import PhotometricAugmentations, ResolutionJitter
from cups.data import collate_function_validation
from cups.data.mots import MOTS, MOTS_THING_CLASSES, MOTS_STUFF_CLASSES
from cups.metrics.panoptic_quality import (
    PanopticQualitySemanticMatching,
    _miou_compute,
    _panoptic_quality_compute,
)
from cups.model.model import prediction_to_standard_format

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")

# ──────────────────────────────────────────────────────────────────────────────
# MOTS class definitions (2 classes)
# ──────────────────────────────────────────────────────────────────────────────

MOTS_CLASSNAMES: List[str] = ["background", "person"]
NUM_TARGET_CLASSES: int = 2


def _mps_empty_cache():
    """Safely clear MPS cache if available."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit number of images (0 = all)")
    parser.add_argument("--output_json", type=str, default="results/mots_eval.json",
                        help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference: cpu or mps (default: cpu)")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"

    # Load config
    config = cups.get_default_config(experiment_config_file=args.experiment_config_file)
    config.defrost()
    config.MODEL.CHECKPOINT = args.checkpoint
    config.SYSTEM.ACCELERATOR = args.device
    config.freeze()
    log.info(config)

    seed_everything(config.SYSTEM.SEED)

    # Validation dataset — num_classes=2 is hardcoded because
    # mots2cs_class_mapping() asserts num_classes==2
    validation_dataset = MOTS(
        root=config.DATA.ROOT,
        resize_scale=config.DATA.VAL_SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        num_classes=2,
    )

    if args.max_images > 0:
        validation_dataset.images = validation_dataset.images[:args.max_images]

    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
    )
    log.info(f"{len(validation_dataset)} validation samples.")

    # Build model with MOTS thing/stuff classes
    model = cups.build_model_self(
        config=config,
        thing_classes=MOTS_THING_CLASSES,
        stuff_classes=MOTS_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=MOTS_CLASSNAMES,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )

    # ── Manual model setup (no Trainer — avoids metric caching OOM) ──────
    device = torch.device(args.device)
    model = model.to(device)
    model.train(False)

    # Extract pseudo class info set during build_model_self
    stuff_pseudo_classes = model.hparams.stuff_pseudo_classes  # tuple
    thing_pseudo_classes = model.hparams.thing_pseudo_classes  # tuple
    num_clusters = len(stuff_pseudo_classes) + len(thing_pseudo_classes)
    log.info(f"Pseudo classes: {len(stuff_pseudo_classes)} stuff + "
             f"{len(thing_pseudo_classes)} things = {num_clusters} clusters")

    # Reuse model's metric instance for helper methods
    pq_helper = model.panoptic_quality

    # ── PASS 1: Cost matrix accumulation (no prediction caching) ─────────
    log.info("\n" + "=" * 60)
    log.info("PASS 1: Accumulating cost matrix (memory-safe)")
    log.info("=" * 60)

    cost_matrix = torch.zeros(num_clusters, NUM_TARGET_CLASSES, dtype=torch.float32)

    for batch_idx, batch in enumerate(tqdm(validation_data_loader, desc="Pass 1 (cost matrix)")):
        images, panoptic_labels, image_names = batch
        images_dev = [{"image": img["image"].to(device)} for img in images]

        with torch.no_grad():
            prediction = model(images_dev)

        panoptic_pred = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=stuff_pseudo_classes,
            thing_classes=thing_pseudo_classes,
        )

        sem_pred = panoptic_pred[..., 0].reshape(-1).cpu()
        sem_target = panoptic_labels[..., 0].reshape(-1).cpu()
        cost_matrix += pq_helper._cost_matrix_update(
            sem_pred, sem_target, NUM_TARGET_CLASSES, num_clusters
        ).cpu()

        del prediction, panoptic_pred, images_dev, sem_pred, sem_target
        if batch_idx % 50 == 0:
            _mps_empty_cache()
            gc.collect()

    _mps_empty_cache()
    gc.collect()

    # ── Compute Hungarian assignments ────────────────────────────────────
    log.info("\nComputing Hungarian assignments...")
    pq_helper.cost_matrix = cost_matrix.to(pq_helper.cost_matrix.device)
    pq_assignments = pq_helper.matching().cpu()
    miou_assignments = pq_helper._matching_no_separation().cpu()
    log.info(f"PQ assignments computed: {pq_assignments.shape}")
    log.info(f"mIoU assignments computed: {miou_assignments.shape}")

    # ── PASS 2: Streaming PQ + mIoU computation ─────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PASS 2: Computing PQ + mIoU (streaming, memory-safe)")
    log.info("=" * 60)

    # Fresh metric for streaming PQ — after remapping, predictions use 2 classes
    fresh_metric = PanopticQualitySemanticMatching(
        things=MOTS_THING_CLASSES,
        stuffs=MOTS_STUFF_CLASSES,
        num_clusters=NUM_TARGET_CLASSES,
        cache_device="cpu",
        disable_matching=True,  # Already remapped via assignments
        sync_on_compute=False,
        dist_sync_on_step=False,
    )

    cost_matrix_matched_miou = torch.zeros(
        NUM_TARGET_CLASSES, NUM_TARGET_CLASSES, dtype=torch.float32
    )

    for batch_idx, batch in enumerate(tqdm(validation_data_loader, desc="Pass 2 (PQ + mIoU)")):
        images, panoptic_labels, image_names = batch
        images_dev = [{"image": img["image"].to(device)} for img in images]

        with torch.no_grad():
            prediction = model(images_dev)

        panoptic_pred = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=stuff_pseudo_classes,
            thing_classes=thing_pseudo_classes,
        )

        # [H, W, 2] -> [1, H, W, 2]
        panoptic_pred_batch = panoptic_pred.unsqueeze(0).cpu()
        panoptic_labels_cpu = panoptic_labels.cpu()

        # PQ: remap with pq_assignments, call parent PanopticQualityTM.update
        pred_pq = PanopticQualitySemanticMatching.map_to_target(
            panoptic_pred_batch, pq_assignments
        )
        if pred_pq.ndim == 3:
            pred_pq = pred_pq.unsqueeze(0)
        if panoptic_labels_cpu.ndim == 3:
            panoptic_labels_cpu = panoptic_labels_cpu.unsqueeze(0)
        PanopticQualityTM.update(fresh_metric, pred_pq, panoptic_labels_cpu)

        # mIoU: remap with miou_assignments, accumulate matched cost matrix
        pred_miou = PanopticQualitySemanticMatching.map_to_target(
            panoptic_pred_batch, miou_assignments
        )
        cost_matrix_matched_miou += pq_helper._cost_matrix_update(
            pred_miou[..., 0].reshape(-1),
            panoptic_labels[..., 0].reshape(-1),
            NUM_TARGET_CLASSES, NUM_TARGET_CLASSES,
        ).cpu()

        del prediction, panoptic_pred, panoptic_pred_batch, pred_pq, pred_miou, images_dev
        if batch_idx % 50 == 0:
            _mps_empty_cache()
            gc.collect()

    _mps_empty_cache()
    gc.collect()

    # ── Compute final metrics ────────────────────────────────────────────
    log.info("\nComputing final metrics...")

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

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "dataset": "mots",
        "checkpoint": args.checkpoint,
        "num_images": len(validation_dataset),
        "metrics": {
            "PQ": round(pq.item(), 5),
            "SQ": round(sq.item(), 5),
            "RQ": round(rq.item(), 5),
            "PQ_things": round(pq_t.item(), 5),
            "SQ_things": round(sq_t.item(), 5),
            "RQ_things": round(rq_t.item(), 5),
            "PQ_stuff": round(pq_s.item(), 5),
            "SQ_stuff": round(sq_s.item(), 5),
            "RQ_stuff": round(rq_s.item(), 5),
            "mIoU": round(miou.item(), 5),
            "Acc": round(acc.item(), 5),
        },
        "per_class": {},
    }
    for i, name in enumerate(MOTS_CLASSNAMES):
        if i < pq_c.shape[0]:
            results["per_class"][name] = {
                "PQ": round(pq_c[i].item(), 5),
                "SQ": round(sq_c[i].item(), 5),
                "RQ": round(rq_c[i].item(), 5),
            }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {args.output_json}")

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("MOTSChallenge Cross-Dataset Results (OOD)")
    log.info("=" * 60)
    log.info(f"PQ:        {pq.item():.4f}")
    log.info(f"SQ:        {sq.item():.4f}")
    log.info(f"RQ:        {rq.item():.4f}")
    log.info(f"PQ_things: {pq_t.item():.4f}")
    log.info(f"PQ_stuff:  {pq_s.item():.4f}")
    log.info(f"mIoU:      {miou.item():.4f}")
    log.info(f"Acc:       {acc.item():.4f}")
    log.info("=" * 60)
    log.info("\nPer-class PQ:")
    for i, name in enumerate(MOTS_CLASSNAMES):
        if i < pq_c.shape[0]:
            marker = "T" if i in MOTS_THING_CLASSES else "S"
            log.info(f"  [{marker}] {name:15s}: PQ={pq_c[i].item():.4f}  "
                     f"SQ={sq_c[i].item():.4f}  RQ={rq_c[i].item():.4f}")


if __name__ == "__main__":
    main()
