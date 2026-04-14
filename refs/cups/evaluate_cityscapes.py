"""Full-dataset evaluation: DINOv3 Stage-3 on Cityscapes val (500 images).

Memory-safe two-pass approach (same as evaluate_mots.py):
  Pass 1: Accumulate cost matrix only (no prediction caching) -> Hungarian matching
  Pass 2: Stream PQ + mIoU computation with known assignments

27-class CAUSE + Hungarian matching — same metric as CUPS paper.
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
from cups.data import (
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_CLASSNAMES,
    CityscapesPanopticValidation,
    collate_function_validation,
)
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

NUM_TARGET_CLASSES: int = 27


def _mps_empty_cache():
    """Safely clear MPS cache if available."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit number of images (0 = all)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results JSON (auto-generated if omitted)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference: cpu or mps")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"

    config = cups.get_default_config(experiment_config_file=args.experiment_config_file)
    config.defrost()
    config.MODEL.CHECKPOINT = args.checkpoint
    config.SYSTEM.ACCELERATOR = args.device
    config.freeze()
    log.info(config)

    seed_everything(config.SYSTEM.SEED)

    validation_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        num_classes=NUM_TARGET_CLASSES,
        resize_scale=config.DATA.VAL_SCALE,
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

    thing_classes = CITYSCAPES_THING_CLASSES
    stuff_classes = CITYSCAPES_STUFF_CLASSES

    model = cups.build_model_self(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=CITYSCAPES_CLASSNAMES,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )

    device = torch.device(args.device)
    model = model.to(device)
    model.train(False)

    stuff_pseudo_classes = model.hparams.stuff_pseudo_classes
    thing_pseudo_classes = model.hparams.thing_pseudo_classes
    num_clusters = len(stuff_pseudo_classes) + len(thing_pseudo_classes)
    log.info(f"Pseudo classes: {len(stuff_pseudo_classes)} stuff + "
             f"{len(thing_pseudo_classes)} things = {num_clusters} clusters")

    pq_helper = model.panoptic_quality

    # ── PASS 1: Cost matrix accumulation ─────────────────────────────────
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
    log.info(f"PQ assignments: {pq_assignments.shape}")
    log.info(f"mIoU assignments: {miou_assignments.shape}")

    # ── PASS 2: Streaming PQ + mIoU computation ─────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PASS 2: Computing PQ + mIoU (streaming, memory-safe)")
    log.info("=" * 60)

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

    # ── Per-class breakdown ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Per-class PQ breakdown")
    log.info("=" * 60)
    log.info(f"{'Class':<20} {'PQ':>8} {'SQ':>8} {'RQ':>8}")
    log.info("-" * 48)
    for i, name in enumerate(CITYSCAPES_CLASSNAMES):
        if i < len(pq_c):
            log.info(f"{name:<20} {pq_c[i].item():8.4f} {sq_c[i].item():8.4f} {rq_c[i].item():8.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info(f"PQ:        {pq.item():.5f} ({pq.item()*100:.2f}%)")
    log.info(f"SQ:        {sq.item():.5f}")
    log.info(f"RQ:        {rq.item():.5f}")
    log.info(f"PQ_things: {pq_t.item():.5f} ({pq_t.item()*100:.2f}%)")
    log.info(f"SQ_things: {sq_t.item():.5f}")
    log.info(f"RQ_things: {rq_t.item():.5f}")
    log.info(f"PQ_stuff:  {pq_s.item():.5f} ({pq_s.item()*100:.2f}%)")
    log.info(f"SQ_stuff:  {sq_s.item():.5f}")
    log.info(f"RQ_stuff:  {rq_s.item():.5f}")
    log.info(f"mIoU:      {miou.item():.5f} ({miou.item()*100:.2f}%)")
    log.info(f"Acc:       {acc.item():.5f} ({acc.item()*100:.2f}%)")

    # ── Save results ─────────────────────────────────────────────────────
    if args.output_json is None:
        ckpt_name = os.path.basename(args.checkpoint).replace(".ckpt", "")
        args.output_json = f"results/cityscapes_{ckpt_name}.json"

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    results = {
        "dataset": "cityscapes_val",
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
        "per_class_pq": {
            name: round(pq_c[i].item(), 5)
            for i, name in enumerate(CITYSCAPES_CLASSNAMES)
            if i < len(pq_c)
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
