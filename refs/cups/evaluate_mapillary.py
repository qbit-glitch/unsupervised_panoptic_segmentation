"""Cross-dataset evaluation: DINOv3 Stage-3 on Mapillary Vistas v2 (OOD evaluation).

Memory-safe two-pass approach (same as evaluate_mots.py):
  Pass 1: Accumulate cost matrix -> Hungarian matching
  Pass 2: Stream PQ + mIoU computation

Mapillary Vistas v2 has 124 classes (72 things, 52 stuff).
We map 19 matched classes to Cityscapes trainIDs.

Usage:
    python refs/cups/evaluate_mapillary.py \
        --experiment_config_file refs/cups/configs/val_dinov3_vitb_k80_local.yaml \
        --checkpoint checkpoints/dinov3_stage3/dinov3_official_stage3_step8000.ckpt \
        --mapillary_root /Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2 \
        --device cpu \
        --max_images 100
"""
import gc
import json
import os
import sys
import logging
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add cups to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cups
from cups.augmentation import PhotometricAugmentations, ResolutionJitter
from cups.data import collate_function_validation
from cups.data.utils import load_image
from cups.metrics.panoptic_quality import (
    PanopticQualitySemanticMatching,
    _miou_compute,
    _panoptic_quality_compute,
)
from cups.model.model import prediction_to_standard_format
from pytorch_lightning import seed_everything

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")

# ──────────────────────────────────────────────────────────────────────────────
# Mapillary → Cityscapes class mapping (19 classes)
# ──────────────────────────────────────────────────────────────────────────────

# Mapillary class index → Cityscapes trainID
# Built from config_v2.0.json analysis
MAPILLARY_TO_CS = {
    21: 0,    # construction--flat--road → road
    19: 1,    # construction--flat--pedestrian-area → sidewalk
    24: 1,    # construction--flat--sidewalk → sidewalk
    27: 2,    # construction--structure--building → building
    12: 3,    # construction--barrier--wall → wall
    5: 4,     # construction--barrier--fence → fence
    85: 5,    # object--support--pole → pole
    88: 5,    # object--support--utility-pole → pole
    90: 6,    # object--traffic-light--general-single → traffic light
    92: 6,    # object--traffic-light--general-upright → traffic light
    93: 6,    # object--traffic-light--general-horizontal → traffic light
    100: 7,   # object--traffic-sign--front → traffic sign
    64: 8,    # nature--vegetation → vegetation
    63: 9,    # nature--terrain → terrain
    61: 10,   # nature--sky → sky
    30: 11,   # human--person--individual → person
    32: 12,   # human--rider--bicyclist → rider
    33: 12,   # human--rider--motorcyclist → rider
    34: 12,   # human--rider--other-rider → rider
    108: 13,  # object--vehicle--car → car
    114: 14,  # object--vehicle--truck → truck
    107: 15,  # object--vehicle--bus → bus
    111: 16,  # object--vehicle--on-rails → train
    110: 17,  # object--vehicle--motorcycle → motorcycle
    105: 18,  # object--vehicle--bicycle → bicycle
}

# Cityscapes class info
CS_THING_IDS = {11, 12, 13, 14, 15, 16, 17, 18}
CS_STUFF_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
CS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
NUM_TARGET_CLASSES = 19


def build_class_mapping(void_id: int = 255) -> Tensor:
    """Build Mapillary → Cityscapes class mapping tensor."""
    # Mapillary has 124 classes (indices 0-123)
    mapping = torch.full((125,), void_id, dtype=torch.long)
    for mv_id, cs_id in MAPILLARY_TO_CS.items():
        if mv_id < 125:
            mapping[mv_id] = cs_id
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MapillaryVistas(Dataset):
    """Mapillary Vistas v2 validation dataset with Cityscapes class mapping."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 0.5,
        void_id: int = 255,
    ):
        super().__init__()
        self.resize_scale = resize_scale
        self.void_id = void_id
        self.class_mapping = build_class_mapping(void_id)

        img_dir = os.path.join(root, "validation", "images")
        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".jpg")
        ])
        self.label_dir = os.path.join(root, "validation", "v2.0", "panoptic")
        log.info(f"MapillaryVistas: {len(self.images)} images, "
                 f"scale={resize_scale}, {NUM_TARGET_CLASSES} target classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img_path = self.images[index]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Load image as [0, 1] tensor
        image = load_image(path=img_path)[None]  # (1, 3, H, W)
        image = F.interpolate(image, scale_factor=self.resize_scale, mode="bilinear")
        # Ensure dimensions are divisible by 32 (required by FPN stride)
        _, _, h, w = image.shape
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        if new_h != h or new_w != w:
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear")

        # Load panoptic label
        label_path = os.path.join(self.label_dir, stem + ".png")
        if not os.path.exists(label_path):
            # Fallback: try with original extension
            label_path = os.path.join(self.label_dir, stem + ".png")

        label = np.array(Image.open(label_path), dtype=np.int64)
        # Mapillary panoptic: RGB encoded as R + G*256 + B*256^2
        if label.ndim == 3:
            label = label[:, :, 0].astype(np.int64) + \
                    label[:, :, 1].astype(np.int64) * 256 + \
                    label[:, :, 2].astype(np.int64) * 256 * 256
        label = torch.from_numpy(label).long()

        # Decode: semantic = label // 1000, instance = label % 1000
        # Wait — Mapillary uses a different encoding.
        # Mapillary panoptic PNG: pixel value = class_id * 256 + instance_id
        # Actually, Mapillary v2 panoptic uses RGB where:
        #   segment_id = R + G*256 + B*65536
        # We need to read the panoptic JSON to get segment_id → category_id mapping

        # Simpler approach: use the semantic label directly from labels/ directory
        sem_label_path = os.path.join(
            os.path.dirname(self.label_dir), "labels", stem + ".png"
        )
        if os.path.exists(sem_label_path):
            sem_label = np.array(Image.open(sem_label_path), dtype=np.int64)
            sem_label = torch.from_numpy(sem_label).long()
        else:
            # Fallback: derive from panoptic
            sem_label = label // 256  # approximate

        # Map to Cityscapes classes
        sem_label = sem_label.clamp(0, 124)
        sem_label = self.class_mapping[sem_label]

        # Instance label from panoptic
        # For now, use a simple instance encoding
        inst_label = torch.zeros_like(sem_label)

        # Resize labels to match image size (new_h, new_w)
        target_h, target_w = image.shape[2], image.shape[3]
        sem_label = F.interpolate(
            sem_label[None][None].float(), size=(target_h, target_w), mode="nearest"
        ).long().squeeze()
        inst_label = F.interpolate(
            inst_label[None][None].float(), size=(target_h, target_w), mode="nearest"
        ).long().squeeze()

        # Stack as panoptic: (H, W, 2) — [semantic, instance]
        panoptic_label = torch.stack([sem_label, inst_label], dim=-1)

        return (
            [{"image": image.squeeze(0)}],
            panoptic_label,
            [stem],
        )


def _mps_empty_cache():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapillary_root", required=True,
                        help="Path to mapillary-vistas-v2 root")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="results/mapillary_eval.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resize_scale", type=float, default=0.5,
                        help="Resize scale for Mapillary images (default 0.5 for memory)")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"

    config = cups.get_default_config(experiment_config_file=args.experiment_config_file)
    config.defrost()
    config.MODEL.CHECKPOINT = args.checkpoint
    config.SYSTEM.ACCELERATOR = args.device
    config.freeze()

    seed_everything(42)

    # Dataset
    dataset = MapillaryVistas(
        root=args.mapillary_root,
        resize_scale=args.resize_scale,
    )
    if args.max_images > 0:
        dataset.images = dataset.images[:args.max_images]

    # Iterate dataset directly (no DataLoader) to avoid collation issues
    # with variable-size Mapillary images

    # Build model
    thing_classes = CS_THING_IDS
    stuff_classes = CS_STUFF_IDS

    model = cups.build_model_self(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=CS_NAMES,
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

    stuff_pseudo = model.hparams.stuff_pseudo_classes
    thing_pseudo = model.hparams.thing_pseudo_classes
    num_clusters = len(stuff_pseudo) + len(thing_pseudo)
    log.info(f"Pseudo classes: {len(stuff_pseudo)} stuff + {len(thing_pseudo)} things")

    pq_helper = model.panoptic_quality

    # ── PASS 1: Cost matrix ─────────────────────────────────────────────
    log.info("\nPASS 1: Cost matrix accumulation")
    cost_matrix = torch.zeros(num_clusters, NUM_TARGET_CLASSES, dtype=torch.float32)

    for batch_idx in tqdm(range(len(dataset)), desc="Pass 1"):
        images, panoptic_labels, names = dataset[batch_idx]
        images_dev = [{"image": img["image"].to(device)} for img in images]

        with torch.no_grad():
            prediction = model(images_dev)

        panoptic_pred = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=stuff_pseudo,
            thing_classes=thing_pseudo,
        )

        sem_pred = panoptic_pred[..., 0].reshape(-1).cpu()
        sem_target = panoptic_labels[..., 0].reshape(-1).cpu()
        cost_matrix += pq_helper._cost_matrix_update(
            sem_pred, sem_target, NUM_TARGET_CLASSES, num_clusters
        )

        del prediction, panoptic_pred, images_dev
        _mps_empty_cache()
        gc.collect()

    # ── Hungarian matching ───────────────────────────────────────────────
    log.info("\nComputing Hungarian assignments...")
    pq_helper.cost_matrix = cost_matrix.to(pq_helper.cost_matrix.device)
    pq_assignments = pq_helper.matching().cpu()
    miou_assignments = pq_helper._matching_no_separation().cpu()
    log.info(f"PQ assignments: {pq_assignments.shape}")
    log.info(f"mIoU assignments: {miou_assignments.shape}")

    # ── PASS 2: PQ + mIoU ──────────────────────────────────────────────
    log.info("\nPASS 2: PQ + mIoU computation")

    from torchmetrics.detection import PanopticQuality as PanopticQualityTM

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

    for batch_idx in tqdm(range(len(dataset)), desc="Pass 2"):
        images, panoptic_labels, names = dataset[batch_idx]
        images_dev = [{"image": img["image"].to(device)} for img in images]

        with torch.no_grad():
            prediction = model(images_dev)

        panoptic_pred = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=stuff_pseudo,
            thing_classes=thing_pseudo,
        )

        panoptic_pred_batch = panoptic_pred.unsqueeze(0).cpu()
        panoptic_labels_cpu = panoptic_labels.unsqueeze(0).cpu()

        # PQ: remap with pq_assignments
        pred_pq = PanopticQualitySemanticMatching.map_to_target(
            panoptic_pred_batch, pq_assignments
        )
        if pred_pq.ndim == 3:
            pred_pq = pred_pq.unsqueeze(0)
        if panoptic_labels_cpu.ndim == 3:
            panoptic_labels_cpu = panoptic_labels_cpu.unsqueeze(0)
        PanopticQualityTM.update(fresh_metric, pred_pq, panoptic_labels_cpu)

        # mIoU: remap with miou_assignments
        pred_miou = PanopticQualitySemanticMatching.map_to_target(
            panoptic_pred_batch, miou_assignments
        )
        cost_matrix_matched_miou += pq_helper._cost_matrix_update(
            pred_miou[..., 0].reshape(-1),
            panoptic_labels[..., 0].reshape(-1),
            NUM_TARGET_CLASSES, NUM_TARGET_CLASSES,
        ).cpu()

        del prediction, panoptic_pred, panoptic_pred_batch, images_dev
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

    pq_overall = pq.item()
    pq_things = pq_t.item()
    pq_stuff = pq_s.item()

    # Results
    results = {
        "dataset": "mapillary_vistas_v2",
        "num_images": len(dataset),
        "PQ": round(pq_overall * 100, 3),
        "PQ_things": round(pq_things * 100, 3),
        "PQ_stuff": round(pq_stuff * 100, 3),
        "SQ": round(sq.item() * 100, 3),
        "RQ": round(rq.item() * 100, 3),
        "mIoU": round(miou.item() * 100, 3),
    }

    log.info("\n" + "=" * 60)
    log.info("MAPILLARY VISTAS v2 RESULTS")
    log.info("=" * 60)
    log.info(f"  PQ       = {results['PQ']:.2f}%")
    log.info(f"  PQ_things= {results['PQ_things']:.2f}%")
    log.info(f"  PQ_stuff = {results['PQ_stuff']:.2f}%")
    log.info(f"  SQ       = {results['SQ']:.2f}%")
    log.info(f"  RQ       = {results['RQ']:.2f}%")
    log.info(f"  mIoU     = {results['mIoU']:.2f}%")
    log.info(f"\n  Acc      = {acc.item()*100:.2f}%")

    # Save
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
