"""Cross-dataset assessment: DINOv3 Stage-3 on COCO-Stuff-27 panoptic segmentation.

Memory-safe two-pass approach:
  Pass 1: Accumulate cost matrix only (no prediction caching) -> Hungarian matching
  Pass 2: Stream PQ + mIoU computation with known assignments

This avoids the ~80GB RAM usage from caching all 5000 predictions.
"""
import gc
import json
import os
import sys
import logging
from argparse import ArgumentParser
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import seed_everything
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection import PanopticQuality as PanopticQualityTM
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import cups
from cups.augmentation import PhotometricAugmentations, ResolutionJitter
from cups.data import collate_function_validation
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
# COCO-Stuff-27 class definitions (27 coarse supercategory classes)
# ──────────────────────────────────────────────────────────────────────────────

COCOSTUFF27_CLASSNAMES: List[str] = [
    # Things (0-11)
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    # Stuff (12-26)
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]

COCOSTUFF27_THING_CLASSES: Set[int] = set(range(12))   # 0-11
COCOSTUFF27_STUFF_CLASSES: Set[int] = set(range(12, 27))  # 12-26

# Mapping from COCO panoptic supercategory name -> coarse class index
SUPERCATEGORY_TO_COARSE: Dict[str, int] = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


# ──────────────────────────────────────────────────────────────────────────────
# COCO-Stuff-27 panoptic validation dataset
# ──────────────────────────────────────────────────────────────────────────────

class COCOStuff27PanopticValidation(Dataset):
    """COCO-Stuff-27 panoptic validation dataset.

    Loads COCO val2017 images and panoptic annotations, mapping 133 fine
    categories to 27 coarse supercategory classes. Returns data in the same
    format as CityscapesPanopticValidation for compatibility with CUPS
    inference pipeline.
    """

    def __init__(
        self,
        root: str,
        crop_resolution: Tuple[int, int] = (512, 1024),
        void_id: int = 255,
        max_images: int = 0,
    ) -> None:
        super().__init__()
        self.void_id = void_id
        self.crop_resolution = crop_resolution  # (H, W)

        images_dir = os.path.join(root, "val2017")
        panoptic_dir = os.path.join(root, "annotations", "panoptic_val2017")
        panoptic_json = os.path.join(root, "annotations", "panoptic_val2017.json")

        # Load panoptic annotations
        with open(panoptic_json) as f:
            panoptic_data = json.load(f)

        # Build category_id -> coarse_27 mapping
        self.catid_to_coarse: Dict[int, int] = {}
        self.catid_to_isthing: Dict[int, bool] = {}
        for cat in panoptic_data["categories"]:
            sc = cat["supercategory"]
            self.catid_to_coarse[cat["id"]] = SUPERCATEGORY_TO_COARSE[sc]
            self.catid_to_isthing[cat["id"]] = bool(cat["isthing"])

        # Build per-image data
        # image_id -> image info
        img_id_to_info = {img["id"]: img for img in panoptic_data["images"]}

        self.samples: List[Dict[str, Any]] = []
        for ann in panoptic_data["annotations"]:
            img_info = img_id_to_info[ann["image_id"]]
            self.samples.append({
                "image_path": os.path.join(images_dir, img_info["file_name"]),
                "panoptic_path": os.path.join(panoptic_dir, ann["file_name"]),
                "segments_info": ann["segments_info"],
                "image_name": img_info["file_name"].replace(".jpg", ""),
            })

        if max_images > 0:
            self.samples = self.samples[:max_images]

        log.info(f"COCOStuff27PanopticValidation: {len(self.samples)} images, "
                 f"27 classes (12 things, 15 stuff)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sample = self.samples[index]

        # Load image
        image = to_tensor(Image.open(sample["image_path"]).convert("RGB"))  # [3, H, W]
        _, h_orig, w_orig = image.shape

        # Load panoptic PNG and decode segment IDs
        panoptic_png = np.array(Image.open(sample["panoptic_path"]), dtype=np.int64)
        segment_ids = (
            panoptic_png[:, :, 0]
            + panoptic_png[:, :, 1] * 256
            + panoptic_png[:, :, 2] * 256 * 256
        )

        # Build semantic and instance label maps
        semantic_label = np.full((h_orig, w_orig), self.void_id, dtype=np.int64)
        instance_label = np.zeros((h_orig, w_orig), dtype=np.int64)
        instance_counter = 0

        for seg_info in sample["segments_info"]:
            seg_id = seg_info["id"]
            cat_id = seg_info["category_id"]
            is_crowd = seg_info.get("iscrowd", 0)

            mask = segment_ids == seg_id
            if not mask.any():
                continue

            # Map to coarse 27-class
            coarse_class = self.catid_to_coarse[cat_id]

            if is_crowd:
                # Mark crowd regions as void
                semantic_label[mask] = self.void_id
            else:
                semantic_label[mask] = coarse_class

                # Assign instance IDs for thing classes
                if self.catid_to_isthing[cat_id]:
                    instance_counter += 1
                    instance_label[mask] = instance_counter

        # Convert to tensors
        semantic_label = torch.from_numpy(semantic_label).long()  # [H, W]
        instance_label = torch.from_numpy(instance_label).long()  # [H, W]

        # Resize: shorter side to target, then center crop
        target_h, target_w = self.crop_resolution
        scale = min(target_h / h_orig, target_w / w_orig)
        # Ensure at least crop_resolution size
        scale = max(scale, target_h / h_orig, target_w / w_orig)
        new_h = int(h_orig * scale)
        new_w = int(w_orig * scale)

        image = F.interpolate(
            image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        )  # [1, 3, new_h, new_w]

        semantic_label = F.interpolate(
            semantic_label.float().unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w), mode="nearest",
        ).long()  # [1, 1, new_h, new_w]

        instance_label = F.interpolate(
            instance_label.float().unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w), mode="nearest",
        ).long()  # [1, 1, new_h, new_w]

        # Center crop to exact resolution
        pad_h = max(0, target_h - new_h)
        pad_w = max(0, target_w - new_w)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
            semantic_label = F.pad(semantic_label, (0, pad_w, 0, pad_h), value=self.void_id)
            instance_label = F.pad(instance_label, (0, pad_w, 0, pad_h), value=0)
            new_h, new_w = new_h + pad_h, new_w + pad_w

        start_h = (new_h - target_h) // 2
        start_w = (new_w - target_w) // 2
        image = image[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
        semantic_label = semantic_label[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
        instance_label = instance_label[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

        return {
            "image_0_l": image,          # [1, 3, H, W]
            "semantic_gt": semantic_label,  # [1, 1, H, W]
            "instance_gt": instance_label,  # [1, 1, H, W]
            "image_name": sample["image_name"],
        }


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
                        help="Limit number of images (0 = all 5000)")
    parser.add_argument("--output_json", type=str, default="results/coco_stuff27_eval.json",
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

    # Validation dataset
    validation_dataset = COCOStuff27PanopticValidation(
        root=config.DATA.ROOT_VAL,
        crop_resolution=tuple(config.DATA.CROP_RESOLUTION),
        max_images=args.max_images,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
    )
    log.info(f"{len(validation_dataset)} validation samples.")

    # Build model with COCO thing/stuff classes
    model = cups.build_model_self(
        config=config,
        thing_classes=COCOSTUFF27_THING_CLASSES,
        stuff_classes=COCOSTUFF27_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=COCOSTUFF27_CLASSNAMES,
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

    cost_matrix = torch.zeros(num_clusters, 27, dtype=torch.float32)

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
            sem_pred, sem_target, 27, num_clusters
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

    # Fresh metric for streaming PQ — after remapping, predictions use 27 classes
    fresh_metric = PanopticQualitySemanticMatching(
        things=COCOSTUFF27_THING_CLASSES,
        stuffs=COCOSTUFF27_STUFF_CLASSES,
        num_clusters=27,
        cache_device="cpu",
        disable_matching=True,  # Already remapped via assignments
        sync_on_compute=False,
        dist_sync_on_step=False,
    )

    cost_matrix_matched_miou = torch.zeros(27, 27, dtype=torch.float32)

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
            27, 27,
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
        "dataset": "coco_stuff27",
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
    for i, name in enumerate(COCOSTUFF27_CLASSNAMES):
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
    log.info("COCO-Stuff-27 Cross-Dataset Results")
    log.info("=" * 60)
    log.info(f"PQ:       {pq.item():.4f}")
    log.info(f"SQ:       {sq.item():.4f}")
    log.info(f"RQ:       {rq.item():.4f}")
    log.info(f"PQ_things: {pq_t.item():.4f}")
    log.info(f"PQ_stuff:  {pq_s.item():.4f}")
    log.info(f"mIoU:     {miou.item():.4f}")
    log.info(f"Acc:      {acc.item():.4f}")
    log.info("=" * 60)
    log.info("\nPer-class PQ:")
    for i, name in enumerate(COCOSTUFF27_CLASSNAMES):
        if i < pq_c.shape[0]:
            marker = "T" if i < 12 else "S"
            log.info(f"  [{marker}] {name:20s}: PQ={pq_c[i].item():.4f}  "
                     f"SQ={sq_c[i].item():.4f}  RQ={rq_c[i].item():.4f}")


if __name__ == "__main__":
    main()
