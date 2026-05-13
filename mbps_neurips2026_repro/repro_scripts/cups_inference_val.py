#!/usr/bin/env python3
"""Run CUPS Stage-2 inference on Cityscapes val and save predictions for our evaluator.

Saves:
  - Semantic predictions as PNGs (27-class CAUSE IDs)
  - Instance predictions as NPZs (masks, scores, boxes, num_valid)

Usage (on remote):
    export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:$PYTHONPATH
    cd /media/santosh/Kuldeep/panoptic_segmentation/refs/cups
    python /media/santosh/Kuldeep/panoptic_segmentation/scripts/cups_inference_val.py \
        --config configs/train_cityscapes_vitb.yaml \
        --checkpoint '/media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/cups_vitb_stage2/Unsupervised Panoptic Segmentation/atyy26o9/checkpoints/ups_checkpoint_step=003500.ckpt' \
        --output_dir /media/santosh/Kuldeep/panoptic_segmentation/cups_stage2_val_preds
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# CUPS imports
import cups
from cups.data import (
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_CLASSNAMES,
    CityscapesPanopticValidation,
    collate_function_validation,
)
from torch.utils.data import DataLoader

# CAUSE 27-class to Cityscapes 19 trainID mapping
CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for c27, t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    CAUSE27_TO_TRAINID[c27] = t19

# Cityscapes 19-class thing IDs
THING_IDS_19 = set(range(11, 19))


def main():
    parser = argparse.ArgumentParser("CUPS Stage-2 val inference")
    parser.add_argument("--config", required=True, help="CUPS config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt")
    parser.add_argument("--output_dir", required=True, help="Output directory for predictions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = cups.get_default_config(experiment_config_file=args.config)
    # Set checkpoint path so build_model_pseudo can infer thing/stuff class counts
    config.defrost()
    config.MODEL.CHECKPOINT = args.checkpoint
    config.freeze()
    print(f"Config loaded: {args.config}")
    print(f"Dataset root: {config.DATA.ROOT}")
    print(f"Num classes: {config.DATA.NUM_CLASSES}")

    # Make val dataset
    val_dataset = CityscapesPanopticValidation(
        root=config.DATA.ROOT,
        resize_scale=config.DATA.VAL_SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        num_classes=config.DATA.NUM_CLASSES,
    )
    print(f"Val dataset: {len(val_dataset)} images")

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )

    # Build model
    thing_classes = CITYSCAPES_THING_CLASSES
    stuff_classes = CITYSCAPES_STUFF_CLASSES

    model = cups.build_model_pseudo(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        copy_paste_augmentation=torch.nn.Identity(),
        resolution_jitter_augmentation=torch.nn.Identity(),
        photometric_augmentation=torch.nn.Identity(),
        use_tta=False,
        class_names=CITYSCAPES_CLASSNAMES,
        classes_mask=None,
    )

    # Checkpoint is already loaded by build_model_pseudo via config.MODEL.CHECKPOINT
    print(f"Model loaded with checkpoint: {args.checkpoint}")

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Output dirs
    sem_dir = os.path.join(args.output_dir, "semantic", "val")
    inst_dir = os.path.join(args.output_dir, "instances", "val")
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(inst_dir, exist_ok=True)

    total_instances = 0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Inference")):
            images, panoptic_labels, image_names = batch

            # Move images to device
            for img_dict in images:
                for k, v in img_dict.items():
                    if isinstance(v, torch.Tensor):
                        img_dict[k] = v.to(device)

            # Forward pass
            predictions = model(images)

            for i, pred in enumerate(predictions):
                name = image_names[i]
                # Parse city and stem from name like "frankfurt/frankfurt_000000_000294"
                parts = name.split("/")
                if len(parts) == 2:
                    city, stem = parts
                else:
                    city = "unknown"
                    stem = name

                # --- Semantic prediction ---
                sem_logits = pred["sem_seg"]  # (C, H, W)
                sem_pred = sem_logits.argmax(dim=0).cpu().numpy().astype(np.uint8)  # 27-class CAUSE IDs

                city_sem_dir = os.path.join(sem_dir, city)
                os.makedirs(city_sem_dir, exist_ok=True)
                Image.fromarray(sem_pred).save(os.path.join(city_sem_dir, stem + ".png"))

                # --- Instance prediction ---
                instances = pred["instances"]
                if hasattr(instances, "pred_masks") and len(instances) > 0:
                    masks = instances.pred_masks.cpu().numpy().astype(bool)  # (M, H, W)
                    scores = instances.scores.cpu().numpy().astype(np.float32)
                    pred_classes = instances.pred_classes.cpu().numpy().astype(np.int32)
                    boxes = instances.pred_boxes.tensor.cpu().numpy().astype(np.float32)
                    total_instances += len(masks)
                else:
                    H, W = sem_pred.shape
                    masks = np.zeros((0, H, W), dtype=bool)
                    scores = np.array([], dtype=np.float32)
                    pred_classes = np.array([], dtype=np.int32)
                    boxes = np.zeros((0, 4), dtype=np.float32)

                city_inst_dir = os.path.join(inst_dir, city)
                os.makedirs(city_inst_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(city_inst_dir, stem + ".npz"),
                    masks=masks,
                    scores=scores,
                    pred_classes=pred_classes,
                    boxes=boxes,
                    num_valid=len(masks),
                )

                count += 1

            if args.limit and count >= args.limit:
                break

    avg = total_instances / max(count, 1)
    print(f"\nDone. {count} images, {total_instances} instances ({avg:.1f} avg/img)")
    print(f"Semantic saved to: {sem_dir}")
    print(f"Instances saved to: {inst_dir}")


if __name__ == "__main__":
    main()
