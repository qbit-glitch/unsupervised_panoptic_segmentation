#!/usr/bin/env python3
"""Diagnose mask quality: predictions vs pseudo-masks vs GT."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

from mbps.data.tfrecord_utils import parse_example
from mbps.models.backbone.dino_vits8 import DINOViTS8
from mbps.models.backbone.weights_converter import convert_dino_weights
from mbps.models.instance.cascade_mask_rcnn import InstanceHead
from scripts.train_cuts3d import interpolate_pos_embed
from scripts.eval_cuts3d_ap import load_cad_checkpoint, extract_gt_instance_masks, compute_mask_iou


def main():
    # Load val images with instance labels
    ds = tf.data.TFRecordDataset(
        tf.io.gfile.glob("gs://mbps-panoptic/datasets/cityscapes/tfrecords/val/*.tfrecord"),
        compression_type="GZIP",
    )
    def parse_fn(raw):
        return parse_example(raw, image_size=(512, 1024), has_semantic=True, has_instance=True)
    ds = ds.map(parse_fn)

    # Find image with thing instances
    for sample_tf in ds:
        sample = {k: v.numpy() for k, v in sample_tf.items()}
        inst = sample["instance_label"]
        inst_resized = np.array(
            Image.fromarray(inst.astype(np.int32), mode="I").resize((256, 128), Image.NEAREST)
        )
        thing_ids = np.unique(inst_resized[inst_resized >= 1000])
        if len(thing_ids) >= 3:
            break

    img_id = sample["image_id"].decode("utf-8")
    image = tf.image.resize(sample["image"], [128, 256]).numpy()
    print(f"Image: {img_id}")
    print(f"GT thing instances: {len(thing_ids)}")
    for tid in thing_ids[:8]:
        cls = tid // 1000
        n_patches = (inst_resized.flatten() == tid).sum()
        pct = n_patches / 512 * 100
        print(f"  class {cls} (id={tid}): {n_patches} patches ({pct:.1f}%)")

    # Init models + load checkpoint
    backbone = DINOViTS8(freeze=True)
    rng = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, 128, 256, 3))
    pretrained = convert_dino_weights(
        os.path.expanduser("~/.cache/dino/dino_vits8_pretrain.pth")
    )
    pretrained["params"]["pos_embed"] = interpolate_pos_embed(
        pretrained["params"]["pos_embed"], 512
    )
    bp = pretrained
    feats = backbone.apply(bp, dummy)

    cad = InstanceHead(max_instances=5, hidden_dim=256, num_refinement_stages=3, num_classes=1)
    cp = load_cad_checkpoint("checkpoints/cuts3d_cad_full_mi5.npz", cad, feats, rng)

    # Run inference
    feats = backbone.apply(bp, jnp.array(image[None]))
    masks, scores = cad.apply(cp, feats, deterministic=True)
    masks_np = np.array(masks[0])
    scores_np = np.array(scores[0])

    sigmoid_masks = 1.0 / (1.0 + np.exp(-masks_np.astype(np.float64)))
    binary_masks = sigmoid_masks > 0.5

    gt_masks = extract_gt_instance_masks(inst_resized, 16, 32)
    print(f"\nGT masks at patch resolution: {gt_masks.shape[0]} instances")

    print(f"\n--- Predicted masks vs GT ---")
    print(f"{'Pred':>5} | {'Active':>8} | {'BestIoU':>8} | {'BestGT':>7}")
    print("-" * 40)
    for p in range(5):
        pred = binary_masks[p]
        n_active = int(pred.sum())
        best_iou = 0.0
        best_gt = -1
        for g in range(len(gt_masks)):
            iou = compute_mask_iou(pred, gt_masks[g])
            if iou > best_iou:
                best_iou = iou
                best_gt = g
        gt_info = f"cls={thing_ids[best_gt]//1000}" if 0 <= best_gt < len(thing_ids) else "none"
        print(f"  {p:>3} | {n_active:>4}/512 | {best_iou:>7.4f} | {gt_info}")

    # Check pseudo-mask quality against GT
    print(f"\n--- Pseudo-mask quality vs GT ---")
    pseudo_dir = Path("data/pseudo_masks_full")
    for f in sorted(pseudo_dir.glob("masks_*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["image_id"]) == img_id:
            pm_masks = d["masks"]
            pm_nv = int(d["num_valid"])
            print(f"Pseudo-masks for {img_id}: {pm_nv} masks")
            for m in range(pm_nv):
                pm_binary = pm_masks[m] > 0.5
                n_active = int(pm_binary.sum())
                best_iou = 0.0
                best_gt = -1
                for g in range(len(gt_masks)):
                    iou = compute_mask_iou(pm_binary, gt_masks[g])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g
                gt_info = f"cls={thing_ids[best_gt]//1000}" if 0 <= best_gt < len(thing_ids) else "none"
                print(f"  PM{m}: {n_active:>4}/512 active, bestIoU={best_iou:.4f}, {gt_info}")
            break


if __name__ == "__main__":
    main()
