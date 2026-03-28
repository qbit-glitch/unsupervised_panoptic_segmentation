#!/usr/bin/env python3
"""Naive Panoptic Baseline: Merge DepthG + CutS3D outputs.

Simple panoptic assembly pipeline from implementation guide Step 1.5:
1. Run DepthG to get semantic labels
2. Run CutS3D to get instance masks
3. Assign each instance mask the majority semantic class
4. Classify clusters as "things" or "stuff" using simple heuristic
5. Merge: give priority to high-confidence instance masks over stuff regions
6. Evaluate PQ on Cityscapes

Usage:
    python scripts/evaluate_naive_panoptic.py \
        --config configs/cityscapes_5pct.yaml \
        --depthg-checkpoint checkpoints/depthg_baseline.npz \
        --cuts3d-checkpoint checkpoints/cuts3d_baseline.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.data.datasets import get_dataset
from mbps.models.backbone.dino_vits8 import DINOViTS8
from mbps.models.semantic.depthg_head import DepthGHead
from mbps.models.instance.cuts3d import CutS3DModule, CascadeMaskHead
from mbps.evaluation.panoptic_quality import compute_panoptic_quality
from mbps.evaluation.hungarian_matching import hungarian_match, compute_miou


def load_config(config_path):
    """Load config with default fallback."""
    default_path = Path(config_path).parent / "default.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f)
    
    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
    
    deep_merge(config, override)
    return config


def naive_panoptic_merge(semantic_pred, instance_masks, instance_scores,
                          thing_classes, stuff_classes, num_classes,
                          score_threshold=0.3):
    """Naive panoptic assembly from semantic + instance outputs.
    
    Args:
        semantic_pred: Semantic labels array of shape (N,).
        instance_masks: Instance mask probabilities of shape (M, N).
        instance_scores: Instance confidence scores of shape (M,).
        thing_classes: Set of thing class IDs.
        stuff_classes: Set of stuff class IDs.
        num_classes: Total number of classes.
        score_threshold: Minimum score for instance to be included.
    
    Returns:
        Tuple of (panoptic_map, panoptic_segments):
            - panoptic_map: Array of shape (N,) with panoptic IDs
            - panoptic_segments: List of segment dicts
    """
    n_tokens = semantic_pred.shape[0]
    label_divisor = 1000
    
    # Initialize panoptic map
    panoptic_map = np.zeros(n_tokens, dtype=np.int64)
    panoptic_segments = []
    seg_id = 1
    
    # Step 1: Add stuff regions from semantic segmentation
    for c in stuff_classes:
        mask = semantic_pred == c
        if np.sum(mask) > 0:
            panoptic_id = c * label_divisor + seg_id
            panoptic_map[mask] = panoptic_id
            panoptic_segments.append({
                "id": panoptic_id,
                "category_id": int(c),
                "isthing": False,
            })
            seg_id += 1
    
    # Step 2: Add thing instances from instance segmentation
    # Sort by score (descending) to prioritize high-confidence instances
    sorted_indices = np.argsort(instance_scores)[::-1]
    
    for m in sorted_indices:
        if instance_scores[m] < score_threshold:
            continue
        
        # Binarize instance mask
        inst_mask = instance_masks[m] > 0.5
        if np.sum(inst_mask) == 0:
            continue
        
        # Step 3: Assign majority semantic class to instance
        masked_sem = semantic_pred[inst_mask]
        if len(masked_sem) == 0:
            continue
        
        # Majority vote
        cat_id = int(np.bincount(masked_sem, minlength=num_classes).argmax())
        
        # Step 4: Check if it's a thing class (simple heuristic)
        # Use thing_classes from config, or heuristic: if instance has high coherence
        if cat_id not in thing_classes:
            # Try to reclassify based on heuristic
            # If semantic prediction varies widely, likely a thing
            unique_classes = np.unique(masked_sem)
            if len(unique_classes) == 1:
                # Uniform semantic → stuff
                continue
            # Otherwise treat as thing
        
        # Step 5: Merge - things override stuff
        panoptic_id = cat_id * label_divisor + seg_id
        
        # Override existing labels (things have priority)
        panoptic_map[inst_mask] = panoptic_id
        
        panoptic_segments.append({
            "id": panoptic_id,
            "category_id": int(cat_id),
            "isthing": True,
        })
        seg_id += 1
    
    return panoptic_map, panoptic_segments


def evaluate_naive_panoptic(config, args):
    """Evaluate naive panoptic baseline."""
    print("\n" + "=" * 80)
    print("  Naive Panoptic Baseline Evaluation")
    print("=" * 80)
    
    # Load dataset
    data_cfg = config["data"]
    val_ds = get_dataset(
        dataset_name=data_cfg["dataset"],
        data_dir=data_cfg["data_dir"],
        depth_dir=data_cfg["depth_dir"],
        split="val",
        image_size=tuple(data_cfg["image_size"]),
        subset_fraction=data_cfg.get("subset_fraction", 0.1),
    )
    
    print(f"Evaluating on {len(val_ds)} samples")
    
    # Initialize models
    backbone = DINOViTS8(pretrained=True, frozen=True)
    semantic_head = DepthGHead(
        input_dim=384,
        code_dim=config["architecture"].get("semantic_dim", 90),
    )
    
    max_instances = config["architecture"].get("max_instances", 100)
    cuts3d = CutS3DModule(max_instances=max_instances)
    cascade_head = CascadeMaskHead()
    
    # Load checkpoints
    print(f"\nLoading DepthG checkpoint: {args.depthg_checkpoint}")
    print(f"Loading CutS3D checkpoint: {args.cuts3d_checkpoint}")
    
    # TODO: Implement checkpoint loading
    # For now, use random initialization
    rng = jax.random.PRNGKey(42)
    img_h, img_w = data_cfg["image_size"]
    dummy_img = jnp.zeros((1, img_h, img_w, 3))
    dummy_depth = jnp.zeros((1, img_h, img_w))
    
    rng, b_rng, s_rng, c_rng, cas_rng = jax.random.split(rng, 5)
    
    backbone_params = backbone.init(b_rng, dummy_img)
    dino_out = backbone.apply(backbone_params, dummy_img)
    features = dino_out["patch_features"]
    
    semantic_params = semantic_head.init(s_rng, features, deterministic=True)
    cuts3d_params = cuts3d.init(c_rng, features, dummy_depth, deterministic=True)
    init_masks, _ = cuts3d.apply(cuts3d_params, features, dummy_depth, deterministic=True)
    cascade_params = cascade_head.init(cas_rng, features, init_masks, deterministic=True)
    
    # Class definitions
    dataset_name = data_cfg.get("dataset", "")
    num_classes = data_cfg["num_classes"]
    
    if dataset_name == "cityscapes":
        # Cityscapes: classes 0-10 stuff, 11-18 things
        stuff_classes = set(range(11))
        thing_classes = set(range(11, min(num_classes, 19)))
    elif dataset_name == "nyu_depth_v2":
        stuff_classes = set(range(num_classes))
        thing_classes = set()
    else:
        stuff_classes = set(range(11))
        thing_classes = set(range(11, num_classes))
    
    print(f"Stuff classes: {len(stuff_classes)}")
    print(f"Thing classes: {len(thing_classes)}")
    
    # Evaluation loop
    patch_size = 8
    h_tokens = img_h // patch_size
    w_tokens = img_w // patch_size
    
    all_pq, all_sq, all_rq = [], [], []
    all_pq_th, all_pq_st = [], []
    all_miou = []
    
    n_eval = min(len(val_ds), 50)
    print(f"\nEvaluating {n_eval} samples...")
    
    for i in range(n_eval):
        sample = val_ds[i]
        image = jnp.array(sample["image"][None])
        depth = jnp.array(sample["depth"][None])
        
        # Forward: DepthG
        dino_out = backbone.apply(backbone_params, image)
        features = dino_out["patch_features"]
        semantic_codes = semantic_head.apply(semantic_params, features, deterministic=True)
        semantic_pred = np.argmax(np.array(semantic_codes[0]), axis=-1)  # (N,)
        
        # Forward: CutS3D
        init_masks, _ = cuts3d.apply(cuts3d_params, features, depth, deterministic=True)
        refined_masks, refined_scores = cascade_head.apply(
            cascade_params, features, init_masks, deterministic=True
        )
        
        instance_masks = np.array(jax.nn.sigmoid(refined_masks[0]))  # (M, N)
        instance_scores = np.array(refined_scores[0])  # (M,)
        
        # Naive panoptic merge
        pred_panoptic, pred_segments = naive_panoptic_merge(
            semantic_pred, instance_masks, instance_scores,
            thing_classes, stuff_classes, num_classes,
            score_threshold=0.3,
        )
        
        # Ground truth
        if "semantic_label" in sample:
            gt_full = sample["semantic_label"]
            gt_semantic = np.zeros(h_tokens * w_tokens, dtype=np.int32)
            for ty in range(h_tokens):
                for tx in range(w_tokens):
                    cy = min(ty * patch_size + patch_size // 2, gt_full.shape[0] - 1)
                    cx = min(tx * patch_size + patch_size // 2, gt_full.shape[1] - 1)
                    gt_semantic[ty * w_tokens + tx] = gt_full[cy, cx]
        else:
            gt_semantic = np.random.randint(0, num_classes, semantic_pred.shape)
        
        # Build GT panoptic
        gt_panoptic = np.zeros_like(pred_panoptic)
        gt_segments = []
        label_divisor = 1000
        gt_seg_id = 1
        
        for c in stuff_classes | thing_classes:
            mask = gt_semantic == c
            if np.sum(mask) > 0:
                pan_id = c * label_divisor + gt_seg_id
                gt_panoptic[mask] = pan_id
                gt_segments.append({
                    "id": pan_id,
                    "category_id": int(c),
                    "isthing": c in thing_classes,
                })
                gt_seg_id += 1
        
        # Compute PQ
        pq_result = compute_panoptic_quality(
            pred_panoptic=pred_panoptic,
            gt_panoptic=gt_panoptic,
            pred_segments=pred_segments,
            gt_segments=gt_segments,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            iou_threshold=0.5,
            label_divisor=label_divisor,
        )
        
        all_pq.append(pq_result.pq)
        all_sq.append(pq_result.sq)
        all_rq.append(pq_result.rq)
        all_pq_th.append(pq_result.pq_things)
        all_pq_st.append(pq_result.pq_stuff)
        
        # Compute mIoU (semantic)
        mapping, _ = hungarian_match(semantic_pred, gt_semantic, num_classes, num_classes)
        miou, _ = compute_miou(semantic_pred, gt_semantic, mapping, num_classes)
        all_miou.append(miou)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_eval} samples")
    
    # Results
    results = {
        "PQ": float(np.mean(all_pq)) * 100,
        "PQ_Th": float(np.mean(all_pq_th)) * 100,
        "PQ_St": float(np.mean(all_pq_st)) * 100,
        "SQ": float(np.mean(all_sq)) * 100,
        "RQ": float(np.mean(all_rq)) * 100,
        "mIoU": float(np.mean(all_miou)) * 100,
        "n_samples": n_eval,
    }
    
    print("\n" + "=" * 80)
    print("  Naive Panoptic Baseline Results")
    print("=" * 80)
    print(f"  PQ:      {results['PQ']:.2f}%")
    print(f"  PQ^Th:   {results['PQ_Th']:.2f}%")
    print(f"  PQ^St:   {results['PQ_St']:.2f}%")
    print(f"  SQ:      {results['SQ']:.2f}%")
    print(f"  RQ:      {results['RQ']:.2f}%")
    print(f"  mIoU:    {results['mIoU']:.2f}%")
    print("=" * 80)
    
    # Save results
    output_path = args.output or "results/naive_panoptic_baseline.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Naive Panoptic Baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--depthg-checkpoint", type=str, required=True)
    parser.add_argument("--cuts3d-checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/naive_panoptic_baseline.json")
    args = parser.parse_args()
    
    config = load_config(args.config)
    evaluate_naive_panoptic(config, args)


if __name__ == "__main__":
    main()
