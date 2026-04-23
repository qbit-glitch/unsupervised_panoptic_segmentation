#!/usr/bin/env python3
"""Multi-seed k-means evaluation for DCFA v2 ablation.

Extracts features ONCE per adapter, then runs k-means with 5 seeds.
Reports mean +/- std mIoU to separate signal from noise.

Usage:
    python scripts/eval_dcfa_v2_multiseed.py \
        --cityscapes_root /path/to/cityscapes \
        --num_seeds 5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.generate_depth_overclustered_semantics import (
    NUM_CLASSES,
    _CS_CLASS_NAMES,
    _CS_ID_TO_TRAIN,
    downsample_depth,
    extract_cause_features_crop,
    get_cityscapes_images,
    load_cause_models,
    load_depth_map,
    sinusoidal_depth_encode,
    sliding_window_features,
    _load_adapter_extras,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_all_features(
    net, segment, cityscapes_root: str, depth_subdir: str,
    device: torch.device, crop_size: int, patch_size: int,
    adapter: Optional[torch.nn.Module] = None,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for all val images. Returns (features, gt_labels)."""
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
    import torch.nn.functional as F
    from scipy.ndimage import sobel

    images = get_cityscapes_images(cityscapes_root, "val")
    all_feats = []
    all_labels = []

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for entry in (pbar := tqdm(images, desc="Extracting features")):
        img_path = os.path.join(
            cityscapes_root, "leftImg8bit", "val",
            entry["city"], f"{entry['stem']}_leftImg8bit.png",
        )
        img = Image.open(img_path).convert("RGB")
        orig_h, orig_w = img.height, img.width
        scale = crop_size / min(orig_h, orig_w)
        new_h = (int(round(orig_h * scale)) // patch_size) * patch_size
        new_w = (int(round(orig_w * scale)) // patch_size) * patch_size
        ph = new_h // patch_size
        pw = new_w // patch_size

        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = preprocess(img_resized).unsqueeze(0).to(device)

        feat_ds = sliding_window_features(
            net, segment, img_tensor, crop_size,
        )
        feats_90 = feat_ds.cpu().numpy().reshape(90, -1).T.astype(np.float64)

        # Apply adapter if provided
        if adapter is not None:
            depth_map = load_depth_map(
                cityscapes_root, depth_subdir, "val",
                entry["city"], entry["stem"],
            )
            if depth_map is not None:
                depth_ds = downsample_depth(depth_map, ph, pw)
                depth_flat = depth_ds.reshape(-1).astype(np.float32)
            else:
                depth_flat = np.zeros(feats_90.shape[0], dtype=np.float32)

            with torch.no_grad():
                codes_t = torch.from_numpy(
                    feats_90.astype(np.float32),
                ).unsqueeze(0).to(device)
                depth_t = torch.from_numpy(depth_flat).unsqueeze(0).to(device)
                if adapter.depth_dim == 16 or adapter.depth_dim > 16:
                    depth_t = sinusoidal_depth_encode(depth_t)

                v2_type = getattr(adapter, "_v2_type", "v3")
                v2_kwargs = getattr(
                    adapter, "_v2_adapter_kwargs",
                    {"depth_dim": adapter.depth_dim},
                )
                fwd_kwargs, geo_concat = _load_adapter_extras(
                    cityscapes_root, "cause_codes_90d", "val",
                    entry["city"], entry["stem"], ph, pw,
                    v2_type, v2_kwargs, device,
                )
                if geo_concat:
                    depth_t = torch.cat([depth_t] + geo_concat, dim=-1)
                adjusted = adapter(
                    codes_t, depth_t, **fwd_kwargs,
                ).squeeze(0).cpu().numpy().astype(np.float64)
            feats_90 = adjusted

        # Depth features for clustering
        depth_map = load_depth_map(
            cityscapes_root, depth_subdir, "val",
            entry["city"], entry["stem"],
        )
        if depth_map is not None:
            depth_ds = downsample_depth(depth_map, ph, pw)
            depth_flat_raw = depth_ds.reshape(-1).astype(np.float32)
            freq_bands = [1, 2, 4, 8, 16, 32, 64, 128]
            encodings = []
            for freq in freq_bands:
                encodings.append(np.sin(freq * np.pi * depth_flat_raw))
                encodings.append(np.cos(freq * np.pi * depth_flat_raw))
            depth_feats = np.stack(encodings, axis=-1).astype(np.float64) * alpha
        else:
            depth_feats = np.zeros(
                (feats_90.shape[0], 16), dtype=np.float64,
            )
        feats = np.concatenate([feats_90, depth_feats], axis=1)

        # GT labels
        gt_path = os.path.join(
            cityscapes_root, "gtFine", "val",
            entry["city"], f"{entry['stem']}_gtFine_labelIds.png",
        )
        if os.path.isfile(gt_path):
            gt = np.array(Image.open(gt_path))
            gt_resized = np.array(
                Image.fromarray(gt).resize((new_w, new_h), Image.NEAREST),
            )
            # Block average to patch grid
            gt_patches = gt_resized.reshape(
                ph, patch_size, pw, patch_size,
            ).transpose(0, 2, 1, 3).reshape(ph * pw, -1)
            gt_majority = np.array([
                np.bincount(p.flatten(), minlength=256).argmax()
                for p in gt_patches
            ])
            gt_train = np.array([
                _CS_ID_TO_TRAIN.get(g, 255) for g in gt_majority
            ], dtype=np.uint8)
        else:
            gt_train = np.full(feats.shape[0], 255, dtype=np.uint8)

        all_feats.append(feats)
        all_labels.append(gt_train)

    return np.concatenate(all_feats), np.concatenate(all_labels)


def run_kmeans_eval(
    feats_norm: np.ndarray,
    all_labels: np.ndarray,
    k: int,
    seed: int,
) -> Dict:
    """Run k-means with a specific seed and compute mIoU."""
    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=10000, max_iter=300,
        random_state=seed, n_init=3,
    )
    kmeans.fit(feats_norm)
    cluster_labels = kmeans.predict(feats_norm)

    # Majority-vote mapping
    conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    for cl, gt in zip(cluster_labels, all_labels):
        if gt < NUM_CLASSES:
            conf[cl, gt] += 1
    cluster_to_class = np.argmax(conf, axis=1).astype(np.uint8)

    # Compute per-class IoU
    pred_classes = cluster_to_class[cluster_labels]
    valid = all_labels < NUM_CLASSES
    pred_valid = pred_classes[valid]
    gt_valid = all_labels[valid]

    per_class_iou = {}
    ious = []
    for c in range(NUM_CLASSES):
        tp = np.sum((pred_valid == c) & (gt_valid == c))
        fp = np.sum((pred_valid == c) & (gt_valid != c))
        fn = np.sum((pred_valid != c) & (gt_valid == c))
        union = tp + fp + fn
        if union > 0:
            iou = tp / union
            ious.append(iou)
            per_class_iou[_CS_CLASS_NAMES[c]] = round(iou * 100, 2)
        else:
            per_class_iou[_CS_CLASS_NAMES[c]] = 0.0

    miou = float(np.mean(ious)) * 100 if ious else 0.0
    return {"miou": round(miou, 2), "per_class_iou": per_class_iou}


def load_adapter(ckpt_path: str, device: torch.device):
    """Load adapter from checkpoint (v1 or v2 format)."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "adapter_type" in raw:
        from mbps_pytorch.models.semantic.depth_adapter_v2 import create_adapter
        adapter = create_adapter(
            raw["adapter_type"], **raw["adapter_kwargs"],
        ).to(device)
        adapter.load_state_dict(raw["state_dict"])
        adapter._v2_type = raw["adapter_type"]
        adapter._v2_adapter_kwargs = raw["adapter_kwargs"]
    else:
        from mbps_pytorch.models.semantic.depth_adapter import DepthAdapter
        state = raw if not isinstance(raw, dict) else raw.get("state_dict", raw)
        first_w = state["mlp.0.weight"]
        n_layers = sum(
            1 for k in state
            if k.startswith("mlp.") and k.endswith(".weight")
            and int(k.split(".")[1]) % 3 == 0
        )
        adapter = DepthAdapter(
            code_dim=90,
            depth_dim=first_w.shape[1] - 90,
            hidden_dim=first_w.shape[0],
            num_layers=n_layers,
        ).to(device)
        adapter.load_state_dict(state)
        adapter._v2_type = "v3"
        adapter._v2_adapter_kwargs = {"depth_dim": first_w.shape[1] - 90}
    adapter.eval()
    return adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed DCFA v2 evaluation")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    parser.add_argument("--k", type=int, default=80)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="CAUSE pretrained dir (default: refs/cause)")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    seeds = [42, 123, 456, 789, 1024][:args.num_seeds]

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(PROJECT_ROOT, "refs", "cause")

    experiments = {
        "V3_baseline": "results/depth_adapter/V3_dd16_h384_l2/best.pt",
        "A1_cross_attn": "results/depth_adapter/DCFA_v2/A1_cross_attn/best.pt",
        "A2_normals": "results/depth_adapter/DCFA_v2/A2_normals/best.pt",
        "A3_gradients": "results/depth_adapter/DCFA_v2/A3_gradients/best.pt",
        "B1_film": "results/depth_adapter/DCFA_v2/B1_film/best.pt",
        "B2_deep": "results/depth_adapter/DCFA_v2/B2_deep/best.pt",
        "B3_window_attn": "results/depth_adapter/DCFA_v2/B3_window_attn/best.pt",
        "C1_contrastive": "results/depth_adapter/DCFA_v2/C1_contrastive/best.pt",
        "C2_cross_image": "results/depth_adapter/DCFA_v2/C2_cross_image/best.pt",
        "X_combined": "results/depth_adapter/DCFA_v2/X_combined/best.pt",
        "no_adapter": None,
    }

    net, segment, cause_args = load_cause_models(
        args.checkpoint_dir, device,
    )

    all_results = {}

    for exp_name, ckpt_path in experiments.items():
        logger.info("=" * 60)
        logger.info("Experiment: %s", exp_name)

        adapter = None
        if ckpt_path and os.path.isfile(ckpt_path):
            adapter = load_adapter(ckpt_path, device)
            logger.info("Loaded adapter from %s", ckpt_path)
        elif ckpt_path:
            logger.warning("Checkpoint not found: %s, skipping", ckpt_path)
            continue

        # Extract features once
        logger.info("Extracting features...")
        all_feats, all_labels = extract_all_features(
            net, segment, args.cityscapes_root, args.depth_subdir,
            device, cause_args.crop_size, cause_args.patch_size,
            adapter=adapter, alpha=0.1,
        )

        # L2 normalize
        norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
        feats_norm = all_feats / np.maximum(norms, 1e-8)
        logger.info("Features: %s", feats_norm.shape)

        # Run k-means with multiple seeds
        seed_results = []
        for seed in seeds:
            result = run_kmeans_eval(feats_norm, all_labels, args.k, seed)
            seed_results.append(result)
            logger.info(
                "  seed=%d: mIoU=%.2f%%", seed, result["miou"],
            )

        mious = [r["miou"] for r in seed_results]
        mean_miou = np.mean(mious)
        std_miou = np.std(mious)
        logger.info(
            "  %s: %.2f +/- %.2f%% (seeds: %s)",
            exp_name, mean_miou, std_miou,
            [f"{m:.2f}" for m in mious],
        )

        # Per-class mean +/- std
        per_class_stats = {}
        for cls in _CS_CLASS_NAMES:
            cls_vals = [r["per_class_iou"].get(cls, 0) for r in seed_results]
            per_class_stats[cls] = {
                "mean": round(float(np.mean(cls_vals)), 2),
                "std": round(float(np.std(cls_vals)), 2),
                "values": cls_vals,
            }

        all_results[exp_name] = {
            "mean_miou": round(float(mean_miou), 2),
            "std_miou": round(float(std_miou), 2),
            "seed_mious": mious,
            "per_class": per_class_stats,
        }

    # Summary table
    print("\n" + "=" * 70)
    print("  Multi-Seed k-means Evaluation (k={}, {} seeds)".format(
        args.k, args.num_seeds,
    ))
    print("=" * 70)
    print(f"\n{'Method':<20s}  {'mean mIoU':>10s}  {'std':>6s}  {'seeds':>30s}")
    print("-" * 70)
    sorted_results = sorted(
        all_results.items(), key=lambda x: -x[1]["mean_miou"],
    )
    for name, r in sorted_results:
        seeds_str = ", ".join(f"{m:.1f}" for m in r["seed_mious"])
        print(
            f"{name:<20s}  {r['mean_miou']:>9.2f}%  "
            f"{r['std_miou']:>5.2f}  [{seeds_str}]",
        )

    # Save
    out_path = "results/depth_adapter/DCFA_v2/multiseed_eval.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
