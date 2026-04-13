#!/usr/bin/env python3
"""Generate PICL instance masks for the train split (iterative refinement loop).

Runs PICL + HDBSCAN on all train images and saves per-image .npz files
compatible with PICLDataset (includes class_ids field).

These become the pseudo-instance training signal for Round N+1 of PICL.

Usage:
    python mbps_pytorch/generate_picl_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/picl/round1/best.pth \
        --output_subdir pseudo_instance_picl_r1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ablate_instance_methods import THING_IDS, discover_files
from instance_methods.picl_embed import picl_instances
from instance_methods.utils import load_features
from train_picl import PICLProjectionHead


def save_picl_instances(instances: list, output_path: Path,
                        H: int = 512, W: int = 1024) -> None:
    """Save PICL instance list to .npz compatible with PICLDataset.

    Stores class_ids explicitly (unlike spidepth format which needs semantic map
    lookup), enabling Round N+1 training without re-running semantic inference.

    Args:
        instances: List of (mask: np.ndarray(H,W) bool, class_id: int, score: float).
        output_path: Destination .npz file path.
        H, W: Full pixel resolution of masks.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(instances) == 0:
        np.savez(
            str(output_path),
            masks=np.zeros((0, H * W), dtype=bool),
            class_ids=np.zeros(0, dtype=np.int32),
            scores=np.zeros(0, dtype=np.float32),
            num_valid=0,
        )
        return

    masks = np.stack([m.reshape(-1) for m, _, _ in instances])  # (N, H*W) bool
    class_ids = np.array([c for _, c, _ in instances], dtype=np.int32)
    scores = np.array([s for _, _, s in instances], dtype=np.float32)

    np.savez(
        str(output_path),
        masks=masks,
        class_ids=class_ids,
        scores=scores,
        num_valid=len(instances),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate PICL instance masks for iterative training"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained PICL checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="train",
                        help="Split to generate instances for (train for next round)")
    parser.add_argument("--output_subdir", type=str, required=True,
                        help="Output subdirectory name, e.g. pseudo_instance_picl_r1")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_mapped_k80")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--feature_subdir", type=str, default="dinov2_features")
    parser.add_argument("--hdbscan_min_cluster", type=int, default=5)
    parser.add_argument("--hdbscan_min_samples", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--eval_h", type=int, default=512)
    parser.add_argument("--eval_w", type=int, default=1024)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit images (for testing)")
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    output_root = cs_root / args.output_subdir / args.split
    eval_hw = (args.eval_h, args.eval_w)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load PICL model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PICLProjectionHead(
        input_dim=cfg.get("input_dim", 771),
        hidden_dim=cfg.get("hidden_dim", 512),
        embed_dim=cfg.get("embed_dim", 128),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    depth_weight = cfg.get("depth_weight", 2.0)
    pos_weight = cfg.get("pos_weight", 0.5)
    print(f"  PQ_things at checkpoint: {ckpt.get('PQ_things', 'N/A')}")

    # Discover files
    files = discover_files(
        cs_root, args.split,
        args.semantic_subdir, args.depth_subdir, args.feature_subdir
    )
    if args.max_images:
        files = files[:args.max_images]
    print(f"Generating instances for {len(files)} {args.split} images")
    print(f"  Output: {output_root}")

    H, W = eval_hw
    t_start = time.time()
    n_errors = 0
    total_instances = 0

    for sem_path, depth_path, _, _, feat_path in tqdm(files, desc="Generating"):
        # Derive city/stem for output path
        sem_path = Path(sem_path)
        city = sem_path.parent.name
        stem = sem_path.stem  # e.g. frankfurt_000000_000294
        out_path = output_root / city / f"{stem}.npz"

        if out_path.exists():
            continue  # resume-safe: skip already-generated

        # Load inputs
        pred_sem = np.array(PILImage.open(sem_path))
        if pred_sem.shape != (H, W):
            pred_sem = np.array(
                PILImage.fromarray(pred_sem).resize((W, H), PILImage.NEAREST)
            )

        depth = np.load(str(depth_path))
        if depth.shape != (H, W):
            depth = np.array(
                PILImage.fromarray(depth).resize((W, H), PILImage.BILINEAR)
            )

        features = load_features(str(feat_path)) if feat_path else None
        if features is None:
            n_errors += 1
            save_picl_instances([], out_path, H, W)
            continue

        try:
            instances = picl_instances(
                pred_sem, depth,
                thing_ids=THING_IDS,
                features=features,
                model=model,
                depth_weight=depth_weight,
                pos_weight=pos_weight,
                hdbscan_min_cluster=args.hdbscan_min_cluster,
                hdbscan_min_samples=args.hdbscan_min_samples,
                min_area=args.min_area,
                dilation_iters=3,
            )
        except Exception as e:
            tqdm.write(f"  [WARN] {stem}: {type(e).__name__}: {e}")
            instances = []
            n_errors += 1

        save_picl_instances(instances, out_path, H, W)
        total_instances += len(instances)

    elapsed = time.time() - t_start
    n_images = len(files)
    print(f"\nDone: {n_images} images in {elapsed:.0f}s "
          f"({elapsed/max(n_images,1):.2f}s/img)")
    print(f"  Total instances: {total_instances} "
          f"({total_instances/max(n_images,1):.1f}/img avg)")
    print(f"  Errors: {n_errors}")
    print(f"  Saved to: {output_root}")


if __name__ == "__main__":
    main()
