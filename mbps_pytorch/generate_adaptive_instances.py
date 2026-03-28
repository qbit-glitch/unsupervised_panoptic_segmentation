#!/usr/bin/env python3
"""Generate instance pseudo-labels using a trained AdaptiveInstanceNet.

Takes a trained checkpoint and generates instance masks using the
learned split map + connected components. Outputs NPZ files compatible
with evaluate_cascade_pseudolabels.py.

Usage:
    python mbps_pytorch/generate_adaptive_instances.py \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/adaptive_instance/best.pth \
        --semantic_subdir pseudo_semantic_cause_crf \
        --output_dir /path/to/cityscapes/adaptive_instances \
        --split val --device auto
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from mbps_pytorch.adaptive_instance_net import AdaptiveInstanceNet

PATCH_H, PATCH_W = 32, 64
EVAL_H, EVAL_W = 512, 1024
NUM_CLASSES = 27

# CAUSE 27-class → 19 trainID
_CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19

DEFAULT_THING_IDS = set(range(11, 19))

CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}


def generate(args):
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"  Checkpoint metrics: PQ={m.get('PQ','?')} "
              f"PQ_things={m.get('PQ_things','?')}")

    # Build model
    model = AdaptiveInstanceNet(
        feature_dim=config.get("feature_dim", 768),
        depth_channels=3,
        semantic_dim=NUM_CLASSES,
        hidden_dim=config.get("hidden_dim", 256),
        embed_dim=config.get("embed_dim", 32),
        num_blocks=config.get("num_blocks", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    os.makedirs(args.output_dir, exist_ok=True)
    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n=== Generating adaptive instances for {split} ===")

        img_dir = os.path.join(args.cityscapes_root, "leftImg8bit", split)
        entries = []
        for city in sorted(os.listdir(img_dir)):
            city_path = os.path.join(img_dir, city)
            if not os.path.isdir(city_path):
                continue
            for fname in sorted(os.listdir(city_path)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                stem = fname.replace("_leftImg8bit.png", "")
                entries.append({"stem": stem, "city": city})

        print(f"Processing {len(entries)} images")
        total_instances = 0
        instance_counts = []
        per_class_counts = {cls: 0 for cls in sorted(DEFAULT_THING_IDS)}
        t0 = time.time()

        with torch.no_grad():
            for entry in tqdm(entries, desc=f"Generating {split}"):
                stem, city = entry["stem"], entry["city"]

                # Load inputs
                features, depth_patch, depth_grads, cause_onehot, sem_trainid = \
                    _load_inputs(args.cityscapes_root, args.semantic_subdir,
                                 split, city, stem, device)

                # Forward pass (model outputs raw logits)
                split_logit, embed = model(
                    features, depth_patch, depth_grads, cause_onehot)
                split_prob = torch.sigmoid(split_logit)

                # Generate instances
                instances = _generate_instances(
                    split_prob.squeeze(0).squeeze(0).cpu().numpy(),
                    embed.squeeze(0).cpu().numpy(),
                    sem_trainid,
                    split_threshold=args.split_threshold,
                    min_area=args.min_area,
                    dilation_iters=args.dilation_iters,
                    use_embeddings=args.use_embeddings,
                    embed_merge_thresh=args.embed_merge_thresh,
                )

                # Save NPZ
                city_dir = os.path.join(args.output_dir, split, city)
                os.makedirs(city_dir, exist_ok=True)
                out_path = os.path.join(city_dir, f"{stem}.npz")
                _save_instances(instances, out_path)

                n = len(instances)
                total_instances += n
                instance_counts.append(n)
                for _, cls, _ in instances:
                    per_class_counts[cls] = per_class_counts.get(cls, 0) + 1

        elapsed = time.time() - t0
        avg_inst = total_instances / max(len(entries), 1)
        print(f"\n{split}: {total_instances} instances from "
              f"{len(entries)} images ({avg_inst:.1f}/img) in {elapsed:.1f}s")

        # Save stats
        stats = {
            "total_images": len(entries),
            "total_instances": total_instances,
            "avg_instances": round(avg_inst, 2),
            "per_class": {CS_NAMES.get(c, str(c)): n
                          for c, n in sorted(per_class_counts.items())},
            "config": {
                "split_threshold": args.split_threshold,
                "min_area": args.min_area,
                "dilation_iters": args.dilation_iters,
                "use_embeddings": args.use_embeddings,
                "embed_merge_thresh": args.embed_merge_thresh,
            },
        }
        stats_path = os.path.join(args.output_dir, split, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats: {json.dumps(stats['per_class'], indent=2)}")

    print("\nDone!")


def _load_inputs(root, semantic_subdir, split, city, stem, device):
    """Load all inputs for one image, return tensors on device."""
    # DINOv2 features
    feat = np.load(
        os.path.join(root, "dinov2_features", split, city,
                     f"{stem}_leftImg8bit.npy")
    ).astype(np.float32)
    feat = feat.reshape(PATCH_H, PATCH_W, -1).transpose(2, 0, 1)
    features = torch.from_numpy(feat).unsqueeze(0).to(device)

    # Depth
    depth_full = np.load(
        os.path.join(root, "depth_spidepth", split, city, f"{stem}.npy"))
    depth_t = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0).float()
    depth_patch = F.interpolate(
        depth_t, size=(PATCH_H, PATCH_W),
        mode="bilinear", align_corners=False)

    # Sobel gradients
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(depth_patch, kx, padding=1)
    grad_y = F.conv2d(depth_patch, ky, padding=1)
    depth_grads = torch.cat([grad_x, grad_y], dim=1)

    # CAUSE semantics: one-hot
    sem_path = os.path.join(root, semantic_subdir, split, city, f"{stem}.png")
    sem_full = np.array(Image.open(sem_path))
    sem_patch = np.array(
        Image.fromarray(sem_full).resize((PATCH_W, PATCH_H), Image.NEAREST))

    smooth = 0.1
    onehot = np.full((NUM_CLASSES, PATCH_H, PATCH_W), smooth / NUM_CLASSES,
                     dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask = sem_patch == c
        onehot[c][mask] = 1.0 - smooth + smooth / NUM_CLASSES
    cause_onehot = torch.from_numpy(onehot).unsqueeze(0).to(device)

    # TrainID semantic at full resolution for CC
    sem_trainid = _CAUSE27_TO_TRAINID[sem_full]
    if sem_trainid.shape != (EVAL_H, EVAL_W):
        sem_trainid = np.array(
            Image.fromarray(sem_trainid).resize((EVAL_W, EVAL_H), Image.NEAREST))

    return (features, depth_patch.to(device), depth_grads.to(device),
            cause_onehot, sem_trainid)


def _generate_instances(
    split_prob_patch, embed_patch, sem_trainid_full,
    split_threshold=0.5, min_area=100, dilation_iters=3,
    use_embeddings=False, embed_merge_thresh=0.7,
):
    """Generate instances from split map + CC at full resolution.

    Args:
        split_prob_patch: (32, 64) float split probability
        embed_patch: (E, 32, 64) float embeddings
        sem_trainid_full: (512, 1024) uint8 trainID semantic map
        split_threshold: threshold for boundary map
        min_area: minimum instance area
        dilation_iters: boundary reclamation iterations
        use_embeddings: whether to use embeddings for CC merging
        embed_merge_thresh: cosine sim threshold for merging CCs

    Returns:
        List of (mask, class_id, score) at EVAL_H × EVAL_W
    """
    H, W = EVAL_H, EVAL_W

    # Upsample split_prob to full resolution
    split_full = np.array(
        Image.fromarray(split_prob_patch.astype(np.float32)).resize(
            (W, H), Image.BILINEAR))
    edge_map = split_full > split_threshold

    # Optionally upsample embeddings for merging
    if use_embeddings:
        E = embed_patch.shape[0]
        embed_t = torch.from_numpy(embed_patch).unsqueeze(0)
        embed_full = F.interpolate(
            embed_t, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0).numpy()  # (E, H, W)

    assigned = np.zeros((H, W), dtype=bool)
    instances = []

    for cls in sorted(DEFAULT_THING_IDS):
        cls_mask = sem_trainid_full == cls
        if cls_mask.sum() < min_area:
            continue

        split_mask = cls_mask & (~edge_map)
        labeled, n_cc = ndimage.label(split_mask)

        # Collect CCs
        ccs = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area >= min_area:
                ccs.append((cc_id, cc_mask, area))
        ccs.sort(key=lambda x: -x[2])

        # Optional: merge CCs with similar embeddings
        if use_embeddings and len(ccs) > 1:
            ccs = _merge_similar_ccs(ccs, embed_full, embed_merge_thresh)

        # Reclaim boundary pixels
        for _, cc_mask, area in ccs:
            if dilation_iters > 0:
                dilated = ndimage.binary_dilation(cc_mask, iterations=dilation_iters)
                reclaimed = dilated & cls_mask & ~assigned
                final = cc_mask | reclaimed
            else:
                final = cc_mask

            if final.sum() < min_area:
                continue

            assigned |= final
            instances.append((final, cls, float(final.sum())))

    # Normalize scores
    instances.sort(key=lambda x: -x[2])
    if instances:
        max_area = instances[0][2]
        instances = [(m, c, s / max_area) for m, c, s in instances]

    return instances


def _merge_similar_ccs(ccs, embed_full, threshold=0.7):
    """Merge connected components with similar mean embeddings."""
    if len(ccs) <= 1:
        return ccs

    E = embed_full.shape[0]
    # Compute mean embedding per CC
    means = []
    for _, cc_mask, _ in ccs:
        pixels = embed_full[:, cc_mask]  # (E, N)
        means.append(pixels.mean(axis=1))
    means = np.array(means)  # (num_cc, E)

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
    means_norm = means / norms

    # Greedy merge: merge pairs with cosine sim > threshold
    merged = list(range(len(ccs)))  # union-find parent
    for i in range(len(ccs)):
        for j in range(i + 1, len(ccs)):
            sim = (means_norm[i] * means_norm[j]).sum()
            if sim > threshold:
                # Merge j into i (smaller index is parent)
                root_i = _find(merged, i)
                root_j = _find(merged, j)
                if root_i != root_j:
                    merged[root_j] = root_i

    # Group by root
    groups = {}
    for idx in range(len(ccs)):
        root = _find(merged, idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(idx)

    # Create merged CCs
    result = []
    for root, indices in groups.items():
        combined_mask = np.zeros_like(ccs[0][1])
        for idx in indices:
            combined_mask |= ccs[idx][1]
        result.append((root, combined_mask, int(combined_mask.sum())))

    return result


def _find(parent, i):
    """Union-find path compression."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _save_instances(instances, output_path):
    """Save instances as NPZ compatible with evaluate_cascade_pseudolabels.py."""
    H, W = EVAL_H, EVAL_W

    if not instances:
        np.savez_compressed(
            output_path,
            masks=np.zeros((0, H * W), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            num_valid=0,
            h_patches=H,
            w_patches=W,
        )
        return

    num = len(instances)
    masks = np.zeros((num, H * W), dtype=bool)
    scores = np.zeros(num, dtype=np.float32)

    for i, (mask, cls, score) in enumerate(instances):
        masks[i] = mask.ravel()
        scores[i] = score

    np.savez_compressed(
        output_path,
        masks=masks,
        scores=scores,
        num_valid=num,
        h_patches=H,
        w_patches=W,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate adaptive instances from trained AdaptiveInstanceNet")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause_crf")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "both"])
    parser.add_argument("--split_threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--dilation_iters", type=int, default=3)
    parser.add_argument("--use_embeddings", action="store_true",
                        help="Use instance embeddings for CC merging")
    parser.add_argument("--embed_merge_thresh", type=float, default=0.7,
                        help="Cosine similarity threshold for merging CCs")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
