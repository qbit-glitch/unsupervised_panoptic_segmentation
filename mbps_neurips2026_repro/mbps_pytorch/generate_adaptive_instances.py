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
from mbps_pytorch.adaptive_instance_semantics import (
    encode_semantic_onehot,
    infer_semantic_spec,
    map_to_trainid,
    validate_semantic_inputs,
)

PATCH_H, PATCH_W = 32, 64
FUSION_H, FUSION_W = 128, 256
EVAL_H, EVAL_W = 512, 1024
SAM3_MAX_MASKS = 20
FUSION_CHANNELS = 9

DEFAULT_THING_IDS = set(range(11, 19))
THING_IDS_LIST = list(range(11, 19))
THING_HEAD_IDX_TO_TRAINID = np.array([-1] + THING_IDS_LIST, dtype=np.int16)
TRAINID_TO_THING_HEAD_IDX = {tid: i + 1 for i, tid in enumerate(THING_IDS_LIST)}
_SAM3_THING_LABELS = frozenset({0, 1, 2, 3, 6, 7, 8, 10, 11, 12})
_SAM3_TO_TRAINID = {
    0: 11,   # person
    1: 18,   # bicycle
    2: 17,   # motorcycle
    3: 12,   # rider
    6: 14,   # truck
    7: 15,   # bus
    8: 16,   # train
    12: 13,  # car
}

CS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}


def _unpack_model_outputs(outputs):
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2]
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1], None
    raise ValueError("AdaptiveInstanceNet must return 2 or 3 tensors")


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

    depth_subdir = args.depth_subdir or config.get("depth_subdir", "depth_spidepth")
    semantic_mode = args.semantic_mode
    if semantic_mode == "auto" and config.get("semantic_mode"):
        semantic_mode = config["semantic_mode"]
    semantic_spec = infer_semantic_spec(
        args.cityscapes_root,
        args.semantic_subdir,
        semantic_mode=semantic_mode,
        centroids_path=args.centroids_path or config.get("centroids_path"),
        num_semantic_classes=(
            args.num_semantic_classes or config.get("semantic_dim")
        ),
        split="train" if args.split in ("train", "both") else args.split,
    )
    print(
        f"Semantic mode: {semantic_spec.mode} | "
        f"semantic_dim={semantic_spec.num_classes} | "
        f"centroids={semantic_spec.centroids_path or 'none'}"
    )
    use_highres_fusion = bool(config.get("use_highres_fusion", False))
    fusion_h = int(config.get("fusion_h", FUSION_H))
    fusion_w = int(config.get("fusion_w", FUSION_W))
    sam3_subdir = args.sam3_subdir or config.get("sam3_subdir", "sam_fine_masks_sam3")
    sam3_mask_threshold = float(config.get("sam3_mask_threshold", 0.25))
    sam3_fusion_dropout = float(config.get("sam3_fusion_dropout", 0.0))
    print(
        f"Depth subdir: {depth_subdir} | edge_mode={args.edge_mode} | "
        f"highres_fusion={use_highres_fusion}"
    )

    # Build model
    model = AdaptiveInstanceNet(
        feature_dim=config.get("feature_dim", 768),
        depth_channels=3,
        semantic_dim=semantic_spec.num_classes,
        hidden_dim=config.get("hidden_dim", 256),
        embed_dim=config.get("embed_dim", 32),
        num_blocks=config.get("num_blocks", 6),
        use_highres_fusion=use_highres_fusion,
        fusion_channels=config.get("fusion_channels", FUSION_CHANNELS),
        fusion_hidden_dim=config.get("fusion_hidden_dim", 64),
        fusion_blocks=config.get("fusion_blocks", 2),
        thing_classes=config.get("thing_classes", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    os.makedirs(args.output_dir, exist_ok=True)
    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n=== Generating adaptive instances for {split} ===")
        stats = validate_semantic_inputs(
            args.cityscapes_root,
            args.semantic_subdir,
            depth_subdir,
            semantic_spec,
            split=split,
            max_samples=args.diagnostic_samples,
        )
        print(
            f"Semantic diagnostics {split}: labels={stats['label_min']}.."
            f"{stats['label_max']} valid_trainID="
            f"{100.0 * stats['valid_trainid_frac']:.1f}% thing_trainID="
            f"{100.0 * stats['thing_trainid_frac']:.1f}%"
        )

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
                (features, depth_patch, depth_grads, cause_onehot,
                 sem_trainid, depth_native, fusion_inputs) = _load_inputs(
                    args.cityscapes_root, args.semantic_subdir,
                    depth_subdir, semantic_spec, split, city, stem, device,
                    use_highres_fusion=use_highres_fusion and not args.no_sam3,
                    fusion_h=fusion_h,
                    fusion_w=fusion_w,
                    sam3_subdir=sam3_subdir,
                    sam3_mask_threshold=sam3_mask_threshold,
                    sam3_fusion_dropout=sam3_fusion_dropout,
                )

                # Forward pass (model outputs raw logits)
                outputs = model(
                    features, depth_patch, depth_grads, cause_onehot,
                    fusion_inputs=fusion_inputs)
                split_logit, embed, thing_logits = _unpack_model_outputs(outputs)
                split_prob = torch.sigmoid(split_logit)
                thing_logits_np = None
                if thing_logits is not None:
                    thing_logits_np = thing_logits.squeeze(0).cpu().numpy()

                # Generate instances
                sam3_proposals = None
                if (not args.no_sam3 and args.use_sam3_proposals
                        and not args.no_sam3_proposals):
                    sam3_proposals = _load_sam3_proposals(
                        args.cityscapes_root, sam3_subdir, split, city, stem,
                        out_hw=(EVAL_H, EVAL_W),
                        mask_threshold=sam3_mask_threshold,
                        min_area=args.min_area,
                    )
                instances = _generate_instances(
                    split_prob.squeeze(0).squeeze(0).cpu().numpy(),
                    embed.squeeze(0).cpu().numpy(),
                    sem_trainid,
                    depth_native=depth_native,
                    sam3_proposals=sam3_proposals,
                    thing_logits_patch=thing_logits_np,
                    split_threshold=args.split_threshold,
                    min_area=args.min_area,
                    dilation_iters=args.dilation_iters,
                    use_embeddings=args.use_embeddings,
                    embed_merge_thresh=args.embed_merge_thresh,
                    edge_mode=args.edge_mode,
                    base_tau=args.base_tau,
                    thing_conf_threshold=args.thing_conf_threshold,
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
                "edge_mode": args.edge_mode,
                "depth_subdir": depth_subdir,
                "thing_conf_threshold": args.thing_conf_threshold,
                "semantic_mode": semantic_spec.mode,
                "semantic_dim": semantic_spec.num_classes,
                "centroids_path": semantic_spec.centroids_path,
                "use_highres_fusion": use_highres_fusion,
                "sam3_subdir": sam3_subdir,
                "use_sam3_proposals": (
                    not args.no_sam3
                    and args.use_sam3_proposals
                    and not args.no_sam3_proposals
                ),
            },
        }
        stats_path = os.path.join(args.output_dir, split, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats: {json.dumps(stats['per_class'], indent=2)}")

    print("\nDone!")


def _load_inputs(root, semantic_subdir, depth_subdir, semantic_spec,
                 split, city, stem, device, use_highres_fusion=False,
                 fusion_h=FUSION_H, fusion_w=FUSION_W,
                 sam3_subdir="sam_fine_masks_sam3",
                 sam3_mask_threshold=0.25,
                 sam3_fusion_dropout=0.0):
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
        os.path.join(root, depth_subdir, split, city, f"{stem}.npy"))
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

    # Semantic one-hot/probabilities.
    sem_path = os.path.join(root, semantic_subdir, split, city, f"{stem}.png")
    sem_full = np.array(Image.open(sem_path))
    sem_patch = np.array(
        Image.fromarray(sem_full).resize((PATCH_W, PATCH_H), Image.NEAREST))

    onehot = encode_semantic_onehot(sem_patch, semantic_spec)
    cause_onehot = torch.from_numpy(onehot).unsqueeze(0).to(device)

    # TrainID semantic at full resolution for CC
    sem_trainid = map_to_trainid(sem_full, semantic_spec)
    if sem_trainid.shape != (EVAL_H, EVAL_W):
        sem_trainid = np.array(
            Image.fromarray(sem_trainid).resize((EVAL_W, EVAL_H), Image.NEAREST))

    fusion_inputs = None
    if use_highres_fusion:
        fusion_np = _build_fusion_inputs(
            root, sam3_subdir, split, city, stem, depth_full,
            fusion_h, fusion_w, sam3_mask_threshold)
        if sam3_fusion_dropout >= 1.0:
            fusion_np[4:] = 0.0
        fusion_inputs = torch.from_numpy(fusion_np).unsqueeze(0).to(device)

    return (features, depth_patch.to(device), depth_grads.to(device),
            cause_onehot, sem_trainid, depth_full, fusion_inputs)


def _build_fusion_inputs(root, sam3_subdir, split, city, stem, depth_full,
                         out_h, out_w, mask_threshold):
    depth = np.array(
        Image.fromarray(depth_full.astype(np.float32)).resize(
            (out_w, out_h), Image.BILINEAR),
        dtype=np.float32,
    )
    d_min, d_max = float(np.nanmin(depth)), float(np.nanmax(depth))
    if d_max > d_min + 1e-6:
        depth = ((depth - d_min) / (d_max - d_min)).astype(np.float32)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)

    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32).reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32).reshape(1, 1, 3, 3)
    gx = F.conv2d(depth_t, kx, padding=1).squeeze().numpy()
    gy = F.conv2d(depth_t, ky, padding=1).squeeze().numpy()
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    p95 = np.percentile(grad_mag, 95)
    if p95 > 1e-6:
        grad_mag = np.clip(grad_mag / p95, 0.0, 1.0)

    boundary, instance_map, ious, smallness = _load_sam3_prior(
        root, sam3_subdir, split, city, stem, out_h, out_w, mask_threshold)
    occupancy = (instance_map >= 0).astype(np.float32)
    confidence = np.zeros_like(occupancy, dtype=np.float32)
    for n, iou in enumerate(ious):
        if iou <= 0:
            continue
        confidence[instance_map == n] = float(iou)

    if occupancy.any():
        inside = ndimage.distance_transform_edt(occupancy > 0)
        outside = ndimage.distance_transform_edt(occupancy <= 0)
        inside = inside / max(float(inside.max()), 1.0)
        outside = outside / max(float(outside.max()), 1.0)
        signed_dist = (inside - outside).astype(np.float32)
    else:
        signed_dist = np.zeros_like(occupancy, dtype=np.float32)

    return np.stack([
        depth.astype(np.float32),
        gx.astype(np.float32),
        gy.astype(np.float32),
        grad_mag.astype(np.float32),
        occupancy,
        boundary.astype(np.float32),
        signed_dist,
        confidence,
        smallness.astype(np.float32),
    ], axis=0)


def _load_sam3_prior(root, sam3_subdir, split, city, stem,
                     out_h, out_w, mask_threshold):
    boundary = np.zeros((out_h, out_w), dtype=bool)
    instance_id = np.full((out_h, out_w), -1, dtype=np.int32)
    smallness = np.zeros((out_h, out_w), dtype=np.float32)
    ious_pad = np.zeros(SAM3_MAX_MASKS, dtype=np.float32)
    npz_path = os.path.join(root, sam3_subdir, split, city, f"{stem}_fine_masks.npz")
    if not os.path.exists(npz_path):
        return boundary, instance_id, ious_pad, smallness

    try:
        data = np.load(npz_path)
        masks_full = data["masks"].astype(bool)
        ious = data["iou_scores"].astype(np.float32)
        cls_labels = data["class_labels"].astype(np.int32)
    except Exception:
        return boundary, instance_id, ious_pad, smallness

    keep = np.array([int(c) in _SAM3_THING_LABELS for c in cls_labels])
    if not keep.any():
        return boundary, instance_id, ious_pad, smallness
    masks_full = masks_full[keep]
    ious = ious[keep]
    order = np.argsort(ious)[::-1][:SAM3_MAX_MASKS]
    masks_full = masks_full[order]
    ious = ious[order]
    ious_pad[:len(ious)] = ious

    for n, mask in enumerate(masks_full):
        m_img = Image.fromarray((mask * 255).astype(np.uint8))
        soft = np.array(m_img.resize((out_w, out_h), Image.BILINEAR)).astype(
            np.float32) / 255.0
        if soft.max() > 0 and not (soft > mask_threshold).any():
            y, x = np.unravel_index(np.argmax(soft), soft.shape)
            soft[y, x] = 1.0
        mask_n = soft > mask_threshold
        claim = mask_n & (instance_id < 0)
        instance_id[claim] = n
        area = max(int(mask_n.sum()), 1)
        smallness[claim] = min(1.0, np.sqrt(64.0 / area) / 4.0)

    h_diff = instance_id[:, 1:] != instance_id[:, :-1]
    boundary[:, :-1] |= h_diff
    boundary[:, 1:] |= h_diff
    v_diff = instance_id[1:, :] != instance_id[:-1, :]
    boundary[:-1, :] |= v_diff
    boundary[1:, :] |= v_diff
    return boundary, instance_id, ious_pad, smallness


def _load_sam3_proposals(root, sam3_subdir, split, city, stem,
                         out_hw=(EVAL_H, EVAL_W), mask_threshold=0.25,
                         min_area=100, max_masks=SAM3_MAX_MASKS):
    """Load SAM3 thing masks as class-aware instance proposals."""
    H, W = out_hw
    npz_path = os.path.join(root, sam3_subdir, split, city, f"{stem}_fine_masks.npz")
    if not os.path.exists(npz_path):
        return []

    try:
        data = np.load(npz_path)
        masks = data["masks"].astype(bool)
        ious = data["iou_scores"].astype(np.float32)
        labels = data["class_labels"].astype(np.int32)
    except Exception:
        return []

    keep = np.array([int(c) in _SAM3_TO_TRAINID for c in labels])
    if not keep.any():
        return []

    masks = masks[keep]
    ious = ious[keep]
    labels = labels[keep]
    order = np.argsort(ious)[::-1][:max_masks]

    proposals = []
    for idx in order:
        cls = _SAM3_TO_TRAINID.get(int(labels[idx]))
        if cls is None:
            continue
        mask = masks[idx]
        if mask.shape != (H, W):
            soft = np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (W, H), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            if soft.max() > 0 and not (soft > mask_threshold).any():
                y, x = np.unravel_index(np.argmax(soft), soft.shape)
                soft[y, x] = 1.0
            mask = soft > mask_threshold
        else:
            mask = mask.astype(bool)

        labeled, n_cc = ndimage.label(mask)
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = int(cc_mask.sum())
            if area < min_area:
                continue
            proposals.append((cc_mask, cls, float(ious[idx]), area))

    proposals.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return proposals


def _generate_instances(
    split_prob_patch, embed_patch, sem_trainid_full,
    depth_native=None,
    sam3_proposals=None,
    thing_logits_patch=None,
    split_threshold=0.5, min_area=100, dilation_iters=3,
    use_embeddings=False, embed_merge_thresh=0.7,
    edge_mode="hybrid_or", base_tau=0.05,
    thing_conf_threshold=0.35,
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
    model_edge = split_full > split_threshold
    if edge_mode == "direct" or depth_native is None:
        edge_map = model_edge
    else:
        depth_edge = _adaptive_depth_edges(
            depth_native, split_full, base_tau=base_tau, out_hw=(H, W))
        if edge_mode == "hybrid":
            edge_map = depth_edge
        elif edge_mode == "hybrid_or":
            edge_map = depth_edge | model_edge
        else:
            raise ValueError(f"Unknown edge_mode: {edge_mode}")

    # Optionally upsample embeddings for merging
    if use_embeddings:
        E = embed_patch.shape[0]
        embed_t = torch.from_numpy(embed_patch).unsqueeze(0)
        embed_full = F.interpolate(
            embed_t, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0).numpy()  # (E, H, W)

    assigned = np.zeros((H, W), dtype=bool)
    instances = []

    thing_trainid_full = thing_conf_full = None
    if thing_logits_patch is not None and thing_logits_patch.shape[0] > 1:
        thing_t = torch.from_numpy(thing_logits_patch).unsqueeze(0).float()
        thing_prob = F.softmax(thing_t, dim=1)
        if thing_prob.shape[-2:] != (H, W):
            thing_prob = F.interpolate(
                thing_prob, size=(H, W), mode="bilinear", align_corners=False)
        thing_np = thing_prob.squeeze(0).numpy()
        thing_idx = thing_np.argmax(axis=0)
        safe_idx = np.clip(thing_idx, 0, len(THING_HEAD_IDX_TO_TRAINID) - 1)
        thing_trainid_full = THING_HEAD_IDX_TO_TRAINID[safe_idx]
        thing_conf_full = thing_np.max(axis=0)

    if sam3_proposals:
        for prop_mask, cls, score, _area in sam3_proposals:
            final = prop_mask & ~assigned
            if final.sum() < min_area:
                continue
            assigned |= final
            instances.append((final, cls, float(score) * float(final.sum())))

    for cls in sorted(DEFAULT_THING_IDS):
        if thing_trainid_full is not None:
            cls_mask = (
                (thing_trainid_full == cls)
                & (thing_conf_full >= thing_conf_threshold)
                & ~assigned
            )
            if cls_mask.sum() < min_area:
                cls_mask = (sem_trainid_full == cls) & ~assigned
        else:
            cls_mask = (sem_trainid_full == cls) & ~assigned
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


def _adaptive_depth_edges(depth_native, split_full, base_tau=0.05,
                          out_hw=(EVAL_H, EVAL_W)):
    """Depth edge map with local threshold modulated by model split prob."""
    H, W = out_hw
    depth = depth_native
    if depth.shape != (H, W):
        depth = np.array(
            Image.fromarray(depth.astype(np.float32)).resize((W, H), Image.BILINEAR)
        )
    depth_smooth = ndimage.gaussian_filter(depth.astype(np.float64), sigma=1.0)
    gx = ndimage.sobel(depth_smooth, axis=1)
    gy = ndimage.sobel(depth_smooth, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    tau_local = base_tau * (1.5 - split_full)
    tau_local = np.clip(tau_local, 0.02, 0.25)
    return grad_mag > tau_local


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
            class_ids=np.zeros((0,), dtype=np.uint8),
            num_valid=0,
            h_patches=H,
            w_patches=W,
        )
        return

    num = len(instances)
    masks = np.zeros((num, H * W), dtype=bool)
    scores = np.zeros(num, dtype=np.float32)
    class_ids = np.zeros(num, dtype=np.uint8)

    for i, (mask, cls, score) in enumerate(instances):
        masks[i] = mask.ravel()
        scores[i] = score
        class_ids[i] = cls

    np.savez_compressed(
        output_path,
        masks=masks,
        scores=scores,
        class_ids=class_ids,
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
    parser.add_argument("--semantic_mode", type=str, default="auto",
                        choices=["auto", "cluster", "cause27", "trainid"],
                        help="How to interpret semantic PNG values")
    parser.add_argument("--centroids_path", type=str, default=None,
                        help="kmeans_centroids.npz for raw cluster semantics")
    parser.add_argument("--num_semantic_classes", type=int, default=None,
                        help="Override semantic input channels; inferred by default")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "both"])
    parser.add_argument("--depth_subdir", type=str, default=None,
                        help="Depth subdir; defaults to checkpoint config")
    parser.add_argument("--sam3_subdir", type=str, default=None,
                        help="SAM3 mask subdir; defaults to checkpoint config")
    parser.add_argument("--no_sam3", action="store_true",
                        help="Disable SAM3 priors for high-res fusion checkpoints")
    parser.add_argument("--no_sam3_proposals", action="store_true",
                        help="Do not emit class-aware SAM3 masks as instance proposals")
    parser.add_argument("--use_sam3_proposals", action="store_true",
                        help="Use SAM3 masks as generation-time proposals (diagnostic only)")
    parser.add_argument("--split_threshold", type=float, default=0.5)
    parser.add_argument("--base_tau", type=float, default=0.05,
                        help="Base depth threshold for hybrid edge modes")
    parser.add_argument("--edge_mode", type=str, default="hybrid_or",
                        choices=["hybrid_or", "hybrid", "direct"],
                        help="How learned boundaries combine with depth edges")
    parser.add_argument("--thing_conf_threshold", type=float, default=0.35,
                        help="Confidence threshold for checkpoint-predicted thing masks")
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--dilation_iters", type=int, default=3)
    parser.add_argument("--diagnostic_samples", type=int, default=32,
                        help="Samples per split for semantic/depth validation")
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
