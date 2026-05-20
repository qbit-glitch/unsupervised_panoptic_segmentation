#!/usr/bin/env python3
"""Generate learnable instance pseudo-labels with SuperpixelAffinityAdapter.

The output format matches ``generate_depth_guided_instances.py`` NPZ files, so
existing evaluation can be reused:

    python mbps_pytorch/evaluate_cascade_pseudolabels.py \\
        --cityscapes_root /path/to/cityscapes \\
        --semantic_subdir pseudo_semantic_raw_k80 \\
        --instance_subdir superpixel_affinity_instances \\
        --num_clusters 80 \\
        --cluster_mapping_path /path/to/kmeans_centroids.npz \\
        --thing_mode maskcut --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.adaptive_instance_semantics import (
    infer_semantic_spec,
    map_to_trainid,
)
from mbps_pytorch.instance_methods.superpixel_affinity import (
    SuperpixelExtractionConfig,
    build_depth_cc_prior,
    build_superpixel_graph,
    instances_from_edge_probs,
    load_feature_grid,
    load_proposal_bank,
    resize_bilinear,
    resize_nearest,
    save_instances_npz,
)
from mbps_pytorch.models.instance.superpixel_affinity_adapter import (
    SuperpixelAffinityAdapter,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_entries(
    cityscapes_root: Path,
    split: str,
    semantic_subdir: str,
    depth_subdir: str,
    feature_subdir: str | None,
    clip_feature_subdir: str | None,
    proposal_bank_dir: str | None,
) -> list[dict]:
    entries = []
    img_root = cityscapes_root / "leftImg8bit" / split
    for image_path in sorted(img_root.rglob("*_leftImg8bit.png")):
        city = image_path.parent.name
        stem = image_path.name.replace("_leftImg8bit.png", "")
        sem_path = resolve_existing([
            cityscapes_root / semantic_subdir / split / city / f"{stem}.png",
            cityscapes_root / semantic_subdir / split / city / f"{stem}_leftImg8bit.png",
        ])
        depth_path = resolve_existing([
            cityscapes_root / depth_subdir / split / city / f"{stem}.npy",
            cityscapes_root / depth_subdir / split / city / f"{stem}_leftImg8bit.npy",
        ])
        feat_path = None
        if feature_subdir:
            feat_path = resolve_existing([
                cityscapes_root / feature_subdir / split / city / f"{stem}_leftImg8bit.npy",
                cityscapes_root / feature_subdir / split / city / f"{stem}.npy",
            ])
        clip_path = None
        if clip_feature_subdir:
            clip_path = resolve_existing([
                cityscapes_root / clip_feature_subdir / split / city / f"{stem}_leftImg8bit.npy",
                cityscapes_root / clip_feature_subdir / split / city / f"{stem}.npy",
            ])
        proposal_path = None
        if proposal_bank_dir:
            proposal_root = Path(proposal_bank_dir)
            if not proposal_root.is_absolute():
                proposal_root = cityscapes_root / proposal_root
            proposal_path = resolve_existing([
                proposal_root / split / city / f"{stem}.npz",
                proposal_root / split / city / f"{stem}_leftImg8bit.npz",
                proposal_root / split / f"{stem}.npz",
                proposal_root / city / f"{stem}.npz",
                proposal_root / f"{stem}.npz",
            ])
        if sem_path is None or depth_path is None:
            continue
        if feature_subdir and feat_path is None:
            continue
        entries.append({
            "city": city,
            "stem": stem,
            "image": image_path,
            "semantic": sem_path,
            "depth": depth_path,
            "feature": feat_path,
            "clip": clip_path,
            "proposal": proposal_path,
        })
    return entries


def load_clip_prototypes(raw) -> dict[int, np.ndarray] | None:
    if not raw:
        return None
    return {int(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def parse_class_min_area(text: str | None) -> dict[int, int] | None:
    """Parse class-specific min areas, e.g. ``11:300,12:300,18:300``."""
    if text is None or str(text).strip() == "":
        return None
    out: dict[int, int] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "class_min_area entries must be formatted as class_id:min_area")
        cls_s, area_s = item.split(":", 1)
        out[int(cls_s.strip())] = int(area_s.strip())
    return out


def generate(args) -> None:
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    extraction = config.get("extraction", {})
    cfg = SuperpixelExtractionConfig(**extraction)
    if args.n_segments is not None:
        cfg.n_segments = args.n_segments
    if args.pseudo_tau is not None:
        cfg.pseudo_tau = args.pseudo_tau
    if args.pseudo_min_area is not None:
        cfg.pseudo_min_area = args.pseudo_min_area

    root = Path(args.cityscapes_root)
    semantic_subdir = args.semantic_subdir or config.get("semantic_subdir")
    depth_subdir = args.depth_subdir or config.get("depth_subdir", "depth_depthpro")
    feature_subdir = args.feature_subdir if args.feature_subdir is not None else config.get("feature_subdir")
    clip_feature_subdir = (
        args.clip_feature_subdir
        if args.clip_feature_subdir is not None
        else config.get("clip_feature_subdir")
    )
    proposal_bank_dir = (
        args.proposal_bank_dir
        if args.proposal_bank_dir is not None
        else config.get("proposal_bank_dir")
    )
    semantic_mode = args.semantic_mode or config.get("semantic_mode", "auto")
    centroids_path = args.centroids_path or config.get("centroids_path")
    semantic_spec = infer_semantic_spec(
        root,
        semantic_subdir,
        semantic_mode=semantic_mode,
        centroids_path=centroids_path,
        num_semantic_classes=config.get("semantic_dim"),
        split="train" if args.split == "both" else args.split,
    )

    model = SuperpixelAffinityAdapter(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.0)),
        use_layer_norm=bool(config.get("use_layer_norm", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feat_mean = torch.from_numpy(np.asarray(ckpt["feature_mean"], dtype=np.float32)).to(device)
    feat_std = torch.from_numpy(np.asarray(ckpt["feature_std"], dtype=np.float32)).to(device)
    clip_prototypes = load_clip_prototypes(ckpt.get("clip_prototypes"))
    dino_projection = ckpt.get("dino_projection")
    clip_projection = ckpt.get("clip_projection")
    if dino_projection is not None:
        dino_projection = np.asarray(dino_projection, dtype=np.float32)
    if clip_projection is not None:
        clip_projection = np.asarray(clip_projection, dtype=np.float32)
    merge_threshold = args.merge_threshold
    if merge_threshold is None:
        merge_threshold = float(config.get("merge_threshold", 0.55))
    min_area = args.min_area if args.min_area is not None else int(config.get("min_area", 1000))
    class_min_area = parse_class_min_area(args.class_min_area)

    log.info("Device: %s", device)
    log.info("Semantic: %s | dim=%d | centroids=%s",
             semantic_spec.mode, semantic_spec.num_classes,
             semantic_spec.centroids_path or "none")
    log.info("Model params: %d | merge_threshold=%.3f | min_area=%d",
             sum(p.numel() for p in model.parameters()), merge_threshold, min_area)
    if class_min_area:
        log.info("Class min area overrides: %s", class_min_area)

    out_root = Path(args.output_dir)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    summary = {}
    for split in splits:
        entries = discover_entries(
            root, split, semantic_subdir, depth_subdir,
            feature_subdir, clip_feature_subdir, proposal_bank_dir)
        if args.max_images is not None:
            entries = entries[:args.max_images]
        log.info("%s: %d images", split, len(entries))
        if cfg.proposal_objectness_enabled:
            log.info(
                "%s proposal files: %d/%d root=%s",
                split,
                sum(1 for e in entries if e.get("proposal") is not None),
                len(entries),
                proposal_bank_dir or "none",
            )
        total_instances = 0
        counts = []
        per_class = {c: 0 for c in range(11, 19)}

        for entry in tqdm(entries, desc=f"Generating {split}", ncols=100):
            image = np.array(Image.open(entry["image"]).convert("RGB"))
            h, w = image.shape[:2]
            sem_raw = np.array(Image.open(entry["semantic"]))
            if sem_raw.shape != (h, w):
                sem_raw = resize_nearest(sem_raw.astype(np.uint8), (h, w))
            sem_trainid = map_to_trainid(sem_raw, semantic_spec)
            depth = np.load(str(entry["depth"])).astype(np.float32)
            if depth.shape != (h, w):
                depth = resize_bilinear(depth, (h, w))
            dino = load_feature_grid(entry.get("feature"))
            clip = load_feature_grid(entry.get("clip"))
            proposal_masks, proposal_scores = load_proposal_bank(
                entry.get("proposal"), image_hw=(h, w))
            pseudo = build_depth_cc_prior(sem_trainid, depth, cfg)

            graph = build_superpixel_graph(
                image_rgb=image,
                semantic_trainid=sem_trainid,
                depth=depth,
                dino_features=dino,
                clip_features=clip,
                proposal_masks=proposal_masks,
                proposal_scores=proposal_scores,
                pseudo_instance_map=pseudo,
                clip_prototypes=clip_prototypes,
                dino_projection=dino_projection,
                clip_projection=clip_projection,
                cfg=cfg,
            )
            if graph.edge_features.shape[0] == 0:
                instances = []
            else:
                x = torch.from_numpy(graph.edge_features).float().to(device)
                x = (x - feat_mean) / feat_std.clamp_min(1e-4)
                with torch.no_grad():
                    probs = torch.sigmoid(model(x)).cpu().numpy()
                instances = instances_from_edge_probs(
                    graph,
                    probs,
                    semantic_trainid=sem_trainid,
                    merge_threshold=merge_threshold,
                    min_area=min_area,
                    class_min_area=class_min_area,
                )

            out_path = out_root / split / entry["city"] / f"{entry['stem']}.npz"
            save_instances_npz(instances, out_path, h, w)
            counts.append(len(instances))
            total_instances += len(instances)
            for _mask, cls, _score in instances:
                per_class[int(cls)] = per_class.get(int(cls), 0) + 1

        summary[split] = {
            "images": len(entries),
            "total_instances": total_instances,
            "avg_instances": float(np.mean(counts)) if counts else 0.0,
            "per_class": per_class,
            "class_min_area": class_min_area or {},
        }
        log.info("%s: %.2f instances/image", split, summary[split]["avg_instances"])

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary: %s", out_root / "generation_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "both"])
    parser.add_argument("--semantic_subdir", default=None)
    parser.add_argument("--semantic_mode", default=None)
    parser.add_argument("--centroids_path", default=None)
    parser.add_argument("--depth_subdir", default=None)
    parser.add_argument("--feature_subdir", default=None)
    parser.add_argument("--clip_feature_subdir", default=None)
    parser.add_argument("--proposal_bank_dir", default=None,
                        help="Optional proposal NPZ root for checkpoints trained with proposal objectness")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--merge_threshold", type=float, default=None)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--class_min_area", default=None,
                        help="Optional class min areas, e.g. 11:300,12:300,18:300")
    parser.add_argument("--n_segments", type=int, default=None)
    parser.add_argument("--pseudo_tau", type=float, default=None)
    parser.add_argument("--pseudo_min_area", type=int, default=None)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
