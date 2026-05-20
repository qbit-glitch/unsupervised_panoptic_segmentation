#!/usr/bin/env python3
"""Generate instance pseudo-labels from a trained SGM adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.adaptive_instance_semantics import infer_semantic_spec, map_to_trainid
from mbps_pytorch.generate_depth_guided_instances import WORK_H, WORK_W
from mbps_pytorch.models.instance.sgm_adapter import SGMAdapter, SGMAdapterConfig
from mbps_pytorch.train_sgm_adapter import THING_TRAIN_IDS, _patch_depth_from_full


def _resolve(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_entries(root: Path, split: str, semantic_subdir: str,
                     depth_subdir: str, dino_subdir: str) -> list[dict]:
    sem_root = root / semantic_subdir / split
    entries = []
    for sem_path in sorted(sem_root.rglob("*.png")):
        city = sem_path.parent.name
        stem = sem_path.stem.replace("_leftImg8bit", "")
        image = _resolve([
            root / "leftImg8bit" / split / city / f"{stem}_leftImg8bit.png",
            root / "leftImg8bit" / split / city / f"{stem}.png",
        ])
        depth = _resolve([
            root / depth_subdir / split / city / f"{stem}.npy",
            root / depth_subdir / split / city / f"{stem}_leftImg8bit.npy",
        ])
        dino = _resolve([
            root / dino_subdir / split / city / f"{stem}.npy",
            root / dino_subdir / split / city / f"{stem}_leftImg8bit.npy",
        ])
        if image is None or depth is None or dino is None:
            continue
        entries.append({
            "city": city,
            "stem": stem,
            "image": image,
            "semantic": sem_path,
            "depth": depth,
            "dino": dino,
        })
    return entries


def load_depth(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.shape != (WORK_H, WORK_W):
        depth = np.array(
            Image.fromarray(depth).resize((WORK_W, WORK_H), Image.BILINEAR),
            dtype=np.float32,
        )
    return depth


def load_semantic(path: Path, semantic_spec) -> np.ndarray:
    semantic = np.array(Image.open(path))
    if semantic.shape != (WORK_H, WORK_W):
        semantic = np.array(
            Image.fromarray(semantic).resize((WORK_W, WORK_H), Image.NEAREST)
        )
    return map_to_trainid(semantic, semantic_spec)


def load_dino(path: Path) -> np.ndarray:
    dino = np.load(path).astype(np.float32)
    if dino.ndim == 2:
        if dino.shape[0] == 32 * 64:
            dino = dino.reshape(32, 64, -1)
        else:
            side = int(np.sqrt(dino.shape[0]))
            if side * side == dino.shape[0]:
                dino = dino.reshape(side, side, -1)
            else:
                hp = 32
                wp = dino.shape[0] // hp
                dino = dino.reshape(hp, wp, -1)
    return dino


def save_instances(instances: list[tuple[np.ndarray, int, float]], output_path: Path,
                   h: int = WORK_H, w: int = WORK_W) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not instances:
        np.savez_compressed(
            str(output_path),
            masks=np.zeros((0, h * w), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int32),
            num_valid=0,
            h_patches=h,
            w_patches=w,
        )
        Image.fromarray(np.zeros((h, w), dtype=np.uint16)).save(
            str(output_path).replace(".npz", "_instance.png")
        )
        return

    masks = np.zeros((len(instances), h * w), dtype=bool)
    scores = np.zeros((len(instances),), dtype=np.float32)
    class_ids = np.zeros((len(instances),), dtype=np.int32)
    vis = np.zeros((h, w), dtype=np.uint16)
    for idx, (mask, cls, score) in enumerate(instances):
        masks[idx] = mask.reshape(-1)
        scores[idx] = float(score)
        class_ids[idx] = int(cls)
        vis[mask] = idx + 1
    np.savez_compressed(
        str(output_path),
        masks=masks,
        scores=scores,
        class_ids=class_ids,
        num_valid=len(instances),
        h_patches=h,
        w_patches=w,
    )
    Image.fromarray(vis).save(str(output_path).replace(".npz", "_instance.png"))


def masks_from_probs(probs: np.ndarray, semantic_trainid: np.ndarray,
                     threshold: float, min_area: int,
                     class_min_area: dict[int, int]) -> list[tuple[np.ndarray, int, float]]:
    instances = []
    for thing_idx, cls in enumerate(THING_TRAIN_IDS):
        cls_prob = probs[thing_idx]
        cls_min_area = int(class_min_area.get(int(cls), min_area))
        binary = (cls_prob >= threshold) & (semantic_trainid == int(cls))
        if binary.sum() < cls_min_area:
            continue
        labeled, n_comp = ndimage.label(binary)
        for comp_id in range(1, n_comp + 1):
            mask = labeled == comp_id
            area = int(mask.sum())
            if area < cls_min_area:
                continue
            score = float(cls_prob[mask].mean())
            instances.append((mask, int(cls), score))
    instances.sort(key=lambda item: item[2], reverse=True)
    return instances


def parse_class_min_area(text: str | None) -> dict[int, int]:
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        out[int(key)] = int(value)
    return out


def generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    root = Path(args.cityscapes_root)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SGMAdapterConfig(**ckpt["adapter_config"])
    model = SGMAdapter(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    semantic_spec = infer_semantic_spec(
        root,
        args.semantic_subdir,
        semantic_mode=args.semantic_mode,
        centroids_path=args.centroids_path,
        num_semantic_classes=args.num_semantic_classes,
        split=args.split,
    )
    entries = discover_entries(
        root, args.split, args.semantic_subdir, args.depth_subdir, args.dino_subdir)
    if args.max_images:
        entries = entries[:args.max_images]
    out_root = Path(args.output_dir)
    class_min_area = parse_class_min_area(args.class_min_area)
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "threshold": args.threshold,
        "min_area": args.min_area,
        "class_min_area": class_min_area,
        "num_images": len(entries),
        "total_instances": 0,
        "per_class": {str(c): 0 for c in THING_TRAIN_IDS},
    }

    for entry in tqdm(entries, desc="Generating SGM instances", ncols=100):
        depth_np = load_depth(entry["depth"])
        semantic = load_semantic(entry["semantic"], semantic_spec)
        dino_np = load_dino(entry["dino"])
        dino_patch = torch.from_numpy(dino_np.reshape(-1, dino_np.shape[-1])).float()
        dino_patch = dino_patch.unsqueeze(0).to(device)
        depth_t = torch.from_numpy(depth_np).float().to(device)
        depth_patch = _patch_depth_from_full(
            depth_t, config.patch_h, config.patch_w).unsqueeze(0)
        with torch.no_grad():
            probs = model(dino_patch, depth_patch, out_hw=(WORK_H, WORK_W))
        probs_np = probs.squeeze(0).cpu().numpy()
        instances = masks_from_probs(
            probs_np, semantic, args.threshold, args.min_area, class_min_area)
        out_path = out_root / args.split / entry["city"] / f"{entry['stem']}.npz"
        save_instances(instances, out_path)
        summary["total_instances"] += len(instances)
        for _mask, cls, _score in instances:
            summary["per_class"][str(cls)] += 1

    summary["avg_instances"] = (
        summary["total_instances"] / max(summary["num_images"], 1)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_raw_k80")
    parser.add_argument("--semantic_mode", default="auto",
                        choices=("auto", "cluster", "cause27", "trainid"))
    parser.add_argument("--centroids_path", default=None)
    parser.add_argument("--num_semantic_classes", type=int, default=None)
    parser.add_argument("--depth_subdir", default="depth_depthpro")
    parser.add_argument("--dino_subdir", default="dinov2_features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=600)
    parser.add_argument("--class_min_area", default="11:300,12:300,18:300")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
