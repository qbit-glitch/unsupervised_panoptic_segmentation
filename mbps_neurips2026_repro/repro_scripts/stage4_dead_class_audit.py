#!/usr/bin/env python3
"""Audit Stage-4 dead-class coverage in a flat CUPS pseudo-label directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from PIL import Image


CITYSCAPES_27 = [
    "road", "sidewalk", "parking", "rail track", "building", "wall", "fence",
    "guard rail", "bridge", "tunnel", "pole", "polegroup", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle",
]


def _normalise(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def _load_png(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _load_distribution(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(root.glob("*.pt"))]
    if not tensors:
        raise FileNotFoundError(f"No .pt distribution files found in {root}")
    inside = torch.stack([t["distribution inside object proposals"] for t in tensors]).sum(dim=0)
    all_pixels = torch.stack([t["distribution all pixels"] for t in tensors]).sum(dim=0)
    return inside.float(), all_pixels.float()


def _thing_stuff_split(root: Path, threshold: float) -> tuple[tuple[int, ...], tuple[int, ...], list[float]]:
    inside, all_pixels = _load_distribution(root)
    distribution, indices = torch.sort(inside / (all_pixels + 1e-6), descending=True)
    distribution = distribution / distribution.sum().clamp_min(1e-6)
    num_instance = int((distribution > threshold).float().argmin().item())
    things = tuple(int(v) for v in indices[:num_instance].tolist())
    stuffs = tuple(int(v) for v in indices[num_instance:].tolist())
    return things, stuffs, [float(v) for v in (all_pixels / all_pixels.sum().clamp_min(1.0)).tolist()]


def _resolve_rare(names: Iterable[str]) -> Dict[str, int | None]:
    name_to_id = {_normalise(name): idx for idx, name in enumerate(CITYSCAPES_27)}
    return {name: name_to_id.get(_normalise(name)) for name in names}


def audit(root: Path, rare_names: list[str], threshold: float) -> dict:
    things, stuffs, distribution = _thing_stuff_split(root, threshold)
    rare = _resolve_rare(rare_names)
    semantic_paths = sorted(root.glob("*_semantic.png"))
    instance_paths = {p.name.replace("_instance.png", ""): p for p in root.glob("*_instance.png")}

    stats = {
        name: {
            "class_id": cid,
            "pixel_count": 0,
            "image_count": 0,
            "instance_count": 0,
            "frequency": distribution[cid] if cid is not None and cid < len(distribution) else 0.0,
            "cups_stuff_target": stuffs.index(cid) + 1 if cid in stuffs else None,
            "cups_thing_target": things.index(cid) if cid in things else None,
            "recoverable": False,
        }
        for name, cid in rare.items()
    }

    for sem_path in semantic_paths:
        sem = _load_png(sem_path)
        stem = sem_path.name.replace("_semantic.png", "")
        inst_path = instance_paths.get(stem)
        inst = _load_png(inst_path) if inst_path is not None else None
        for name, cid in rare.items():
            if cid is None:
                continue
            mask = sem == cid
            count = int(mask.sum())
            if count > 0:
                stats[name]["pixel_count"] += count
                stats[name]["image_count"] += 1
            if inst is not None and count > 0:
                for inst_id in np.unique(inst[mask]):
                    if inst_id != 0:
                        stats[name]["instance_count"] += 1

    for item in stats.values():
        item["recoverable"] = bool(item["pixel_count"] > 0 or item["instance_count"] > 0)

    return {
        "pseudo_dir": str(root),
        "thing_stuff_threshold": threshold,
        "things_classes": things,
        "stuff_classes": stuffs,
        "rare_classes": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_dir", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument(
        "--rare_classes",
        nargs="+",
        default=["guard rail", "tunnel", "polegroup", "caravan", "trailer"],
    )
    parser.add_argument("--thing_stuff_threshold", type=float, default=0.05)
    parser.add_argument(
        "--candidate_output_dir",
        type=Path,
        default=None,
        help="Optional directory to populate with the current pseudo labels as a Stage-4 candidate root.",
    )
    args = parser.parse_args()

    report = audit(args.pseudo_dir, args.rare_classes, args.thing_stuff_threshold)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.candidate_output_dir is not None:
        args.candidate_output_dir.mkdir(parents=True, exist_ok=True)
        for path in args.pseudo_dir.iterdir():
            if path.suffix.lower() in {".png", ".pt"}:
                shutil.copy2(path, args.candidate_output_dir / path.name)

    print(json.dumps(report["rare_classes"], indent=2))


if __name__ == "__main__":
    main()
