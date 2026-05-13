"""Semantic-label handling for AdaptiveInstanceNet.

The adaptive instance adapter is used with both CAUSE-27 semantic maps and
raw k-means cluster maps. Keep that distinction explicit so a raw k=80 map
cannot be silently treated as CAUSE labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


TRAINID_NUM_CLASSES = 19
CAUSE27_NUM_CLASSES = 27

THING_TRAIN_IDS = set(range(11, 19))
STUFF_TRAIN_IDS = set(range(0, 11))

CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    CAUSE27_TO_TRAINID[_c27] = _t19


@dataclass(frozen=True)
class SemanticSpec:
    """Resolved semantic-label contract for adapter training/generation."""

    mode: str
    num_classes: int
    label_to_trainid: np.ndarray
    centroids_path: Optional[str] = None

    def to_config(self) -> dict:
        return {
            "semantic_mode": self.mode,
            "semantic_dim": self.num_classes,
            "centroids_path": self.centroids_path,
        }


def infer_semantic_spec(
    cityscapes_root: str | Path,
    semantic_subdir: str,
    semantic_mode: str = "auto",
    centroids_path: Optional[str] = None,
    num_semantic_classes: Optional[int] = None,
    split: str = "train",
) -> SemanticSpec:
    """Resolve how semantic PNG values should be encoded and mapped.

    Modes:
      - cause27: labels are CAUSE-27 ids, encoded as 27 channels.
      - trainid: labels are Cityscapes trainIDs, encoded as 19 channels.
      - cluster: labels are raw cluster ids, encoded as k channels and mapped
        through kmeans_centroids.npz::cluster_to_class for trainID logic.
      - auto: use cluster mode when centroids are present, otherwise infer from
        sampled label range.
    """
    root = Path(cityscapes_root)
    centroid_candidate = (
        Path(centroids_path)
        if centroids_path
        else root / semantic_subdir / "kmeans_centroids.npz"
    )
    max_label = _sample_max_label(root, semantic_subdir, split)

    mode = semantic_mode.lower()
    if mode == "auto":
        if centroid_candidate.is_file():
            mode = "cluster"
        elif max_label <= 18:
            mode = "trainid"
        elif max_label <= 26:
            mode = "cause27"
        else:
            raise ValueError(
                f"Semantic labels in {semantic_subdir}/{split} reach "
                f"{max_label}, but no k-means centroids were found. Pass "
                "--centroids_path or place kmeans_centroids.npz in the "
                "semantic directory."
            )

    if mode == "cluster":
        if not centroid_candidate.is_file():
            raise ValueError(
                f"Cluster semantic mode requires kmeans centroids, missing: "
                f"{centroid_candidate}"
            )
        data = np.load(str(centroid_candidate))
        if "cluster_to_class" not in data.files:
            raise ValueError(
                f"{centroid_candidate} does not contain cluster_to_class"
            )
        cluster_to_class = data["cluster_to_class"].astype(np.int64)
        inferred_classes = int(len(cluster_to_class))
        num_classes = int(num_semantic_classes or inferred_classes)
        if num_classes < inferred_classes:
            raise ValueError(
                f"num_semantic_classes={num_classes} is smaller than "
                f"cluster_to_class length {inferred_classes}"
            )
        lut = np.full(256, 255, dtype=np.uint8)
        valid = (cluster_to_class >= 0) & (cluster_to_class < TRAINID_NUM_CLASSES)
        cluster_ids = np.arange(inferred_classes)[valid]
        lut[cluster_ids] = cluster_to_class[valid].astype(np.uint8)
        spec = SemanticSpec(
            mode=mode,
            num_classes=num_classes,
            label_to_trainid=lut,
            centroids_path=str(centroid_candidate),
        )
    elif mode == "cause27":
        num_classes = int(num_semantic_classes or CAUSE27_NUM_CLASSES)
        if num_classes != CAUSE27_NUM_CLASSES:
            raise ValueError("CAUSE-27 mode requires num_semantic_classes=27")
        spec = SemanticSpec(
            mode=mode,
            num_classes=CAUSE27_NUM_CLASSES,
            label_to_trainid=CAUSE27_TO_TRAINID.copy(),
        )
    elif mode == "trainid":
        num_classes = int(num_semantic_classes or TRAINID_NUM_CLASSES)
        if num_classes < TRAINID_NUM_CLASSES:
            raise ValueError("trainID mode requires at least 19 channels")
        lut = np.full(256, 255, dtype=np.uint8)
        lut[:TRAINID_NUM_CLASSES] = np.arange(TRAINID_NUM_CLASSES, dtype=np.uint8)
        spec = SemanticSpec(mode=mode, num_classes=num_classes, label_to_trainid=lut)
    else:
        raise ValueError(
            f"Unknown semantic mode {semantic_mode!r}; use auto, cluster, "
            "cause27, or trainid."
        )

    if max_label >= spec.num_classes:
        raise ValueError(
            f"Semantic labels in {semantic_subdir}/{split} reach {max_label}, "
            f"but resolved mode {spec.mode!r} has only {spec.num_classes} "
            "channels. This would silently drop labels."
        )
    return spec


def encode_semantic_onehot(
    label_patch: np.ndarray,
    spec: SemanticSpec,
    smooth: float = 0.1,
) -> np.ndarray:
    """Encode semantic labels as smoothed one-hot channels."""
    onehot = np.full(
        (spec.num_classes, label_patch.shape[0], label_patch.shape[1]),
        smooth / spec.num_classes,
        dtype=np.float32,
    )
    valid = label_patch < spec.num_classes
    if valid.any():
        yy, xx = np.nonzero(valid)
        cls = label_patch[yy, xx].astype(np.int64)
        onehot[cls, yy, xx] = 1.0 - smooth + smooth / spec.num_classes
    return onehot


def map_to_trainid(label_map: np.ndarray, spec: SemanticSpec) -> np.ndarray:
    """Map semantic labels to Cityscapes trainIDs, using 255 for unknown."""
    label_uint = label_map.astype(np.int64, copy=False)
    out = np.full(label_uint.shape, 255, dtype=np.uint8)
    valid = (label_uint >= 0) & (label_uint < len(spec.label_to_trainid))
    out[valid] = spec.label_to_trainid[label_uint[valid]]
    return out


def validate_semantic_inputs(
    cityscapes_root: str | Path,
    semantic_subdir: str,
    depth_subdir: str,
    spec: SemanticSpec,
    split: str,
    max_samples: int = 32,
) -> dict:
    """Validate semantic/depth wiring and return summary stats."""
    root = Path(cityscapes_root)
    entries = _sample_entries(root, split, max_samples)
    if not entries:
        raise ValueError(f"No Cityscapes image entries found for split {split}")

    total = 0
    unknown_channel = 0
    valid_trainid = 0
    thing_trainid = 0
    label_min = 255
    label_max = 0
    missing_depth = []
    missing_semantic = []

    for city, stem in entries:
        sem_path = root / semantic_subdir / split / city / f"{stem}.png"
        depth_path = root / depth_subdir / split / city / f"{stem}.npy"
        if not sem_path.exists():
            missing_semantic.append(str(sem_path))
            continue
        if not depth_path.exists():
            missing_depth.append(str(depth_path))

        labels = np.array(Image.open(sem_path))
        mapped = map_to_trainid(labels, spec)
        total += labels.size
        label_min = min(label_min, int(labels.min()))
        label_max = max(label_max, int(labels.max()))
        unknown_channel += int((labels >= spec.num_classes).sum())
        valid_trainid += int((mapped < TRAINID_NUM_CLASSES).sum())
        thing_trainid += int(((mapped >= 11) & (mapped < 19)).sum())

    if missing_semantic:
        raise FileNotFoundError(
            f"Missing semantic PNGs for {split}; first missing: "
            f"{missing_semantic[0]}"
        )
    if missing_depth:
        raise FileNotFoundError(
            f"Missing depth maps for {split}/{depth_subdir}; first missing: "
            f"{missing_depth[0]}"
        )
    if total == 0:
        raise ValueError(f"No readable semantic labels found for split {split}")
    if unknown_channel:
        raise ValueError(
            f"{unknown_channel / total:.2%} of sampled {split} semantic pixels "
            f"are >= semantic_dim={spec.num_classes}. Refusing to train with "
            "dropped labels."
        )

    return {
        "split": split,
        "samples": len(entries),
        "label_min": int(label_min),
        "label_max": int(label_max),
        "valid_trainid_frac": valid_trainid / total,
        "thing_trainid_frac": thing_trainid / total,
    }


def _sample_max_label(root: Path, semantic_subdir: str, split: str) -> int:
    for city, stem in _sample_entries(root, split, max_samples=16):
        sem_path = root / semantic_subdir / split / city / f"{stem}.png"
        if sem_path.exists():
            return int(np.array(Image.open(sem_path)).max())
    raise FileNotFoundError(
        f"Could not find semantic PNGs under {root / semantic_subdir / split}"
    )


def _sample_entries(root: Path, split: str, max_samples: int) -> list[tuple[str, str]]:
    img_dir = root / "leftImg8bit" / split
    entries: list[tuple[str, str]] = []
    if not img_dir.exists():
        return entries
    for city_path in sorted(p for p in img_dir.iterdir() if p.is_dir()):
        for img_path in sorted(city_path.glob("*_leftImg8bit.png")):
            stem = img_path.name.replace("_leftImg8bit.png", "")
            entries.append((city_path.name, stem))
            if len(entries) >= max_samples:
                return entries
    return entries
