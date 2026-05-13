"""Train-only boundary-evidence noise sweep for depth pseudo-instances.

This audit is designed to be closer to instance-training noise than raw
instance counts. For adjacent same-class thing patches, it compares the
pseudo-instance boundary target against independent boundary evidence from
frozen DINO features, RGB contrast, and depth continuity.

Noise definitions:
  - Unsupported split: pseudo boundary exists, but fewer than two evidence
    cues support an object boundary.
  - Missed boundary: no pseudo boundary within one pseudo-instance, but at
    least two evidence cues indicate a boundary.

No ground-truth annotations are read by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbps_pytorch.convert_to_cups_format import (  # noqa: E402
    build_instance_map_depth_cc,
    determine_thing_cluster_ids,
)


THING_TRAIN_IDS = set(range(11, 19))


def parse_floats(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def progress_line(label: str, index: int, total: int, start_time: float) -> str:
    elapsed = time.monotonic() - start_time
    pct = 100.0 * index / max(total, 1)
    rate = index / elapsed if elapsed > 0 else 0.0
    eta = (total - index) / rate if rate > 0 else 0.0
    return (
        f"{label} {index}/{total} ({pct:.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def semantic_files(semantic_root: Path, split: str) -> list[Path]:
    files: list[Path] = []
    for city_dir in sorted((semantic_root / split).iterdir()):
        if city_dir.is_dir():
            files.extend(sorted(city_dir.glob("*.png")))
    return files


def infer_feature_grid(n_tokens: int) -> tuple[int, int]:
    h = int(round((n_tokens / 2) ** 0.5))
    w = int(round(n_tokens / h))
    if h * w != n_tokens:
        raise ValueError(f"Cannot infer 2:1 feature grid for {n_tokens} tokens")
    return h, w


def load_cluster_to_class(centroids_path: Path) -> np.ndarray:
    data = np.load(centroids_path)
    raw = data["cluster_to_class"].astype(np.uint8)
    lut = np.full(256, 255, dtype=np.uint8)
    lut[: len(raw)] = raw
    return lut


def depth_path(depth_root: Path, split: str, city: str, stem: str) -> Path:
    candidates = [
        depth_root / split / city / f"{stem}.npy",
        depth_root / split / city / f"{stem}_leftImg8bit.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No depth map found for {split}/{city}/{stem}")


def feature_path(feature_root: Path, split: str, city: str, stem: str) -> Path:
    candidates = [
        feature_root / split / city / f"{stem}_leftImg8bit.npy",
        feature_root / split / city / f"{stem}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No feature map found for {split}/{city}/{stem}")


def image_path(image_root: Path, split: str, city: str, stem: str) -> Path:
    candidates = [
        image_root / split / city / f"{stem}_leftImg8bit.png",
        image_root / split / city / f"{stem}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No RGB image found for {split}/{city}/{stem}")


def resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_linear(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape:
        return arr.astype(np.float32)
    return cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def normalize_features(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    return features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)


def pair_arrays(arr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if arr.ndim == 3:
        return [
            (arr[:, :-1, :].reshape(-1, arr.shape[-1]), arr[:, 1:, :].reshape(-1, arr.shape[-1])),
            (arr[:-1, :, :].reshape(-1, arr.shape[-1]), arr[1:, :, :].reshape(-1, arr.shape[-1])),
        ]
    return [
        (arr[:, :-1].reshape(-1), arr[:, 1:].reshape(-1)),
        (arr[:-1, :].reshape(-1), arr[1:, :].reshape(-1)),
    ]


def feature_pair_arrays(feat: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (feat[:, :-1, :].reshape(-1, feat.shape[-1]), feat[:, 1:, :].reshape(-1, feat.shape[-1])),
        (feat[:-1, :, :].reshape(-1, feat.shape[-1]), feat[1:, :, :].reshape(-1, feat.shape[-1])),
    ]


def robust_threshold(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.percentile(values, percentile))


def evidence_cache_for_image(
    sem: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    features: np.ndarray,
    cluster_to_class: np.ndarray,
    evidence_percentile: float,
    min_pairs_for_percentile: int,
) -> dict:
    feat_h, feat_w = infer_feature_grid(features.shape[0])
    feat_grid = normalize_features(features).reshape(feat_h, feat_w, -1)
    sem_s = resize_nearest(sem, (feat_h, feat_w))
    cls_s = cluster_to_class[sem_s].astype(np.int16)
    depth_s = resize_linear(depth, (feat_h, feat_w))
    rgb_s = resize_linear(rgb, (feat_h, feat_w * 2)) if False else resize_linear(rgb, (feat_h, feat_w))
    if rgb_s.ndim == 2:
        rgb_s = rgb_s[..., None]
    rgb_s = rgb_s.astype(np.float32) / 255.0

    entries = []
    all_dino, all_depth, all_rgb = [], [], []
    for (ia, ib), (ca, cb), (da, db), (ra, rb), (fa, fb) in zip(
        pair_arrays(np.zeros((feat_h, feat_w), dtype=np.uint8)),
        pair_arrays(cls_s),
        pair_arrays(depth_s),
        pair_arrays(rgb_s),
        feature_pair_arrays(feat_grid),
    ):
        # ia/ib are placeholders only to keep orientation counts aligned.
        del ia, ib
        same_thing_class = (ca == cb) & np.isin(ca, list(THING_TRAIN_IDS))
        dino_delta = 1.0 - np.einsum("ij,ij->i", fa, fb)
        depth_delta = np.abs(da - db)
        rgb_delta = np.linalg.norm(ra - rb, axis=1)
        entries.append(
            {
                "same_thing_class": same_thing_class,
                "dino_delta": dino_delta,
                "depth_delta": depth_delta,
                "rgb_delta": rgb_delta,
            }
        )
        valid = same_thing_class
        if valid.any():
            all_dino.append(dino_delta[valid])
            all_depth.append(depth_delta[valid])
            all_rgb.append(rgb_delta[valid])

    dino_vals = np.concatenate(all_dino) if all_dino else np.array([], dtype=np.float32)
    depth_vals = np.concatenate(all_depth) if all_depth else np.array([], dtype=np.float32)
    rgb_vals = np.concatenate(all_rgb) if all_rgb else np.array([], dtype=np.float32)
    if len(dino_vals) < min_pairs_for_percentile:
        # This image has too little same-class thing adjacency to be useful.
        return {"skip": True}

    return {
        "skip": False,
        "feat_shape": (feat_h, feat_w),
        "entries": entries,
        "thresholds": {
            "dino_delta": robust_threshold(dino_vals, evidence_percentile),
            "depth_delta": robust_threshold(depth_vals, evidence_percentile),
            "rgb_delta": robust_threshold(rgb_vals, evidence_percentile),
        },
    }


def score_instance_boundaries(inst: np.ndarray, cache: dict, evidence_votes: int) -> dict:
    feat_h, feat_w = cache["feat_shape"]
    inst_s = resize_nearest(inst, (feat_h, feat_w)).astype(np.int64)
    thresholds = cache["thresholds"]
    stats = {
        "eligible_pairs": 0,
        "pseudo_boundaries": 0,
        "consensus_boundaries": 0,
        "unsupported_split_pairs": 0,
        "missed_boundary_pairs": 0,
        "instances": int(len(np.unique(inst[inst > 0]))),
    }

    for entry, (inst_a, inst_b) in zip(cache["entries"], pair_arrays(inst_s)):
        valid = entry["same_thing_class"] & (inst_a > 0) & (inst_b > 0)
        if not valid.any():
            continue
        pseudo_boundary = valid & (inst_a != inst_b)
        same_instance = valid & (inst_a == inst_b)
        votes = (
            (entry["dino_delta"] >= thresholds["dino_delta"]).astype(np.int16)
            + (entry["depth_delta"] >= thresholds["depth_delta"]).astype(np.int16)
            + (entry["rgb_delta"] >= thresholds["rgb_delta"]).astype(np.int16)
        )
        consensus_boundary = valid & (votes >= evidence_votes)

        unsupported_split = pseudo_boundary & (~consensus_boundary)
        missed_boundary = same_instance & consensus_boundary

        stats["eligible_pairs"] += int(valid.sum())
        stats["pseudo_boundaries"] += int(pseudo_boundary.sum())
        stats["consensus_boundaries"] += int(consensus_boundary.sum())
        stats["unsupported_split_pairs"] += int(unsupported_split.sum())
        stats["missed_boundary_pairs"] += int(missed_boundary.sum())
    return stats


def add_stats(dst: dict, src: dict) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def summarize(total: dict, n_images: int, tau: float, a_min: int) -> dict:
    pseudo = max(int(total.get("pseudo_boundaries", 0)), 1)
    consensus = max(int(total.get("consensus_boundaries", 0)), 1)
    eligible = max(int(total.get("eligible_pairs", 0)), 1)
    unsupported = int(total.get("unsupported_split_pairs", 0))
    missed = int(total.get("missed_boundary_pairs", 0))
    noise = unsupported + missed
    return {
        "tau": tau,
        "A_min": a_min,
        "n_images": n_images,
        "instances_per_image": total.get("instances", 0) / max(n_images, 1),
        "eligible_pairs": int(total.get("eligible_pairs", 0)),
        "pseudo_boundaries": int(total.get("pseudo_boundaries", 0)),
        "consensus_boundaries": int(total.get("consensus_boundaries", 0)),
        "unsupported_split_pairs": unsupported,
        "unsupported_split_rate_pct": 100.0 * unsupported / pseudo,
        "unsupported_split_pairs_per_image": unsupported / max(n_images, 1),
        "missed_boundary_pairs": missed,
        "missed_boundary_rate_pct": 100.0 * missed / consensus,
        "missed_boundary_pairs_per_image": missed / max(n_images, 1),
        "boundary_noise_pairs": noise,
        "boundary_noise_pairs_per_image": noise / max(n_images, 1),
        "boundary_noise_rate_pct": 100.0 * noise / eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_root", type=Path, required=True)
    parser.add_argument("--depth_root", type=Path, required=True)
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--centroids_path", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tau_values", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--A_min_values", default="500,1000,2000")
    parser.add_argument("--depth_blur_sigma", type=float, default=0.0)
    parser.add_argument("--dilation_iters", type=int, default=3)
    parser.add_argument("--evidence_percentile", type=float, default=75.0)
    parser.add_argument("--evidence_votes", type=int, default=2)
    parser.add_argument("--min_pairs_for_percentile", type=int, default=20)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--config_progress_every", type=int, default=0)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    taus = parse_floats(args.tau_values)
    a_mins = parse_ints(args.A_min_values)
    configs = [(tau, a_min) for tau in taus for a_min in a_mins]

    files = semantic_files(args.semantic_root, args.split)
    if args.max_images > 0:
        if args.sample_seed:
            rng = random.Random(args.sample_seed)
            rng.shuffle(files)
        files = files[: args.max_images]

    thing_ids = determine_thing_cluster_ids(args.centroids_path)
    cluster_to_class = load_cluster_to_class(args.centroids_path)
    totals = {cfg: {} for cfg in configs}
    missing: list[str] = []
    skipped_low_pair = 0
    processed = 0
    start_time = time.monotonic()

    for index, sem_path in enumerate(files, start=1):
        city = sem_path.parent.name
        stem = sem_path.stem
        should_log = (
            args.progress_every > 0
            and (index == 1 or index % args.progress_every == 0 or index == len(files))
        )
        if should_log:
            print(
                progress_line("start", index, len(files), start_time)
                + f" file={city}/{stem} usable={processed} configs={len(configs)}",
                flush=True,
            )
        try:
            depth = np.load(depth_path(args.depth_root, args.split, city, stem))
            features = np.load(feature_path(args.feature_root, args.split, city, stem))
            rgb = np.array(Image.open(image_path(args.image_root, args.split, city, stem)).convert("RGB"))
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue

        sem = np.array(Image.open(sem_path))
        depth_full = resize_linear(depth, sem.shape)
        cache = evidence_cache_for_image(
            sem=sem,
            depth=depth_full,
            rgb=rgb,
            features=features,
            cluster_to_class=cluster_to_class,
            evidence_percentile=args.evidence_percentile,
            min_pairs_for_percentile=args.min_pairs_for_percentile,
        )
        if cache.get("skip"):
            skipped_low_pair += 1
            if should_log:
                print(
                    progress_line("skip ", index, len(files), start_time)
                    + f" file={city}/{stem} skipped_low_pair={skipped_low_pair}",
                    flush=True,
                )
            continue

        for cfg_index, (tau, a_min) in enumerate(configs, start=1):
            if (
                should_log
                and args.config_progress_every > 0
                and (
                    cfg_index == 1
                    or cfg_index % args.config_progress_every == 0
                    or cfg_index == len(configs)
                )
            ):
                global_step = (index - 1) * len(configs) + cfg_index
                global_total = len(files) * len(configs)
                print(
                    progress_line("config", global_step, global_total, start_time)
                    + f" file={index}/{len(files)}:{city}/{stem}"
                    + f" cfg={cfg_index}/{len(configs)}"
                    + f" tau={tau:.2f} A_min={a_min}",
                    flush=True,
                )
            inst = build_instance_map_depth_cc(
                sem,
                depth_full,
                thing_ids,
                min_area=a_min,
                grad_threshold=tau,
                depth_blur_sigma=args.depth_blur_sigma,
                dilation_iters=args.dilation_iters,
            )
            stats = score_instance_boundaries(inst, cache, args.evidence_votes)
            add_stats(totals[(tau, a_min)], stats)

        processed += 1
        if should_log:
            print(
                progress_line("done ", index, len(files), start_time)
                + f" file={city}/{stem} usable={processed}",
                flush=True,
            )

    summaries = [
        summarize(totals[(tau, a_min)], processed, tau, a_min)
        for tau, a_min in configs
    ]
    result = {
        "semantic_root": str(args.semantic_root),
        "depth_root": str(args.depth_root),
        "feature_root": str(args.feature_root),
        "image_root": str(args.image_root),
        "centroids_path": str(args.centroids_path),
        "split": args.split,
        "processed_images": processed,
        "max_images": args.max_images,
        "sample_seed": args.sample_seed,
        "evidence_percentile": args.evidence_percentile,
        "evidence_votes": args.evidence_votes,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "skipped_low_pair": skipped_low_pair,
        "summaries": summaries,
        "note": (
            "Train-only label-free metric. Unsupported split = pseudo boundary "
            "without multi-cue RGB/DINO/depth support. Missed boundary = strong "
            "multi-cue evidence hidden inside one pseudo-instance."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(summaries[0].keys()) if summaries else []
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
