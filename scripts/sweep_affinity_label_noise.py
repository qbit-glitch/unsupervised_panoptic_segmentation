"""Label-free affinity-noise audit for depth-guided pseudo-instances.

The metric here is tied to the supervision an instance learner receives:
adjacent patch pairs inside the same pseudo-instance are positive affinity
targets, while adjacent same-class patch pairs across pseudo-instances are
negative affinity targets. We count contradictions between those pseudo
targets and frozen evidence from DINO features plus depth continuity.

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
    # Cityscapes feature grids preserve the 2:1 image aspect ratio.
    h = int(round((n_tokens / 2) ** 0.5))
    w = int(round(n_tokens / h))
    if h * w != n_tokens:
        side = int(round(n_tokens**0.5))
        if side * side == n_tokens:
            return side, side
        raise ValueError(f"Cannot infer feature grid for {n_tokens} tokens")
    return h, w


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
    raise FileNotFoundError(f"No features found for {split}/{city}/{stem}")


def resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_linear(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape:
        return arr
    return cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def depth_gradient(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.all():
        fill = float(np.nanmedian(depth[finite])) if finite.any() else 0.0
        depth = np.where(finite, depth, fill)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def load_cluster_to_class(centroids_path: Path) -> np.ndarray:
    data = np.load(centroids_path)
    raw = data["cluster_to_class"].astype(np.uint8)
    lut = np.full(256, 255, dtype=np.uint8)
    lut[: len(raw)] = raw
    return lut


def normalize_features(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    norms = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
    return features / norms


def patch_pair_stats(
    sem: np.ndarray,
    inst: np.ndarray,
    depth: np.ndarray,
    features: np.ndarray,
    cluster_to_class: np.ndarray,
    split_sim_threshold: float,
    split_depth_delta: float,
    weak_boundary_threshold: float,
    merge_low_sim_threshold: float,
    merge_depth_delta: float,
    strong_boundary_threshold: float,
    min_same_votes: int,
    min_diff_votes: int,
) -> dict:
    feat_h, feat_w = infer_feature_grid(features.shape[0])
    feat_grid = normalize_features(features).reshape(feat_h, feat_w, -1)
    inst_s = resize_nearest(inst, (feat_h, feat_w)).astype(np.int64)
    sem_s = resize_nearest(sem, (feat_h, feat_w))
    cls_s = cluster_to_class[sem_s].astype(np.int16)
    depth_s = resize_linear(depth, (feat_h, feat_w))
    grad_s = depth_gradient(depth_s)

    stats = {
        "instances": int(len(np.unique(inst[inst > 0]))),
        "positive_pairs": 0,
        "under_merge_noise_pairs": 0,
        "negative_same_class_pairs": 0,
        "over_split_noise_pairs": 0,
        "positive_sim_sum": 0.0,
        "positive_depth_delta_sum": 0.0,
        "negative_sim_sum": 0.0,
        "negative_depth_delta_sum": 0.0,
    }

    pair_slices = [
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ]
    for a_sl, b_sl in pair_slices:
        inst_a = inst_s[a_sl].reshape(-1)
        inst_b = inst_s[b_sl].reshape(-1)
        valid = (inst_a > 0) & (inst_b > 0)
        if not valid.any():
            continue

        cls_a = cls_s[a_sl].reshape(-1)
        cls_b = cls_s[b_sl].reshape(-1)
        same_class = (cls_a == cls_b) & (cls_a != 255)
        same_inst = inst_a == inst_b

        feat_a = feat_grid[a_sl].reshape(-1, feat_grid.shape[-1])
        feat_b = feat_grid[b_sl].reshape(-1, feat_grid.shape[-1])
        sim = np.einsum("ij,ij->i", feat_a, feat_b)
        depth_delta = np.abs(depth_s[a_sl].reshape(-1) - depth_s[b_sl].reshape(-1))
        boundary_strength = 0.5 * (grad_s[a_sl].reshape(-1) + grad_s[b_sl].reshape(-1))

        pos = valid & same_inst
        if pos.any():
            diff_votes = (
                (sim < merge_low_sim_threshold).astype(np.int16)
                + (depth_delta > merge_depth_delta).astype(np.int16)
                + (boundary_strength > strong_boundary_threshold).astype(np.int16)
            )
            noisy = pos & (diff_votes >= min_diff_votes)
            stats["positive_pairs"] += int(pos.sum())
            stats["under_merge_noise_pairs"] += int(noisy.sum())
            stats["positive_sim_sum"] += float(sim[pos].sum())
            stats["positive_depth_delta_sum"] += float(depth_delta[pos].sum())

        neg_same_class = valid & (~same_inst) & same_class
        if neg_same_class.any():
            same_votes = (
                (sim > split_sim_threshold).astype(np.int16)
                + (depth_delta < split_depth_delta).astype(np.int16)
                + (boundary_strength < weak_boundary_threshold).astype(np.int16)
            )
            noisy = neg_same_class & (same_votes >= min_same_votes)
            stats["negative_same_class_pairs"] += int(neg_same_class.sum())
            stats["over_split_noise_pairs"] += int(noisy.sum())
            stats["negative_sim_sum"] += float(sim[neg_same_class].sum())
            stats["negative_depth_delta_sum"] += float(depth_delta[neg_same_class].sum())

    return stats


def add_stats(dst: dict, src: dict) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def summarize(config_stats: dict, n_images: int, tau: float, a_min: int) -> dict:
    pos = max(int(config_stats.get("positive_pairs", 0)), 1)
    neg = max(int(config_stats.get("negative_same_class_pairs", 0)), 1)
    under = int(config_stats.get("under_merge_noise_pairs", 0))
    over = int(config_stats.get("over_split_noise_pairs", 0))
    total_pairs = pos + neg
    total_noise = under + over
    return {
        "tau": tau,
        "A_min": a_min,
        "n_images": n_images,
        "instances_per_image": config_stats.get("instances", 0) / max(n_images, 1),
        "positive_pairs": int(config_stats.get("positive_pairs", 0)),
        "negative_same_class_pairs": int(config_stats.get("negative_same_class_pairs", 0)),
        "over_split_noise_pairs": over,
        "over_split_noise_rate_pct": 100.0 * over / neg,
        "over_split_noise_pairs_per_image": over / max(n_images, 1),
        "under_merge_noise_pairs": under,
        "under_merge_noise_rate_pct": 100.0 * under / pos,
        "under_merge_noise_pairs_per_image": under / max(n_images, 1),
        "affinity_noise_rate_pct": 100.0 * total_noise / max(total_pairs, 1),
        "affinity_noise_pairs_per_image": total_noise / max(n_images, 1),
        "positive_feature_sim_mean": config_stats.get("positive_sim_sum", 0.0) / pos,
        "negative_same_class_feature_sim_mean": config_stats.get("negative_sim_sum", 0.0) / neg,
        "positive_depth_delta_mean": config_stats.get("positive_depth_delta_sum", 0.0) / pos,
        "negative_same_class_depth_delta_mean": config_stats.get("negative_depth_delta_sum", 0.0) / neg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_root", type=Path, required=True)
    parser.add_argument("--depth_root", type=Path, required=True)
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--centroids_path", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tau_values", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--A_min_values", default="500,1000,2000")
    parser.add_argument("--depth_blur_sigma", type=float, default=0.0)
    parser.add_argument("--dilation_iters", type=int, default=3)
    parser.add_argument("--split_sim_threshold", type=float, default=0.85)
    parser.add_argument("--split_depth_delta", type=float, default=0.05)
    parser.add_argument("--weak_boundary_threshold", type=float, default=0.20)
    parser.add_argument("--merge_low_sim_threshold", type=float, default=0.70)
    parser.add_argument("--merge_depth_delta", type=float, default=0.15)
    parser.add_argument("--strong_boundary_threshold", type=float, default=0.20)
    parser.add_argument("--min_same_votes", type=int, default=3)
    parser.add_argument("--min_diff_votes", type=int, default=2)
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
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue

        sem = np.array(Image.open(sem_path))
        if depth.shape != sem.shape:
            depth_full = resize_linear(depth, sem.shape)
        else:
            depth_full = depth.astype(np.float32)

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
            stats = patch_pair_stats(
                sem=sem,
                inst=inst,
                depth=depth_full,
                features=features,
                cluster_to_class=cluster_to_class,
                split_sim_threshold=args.split_sim_threshold,
                split_depth_delta=args.split_depth_delta,
                weak_boundary_threshold=args.weak_boundary_threshold,
                merge_low_sim_threshold=args.merge_low_sim_threshold,
                merge_depth_delta=args.merge_depth_delta,
                strong_boundary_threshold=args.strong_boundary_threshold,
                min_same_votes=args.min_same_votes,
                min_diff_votes=args.min_diff_votes,
            )
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
        "centroids_path": str(args.centroids_path),
        "split": args.split,
        "max_images": args.max_images,
        "sample_seed": args.sample_seed,
        "processed_images": processed,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "thresholds": {
            "split_sim_threshold": args.split_sim_threshold,
            "split_depth_delta": args.split_depth_delta,
            "weak_boundary_threshold": args.weak_boundary_threshold,
            "merge_low_sim_threshold": args.merge_low_sim_threshold,
            "merge_depth_delta": args.merge_depth_delta,
            "strong_boundary_threshold": args.strong_boundary_threshold,
            "min_same_votes": args.min_same_votes,
            "min_diff_votes": args.min_diff_votes,
        },
        "summaries": summaries,
        "note": (
            "Over-split noise: adjacent same-class negative pseudo-affinity "
            "pairs that DINO/depth/boundary evidence says should be positive. "
            "Under-merge noise: adjacent positive pseudo-affinity pairs that "
            "frozen evidence says should be separated. No GT labels are used."
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
