"""Label-free BoxTeacher-style audit over depth-splitting thresholds.

This script generates depth-CC instance maps on the fly for each
``(tau, A_min)`` pair and scores them with a fixed BoxTeacher-inspired
quality proxy. It does not consult ground-truth annotations.
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

from mbps_pytorch.convert_to_cups_format import (
    build_instance_map_depth_cc,
    determine_thing_cluster_ids,
)
from scripts.score_boxteacher_instance_quality import _bbox, _boundary, _depth_grad


def _parse_floats(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _resize_like(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def _progress_line(label: str, index: int, total: int, start_time: float) -> str:
    elapsed = time.monotonic() - start_time
    pct = 100.0 * index / max(total, 1)
    rate = index / elapsed if elapsed > 0 else 0.0
    eta = (total - index) / rate if rate > 0 else 0.0
    return (
        f"{label} {index}/{total} ({pct:.1f}%) "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
    )


def _depth_path(depth_root: Path, split: str, city: str, stem: str) -> Path:
    candidates = [
        depth_root / split / city / f"{stem}.npy",
        depth_root / split / city / f"{stem}_leftImg8bit.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No depth map found for {split}/{city}/{stem}")


def _semantic_files(semantic_root: Path, split: str) -> list[Path]:
    split_dir = semantic_root / split
    files: list[Path] = []
    for city_dir in sorted(split_dir.iterdir()):
        if city_dir.is_dir():
            files.extend(sorted(city_dir.glob("*.png")))
    return files


def score_instance_map(
    stem: str,
    inst: np.ndarray,
    sem: np.ndarray,
    depth: np.ndarray,
    score_tau: float,
) -> list[dict]:
    grad = _depth_grad(depth)
    depth_prob = 1.0 - np.clip(grad / max(score_tau, 1e-6), 0.0, 1.0)

    rows: list[dict] = []
    for iid in np.unique(inst):
        if int(iid) == 0:
            continue
        mask = inst == iid
        area = int(mask.sum())
        if area == 0:
            continue

        y0, x0, y1, x1 = _bbox(mask)
        box_area = max((y1 - y0) * (x1 - x0), 1)
        fill_ratio = area / box_area

        sem_vals, sem_counts = np.unique(sem[mask], return_counts=True)
        mode_idx = int(np.argmax(sem_counts))
        semantic_purity = float(sem_counts[mode_idx] / area)

        mean_positive_prob = float(depth_prob[mask].mean())
        mask_score = float(np.sqrt(max(semantic_purity * mean_positive_prob, 0.0)))

        bnd = _boundary(mask)
        boundary_support = (
            float(np.clip(grad[bnd] / max(score_tau, 1e-6), 0.0, 1.0).mean())
            if bnd.any()
            else 0.0
        )
        depth_vals = depth[mask]
        med = float(np.median(depth_vals))
        mad = float(np.median(np.abs(depth_vals - med)))
        depth_mad_norm = float(mad / (abs(med) + 1e-6))

        rows.append(
            {
                "image": stem,
                "instance_id": int(iid),
                "area": area,
                "bbox_area": int(box_area),
                "fill_ratio": float(fill_ratio),
                "semantic_purity": semantic_purity,
                "mean_depth_interior_prob": mean_positive_prob,
                "boxteacher_style_score": mask_score,
                "noise_score": float(1.0 - mask_score),
                "boundary_support": boundary_support,
                "depth_mad_norm": depth_mad_norm,
            }
        )
    return rows


def summarize(rows: list[dict], n_images: int, tau: float, a_min: int, score_tau: float) -> dict:
    if not rows:
        return {
            "tau": tau,
            "A_min": a_min,
            "score_tau": score_tau,
            "n_images": n_images,
            "n_instances": 0,
            "instances_per_image": 0.0,
        }

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=np.float64)

    scores = arr("boxteacher_style_score")
    areas = arr("area")
    fill = arr("fill_ratio")
    depth_mad = arr("depth_mad_norm")
    boundary = arr("boundary_support")
    return {
        "tau": tau,
        "A_min": a_min,
        "score_tau": score_tau,
        "n_images": n_images,
        "n_instances": len(rows),
        "instances_per_image": len(rows) / max(n_images, 1),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_p05": float(np.percentile(scores, 5)),
        "score_p10": float(np.percentile(scores, 10)),
        "score_p25": float(np.percentile(scores, 25)),
        "score_lt_0_60_pct": float((scores < 0.60).mean() * 100.0),
        "score_lt_0_70_pct": float((scores < 0.70).mean() * 100.0),
        "score_lt_0_80_pct": float((scores < 0.80).mean() * 100.0),
        "score_lt_0_90_pct": float((scores < 0.90).mean() * 100.0),
        "area_median": float(np.median(areas)),
        "area_p10": float(np.percentile(areas, 10)),
        "fill_ratio_mean": float(fill.mean()),
        "fill_ratio_p10": float(np.percentile(fill, 10)),
        "fill_ratio_lt_0_10_pct": float((fill < 0.10).mean() * 100.0),
        "depth_mad_norm_mean": float(depth_mad.mean()),
        "depth_mad_norm_p90": float(np.percentile(depth_mad, 90)),
        "boundary_support_mean": float(boundary.mean()),
        "worst_10": sorted(rows, key=lambda r: r["boxteacher_style_score"])[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_root", type=Path, required=True)
    parser.add_argument("--depth_root", type=Path, required=True)
    parser.add_argument("--centroids_path", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tau_values", default="0.05,0.10,0.15,0.20,0.30,0.50")
    parser.add_argument("--A_min_values", default="250,500,1000,2000")
    parser.add_argument("--score_tau", type=float, default=0.20)
    parser.add_argument("--depth_blur_sigma", type=float, default=0.0)
    parser.add_argument("--dilation_iters", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--config_progress_every", type=int, default=0)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    taus = _parse_floats(args.tau_values)
    a_mins = _parse_ints(args.A_min_values)
    configs = [(tau, a_min) for tau in taus for a_min in a_mins]
    files = _semantic_files(args.semantic_root, args.split)
    if args.max_images > 0:
        if args.sample_seed:
            rng = random.Random(args.sample_seed)
            rng.shuffle(files)
        files = files[: args.max_images]

    thing_ids = determine_thing_cluster_ids(args.centroids_path)
    rows_by_config: dict[tuple[float, int], list[dict]] = {cfg: [] for cfg in configs}
    missing: list[str] = []
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
                _progress_line("start", index, len(files), start_time)
                + f" image={city}/{stem} configs={len(configs)}",
                flush=True,
            )
        sem = np.array(Image.open(sem_path))
        try:
            depth = np.load(_depth_path(args.depth_root, args.split, city, stem))
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        depth = _resize_like(depth, sem.shape)

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
                    _progress_line("config", global_step, global_total, start_time)
                    + f" image={index}/{len(files)}:{city}/{stem}"
                    + f" cfg={cfg_index}/{len(configs)}"
                    + f" tau={tau:.2f} A_min={a_min}",
                    flush=True,
                )
            inst = build_instance_map_depth_cc(
                sem,
                depth,
                thing_ids,
                min_area=a_min,
                grad_threshold=tau,
                depth_blur_sigma=args.depth_blur_sigma,
                dilation_iters=args.dilation_iters,
            )
            rows_by_config[(tau, a_min)].extend(
                score_instance_map(stem, inst, sem, depth, score_tau=args.score_tau)
            )
        if should_log:
            total_instances = sum(len(rows) for rows in rows_by_config.values())
            print(
                _progress_line("done ", index, len(files), start_time)
                + f" image={city}/{stem} accumulated_instances={total_instances}",
                flush=True,
            )

    n_images = len(files) - len(missing)
    summaries = [
        summarize(rows_by_config[(tau, a_min)], n_images, tau, a_min, args.score_tau)
        for tau, a_min in configs
    ]
    result = {
        "semantic_root": str(args.semantic_root),
        "depth_root": str(args.depth_root),
        "centroids_path": str(args.centroids_path),
        "split": args.split,
        "score_tau": args.score_tau,
        "depth_blur_sigma": args.depth_blur_sigma,
        "dilation_iters": args.dilation_iters,
        "max_images": args.max_images,
        "sample_seed": args.sample_seed,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "summaries": summaries,
        "note": (
            "Generated depth-CC masks for each tau/A_min pair. Scores are "
            "BoxTeacher-inspired label-free proxies and use a fixed score_tau "
            "so generation thresholds are compared on the same confidence scale."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [k for k in summaries[0].keys() if k != "worst_10"]
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                writer.writerow({k: summary.get(k) for k in fieldnames})

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
