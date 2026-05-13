"""BoxTeacher-style pseudo-instance quality scoring.

BoxTeacher scores pseudo masks as sqrt(det_confidence * mean_positive_mask_prob).
Our depth-CC pseudo-labels are hard masks without detector logits, so this script
uses a label-free analogue:

  confidence        := semantic purity inside the instance mask
  mask probability  := depth-interior certainty, 1 - clipped(depth_grad / tau_d)

The resulting score is not BoxTeacher's learned score; it is a BoxTeacher-inspired
mask-aware quality proxy for comparing pseudo-instance sets without GT labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _stem_from_instance(path: Path) -> str:
    return path.name.replace("_instance.png", "")


def _city_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def _depth_path(depth_root: Path, split: str, stem: str) -> Path:
    city = _city_from_stem(stem)
    base = stem.replace("_leftImg8bit", "")
    candidates = [
        depth_root / split / city / f"{base}.npy",
        depth_root / split / city / f"{stem}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No depth map found for {stem} under {depth_root}")


def _resize_like(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _depth_grad(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.all():
        fill = float(np.nanmedian(depth[finite])) if finite.any() else 0.0
        depth = np.where(finite, depth, fill)
    lo, hi = np.percentile(depth, [1, 99])
    if hi > lo:
        depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    depth = depth.astype(np.float32, copy=False)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    return (m > 0) & (eroded == 0)


def score_image(
    instance_path: Path,
    semantic_path: Path,
    depth_root: Path,
    split: str,
    tau_d: float,
) -> list[dict]:
    stem = _stem_from_instance(instance_path)
    inst = np.array(Image.open(instance_path))
    sem = np.array(Image.open(semantic_path))
    depth = np.load(_depth_path(depth_root, split, stem))
    depth = _resize_like(depth, inst.shape)
    grad = _depth_grad(depth)
    depth_prob = 1.0 - np.clip(grad / max(tau_d, 1e-6), 0.0, 1.0)

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
        majority_semantic = int(sem_vals[mode_idx])

        positive_prob = depth_prob[mask]
        mean_positive_prob = float(positive_prob.mean()) if positive_prob.size else 0.0
        mask_aware_score = float(np.sqrt(max(semantic_purity * mean_positive_prob, 0.0)))

        bnd = _boundary(mask)
        boundary_support = float(np.clip(grad[bnd] / max(tau_d, 1e-6), 0.0, 1.0).mean()) if bnd.any() else 0.0
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
                "majority_semantic": majority_semantic,
                "semantic_purity": semantic_purity,
                "mean_depth_interior_prob": mean_positive_prob,
                "boxteacher_style_score": mask_aware_score,
                "noise_score": float(1.0 - mask_aware_score),
                "boundary_support": boundary_support,
                "depth_mad_norm": depth_mad_norm,
            }
        )
    return rows


def summarize(rows: list[dict], tau_d: float, a_min: int) -> dict:
    if not rows:
        return {"tau_d": tau_d, "A_min": a_min, "n_instances": 0}

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=np.float64)

    scores = arr("boxteacher_style_score")
    noise = arr("noise_score")
    areas = arr("area")
    images = sorted({r["image"] for r in rows})
    return {
        "tau_d": tau_d,
        "A_min": a_min,
        "n_images": len(images),
        "n_instances": len(rows),
        "instances_per_image": len(rows) / max(len(images), 1),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_p10": float(np.percentile(scores, 10)),
        "score_p25": float(np.percentile(scores, 25)),
        "score_p75": float(np.percentile(scores, 75)),
        "score_p90": float(np.percentile(scores, 90)),
        "noise_mean": float(noise.mean()),
        "noise_median": float(np.median(noise)),
        "low_quality_score_lt_0_50_pct": float((scores < 0.50).mean() * 100.0),
        "low_quality_score_lt_0_60_pct": float((scores < 0.60).mean() * 100.0),
        "small_area_lt_Amin_pct": float((areas < a_min).mean() * 100.0),
        "area_median": float(np.median(areas)),
        "area_p10": float(np.percentile(areas, 10)),
        "semantic_purity_mean": float(arr("semantic_purity").mean()),
        "mean_depth_interior_prob_mean": float(arr("mean_depth_interior_prob").mean()),
        "boundary_support_mean": float(arr("boundary_support").mean()),
        "depth_mad_norm_mean": float(arr("depth_mad_norm").mean()),
        "worst_20": sorted(rows, key=lambda r: r["boxteacher_style_score"])[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_dir", type=Path, required=True)
    parser.add_argument("--depth_root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tau_d", type=float, default=0.20)
    parser.add_argument("--A_min", type=int, default=1000)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    instance_paths = sorted(args.pseudo_dir.glob("*_instance.png"))
    if args.max_images > 0:
        instance_paths = instance_paths[: args.max_images]

    all_rows: list[dict] = []
    missing = []
    for instance_path in instance_paths:
        stem = _stem_from_instance(instance_path)
        semantic_path = args.pseudo_dir / f"{stem}_semantic.png"
        if not semantic_path.exists():
            missing.append(str(semantic_path))
            continue
        try:
            all_rows.extend(
                score_image(
                    instance_path=instance_path,
                    semantic_path=semantic_path,
                    depth_root=args.depth_root,
                    split=args.split,
                    tau_d=args.tau_d,
                )
            )
        except FileNotFoundError as exc:
            missing.append(str(exc))

    result = summarize(all_rows, tau_d=args.tau_d, a_min=args.A_min)
    result["pseudo_dir"] = str(args.pseudo_dir)
    result["depth_root"] = str(args.depth_root)
    result["split"] = args.split
    result["scoring_note"] = (
        "BoxTeacher-inspired proxy: sqrt(semantic_purity * mean_depth_interior_prob). "
        "This adapts BoxTeacher's sqrt(det_confidence * mean_positive_mask_prob) to "
        "hard depth-CC masks without detector logits."
    )
    result["missing_count"] = len(missing)
    result["missing_examples"] = missing[:10]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
