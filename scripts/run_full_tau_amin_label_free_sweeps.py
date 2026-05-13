"""Run full train-set label-free tau/A_min sweeps.

This is a thin orchestrator over the three audit scripts:
  1. BoxTeacher-style mask coherence / fragmentation audit
  2. Patch-affinity label-noise audit
  3. Boundary-evidence label-noise audit

It intentionally runs on the train split and does not pass any ground-truth
paths to the child scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TAUS = "0.01,0.02,0.05,0.10,0.20,0.30,0.40"
DEFAULT_A_MINS = "100,200,500,1000,1500,2000"


def run_step(name: str, cmd: list[str], output_json: Path, overwrite: bool) -> None:
    print("\n" + "=" * 88, flush=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {name}", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 88, flush=True)

    if output_json.exists() and not overwrite:
        print(f"[SKIP] {name}: output exists at {output_json}", flush=True)
        return

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    dt = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with return code {result.returncode}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE {name} in {dt/3600:.2f} h", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=Path, default=Path("/Users/qbit-glitch/Desktop/datasets/cityscapes"))
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_adapter_V3_k80")
    parser.add_argument("--depth_subdir", default="depth_depthpro")
    parser.add_argument("--feature_subdir", default="dinov3_features")
    parser.add_argument("--image_subdir", default="leftImg8bit")
    parser.add_argument("--centroids_subpath", default="pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tau_values", default=DEFAULT_TAUS)
    parser.add_argument("--A_min_values", default=DEFAULT_A_MINS)
    parser.add_argument("--out_dir", type=Path, default=Path("results/instance_quality/full_train_tau_amin_depthpro"))
    parser.add_argument("--audits", default="boxteacher,affinity,boundary")
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--config_progress_every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    semantic_root = args.cityscapes_root / args.semantic_subdir
    depth_root = args.cityscapes_root / args.depth_subdir
    feature_root = args.cityscapes_root / args.feature_subdir
    image_root = args.cityscapes_root / args.image_subdir
    centroids_path = args.cityscapes_root / args.centroids_subpath
    audits = {a.strip().lower() for a in args.audits.split(",") if a.strip()}

    tau_count = len([v for v in args.tau_values.split(",") if v.strip()])
    amin_count = len([v for v in args.A_min_values.split(",") if v.strip()])
    print("Full label-free sweep configuration", flush=True)
    print(f"  split:          {args.split}", flush=True)
    print(f"  semantic_root:  {semantic_root}", flush=True)
    print(f"  depth_root:     {depth_root}", flush=True)
    print(f"  feature_root:   {feature_root}", flush=True)
    print(f"  image_root:     {image_root}", flush=True)
    print(f"  centroids_path: {centroids_path}", flush=True)
    print(f"  tau_values:     {args.tau_values}", flush=True)
    print(f"  A_min_values:   {args.A_min_values}", flush=True)
    print(f"  grid:           {tau_count} x {amin_count} = {tau_count * amin_count} configs per audit", flush=True)
    print(f"  audits:         {','.join(sorted(audits))}", flush=True)
    print(f"  progress_every: {args.progress_every}", flush=True)
    print(f"  config_every:   {args.config_progress_every}", flush=True)
    print(f"  out_dir:        {out_dir}", flush=True)

    steps: list[tuple[str, list[str], Path]] = []
    if "boxteacher" in audits:
        output_json = out_dir / "boxteacher_depthpro_train_full.json"
        steps.append(
            (
                "boxteacher",
                [
                    sys.executable,
                    "-u",
                    "scripts/sweep_boxteacher_instance_quality.py",
                    "--semantic_root",
                    str(semantic_root),
                    "--depth_root",
                    str(depth_root),
                    "--centroids_path",
                    str(centroids_path),
                    "--split",
                    args.split,
                    "--tau_values",
                    args.tau_values,
                    "--A_min_values",
                    args.A_min_values,
                    "--progress_every",
                    str(args.progress_every),
                    "--config_progress_every",
                    str(args.config_progress_every),
                    "--output_json",
                    str(output_json),
                    "--output_csv",
                    str(out_dir / "boxteacher_depthpro_train_full.csv"),
                ],
                output_json,
            )
        )

    if "affinity" in audits:
        output_json = out_dir / "affinity_depthpro_dinov3_train_full.json"
        steps.append(
            (
                "affinity",
                [
                    sys.executable,
                    "-u",
                    "scripts/sweep_affinity_label_noise.py",
                    "--semantic_root",
                    str(semantic_root),
                    "--depth_root",
                    str(depth_root),
                    "--feature_root",
                    str(feature_root),
                    "--centroids_path",
                    str(centroids_path),
                    "--split",
                    args.split,
                    "--tau_values",
                    args.tau_values,
                    "--A_min_values",
                    args.A_min_values,
                    "--progress_every",
                    str(args.progress_every),
                    "--config_progress_every",
                    str(args.config_progress_every),
                    "--output_json",
                    str(output_json),
                    "--output_csv",
                    str(out_dir / "affinity_depthpro_dinov3_train_full.csv"),
                ],
                output_json,
            )
        )

    if "boundary" in audits:
        output_json = out_dir / "boundary_depthpro_dinov3_rgb_train_full.json"
        steps.append(
            (
                "boundary",
                [
                    sys.executable,
                    "-u",
                    "scripts/sweep_boundary_evidence_noise.py",
                    "--semantic_root",
                    str(semantic_root),
                    "--depth_root",
                    str(depth_root),
                    "--feature_root",
                    str(feature_root),
                    "--image_root",
                    str(image_root),
                    "--centroids_path",
                    str(centroids_path),
                    "--split",
                    args.split,
                    "--tau_values",
                    args.tau_values,
                    "--A_min_values",
                    args.A_min_values,
                    "--progress_every",
                    str(args.progress_every),
                    "--config_progress_every",
                    str(args.config_progress_every),
                    "--output_json",
                    str(output_json),
                    "--output_csv",
                    str(out_dir / "boundary_depthpro_dinov3_rgb_train_full.csv"),
                ],
                output_json,
            )
        )

    if not steps:
        raise SystemExit("No audits selected. Use --audits boxteacher,affinity,boundary")

    t0 = time.time()
    for name, cmd, output_json in steps:
        run_step(name, cmd, output_json, overwrite=args.overwrite)
    print("\nAll requested audits finished.", flush=True)
    print(f"Total elapsed: {(time.time() - t0) / 3600:.2f} h", flush=True)
    print(f"Outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
