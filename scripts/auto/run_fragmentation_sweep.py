#!/usr/bin/env python3
"""Fragmentation sweep: evaluate panoptic PQ across tau and depth models.

Runs evaluate_panoptic_combined.py sequentially for each configuration.
Results saved to results/auto_fragmentation/sweep_results.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_config(sem_dir: Path, depth_subdir: str, tau: float, output_json: Path,
               cityscapes_root: Path, min_area: int = 1000, sigma: float = 0.0, dilation: int = 3) -> dict:
    """Run one fragmentation configuration."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent.parent / "mbps_pytorch" / "evaluate_panoptic_combined.py"),
        "--sem_dir", str(sem_dir),
        "--depth_subdir", depth_subdir,
        "--cityscapes_root", str(cityscapes_root),
        "--tau", str(tau),
        "--min_area", str(min_area),
        "--sigma", str(sigma),
        "--dilation", str(dilation),
        "--output", str(output_json),
    ]
    print(f"RUN: tau={tau}, depth={depth_subdir}, sigma={sigma}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode == 0 and output_json.exists():
            with open(output_json) as f:
                data = json.load(f)
            print(f"  OK: PQ={data.get('PQ', 'N/A')}, PQ_th={data.get('PQ_things', 'N/A')}")
            return {"status": "ok", "data": data}
        else:
            print(f"  FAIL: rc={result.returncode}")
            if result.stderr:
                print(f"  STDERR: {result.stderr[:300]}")
            return {"status": "fail", "stderr": result.stderr[:500]}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=Path, default=Path("/Users/qbit-glitch/Desktop/datasets/cityscapes"))
    parser.add_argument("--sem_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("results/auto_fragmentation"))
    parser.add_argument("--depth_models", default="depth_depthpro,depth_dav3")
    parser.add_argument("--tau_values", default="0.05,0.10,0.15,0.20,0.30,0.50")
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--dilation", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Default semantic dir: V3 adapter k=80
    sem_dir = args.sem_dir or args.cityscapes_root / "pseudo_semantic_adapter_V3_k80" / "val"

    depth_models = [d.strip() for d in args.depth_models.split(",")]
    tau_values = [float(t.strip()) for t in args.tau_values.split(",")]

    results = []
    for depth in depth_models:
        for tau in tau_values:
            config_id = f"{depth.replace('depth_', '')}_tau{tau:.2f}"
            output_json = args.output_dir / f"{config_id}.json"

            if output_json.exists():
                print(f"SKIP {config_id}: already evaluated")
                with open(output_json) as f:
                    data = json.load(f)
                results.append({"config": config_id, "depth": depth, "tau": tau, "status": "cached", "data": data})
                continue

            res = run_config(sem_dir, depth, tau, output_json, args.cityscapes_root,
                            args.min_area, args.sigma, args.dilation)
            results.append({"config": config_id, "depth": depth, "tau": tau, **res})
            time.sleep(2)

    # Save aggregate
    summary = args.output_dir / "sweep_results.json"
    with open(summary, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r.get("status") in ("ok", "cached"))
    print(f"\nDone: {ok}/{len(results)} configurations evaluated")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
