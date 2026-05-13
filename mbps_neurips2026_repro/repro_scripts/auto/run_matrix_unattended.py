#!/usr/bin/env python3
"""Master orchestrator: runs the full composition matrix unattended.

Chains experiments sequentially with cooldown periods.
Each experiment has its own resume logic via state.json.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Define the composition matrix
MATRIX = [
    # Format: (semantic_method, instance_method, dcfa, simcf_steps)
    # CAUSE-TR + DepthPro
    ("cause_k80", "depthpro_tau020", False, []),
    ("cause_k80", "depthpro_tau020", True, []),
    ("cause_k80", "depthpro_tau020", False, ["A", "B", "C"]),
    ("cause_k80", "depthpro_tau020", True, ["A", "B", "C"]),
    # DepthG + DepthPro
    ("depthg_k80", "depthpro_tau020", False, []),
    ("depthg_k80", "depthpro_tau020", True, ["A", "B", "C"]),
    # DINOv3 raw + DepthPro
    ("dinov3_k80", "depthpro_tau020", False, []),
    ("dinov3_k80", "depthpro_tau020", True, ["A", "B", "C"]),
    # CAUSE-TR + DAv3
    ("cause_k80", "dav3", False, []),
    ("cause_k80", "dav3", True, ["A", "B", "C"]),
    # CAUSE-TR + CutS3D
    ("cause_k80", "cuts3d", False, []),
    ("cause_k80", "cuts3d", False, ["A", "B", "C"]),
]


def make_exp_id(semantic: str, instance: str, dcfa: bool, simcf_steps: list) -> str:
    parts = [semantic, instance]
    if dcfa:
        parts.append("dcfa")
    if simcf_steps:
        parts.append("simcf" + "".join(simcf_steps))
    return "_".join(parts)


def check_already_completed(output_dir: Path, exp_id: str) -> bool:
    """Check if this experiment was already completed."""
    state_file = output_dir / exp_id / "state.json"
    if not state_file.exists():
        return False
    try:
        with open(state_file) as f:
            state = json.load(f)
        return state.get("status") == "completed"
    except Exception:
        return False


def run_experiment(semantic: str, instance: str, dcfa: bool, simcf_steps: list,
                   cityscapes_root: Path, project_root: Path, output_root: Path) -> bool:
    """Run a single matrix cell."""
    exp_id = make_exp_id(semantic, instance, dcfa, simcf_steps)
    exp_output = output_root / exp_id

    if check_already_completed(output_root, exp_id):
        logger.info(f"SKIP {exp_id}: Already completed")
        return True

    logger.info("=" * 70)
    logger.info(f"START {exp_id}")
    logger.info("=" * 70)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_single_experiment.py"),
        "--semantic_method", semantic,
        "--instance_method", instance,
        "--cityscapes_root", str(cityscapes_root),
        "--project_root", str(project_root),
        "--output_dir", str(exp_output),
        "--resume",
    ]
    if dcfa:
        cmd.append("--dcfa")
    if simcf_steps:
        cmd.extend(["--simcf_steps", ",".join(simcf_steps)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"SUCCESS {exp_id}")
            return True
        else:
            logger.error(f"FAILED {exp_id}: returncode={result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr[:1000]}")
            return False
    except Exception as e:
        logger.error(f"EXCEPTION {exp_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run composition matrix unattended")
    parser.add_argument("--cityscapes_root", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, default=Path("logs/auto"))
    parser.add_argument("--cooldown", type=int, default=60, help="Seconds between experiments")
    parser.add_argument("--max_failures", type=int, default=3, help="Abort after N consecutive failures")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    fh = logging.FileHandler(args.log_dir / "matrix_master.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("MATRIX ORCHESTRATOR STARTING")
    logger.info(f"Total experiments: {len(MATRIX)}")
    logger.info(f"Output: {args.output_root}")
    logger.info("=" * 70)

    results = []
    consecutive_failures = 0

    for i, (semantic, instance, dcfa, simcf_steps) in enumerate(MATRIX, 1):
        exp_id = make_exp_id(semantic, instance, dcfa, simcf_steps)
        logger.info(f"[{i}/{len(MATRIX)}] Running {exp_id}")

        success = run_experiment(
            semantic, instance, dcfa, simcf_steps,
            args.cityscapes_root, args.project_root, args.output_root
        )

        results.append({"exp_id": exp_id, "success": success})

        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= args.max_failures:
                logger.error(f"ABORTING: {consecutive_failures} consecutive failures")
                break

        if i < len(MATRIX):
            logger.info(f"Cooldown: {args.cooldown}s")
            time.sleep(args.cooldown)

    # Save summary
    summary_file = args.output_root / "matrix_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total": len(MATRIX),
            "completed": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results,
        }, f, indent=2)

    logger.info("=" * 70)
    logger.info("MATRIX ORCHESTRATOR FINISHED")
    logger.info(f"Completed: {sum(1 for r in results if r['success'])}/{len(MATRIX)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
