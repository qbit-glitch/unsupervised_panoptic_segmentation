"""Run Ablation Studies (PyTorch).

Usage:
    python scripts/run_ablations.py --config configs/cityscapes.yaml
    python scripts/run_ablations.py --config configs/cityscapes.yaml --ablations no_mamba no_bicms
    python scripts/run_ablations.py --config configs/cityscapes.yaml --seeds 42 123 456

Runs all ablation experiments defined in configs/ablations/.
Each ablation merges the base config with the ablation override,
then launches scripts/train.py as a subprocess with proper env vars.

Ablations:
    1. no_mamba - Replace Mamba2 with MLP fusion
    2. no_depth_cond - Disable depth conditioning
    3. no_bicms - Disable bidirectional scan
    4. no_consistency - Disable cross-branch consistency losses
    5. oracle_stuff_things - Use GT stuff/things labels
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)


ABLATION_CONFIGS = [
    ("no_mamba", "No Mamba2 Bridge (MLP fusion)"),
    ("no_depth_cond", "No Depth Conditioning"),
    ("no_bicms", "No Bidirectional Scan"),
    ("no_consistency", "No Consistency Losses"),
    ("oracle_stuff_things", "Oracle Stuff-Things Labels"),
]

SEEDS = [42, 123, 456]


def merge_configs(base_path: str, ablation_path: str) -> Dict[str, Any]:
    """Load base config, merge ablation override, return merged dict.

    Args:
        base_path: Path to base dataset config (e.g. cityscapes.yaml).
        ablation_path: Path to ablation config.

    Returns:
        Merged configuration dict.
    """
    default_path = os.path.join(os.path.dirname(base_path), "default.yaml")
    with open(default_path) as f:
        config = yaml.safe_load(f)
    with open(base_path) as f:
        base_override = yaml.safe_load(f)
    with open(ablation_path) as f:
        abl_override = yaml.safe_load(f)

    def deep_merge(base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(config, base_override)
    deep_merge(config, abl_override)
    return config


def run_ablation(
    base_config: str,
    ablation_config: str,
    ablation_name: str,
    seed: int,
    vm_name: str,
    output_dir: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single ablation experiment by launching train.py.

    Args:
        base_config: Path to base dataset config.
        ablation_config: Path to ablation config.
        ablation_name: Name of the ablation.
        seed: Random seed.
        vm_name: VM name for W&B and checkpoints.
        output_dir: Output directory for results.
        dry_run: If True, log command without executing.

    Returns:
        Result dict with status and timing info.
    """
    experiment_name = f"ablation_{ablation_name}_seed{seed}"

    train_cmd = [
        sys.executable, "scripts/train.py",
        "--config", base_config,
        "--ablation", ablation_config,
        "--seed", str(seed),
        "--vm_name", vm_name,
        "--experiment", experiment_name,
    ]

    logger.info(f"[{ablation_name}/seed={seed}] Command: {' '.join(train_cmd)}")

    result = {
        "ablation": ablation_name,
        "seed": seed,
        "experiment": experiment_name,
        "config": ablation_config,
        "command": " ".join(train_cmd),
    }

    if dry_run:
        result["status"] = "dry_run"
        logger.info(f"  [DRY RUN] Would execute above command")
        return result

    # Set env vars for child process
    env = os.environ.copy()
    env["MBPS_VM_NAME"] = vm_name
    env["MBPS_EXPERIMENT"] = experiment_name

    start_time = time.time()
    try:
        proc = subprocess.run(
            train_cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time
        result["status"] = "success" if proc.returncode == 0 else "failed"
        result["returncode"] = proc.returncode
        result["elapsed_s"] = elapsed

        if proc.returncode != 0:
            result["stderr_tail"] = proc.stderr[-2000:] if proc.stderr else ""
            logger.error(
                f"  [{ablation_name}/seed={seed}] FAILED (rc={proc.returncode})"
            )
        else:
            logger.info(
                f"  [{ablation_name}/seed={seed}] SUCCESS in {elapsed:.0f}s"
            )
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  [{ablation_name}/seed={seed}] ERROR: {e}")

    # Save per-run result
    result_dir = os.path.join(output_dir, ablation_name)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"result_seed{seed}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    """Main entry point for ablation runner."""
    parser = argparse.ArgumentParser(description="Run MBPS Ablations")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default="results/ablations",
    )
    parser.add_argument(
        "--ablations", type=str, nargs="*", default=None,
        help="Specific ablations to run (default: all)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=SEEDS,
        help="Random seeds to run (default: 42 123 456)",
    )
    parser.add_argument(
        "--vm_name", type=str,
        default=os.environ.get("MBPS_VM_NAME", "local"),
        help="VM name for W&B and checkpoints",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ablation_dir = os.path.join(
        os.path.dirname(args.config), "ablations"
    )

    if args.ablations:
        selected = [
            (name, desc) for name, desc in ABLATION_CONFIGS
            if name in args.ablations
        ]
    else:
        selected = ABLATION_CONFIGS

    total_runs = len(selected) * len(args.seeds)
    logger.info(
        f"Running {len(selected)} ablation(s) x {len(args.seeds)} seed(s) "
        f"= {total_runs} total runs"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    results_summary: List[Dict[str, Any]] = []
    completed = 0

    for abl_name, abl_desc in selected:
        abl_config = os.path.join(ablation_dir, f"{abl_name}.yaml")

        if not os.path.exists(abl_config):
            logger.warning(f"Config not found: {abl_config}, skipping")
            continue

        for seed in args.seeds:
            completed += 1
            logger.info(
                f"\n{'='*60}\n"
                f"Run {completed}/{total_runs}: {abl_name} (seed={seed})\n"
                f"{'='*60}"
            )

            result = run_ablation(
                base_config=args.config,
                ablation_config=abl_config,
                ablation_name=abl_name,
                seed=seed,
                vm_name=args.vm_name,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
            result["description"] = abl_desc
            results_summary.append(result)

    # Save summary
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nAblation summary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  MBPS Ablation Results ({len(results_summary)} runs)")
    print(f"{'='*70}")
    for r in results_summary:
        status = r.get("status", "?")
        elapsed = r.get("elapsed_s", 0)
        print(
            f"  {r['ablation']:25s} seed={r['seed']}  "
            f"status={status:8s}  time={elapsed:.0f}s"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
