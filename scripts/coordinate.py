"""Coordinator script for aggregating results from worker VMs.

Runs on a single coordinator v4-8 VM. Polls GCS for new checkpoints
from worker VMs, evaluates them, and logs aggregated metrics to W&B.

Usage:
    python scripts/coordinate.py --config configs/cityscapes_gcs.yaml \
        --experiment cityscapes_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import tensorflow as tf
import yaml
from absl import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and merge YAML config with defaults."""
    default_path = os.path.join(
        os.path.dirname(config_path), "default.yaml"
    )
    with open(default_path) as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f)

    def deep_merge(base: dict, over: dict) -> None:
        for k, v in over.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(config, override)
    return config


def discover_worker_checkpoints(
    gcs_ckpt_dir: str,
    experiment: str,
    worker_vms: List[str],
) -> Dict[str, List[str]]:
    """Discover checkpoints for each worker VM on GCS.

    Args:
        gcs_ckpt_dir: GCS checkpoint root (e.g. gs://mbps-panoptic/checkpoints).
        experiment: Experiment name.
        worker_vms: List of worker VM names.

    Returns:
        Dict mapping vm_name -> sorted list of checkpoint paths.
    """
    result = {}
    for vm in worker_vms:
        vm_ckpt_dir = os.path.join(gcs_ckpt_dir, experiment, vm)
        if not tf.io.gfile.exists(vm_ckpt_dir):
            result[vm] = []
            continue
        ckpts = sorted([
            os.path.join(vm_ckpt_dir, d)
            for d in tf.io.gfile.listdir(vm_ckpt_dir)
            if d.startswith("checkpoint_epoch_")
        ])
        result[vm] = ckpts
    return result


def load_eval_result(result_path: str) -> Dict[str, Any] | None:
    """Load a JSON eval result from GCS or local path."""
    if not tf.io.gfile.exists(result_path):
        return None
    with tf.io.gfile.GFile(result_path, "r") as f:
        return json.load(f)


def save_aggregated_results(
    results: Dict[str, Any],
    gcs_results_dir: str,
    experiment: str,
) -> str:
    """Save aggregated results to GCS.

    Args:
        results: Aggregated results dict.
        gcs_results_dir: GCS results root.
        experiment: Experiment name.

    Returns:
        Path where results were saved.
    """
    out_dir = os.path.join(gcs_results_dir, experiment, "aggregated")
    tf.io.gfile.makedirs(out_dir)
    out_path = os.path.join(out_dir, "aggregated_results.json")
    with tf.io.gfile.GFile(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path


def evaluate_checkpoint(
    config: Dict[str, Any],
    checkpoint_path: str,
    num_devices: int,
    use_crf: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint.

    Imports evaluate_model lazily to avoid loading JAX on every poll cycle.
    """
    import jax
    import jax.numpy as jnp
    from mbps.data.datasets import get_dataset
    from mbps.data.transforms import EvalTransform
    from mbps.models.mbps_model import MBPSModel
    from mbps.training.checkpointing import CheckpointManager
    from scripts.evaluate import evaluate_model

    arch = config["architecture"]
    mamba_cfg = arch.get("mamba", {})
    model = MBPSModel(
        num_classes=arch.get("num_classes", config["data"].get("num_classes", 27)),
        semantic_dim=arch.get("semantic_dim", 90),
        feature_dim=arch.get("backbone_dim", 384),
        bridge_dim=arch.get("bridge_dim", 192),
        max_instances=arch.get("max_instances", 100),
        mamba_layers=mamba_cfg.get("num_layers", 4),
        mamba_state_dim=mamba_cfg.get("state_dim", 64),
        chunk_size=mamba_cfg.get("chunk_size", 128),
        use_depth_conditioning=arch.get("use_depth_conditioning", True),
        use_mamba_bridge=arch.get("use_mamba_bridge", True),
        use_bidirectional=mamba_cfg.get("use_bidirectional", True),
    )

    ckpt_manager = CheckpointManager(checkpoint_dir="")
    ckpt = ckpt_manager.load(checkpoint_path)
    params = ckpt["params"]

    eval_split = config.get("coordinator", {}).get("eval_split", "val")
    transform = EvalTransform(image_size=tuple(config["data"]["image_size"]))
    dataset = get_dataset(
        name=config["data"].get("dataset", "cityscapes"),
        data_dir=config["data"]["data_dir"],
        depth_dir=config["data"].get("depth_dir", ""),
        split=eval_split,
        image_size=tuple(config["data"]["image_size"]),
        transforms=transform,
    )

    results = evaluate_model(
        model, params, dataset, config,
        use_crf=use_crf, num_devices=num_devices,
    )
    return results


def main():
    """Coordinator main loop."""
    parser = argparse.ArgumentParser(description="MBPS Coordinator")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "--num_devices", type=int, default=None,
        help="Devices for evaluation (default: all)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one poll cycle and exit (instead of looping)",
    )
    args = parser.parse_args()

    logging.set_verbosity(logging.INFO)
    config = load_config(args.config)

    coord_cfg = config.get("coordinator", {})
    poll_interval = coord_cfg.get("poll_interval_s", 300)
    worker_vms = coord_cfg.get("worker_vms", [])
    use_crf = coord_cfg.get("eval_use_crf", False)

    gcs_cfg = config.get("gcs", {})
    gcs_ckpt_dir = gcs_cfg.get("checkpoint_dir")
    gcs_results_dir = gcs_cfg.get("results_dir")

    if not gcs_ckpt_dir or not gcs_results_dir:
        logging.error("GCS checkpoint_dir and results_dir must be set in config")
        sys.exit(1)

    if not worker_vms:
        logging.error("No worker_vms configured in coordinator section")
        sys.exit(1)

    import jax
    num_devices = args.num_devices or jax.device_count()

    # Initialize W&B for coordinator
    if HAS_WANDB and config.get("logging", {}).get("use_wandb", False):
        log_cfg = config.get("logging", {})
        wandb.init(
            project=log_cfg.get("wandb_project", "mbps"),
            entity=log_cfg.get("wandb_entity"),
            group=log_cfg.get("wandb_group") or args.experiment,
            name=f"{args.experiment}/coordinator",
            tags=["coordinator", args.experiment],
            config=config,
        )

    # Track which checkpoints we've already evaluated
    evaluated: Dict[str, set] = {vm: set() for vm in worker_vms}

    logging.info(
        f"Coordinator started: experiment={args.experiment}, "
        f"workers={worker_vms}, poll_interval={poll_interval}s"
    )

    while True:
        logging.info("Polling for new checkpoints...")
        worker_ckpts = discover_worker_checkpoints(
            gcs_ckpt_dir, args.experiment, worker_vms,
        )

        new_evals = {}
        for vm, ckpts in worker_ckpts.items():
            new_ckpts = [c for c in ckpts if c not in evaluated[vm]]
            if not new_ckpts:
                continue

            # Evaluate only the latest new checkpoint per worker
            latest = new_ckpts[-1]
            logging.info(f"Evaluating {vm}: {os.path.basename(latest)}")

            try:
                result = evaluate_checkpoint(
                    config, latest, num_devices, use_crf,
                )
                result["vm_name"] = vm
                result["checkpoint"] = latest
                new_evals[vm] = result

                # Save per-worker result to GCS
                vm_result_dir = os.path.join(
                    gcs_results_dir, args.experiment, vm
                )
                tf.io.gfile.makedirs(vm_result_dir)
                epoch_name = os.path.basename(latest)
                result_path = os.path.join(
                    vm_result_dir, f"eval_{epoch_name}.json"
                )
                with tf.io.gfile.GFile(result_path, "w") as f:
                    json.dump(result, f, indent=2)
                logging.info(f"  Saved: {result_path}")

                # Log to W&B
                if HAS_WANDB and wandb.run is not None:
                    wandb_metrics = {
                        f"{vm}/miou": result.get("miou", 0),
                        f"{vm}/accuracy": result.get("accuracy", 0),
                        f"{vm}/eval_time_s": result.get("eval_time_s", 0),
                    }
                    wandb.log(wandb_metrics)

                # Mark all new checkpoints as seen (we only eval latest)
                evaluated[vm].update(new_ckpts)

            except Exception as e:
                logging.error(f"  Failed to evaluate {vm}/{latest}: {e}")
                # Still mark as seen to avoid retrying broken checkpoints
                evaluated[vm].update(new_ckpts)

        # Aggregate across workers
        if new_evals:
            all_miou = [r.get("miou", 0) for r in new_evals.values() if "miou" in r]
            aggregated = {
                "experiment": args.experiment,
                "num_workers_evaluated": len(new_evals),
                "per_worker": {vm: r for vm, r in new_evals.items()},
            }
            if all_miou:
                import numpy as np
                aggregated["mean_miou"] = float(np.mean(all_miou))
                aggregated["std_miou"] = float(np.std(all_miou))

            out_path = save_aggregated_results(
                aggregated, gcs_results_dir, args.experiment,
            )
            logging.info(f"Aggregated results → {out_path}")

            if HAS_WANDB and wandb.run is not None and all_miou:
                wandb.log({
                    "aggregated/mean_miou": aggregated["mean_miou"],
                    "aggregated/std_miou": aggregated["std_miou"],
                })
        else:
            logging.info("No new checkpoints found.")

        if args.once:
            break

        logging.info(f"Sleeping {poll_interval}s...")
        time.sleep(poll_interval)

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()

    logging.info("Coordinator finished.")


if __name__ == "__main__":
    main()
