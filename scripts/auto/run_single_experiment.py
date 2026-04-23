#!/usr/bin/env python3
"""Run a single experiment (one matrix cell) with full resume capability.

Pipeline:
    prerequisites -> generate_semantic -> generate_instance -> train_dcfa
    -> regenerate_semantic -> assemble_panoptic -> apply_simcf -> evaluate -> save
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages one experiment with state tracking and resume."""

    def __init__(self, config: dict):
        self.config = config
        self.exp_id = self._make_exp_id()
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "state.json"
        self.state = self._load_state()
        self.log_file = self.output_dir / "experiment.log"
        self._setup_file_logging()

    def _make_exp_id(self) -> str:
        parts = [self.config["semantic_method"], self.config["instance_method"]]
        if self.config.get("dcfa"):
            parts.append("dcfa")
        if self.config.get("simcf_steps"):
            parts.append("simcf" + "".join(self.config["simcf_steps"]))
        return "_".join(parts)

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"status": "pending", "steps": {}}

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _setup_file_logging(self):
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    def _run_step(self, step_name: str, cmd: list, env: dict = None) -> bool:
        """Run a shell command step with retry logic."""
        if step_name in self.state["steps"] and self.state["steps"][step_name].get("status") == "completed":
            logger.info(f"STEP {step_name}: Already completed, skipping")
            return True

        logger.info(f"STEP {step_name}: Starting")
        self.state["steps"][step_name] = {"status": "in_progress"}
        self._save_state()

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            logger.info(f"STEP {step_name}: Attempt {attempt}/{max_retries}")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env={**os.environ, **(env or {})},
                    timeout=None,
                )
                if result.returncode == 0:
                    logger.info(f"STEP {step_name}: Completed successfully")
                    self.state["steps"][step_name] = {"status": "completed"}
                    self._save_state()
                    return True
                else:
                    logger.error(f"STEP {step_name}: Failed with code {result.returncode}")
                    logger.error(f"STDERR: {result.stderr[:500]}")
            except Exception as e:
                logger.error(f"STEP {step_name}: Exception: {e}")
            time.sleep(5)

        logger.error(f"STEP {step_name}: All {max_retries} attempts failed")
        self.state["steps"][step_name] = {"status": "failed"}
        self._save_state()
        return False

    def run(self) -> bool:
        """Execute full pipeline with resume."""
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT: {self.exp_id}")
        logger.info("=" * 60)

        c = self.config
        cityscapes = c["cityscapes_root"]
        out = str(self.output_dir)
        project = c["project_root"]

        # Step 1: Prerequisites
        prereq_cmd = [
            sys.executable,
            str(Path(__file__).parent / "check_prerequisites.py"),
            "--cityscapes_root", cityscapes,
            "--output_dir", out,
            "--min_disk_gb", "20",
        ]
        if not self._run_step("prerequisites", prereq_cmd):
            return False

        # Step 2: Generate semantic pseudo-labels
        sem_method = c["semantic_method"]
        sem_out = Path(out) / "semantic"
        if sem_method == "cause_k80":
            sem_cmd = [
                sys.executable,
                str(Path(project) / "mbps_pytorch" / "generate_semantic_pseudolabels_cause.py"),
                "--mode", "generate",
                "--cityscapes_root", cityscapes,
                "--output_dir", str(sem_out),
                "--split", "train",
            ]
        elif sem_method == "depthg_k80":
            sem_cmd = [
                sys.executable,
                str(Path(project) / "mbps_pytorch" / "generate_depthg_semantic_pseudolabels.py"),
                "--cityscapes_root", cityscapes,
                "--output_dir", str(sem_out),
                "--checkpoint", str(Path(project) / "weights" / "depthg.ckpt"),
            ]
        elif sem_method == "dinov3_k80":
            sem_cmd = [
                sys.executable,
                str(Path(project) / "mbps_pytorch" / "generate_dinov3_kmeans.py"),
                "--cityscapes_root", cityscapes,
                "--output_dir", str(sem_out),
                "--k", "80",
            ]
        else:
            logger.error(f"Unknown semantic method: {sem_method}")
            return False

        if not self._run_step("generate_semantic", sem_cmd):
            return False

        # Step 3: Generate instance pseudo-labels
        inst_method = c["instance_method"]
        inst_out = Path(out) / "instance"
        if inst_method == "depthpro_tau020":
            inst_cmd = [
                sys.executable,
                str(Path(project) / "mbps_pytorch" / "generate_depth_guided_instances.py"),
                "--cityscapes_root", cityscapes,
                "--output_dir", str(inst_out),
                "--depth_model", "depthpro",
                "--tau", "0.20",
            ]
        elif inst_method == "dav3":
            inst_cmd = [
                sys.executable,
                str(Path(project) / "mbps_pytorch" / "generate_depth_guided_instances.py"),
                "--cityscapes_root", cityscapes,
                "--output_dir", str(inst_out),
                "--depth_model", "dav3",
                "--tau", "0.20",
            ]
        elif inst_method == "cuts3d":
            inst_cmd = [
                sys.executable,
                str(Path(project) / "mbps_pytorch" / "generate_cuts3d_instances.py"),
                "--cityscapes_root", cityscapes,
                "--output_dir", str(inst_out),
            ]
        else:
            logger.error(f"Unknown instance method: {inst_method}")
            return False

        if not self._run_step("generate_instance", inst_cmd):
            return False

        # Step 4: Train DCFA if needed
        if c.get("dcfa"):
            dcfa_ckpt = Path(out) / "dcfa" / "best.pt"
            dcfa_ckpt.parent.mkdir(parents=True, exist_ok=True)
            dcfa_cmd = [
                sys.executable,
                str(Path(project) / "mbps_pytorch" / "train_depth_adapter.py"),
                "--cityscapes_root", cityscapes,
                "--feature_dir", str(sem_out),
                "--output_dir", str(dcfa_ckpt.parent),
                "--epochs", "20",
                "--lambda_preserve", "20.0",
            ]
            if not self._run_step("train_dcfa", dcfa_cmd):
                return False

        # Step 5: Apply SIMCF if needed
        simcf_steps = c.get("simcf_steps", [])
        if simcf_steps:
            panoptic_in = Path(out) / "panoptic_raw"
            panoptic_out = Path(out) / "panoptic_refined"
            for step in simcf_steps:
                simcf_cmd = [
                    sys.executable,
                    str(Path(project) / "mbps_pytorch" / "apply_simcf.py"),
                    "--input_dir", str(panoptic_in),
                    "--output_dir", str(panoptic_out),
                    "--step", step,
                ]
                if not self._run_step(f"apply_simcf_{step}", simcf_cmd):
                    return False
                panoptic_in = panoptic_out

        # Step 6: Evaluate
        eval_out = Path(out) / "eval_results.json"
        eval_cmd = [
            sys.executable,
            str(Path(project) / "mbps_pytorch" / "evaluate_pseudolabels.py"),
            "--cityscapes_root", cityscapes,
            "--pseudo_dir", str(panoptic_out if simcf_steps else Path(out) / "panoptic"),
            "--output", str(eval_out),
        ]
        if not self._run_step("evaluate", eval_cmd):
            return False

        self.state["status"] = "completed"
        self._save_state()
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT {self.exp_id}: COMPLETED")
        logger.info("=" * 60)
        return True


def main():
    parser = argparse.ArgumentParser(description="Run single experiment with resume")
    parser.add_argument("--semantic_method", required=True, choices=["cause_k80", "depthg_k80", "dinov3_k80"])
    parser.add_argument("--instance_method", required=True, choices=["depthpro_tau020", "dav3", "cuts3d"])
    parser.add_argument("--dcfa", action="store_true")
    parser.add_argument("--simcf_steps", default="", help="Comma-separated: A,B,C")
    parser.add_argument("--cityscapes_root", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = {
        "semantic_method": args.semantic_method,
        "instance_method": args.instance_method,
        "dcfa": args.dcfa,
        "simcf_steps": [s.strip() for s in args.simcf_steps.split(",") if s.strip()],
        "cityscapes_root": str(args.cityscapes_root),
        "project_root": str(args.project_root),
        "output_dir": str(args.output_dir),
    }

    runner = ExperimentRunner(config)
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
