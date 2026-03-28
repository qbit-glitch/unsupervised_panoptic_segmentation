"""MBPS TPU v6e Experiment Orchestrator.

Deploys all 36 training experiments across 16 TPU v6e-8 spot VMs
in two zones: europe-west4-a (8 VMs) and us-east1-d (8 VMs).

Usage:
    python scripts/orchestrate_v6e.py --phase all --dry_run    # Preview all commands
    python scripts/orchestrate_v6e.py --phase all               # Full pipeline
    python scripts/orchestrate_v6e.py --phase create             # Create 16 VMs
    python scripts/orchestrate_v6e.py --phase setup              # Install deps + sync code
    python scripts/orchestrate_v6e.py --phase launch             # Launch all 3 waves
    python scripts/orchestrate_v6e.py --phase launch --waves 1   # Wave 1 only
    python scripts/orchestrate_v6e.py --phase status             # Check all job progress
    python scripts/orchestrate_v6e.py --phase monitor            # Live dashboard
    python scripts/orchestrate_v6e.py --phase cleanup --force    # Delete all VMs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT = "unsupervised-panoptic-segment"
ZONE_EU = "europe-west4-a"
ZONE_US = "us-east1-d"
BUCKET = "mbps-panoptic"
LOCAL_CODE = "/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
SOFTWARE_VERSION = "v2-alpha-tpuv6e"
ACCELERATOR_TYPE = "v6e-8"
DISK_SIZE_GB = 50
MAX_WORKERS = 16

ABLATIONS = [
    "no_mamba",
    "no_depth_cond",
    "no_bicms",
    "no_consistency",
    "oracle_stuff_things",
]
SEEDS = [42, 123, 456]
TARGET_EPOCHS = 75  # 60 (phases A-C) + 15 (self-training)

RSYNC_EXCLUDES = [
    ".claude", ".git", "__pycache__", "*.pyc", ".DS_Store", "refs/",
    "logs/", "*.log", "checkpoints/",
]

log = logging.getLogger("orchestrate_v6e")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VMSpec:
    """TPU VM specification."""
    vm_name: str
    zone: str
    accelerator_type: str = ACCELERATOR_TYPE
    spot: bool = True  # All v6e are spot
    disk_size_gb: int = DISK_SIZE_GB


@dataclass
class Job:
    """Training job specification."""
    job_id: str
    dataset: str          # "cityscapes" or "coco"
    experiment_type: str  # "full" or ablation name
    seed: int
    config: str           # e.g. "cityscapes_gcs.yaml"
    ablation: Optional[str]
    vm_name: str
    zone: str
    wave: int

    @property
    def experiment_name(self) -> str:
        if self.experiment_type == "full":
            return f"{self.dataset}_full"
        return f"{self.dataset}_ablation_{self.experiment_type}"

    @property
    def gcs_checkpoint_dir(self) -> str:
        return f"gs://{BUCKET}/checkpoints/{self.experiment_name}/{self.vm_name}"

    @property
    def train_command(self) -> str:
        parts = [
            "cd ~/mbps_panoptic_segmentation && mkdir -p logs &&",
            f"MBPS_VM_NAME={self.vm_name} MBPS_EXPERIMENT={self.experiment_name}",
            "nohup python3 scripts/train.py",
            f"--config configs/{self.config}",
            f"--seed {self.seed}",
            f"--vm_name {self.vm_name}",
            f"--experiment {self.experiment_name}",
        ]
        if self.ablation:
            parts.append(f"--ablation configs/ablations/{self.ablation}.yaml")
        parts.append(f"> logs/{self.job_id}.log 2>&1 &")
        return " ".join(parts)

    @property
    def resume_command(self) -> str:
        """Build resume command using latest GCS checkpoint."""
        parts = [
            "cd ~/mbps_panoptic_segmentation && mkdir -p logs &&",
            f"MBPS_VM_NAME={self.vm_name} MBPS_EXPERIMENT={self.experiment_name}",
            "nohup python3 scripts/train.py",
            f"--config configs/{self.config}",
            f"--seed {self.seed}",
            f"--vm_name {self.vm_name}",
            f"--experiment {self.experiment_name}",
        ]
        if self.ablation:
            parts.append(f"--ablation configs/ablations/{self.ablation}.yaml")
        # Resume flag will be added by the caller with the actual checkpoint path
        parts.append(f"> logs/{self.job_id}_resumed.log 2>&1 &")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# VM + Job Matrix Generation
# ---------------------------------------------------------------------------

def generate_vm_specs() -> List[VMSpec]:
    """Generate 16 v6e-8 spot VM specs: 8 in EU, 8 in US."""
    specs: List[VMSpec] = []
    for i in range(8):
        specs.append(VMSpec(
            vm_name=f"mbps-v6e-eu-{i}",
            zone=ZONE_EU,
        ))
    for i in range(8):
        specs.append(VMSpec(
            vm_name=f"mbps-v6e-us-{i}",
            zone=ZONE_US,
        ))
    return specs


def build_job_matrix() -> List[Job]:
    """Build 36 jobs across 3 waves on 16 v6e-8 VMs.

    Layout:
      EU VMs (8): Cityscapes experiments
      US VMs (8): COCO experiments

    Wave 1 (16 jobs): 3 full + 5 ablations(seed=42) per dataset
    Wave 2 (16 jobs): 5 ablations(seed=123) + 3 ablations(seed=456) per dataset
    Wave 3 (4 jobs):  2 remaining ablations(seed=456) per dataset
    """
    jobs: List[Job] = []

    def _job(
        dataset: str, exp_type: str, seed: int,
        vm_name: str, zone: str, wave: int,
    ) -> Job:
        config = "cityscapes_gcs.yaml" if dataset == "cityscapes" else "coco_stuff27_gcs.yaml"
        ablation = None if exp_type == "full" else exp_type
        jid = f"{dataset}_{exp_type}_seed{seed}"
        return Job(jid, dataset, exp_type, seed, config, ablation, vm_name, zone, wave)

    # ── Wave 1: 16 jobs ──
    # Cityscapes on EU VMs
    for i, seed in enumerate(SEEDS):
        jobs.append(_job("cityscapes", "full", seed, f"mbps-v6e-eu-{i}", ZONE_EU, 1))
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("cityscapes", abl, 42, f"mbps-v6e-eu-{i + 3}", ZONE_EU, 1))

    # COCO on US VMs
    for i, seed in enumerate(SEEDS):
        jobs.append(_job("coco", "full", seed, f"mbps-v6e-us-{i}", ZONE_US, 1))
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("coco", abl, 42, f"mbps-v6e-us-{i + 3}", ZONE_US, 1))

    # ── Wave 2: 16 jobs ──
    # Cityscapes ablations (seed 123) + first 3 (seed 456)
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("cityscapes", abl, 123, f"mbps-v6e-eu-{i}", ZONE_EU, 2))
    for i, abl in enumerate(ABLATIONS[:3]):
        jobs.append(_job("cityscapes", abl, 456, f"mbps-v6e-eu-{i + 5}", ZONE_EU, 2))

    # COCO ablations (seed 123) + first 3 (seed 456)
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("coco", abl, 123, f"mbps-v6e-us-{i}", ZONE_US, 2))
    for i, abl in enumerate(ABLATIONS[:3]):
        jobs.append(_job("coco", abl, 456, f"mbps-v6e-us-{i + 5}", ZONE_US, 2))

    # ── Wave 3: 4 jobs ──
    # Last 2 Cityscapes + last 2 COCO ablations (seed 456)
    for i, abl in enumerate(ABLATIONS[3:]):
        jobs.append(_job("cityscapes", abl, 456, f"mbps-v6e-eu-{i}", ZONE_EU, 3))
    for i, abl in enumerate(ABLATIONS[3:]):
        jobs.append(_job("coco", abl, 456, f"mbps-v6e-us-{i}", ZONE_US, 3))

    return jobs


def get_wave_jobs(jobs: List[Job], wave: int) -> List[Job]:
    return [j for j in jobs if j.wave == wave]


# ---------------------------------------------------------------------------
# Shell Utilities
# ---------------------------------------------------------------------------

def run_command(
    cmd: List[str],
    timeout: int = 600,
    dry_run: bool = False,
    label: str = "",
) -> Tuple[int, str, str]:
    cmd_str = " ".join(cmd)
    if dry_run:
        log.info(f"[DRY RUN] {label}: {cmd_str}")
        return 0, "", ""

    log.debug(f"Running: {cmd_str}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            log.debug(f"Failed (rc={proc.returncode}): {proc.stderr[:500]}")
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout ({timeout}s): {label}")
        return -1, "", "TIMEOUT"
    except Exception as e:
        log.error(f"Error: {e}")
        return -1, "", str(e)


def gcloud_ssh(
    vm_name: str,
    zone: str,
    command: str,
    timeout: int = 600,
    retries: int = 3,
    dry_run: bool = False,
) -> Tuple[int, str, str]:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", vm_name,
        f"--zone={zone}",
        f"--project={PROJECT}",
        f"--command={command}",
        "--strict-host-key-checking=no",
    ]
    for attempt in range(retries):
        rc, out, err = run_command(cmd, timeout=timeout, dry_run=dry_run, label=f"SSH {vm_name}")
        if dry_run or rc == 0:
            return rc, out, err
        if "not found" in err.lower() or "does not exist" in err.lower():
            return rc, out, err
        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            log.warning(f"SSH {vm_name} failed (attempt {attempt + 1}), retrying in {wait}s...")
            time.sleep(wait)
    return rc, out, err


def check_vm_state(vm_name: str, zone: str, dry_run: bool = False) -> str:
    if dry_run:
        return "READY"
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "describe", vm_name,
        f"--zone={zone}", f"--project={PROJECT}", "--format=json",
    ]
    rc, out, err = run_command(cmd, timeout=30, label=f"Describe {vm_name}")
    if rc != 0:
        return "NOT_FOUND"
    try:
        data = json.loads(out)
        return data.get("state", "UNKNOWN")
    except (json.JSONDecodeError, KeyError):
        return "UNKNOWN"


def get_vm_ip(vm_name: str, zone: str) -> Optional[str]:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "describe", vm_name,
        f"--zone={zone}", f"--project={PROJECT}",
        "--format=value(networkEndpoints[0].accessConfig.externalIp)",
    ]
    rc, out, _ = run_command(cmd, timeout=30, label=f"IP {vm_name}")
    if rc == 0 and out.strip():
        return out.strip()
    return None


# ---------------------------------------------------------------------------
# VM Lifecycle
# ---------------------------------------------------------------------------

def create_vm(spec: VMSpec, dry_run: bool = False) -> bool:
    state = check_vm_state(spec.vm_name, spec.zone, dry_run)
    if state == "READY":
        log.info(f"VM {spec.vm_name} already READY, skipping")
        return True

    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "create", spec.vm_name,
        f"--zone={spec.zone}",
        f"--accelerator-type={spec.accelerator_type}",
        f"--version={SOFTWARE_VERSION}",
        f"--boot-disk-size={spec.disk_size_gb}GB",
        f"--project={PROJECT}",
        "--spot",  # All v6e are spot
    ]

    label = f"Create {spec.vm_name} (v6e-8 spot, {spec.zone})"
    rc, out, err = run_command(cmd, timeout=600, dry_run=dry_run, label=label)
    if rc == 0 or dry_run:
        log.info(f"Created: {spec.vm_name}")
        return True
    log.error(f"Failed to create {spec.vm_name}: {err[:300]}")
    return False


def setup_vm(spec: VMSpec, dry_run: bool = False) -> bool:
    """Sync code + install v6e deps + verify JAX."""
    vm, zone = spec.vm_name, spec.zone

    # Step 1: Rsync code
    ip = get_vm_ip(vm, zone) if not dry_run else "DRY_RUN_IP"
    if not ip and not dry_run:
        log.error(f"Cannot get IP for {vm}")
        return False

    exclude_args = []
    for exc in RSYNC_EXCLUDES:
        exclude_args.extend(["--exclude", exc])

    rsync_cmd = [
        "rsync", "-avz", "--compress",
        *exclude_args,
        "-e", "ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no",
        f"{LOCAL_CODE}/",
        f"qbit-glitch@{ip}:~/mbps_panoptic_segmentation/",
    ]
    rc, _, err = run_command(rsync_cmd, timeout=300, dry_run=dry_run, label=f"Rsync → {vm}")
    if rc != 0 and not dry_run:
        log.error(f"Rsync to {vm} failed: {err[:300]}")
        return False
    log.info(f"Code synced to {vm}")

    # Step 2: Install v6e-specific dependencies
    rc, out, err = gcloud_ssh(
        vm, zone,
        "bash ~/mbps_panoptic_segmentation/scripts/install_deps_v6e.sh",
        timeout=900,  # Allow up to 15 min for deps
        dry_run=dry_run,
    )
    if rc != 0 and not dry_run:
        log.error(f"Deps install on {vm} failed: {err[:300]}")
        return False
    log.info(f"Dependencies installed on {vm}")

    # Step 3: Verify JAX sees TPU devices
    rc, out, err = gcloud_ssh(
        vm, zone,
        'python3 -c "import jax; print(f\'JAX {jax.__version__}: {jax.device_count()} devices ({jax.devices()[0].platform})\')"',
        timeout=120, dry_run=dry_run,
    )
    if rc != 0 and not dry_run:
        log.error(f"JAX verify failed on {vm}: {err[:300]}")
        return False
    if not dry_run:
        log.info(f"JAX verified on {vm}: {out.strip()}")
    return True


def delete_vm(spec: VMSpec, dry_run: bool = False) -> bool:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "delete", spec.vm_name,
        f"--zone={spec.zone}", f"--project={PROJECT}", "--quiet",
    ]
    rc, _, err = run_command(cmd, timeout=300, dry_run=dry_run, label=f"Delete {spec.vm_name}")
    if rc == 0 or dry_run:
        log.info(f"Deleted: {spec.vm_name}")
        return True
    log.warning(f"Delete {spec.vm_name} failed: {err[:200]}")
    return False


# ---------------------------------------------------------------------------
# Job Execution & Monitoring
# ---------------------------------------------------------------------------

def launch_job(job: Job, dry_run: bool = False) -> bool:
    rc, _, err = gcloud_ssh(
        job.vm_name, job.zone, job.train_command,
        timeout=60, dry_run=dry_run,
    )
    if rc == 0 or dry_run:
        log.info(f"Launched: {job.job_id} on {job.vm_name}")
        return True
    log.error(f"Failed to launch {job.job_id}: {err[:300]}")
    return False


def check_job_progress(job: Job, dry_run: bool = False) -> Optional[int]:
    """Check latest checkpoint epoch on GCS."""
    if dry_run:
        return None
    cmd = ["gsutil", "ls", f"{job.gcs_checkpoint_dir}/"]
    rc, out, _ = run_command(cmd, timeout=30, label=f"Progress {job.job_id}")
    if rc != 0 or not out.strip():
        return None
    epochs = []
    for line in out.strip().split("\n"):
        match = re.search(r"checkpoint_epoch_(\d{4})", line)
        if match:
            epochs.append(int(match.group(1)))
    return max(epochs) if epochs else None


def is_job_complete(job: Job, dry_run: bool = False) -> bool:
    epoch = check_job_progress(job, dry_run)
    return epoch is not None and epoch >= TARGET_EPOCHS


def get_latest_checkpoint(job: Job) -> Optional[str]:
    epoch = check_job_progress(job)
    if epoch is None:
        return None
    return f"{job.gcs_checkpoint_dir}/checkpoint_epoch_{epoch:04d}"


# ---------------------------------------------------------------------------
# Preemption Recovery
# ---------------------------------------------------------------------------

def handle_preemption(job: Job, spec: VMSpec, dry_run: bool = False) -> bool:
    """Recreate preempted VM, re-setup, resume from last checkpoint."""
    log.warning(f"Preemption recovery: {job.vm_name} ({job.job_id})")

    delete_vm(spec, dry_run)
    time.sleep(5)

    if not create_vm(spec, dry_run):
        return False

    if not dry_run:
        for _ in range(30):
            if check_vm_state(spec.vm_name, spec.zone) == "READY":
                break
            time.sleep(10)
        else:
            log.error(f"{spec.vm_name} not READY after recreation")
            return False

    if not setup_vm(spec, dry_run):
        return False

    latest_ckpt = get_latest_checkpoint(job) if not dry_run else None

    resume_parts = [
        "cd ~/mbps_panoptic_segmentation && mkdir -p logs &&",
        f"MBPS_VM_NAME={job.vm_name} MBPS_EXPERIMENT={job.experiment_name}",
        "nohup python3 scripts/train.py",
        f"--config configs/{job.config}",
        f"--seed {job.seed}",
        f"--vm_name {job.vm_name}",
        f"--experiment {job.experiment_name}",
    ]
    if job.ablation:
        resume_parts.append(f"--ablation configs/ablations/{job.ablation}.yaml")
    if latest_ckpt:
        resume_parts.append(f"--resume {latest_ckpt}")
    resume_parts.append(f"> logs/{job.job_id}_resumed.log 2>&1 &")

    rc, _, err = gcloud_ssh(
        job.vm_name, job.zone, " ".join(resume_parts),
        timeout=60, dry_run=dry_run,
    )
    if rc == 0 or dry_run:
        epoch = check_job_progress(job) or 0
        log.info(f"Resumed {job.job_id} on {job.vm_name} (from epoch {epoch})")
        return True
    log.error(f"Resume failed for {job.job_id}: {err[:300]}")
    return False


# ---------------------------------------------------------------------------
# Wave Orchestration
# ---------------------------------------------------------------------------

def schedule_wave(wave_jobs: List[Job], dry_run: bool = False) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(launch_job, job, dry_run): job
            for job in wave_jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results[job.job_id] = future.result()
            except Exception as e:
                log.error(f"Exception launching {job.job_id}: {e}")
                results[job.job_id] = False
    return results


def wait_for_wave(
    wave_jobs: List[Job],
    vm_specs: List[VMSpec],
    poll_interval_s: int = 300,
    max_wait_hours: float = 96,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        log.info(f"[DRY RUN] Would poll {len(wave_jobs)} jobs every {poll_interval_s}s")
        return True

    spec_map = {s.vm_name: s for s in vm_specs}
    completed: Set[str] = set()
    failed: Set[str] = set()
    preemption_counts: Dict[str, int] = {j.job_id: 0 for j in wave_jobs}
    last_epochs: Dict[str, Optional[int]] = {j.job_id: None for j in wave_jobs}
    deadline = time.time() + max_wait_hours * 3600

    wave_num = wave_jobs[0].wave if wave_jobs else 0
    log.info(f"Monitoring wave {wave_num}: {len(wave_jobs)} jobs, poll {poll_interval_s}s")

    while time.time() < deadline:
        pending = [j for j in wave_jobs if j.job_id not in completed and j.job_id not in failed]
        if not pending:
            break

        for job in pending:
            state = check_vm_state(job.vm_name, job.zone)

            if state == "PREEMPTED":
                preemption_counts[job.job_id] += 1
                if preemption_counts[job.job_id] > 5:
                    log.error(f"{job.job_id}: Too many preemptions, FAILED")
                    failed.add(job.job_id)
                    continue
                spec = spec_map.get(job.vm_name)
                if spec:
                    handle_preemption(job, spec)
                continue

            epoch = check_job_progress(job)
            if epoch is not None and epoch >= TARGET_EPOCHS:
                completed.add(job.job_id)
                log.info(f"COMPLETE: {job.job_id} (epoch {epoch})")
                continue

            if epoch == last_epochs[job.job_id] and epoch is not None:
                pass  # Could be training between checkpoints
            last_epochs[job.job_id] = epoch

        n_total = len(wave_jobs)
        n_done = len(completed)
        n_fail = len(failed)
        log.info(
            f"Wave {wave_num}: {n_done}/{n_total} complete, "
            f"{n_total - n_done - n_fail} running, {n_fail} failed"
        )

        if n_done + n_fail >= n_total:
            break
        time.sleep(poll_interval_s)

    return len(failed) == 0 and len(completed) == len(wave_jobs)


# ---------------------------------------------------------------------------
# Status Display
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_status_table(
    all_jobs: List[Job],
    dry_run: bool = False,
) -> None:
    header = (
        f"{BOLD}{'Wave':>4}  {'VM':<18} {'Zone':<16} "
        f"{'Experiment':<35} {'Seed':>4}  {'Status':<10} {'Epoch':>5}{RESET}"
    )
    print(f"\n{'=' * 100}")
    print(f"{BOLD}  MBPS v6e Experiment Status  ({datetime.now().strftime('%H:%M:%S')}){RESET}")
    print(f"{'=' * 100}")
    print(header)
    print("-" * 100)

    for wave_num in [1, 2, 3]:
        wave_jobs = get_wave_jobs(all_jobs, wave_num)
        if not wave_jobs:
            continue

        for job in wave_jobs:
            zone_short = "eu-west4" if "eu" in job.vm_name else "us-east1"

            if dry_run:
                status = f"{DIM}  PEND{RESET}"
                epoch_str = "  -"
            else:
                vm_state = check_vm_state(job.vm_name, job.zone)
                epoch = check_job_progress(job)

                if epoch is not None and epoch >= TARGET_EPOCHS:
                    status = f"{GREEN}  DONE{RESET}"
                    epoch_str = f"{epoch:>5}"
                elif vm_state == "PREEMPTED":
                    status = f"{RED}  PRE!{RESET}"
                    epoch_str = f"{epoch or 0:>5}"
                elif vm_state == "READY" and epoch is not None:
                    status = f"{CYAN}  RUN {RESET}"
                    epoch_str = f"{epoch:>5}"
                elif vm_state == "READY":
                    status = f"{YELLOW}  WAIT{RESET}"
                    epoch_str = "  -"
                elif vm_state == "NOT_FOUND":
                    status = f"{DIM}  PEND{RESET}"
                    epoch_str = "  -"
                else:
                    status = f"{DIM}  {vm_state[:4]}{RESET}"
                    epoch_str = "  -"

            print(
                f"{job.wave:>4}  {job.vm_name:<18} {zone_short:<16} "
                f"{job.experiment_name[:35]:<35} {job.seed:>4}  {status:<10} {epoch_str}"
            )

        if wave_num < 3:
            print(f"{DIM}{'·' * 100}{RESET}")

    print(f"{'=' * 100}")

    if not dry_run:
        total = len(all_jobs)
        done = sum(1 for j in all_jobs if is_job_complete(j))
        print(f"\n  Total: {total} | Complete: {GREEN}{done}{RESET} | Remaining: {total - done}\n")


# ---------------------------------------------------------------------------
# Phase Executors
# ---------------------------------------------------------------------------

def phase_create(vm_specs: List[VMSpec], dry_run: bool = False) -> bool:
    log.info(f"Creating {len(vm_specs)} v6e-8 spot VMs...")
    results: Dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(create_vm, spec, dry_run): spec
            for spec in vm_specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                results[spec.vm_name] = future.result()
            except Exception as e:
                log.error(f"Create {spec.vm_name} error: {e}")
                results[spec.vm_name] = False

    success = sum(results.values())
    failed_vms = [name for name, ok in results.items() if not ok]
    log.info(f"VM creation: {success}/{len(vm_specs)} successful")
    if failed_vms:
        log.warning(f"Failed VMs: {', '.join(failed_vms)}")
    return success == len(vm_specs)


def phase_setup(vm_specs: List[VMSpec], dry_run: bool = False) -> bool:
    # Wait for VMs to be READY first
    if not dry_run:
        log.info("Waiting for all VMs to be READY...")
        for spec in vm_specs:
            for _ in range(30):
                state = check_vm_state(spec.vm_name, spec.zone)
                if state == "READY":
                    break
                time.sleep(10)
            else:
                log.warning(f"{spec.vm_name} not READY, skipping setup")

    log.info(f"Setting up {len(vm_specs)} VMs (code sync + deps + verify)...")
    results: Dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(setup_vm, spec, dry_run): spec
            for spec in vm_specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                results[spec.vm_name] = future.result()
            except Exception as e:
                log.error(f"Setup {spec.vm_name} error: {e}")
                results[spec.vm_name] = False

    success = sum(results.values())
    log.info(f"VM setup: {success}/{len(vm_specs)} successful")
    return success == len(vm_specs)


def phase_launch(
    all_jobs: List[Job],
    vm_specs: List[VMSpec],
    waves: Optional[List[int]] = None,
    poll_interval_s: int = 300,
    max_wait_hours: float = 96,
    dry_run: bool = False,
) -> bool:
    target_waves = waves or [1, 2, 3]
    all_success = True

    for wave_num in target_waves:
        wave_jobs = get_wave_jobs(all_jobs, wave_num)
        if not wave_jobs:
            continue

        log.info(f"\n{'=' * 60}")
        log.info(f"WAVE {wave_num}: Launching {len(wave_jobs)} jobs on v6e")
        log.info(f"{'=' * 60}")

        results = schedule_wave(wave_jobs, dry_run)
        launched = sum(results.values())
        log.info(f"Wave {wave_num}: {launched}/{len(wave_jobs)} launched")

        success = wait_for_wave(
            wave_jobs, vm_specs, poll_interval_s, max_wait_hours, dry_run,
        )
        if not success:
            log.warning(f"Wave {wave_num} did not fully complete")
            all_success = False

    return all_success


def phase_monitor(all_jobs: List[Job], poll_interval_s: int = 60) -> None:
    log.info("Live monitor (Ctrl+C to exit)...")
    try:
        while True:
            print("\033[2J\033[H")
            print_status_table(all_jobs)
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def phase_cleanup(
    vm_specs: List[VMSpec],
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    if not force and not dry_run:
        answer = input(f"Delete ALL {len(vm_specs)} v6e VMs? [y/N]: ")
        if answer.lower() != "y":
            log.info("Cleanup cancelled.")
            return False

    log.info(f"Deleting {len(vm_specs)} v6e VMs...")
    results: Dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(delete_vm, spec, dry_run): spec
            for spec in vm_specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            results[spec.vm_name] = future.result()

    success = sum(results.values())
    log.info(f"Cleanup: {success}/{len(vm_specs)} deleted")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MBPS TPU v6e Experiment Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/orchestrate_v6e.py --phase all --dry_run
  python scripts/orchestrate_v6e.py --phase all
  python scripts/orchestrate_v6e.py --phase create
  python scripts/orchestrate_v6e.py --phase setup
  python scripts/orchestrate_v6e.py --phase launch --waves 1
  python scripts/orchestrate_v6e.py --phase status
  python scripts/orchestrate_v6e.py --phase monitor
  python scripts/orchestrate_v6e.py --phase cleanup --force
""",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "create", "setup", "launch", "status", "monitor", "cleanup"],
        default="all",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--waves", type=int, nargs="*", default=None)
    parser.add_argument("--poll_interval", type=int, default=300)
    parser.add_argument("--max_wait_hours", type=float, default=96)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(
            f"logs/orchestrate_v6e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception:
        pass

    vm_specs = generate_vm_specs()
    all_jobs = build_job_matrix()

    log.info(f"MBPS v6e Orchestrator: {len(all_jobs)} jobs across {len(vm_specs)} VMs")
    log.info(f"  EU ({ZONE_EU}): 8 VMs → Cityscapes")
    log.info(f"  US ({ZONE_US}): 8 VMs → COCO")
    if args.dry_run:
        log.info("DRY RUN MODE")

    for wn in [1, 2, 3]:
        wj = get_wave_jobs(all_jobs, wn)
        log.info(f"  Wave {wn}: {len(wj)} jobs")

    if args.phase == "all":
        log.info("\n=== PHASE: CREATE VMs ===")
        if not phase_create(vm_specs, args.dry_run):
            log.warning("Some VMs failed to create. Continuing with available.")

        log.info("\n=== PHASE: SETUP VMs ===")
        if not phase_setup(vm_specs, args.dry_run):
            log.warning("Some VMs failed setup. Continuing with available.")

        log.info("\n=== PHASE: LAUNCH EXPERIMENTS ===")
        phase_launch(
            all_jobs, vm_specs,
            waves=args.waves,
            poll_interval_s=args.poll_interval,
            max_wait_hours=args.max_wait_hours,
            dry_run=args.dry_run,
        )

        log.info("\n=== PHASE: FINAL STATUS ===")
        print_status_table(all_jobs, args.dry_run)

        # Auto-cleanup after all waves complete
        if not args.dry_run:
            log.info("\nAll waves complete. Cleaning up v6e VMs...")
            phase_cleanup(vm_specs, force=True, dry_run=False)
            log.info("All v6e VMs deleted. Checkpoints saved on GCS.")

    elif args.phase == "create":
        phase_create(vm_specs, args.dry_run)

    elif args.phase == "setup":
        phase_setup(vm_specs, args.dry_run)

    elif args.phase == "launch":
        phase_launch(
            all_jobs, vm_specs,
            waves=args.waves,
            poll_interval_s=args.poll_interval,
            max_wait_hours=args.max_wait_hours,
            dry_run=args.dry_run,
        )

    elif args.phase == "status":
        print_status_table(all_jobs, args.dry_run)

    elif args.phase == "monitor":
        phase_monitor(all_jobs, poll_interval_s=args.poll_interval)

    elif args.phase == "cleanup":
        phase_cleanup(vm_specs, force=args.force, dry_run=args.dry_run)

    log.info("v6e Orchestrator finished.")


if __name__ == "__main__":
    main()
