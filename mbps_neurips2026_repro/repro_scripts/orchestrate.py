"""MBPS TPU Experiment Orchestrator.

Single-command orchestration of 36 training experiments across 16 TPU VMs.

Usage:
    python scripts/orchestrate.py --phase all                # Smoke test + full pipeline
    python scripts/orchestrate.py --phase all --skip_smoke   # Full pipeline (no smoke)
    python scripts/orchestrate.py --phase all --dry_run      # Print commands only
    python scripts/orchestrate.py --phase smoke              # Smoke test only (2 VMs)
    python scripts/orchestrate.py --phase smoke --dry_run    # Print smoke commands only
    python scripts/orchestrate.py --phase create             # Create 16 VMs
    python scripts/orchestrate.py --phase setup              # Install deps + sync code
    python scripts/orchestrate.py --phase launch             # Launch all 3 waves
    python scripts/orchestrate.py --phase launch --waves 1   # Wave 1 only
    python scripts/orchestrate.py --phase monitor            # Live status dashboard
    python scripts/orchestrate.py --phase status             # One-shot status
    python scripts/orchestrate.py --phase cleanup --force    # Delete all VMs
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT = "unsupervised-panoptic-segment"
ZONE_V4 = "us-central2-b"
ZONE_V5E = "us-central1-a"
BUCKET = "mbps-panoptic"
LOCAL_CODE = "/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
SOFTWARE_VERSION = "tpu-ubuntu2204-base"
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

SMOKE_TEST_TARGET_EPOCHS = 7   # 6 training + 1 self-training round
SMOKE_TEST_TIMEOUT_MINUTES = 45
SMOKE_TEST_POLL_INTERVAL_S = 30

RSYNC_EXCLUDES = [
    ".claude", ".git", "__pycache__", "*.pyc", ".DS_Store", "refs/",
    "logs/", "*.log", "checkpoints/",
]

log = logging.getLogger("orchestrate")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VMSpec:
    """TPU VM specification."""
    vm_name: str
    zone: str
    accelerator_type: str
    spot: bool
    disk_size_gb: int = DISK_SIZE_GB


@dataclass
class Job:
    """Training job specification."""
    job_id: str
    dataset: str          # "cityscapes" or "coco"
    experiment_type: str  # "full" or ablation name
    seed: int
    config: str           # e.g. "cityscapes_gcs.yaml"
    ablation: Optional[str]  # e.g. "no_mamba" or None
    vm_name: str
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


# ---------------------------------------------------------------------------
# VM + Job Matrix Generation
# ---------------------------------------------------------------------------

def generate_vm_specs() -> List[VMSpec]:
    """Generate 16 VM specifications (8 v4-8 + 8 v5e-8).

    Quota (TRC grant):
      - 32 on-demand v4 chips = 4x v4-8 on-demand (us-central2-b)
      - 32 spot v4 chips      = 4x v4-8 spot      (us-central2-b)
      - 64 spot v5e chips      = 8x v5e-8 spot     (us-central1-a)
    """
    specs: List[VMSpec] = []
    for i in range(8):
        specs.append(VMSpec(
            vm_name=f"mbps-v4-{i}",
            zone=ZONE_V4,
            accelerator_type="v4-8",
            spot=(i >= 4),  # v4-0..3 on-demand, v4-4..7 spot
        ))
    for i in range(8):
        specs.append(VMSpec(
            vm_name=f"mbps-v5e-{i}",
            zone=ZONE_V5E,
            accelerator_type="v5litepod-8",
            spot=True,  # All v5e are spot (no on-demand quota)
        ))
    return specs


def generate_smoke_vm_specs() -> List[VMSpec]:
    """Generate 2 VM specs for smoke testing (1 v4-8 + 1 v5e-8)."""
    return [
        VMSpec(
            vm_name="mbps-v4-0",
            zone=ZONE_V4,
            accelerator_type="v4-8",
            spot=False,  # On-demand v4 (have 32 on-demand chips)
        ),
        VMSpec(
            vm_name="mbps-v5e-0",
            zone=ZONE_V5E,
            accelerator_type="v5litepod-8",
            spot=True,  # Spot only (no on-demand v5e quota)
        ),
    ]


def build_smoke_jobs() -> List[Job]:
    """Build 2 smoke test jobs: Cityscapes on v4, COCO on v5e."""
    return [
        Job(
            job_id="smoke_cityscapes_v4",
            dataset="cityscapes",
            experiment_type="smoke",
            seed=42,
            config="proxy_cityscapes_gcs.yaml",
            ablation=None,
            vm_name="mbps-v4-0",
            wave=0,
        ),
        Job(
            job_id="smoke_coco_v5e",
            dataset="coco",
            experiment_type="smoke",
            seed=42,
            config="proxy_coco_gcs.yaml",
            ablation=None,
            vm_name="mbps-v5e-0",
            wave=0,
        ),
    ]


def build_job_matrix() -> List[Job]:
    """Build the complete 36-job matrix across 3 waves."""
    jobs: List[Job] = []

    # Helper to create a job
    def _job(
        dataset: str, exp_type: str, seed: int,
        vm_name: str, wave: int,
    ) -> Job:
        config = "cityscapes_gcs.yaml" if dataset == "cityscapes" else "coco_stuff27_gcs.yaml"
        ablation = None if exp_type == "full" else exp_type
        jid = f"{dataset}_{exp_type}_seed{seed}"
        return Job(jid, dataset, exp_type, seed, config, ablation, vm_name, wave)

    # ── Wave 1: 16 jobs ──
    # Cityscapes: 3 full + 5 ablations (seed 42)
    for i, seed in enumerate(SEEDS):
        jobs.append(_job("cityscapes", "full", seed, f"mbps-v4-{i}", 1))
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("cityscapes", abl, 42, f"mbps-v4-{i + 3}", 1))
    # COCO: 3 full + 5 ablations (seed 42)
    for i, seed in enumerate(SEEDS):
        jobs.append(_job("coco", "full", seed, f"mbps-v5e-{i}", 1))
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("coco", abl, 42, f"mbps-v5e-{i + 3}", 1))

    # ── Wave 2: 16 jobs ──
    # Cityscapes: 5 ablations (seed 123) + first 3 ablations (seed 456)
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("cityscapes", abl, 123, f"mbps-v4-{i}", 2))
    for i, abl in enumerate(ABLATIONS[:3]):
        jobs.append(_job("cityscapes", abl, 456, f"mbps-v4-{i + 5}", 2))
    # COCO: 5 ablations (seed 123) + first 3 ablations (seed 456)
    for i, abl in enumerate(ABLATIONS):
        jobs.append(_job("coco", abl, 123, f"mbps-v5e-{i}", 2))
    for i, abl in enumerate(ABLATIONS[:3]):
        jobs.append(_job("coco", abl, 456, f"mbps-v5e-{i + 5}", 2))

    # ── Wave 3: 4 jobs ──
    # Last 2 Cityscapes ablations (seed 456) + last 2 COCO ablations (seed 456)
    for i, abl in enumerate(ABLATIONS[3:]):
        jobs.append(_job("cityscapes", abl, 456, f"mbps-v4-{i}", 3))
    for i, abl in enumerate(ABLATIONS[3:]):
        jobs.append(_job("coco", abl, 456, f"mbps-v5e-{i}", 3))

    return jobs


def get_wave_jobs(jobs: List[Job], wave: int) -> List[Job]:
    """Filter jobs by wave number."""
    return [j for j in jobs if j.wave == wave]


def get_vm_zone(vm_name: str) -> str:
    """Get zone for a VM name."""
    return ZONE_V4 if "v4" in vm_name else ZONE_V5E


# ---------------------------------------------------------------------------
# SSH / GCloud Utilities
# ---------------------------------------------------------------------------

def run_command(
    cmd: List[str],
    timeout: int = 600,
    dry_run: bool = False,
    label: str = "",
) -> Tuple[int, str, str]:
    """Run a shell command with timeout.

    Args:
        cmd: Command as list of strings.
        timeout: Timeout in seconds.
        dry_run: If True, print command and return success.
        label: Label for logging.

    Returns:
        (returncode, stdout, stderr)
    """
    cmd_str = " ".join(cmd)
    if dry_run:
        log.info(f"[DRY RUN] {label}: {cmd_str}")
        return 0, "", ""

    log.debug(f"Running: {cmd_str}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            log.debug(f"Command failed (rc={proc.returncode}): {proc.stderr[:500]}")
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        log.warning(f"Command timed out ({timeout}s): {label}")
        return -1, "", "TIMEOUT"
    except Exception as e:
        log.error(f"Command error: {e}")
        return -1, "", str(e)


def gcloud_ssh(
    vm_name: str,
    zone: str,
    command: str,
    timeout: int = 600,
    retries: int = 3,
    dry_run: bool = False,
) -> Tuple[int, str, str]:
    """SSH into TPU VM via gcloud with retry.

    Args:
        vm_name: VM name.
        zone: GCP zone.
        command: Shell command to execute on VM.
        timeout: Timeout per attempt.
        retries: Number of retries.
        dry_run: Print command only.

    Returns:
        (returncode, stdout, stderr)
    """
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", vm_name,
        f"--zone={zone}",
        f"--project={PROJECT}",
        f"--command={command}",
        "--strict-host-key-checking=no",
    ]

    for attempt in range(retries):
        rc, out, err = run_command(
            cmd, timeout=timeout, dry_run=dry_run, label=f"SSH {vm_name}",
        )
        if dry_run or rc == 0:
            return rc, out, err
        if "not found" in err.lower() or "does not exist" in err.lower():
            return rc, out, err  # Don't retry if VM doesn't exist
        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            log.warning(f"SSH to {vm_name} failed (attempt {attempt + 1}), retrying in {wait}s...")
            time.sleep(wait)

    return rc, out, err


def check_vm_state(
    vm_name: str, zone: str, dry_run: bool = False,
) -> str:
    """Check TPU VM state.

    Returns:
        "READY", "CREATING", "PREEMPTED", "STOPPING", or "NOT_FOUND".
    """
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
    """Get external IP of a TPU VM."""
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
    """Create a single TPU VM."""
    # Check if already exists
    state = check_vm_state(spec.vm_name, spec.zone, dry_run)
    if state == "READY":
        log.info(f"VM {spec.vm_name} already exists (READY), skipping creation")
        return True

    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "create", spec.vm_name,
        f"--zone={spec.zone}",
        f"--accelerator-type={spec.accelerator_type}",
        f"--version={SOFTWARE_VERSION}",
        f"--boot-disk-size={spec.disk_size_gb}GB",
        f"--project={PROJECT}",
    ]
    if spec.spot:
        cmd.append("--spot")

    label = f"Create {spec.vm_name} ({spec.accelerator_type}"
    label += ", spot)" if spec.spot else ")"

    rc, out, err = run_command(cmd, timeout=600, dry_run=dry_run, label=label)
    if rc == 0 or dry_run:
        log.info(f"Created VM: {spec.vm_name}")
        return True

    log.error(f"Failed to create {spec.vm_name}: {err[:300]}")
    return False


def setup_vm(spec: VMSpec, dry_run: bool = False) -> bool:
    """Setup VM: sync code + install deps + verify JAX.

    Args:
        spec: VM specification.
        dry_run: Print commands only.

    Returns:
        True if setup succeeds.
    """
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
        "-e", f"ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no",
        f"{LOCAL_CODE}/",
        f"qbit-glitch@{ip}:~/mbps_panoptic_segmentation/",
    ]
    rc, _, err = run_command(rsync_cmd, timeout=300, dry_run=dry_run, label=f"Rsync → {vm}")
    if rc != 0 and not dry_run:
        log.error(f"Rsync to {vm} failed: {err[:300]}")
        return False
    log.info(f"Code synced to {vm}")

    # Step 2: Install dependencies
    rc, out, err = gcloud_ssh(
        vm, zone,
        "bash ~/mbps_panoptic_segmentation/scripts/install_deps.sh",
        timeout=600, dry_run=dry_run,
    )
    if rc != 0 and not dry_run:
        log.error(f"Dep install on {vm} failed: {err[:300]}")
        return False
    log.info(f"Dependencies installed on {vm}")

    # Step 3: Verify JAX
    rc, out, err = gcloud_ssh(
        vm, zone,
        "python3 -c \"import jax; print(f'JAX devices: {jax.device_count()}')\"",
        timeout=120, dry_run=dry_run,
    )
    if rc != 0 and not dry_run:
        log.error(f"JAX verification failed on {vm}: {err[:300]}")
        return False
    if not dry_run:
        log.info(f"JAX verified on {vm}: {out.strip()}")

    return True


def delete_vm(spec: VMSpec, dry_run: bool = False) -> bool:
    """Delete a TPU VM."""
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "delete", spec.vm_name,
        f"--zone={spec.zone}", f"--project={PROJECT}", "--quiet",
    ]
    rc, _, err = run_command(cmd, timeout=300, dry_run=dry_run, label=f"Delete {spec.vm_name}")
    if rc == 0 or dry_run:
        log.info(f"Deleted VM: {spec.vm_name}")
        return True
    log.warning(f"Failed to delete {spec.vm_name}: {err[:200]}")
    return False


# ---------------------------------------------------------------------------
# Job Execution
# ---------------------------------------------------------------------------

def launch_job(job: Job, dry_run: bool = False) -> bool:
    """Launch a training job on its assigned VM via nohup."""
    zone = get_vm_zone(job.vm_name)
    rc, _, err = gcloud_ssh(
        job.vm_name, zone, job.train_command,
        timeout=60, dry_run=dry_run,
    )
    if rc == 0 or dry_run:
        log.info(f"Launched: {job.job_id} on {job.vm_name}")
        return True
    log.error(f"Failed to launch {job.job_id} on {job.vm_name}: {err[:300]}")
    return False


def check_job_progress(job: Job, dry_run: bool = False) -> Optional[int]:
    """Check latest checkpoint epoch for a job on GCS.

    Returns:
        Latest epoch number or None if no checkpoints found.
    """
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
    """Check if job has reached target epochs."""
    epoch = check_job_progress(job, dry_run)
    return epoch is not None and epoch >= TARGET_EPOCHS


def get_latest_checkpoint(job: Job) -> Optional[str]:
    """Get full GCS path to latest checkpoint for resume."""
    epoch = check_job_progress(job)
    if epoch is None:
        return None
    return f"{job.gcs_checkpoint_dir}/checkpoint_epoch_{epoch:04d}"


# ---------------------------------------------------------------------------
# Smoke Test Validation
# ---------------------------------------------------------------------------

def validate_smoke_job(
    job: Job,
    timeout_minutes: int = SMOKE_TEST_TIMEOUT_MINUTES,
    poll_interval_s: int = SMOKE_TEST_POLL_INTERVAL_S,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Wait for a smoke test job to complete and validate.

    Validation criteria:
      1. Training process has finished running.
      2. At least 1 checkpoint exists on GCS.
      3. Final checkpoint epoch >= SMOKE_TEST_TARGET_EPOCHS.

    Returns:
        (passed, reason) tuple.
    """
    if dry_run:
        return True, "DRY_RUN"

    zone = get_vm_zone(job.vm_name)
    deadline = time.time() + timeout_minutes * 60

    log.info(f"Waiting for smoke job: {job.job_id} (timeout {timeout_minutes}m)")

    while time.time() < deadline:
        # Check if training process is still running
        rc, out, _ = gcloud_ssh(
            job.vm_name, zone,
            f"pgrep -f 'train.py.*{job.config}' > /dev/null 2>&1 && echo RUNNING || echo STOPPED",
            timeout=30, retries=1, dry_run=False,
        )
        process_status = out.strip() if rc == 0 else "UNKNOWN"

        # Check checkpoint progress
        epoch = check_job_progress(job)

        if process_status == "STOPPED":
            if epoch is not None and epoch >= SMOKE_TEST_TARGET_EPOCHS:
                return True, f"Completed at epoch {epoch}"
            elif epoch is not None:
                return False, (
                    f"Process exited but only reached epoch {epoch}, "
                    f"expected >= {SMOKE_TEST_TARGET_EPOCHS}"
                )
            else:
                # Process stopped with no checkpoints -- fetch log tail
                rc2, out2, _ = gcloud_ssh(
                    job.vm_name, zone,
                    f"tail -20 ~/mbps_panoptic_segmentation/logs/{job.job_id}.log 2>/dev/null || echo NO_LOG",
                    timeout=30, retries=1, dry_run=False,
                )
                return False, f"Process exited with no checkpoints. Log tail:\n{out2.strip()[:500]}"

        if epoch is not None:
            log.info(f"  {job.job_id}: epoch {epoch}/{SMOKE_TEST_TARGET_EPOCHS} (running)")

        time.sleep(poll_interval_s)

    return False, f"Timed out after {timeout_minutes} minutes"


# ---------------------------------------------------------------------------
# Preemption Handling
# ---------------------------------------------------------------------------

def handle_preemption(
    job: Job,
    spec: VMSpec,
    dry_run: bool = False,
) -> bool:
    """Handle a preempted spot VM: recreate, setup, resume.

    Args:
        job: The job that was interrupted.
        spec: VM spec for recreation.
        dry_run: Print commands only.

    Returns:
        True if recovery and relaunch succeed.
    """
    log.warning(f"Handling preemption: {job.vm_name} ({job.job_id})")

    # Delete the preempted VM
    delete_vm(spec, dry_run)
    time.sleep(5)  # Brief pause before recreation

    # Recreate
    if not create_vm(spec, dry_run):
        log.error(f"Failed to recreate {spec.vm_name}")
        return False

    if not dry_run:
        # Wait for VM to be ready
        for _ in range(30):
            state = check_vm_state(spec.vm_name, spec.zone)
            if state == "READY":
                break
            time.sleep(10)
        else:
            log.error(f"VM {spec.vm_name} not ready after recreation")
            return False

    # Re-setup
    if not setup_vm(spec, dry_run):
        log.error(f"Failed to re-setup {spec.vm_name}")
        return False

    # Find latest checkpoint for resume
    latest_ckpt = get_latest_checkpoint(job) if not dry_run else None

    # Build resume command
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

    zone = get_vm_zone(job.vm_name)
    rc, _, err = gcloud_ssh(
        job.vm_name, zone, " ".join(resume_parts),
        timeout=60, dry_run=dry_run,
    )
    if rc == 0 or dry_run:
        log.info(f"Resumed {job.job_id} on {job.vm_name} (from epoch {check_job_progress(job) or 0})")
        return True

    log.error(f"Failed to resume {job.job_id}: {err[:300]}")
    return False


# ---------------------------------------------------------------------------
# Wave Orchestration
# ---------------------------------------------------------------------------

def schedule_wave(
    wave_jobs: List[Job],
    dry_run: bool = False,
) -> Dict[str, bool]:
    """Launch all jobs in a wave in parallel.

    Returns:
        Dict mapping job_id -> success.
    """
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
    """Poll until all jobs in wave complete or timeout.

    Args:
        wave_jobs: Jobs to monitor.
        vm_specs: All VM specs (for preemption recovery).
        poll_interval_s: Poll interval.
        max_wait_hours: Maximum wait time.
        dry_run: Skip actual polling.

    Returns:
        True if all jobs completed.
    """
    if dry_run:
        log.info(f"[DRY RUN] Would poll {len(wave_jobs)} jobs every {poll_interval_s}s")
        return True

    spec_map = {s.vm_name: s for s in vm_specs}
    completed: Set[str] = set()
    failed: Set[str] = set()
    preemption_counts: Dict[str, int] = {j.job_id: 0 for j in wave_jobs}
    stale_counts: Dict[str, int] = {j.job_id: 0 for j in wave_jobs}
    last_epochs: Dict[str, Optional[int]] = {j.job_id: None for j in wave_jobs}
    deadline = time.time() + max_wait_hours * 3600

    wave_num = wave_jobs[0].wave if wave_jobs else 0
    log.info(f"Monitoring wave {wave_num}: {len(wave_jobs)} jobs, poll every {poll_interval_s}s")

    while time.time() < deadline:
        pending = [j for j in wave_jobs if j.job_id not in completed and j.job_id not in failed]
        if not pending:
            break

        for job in pending:
            # Check VM state
            zone = get_vm_zone(job.vm_name)
            state = check_vm_state(job.vm_name, zone)

            if state == "PREEMPTED":
                preemption_counts[job.job_id] += 1
                if preemption_counts[job.job_id] > 5:
                    log.error(f"{job.job_id}: Too many preemptions (5), marking FAILED")
                    failed.add(job.job_id)
                    continue
                spec = spec_map.get(job.vm_name)
                if spec:
                    handle_preemption(job, spec)
                continue

            # Check progress
            epoch = check_job_progress(job)
            if epoch is not None and epoch >= TARGET_EPOCHS:
                completed.add(job.job_id)
                log.info(f"COMPLETE: {job.job_id} (epoch {epoch})")
                continue

            # Detect stale jobs (no progress in 2 cycles)
            if epoch == last_epochs[job.job_id]:
                stale_counts[job.job_id] += 1
            else:
                stale_counts[job.job_id] = 0
            last_epochs[job.job_id] = epoch

        # Print summary
        n_total = len(wave_jobs)
        n_done = len(completed)
        n_fail = len(failed)
        n_run = n_total - n_done - n_fail
        log.info(
            f"Wave {wave_num}: {n_done}/{n_total} complete, "
            f"{n_run} running, {n_fail} failed"
        )

        if n_done + n_fail >= n_total:
            break

        time.sleep(poll_interval_s)

    success = len(failed) == 0 and len(completed) == len(wave_jobs)
    if not success:
        log.warning(
            f"Wave {wave_num} ended: {len(completed)} complete, "
            f"{len(failed)} failed out of {len(wave_jobs)}"
        )
    return success


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def launch_coordinator(
    vm_name: str,
    config: str,
    experiment: str,
    dry_run: bool = False,
) -> bool:
    """Launch coordinate.py on a VM."""
    zone = get_vm_zone(vm_name)
    cmd = (
        f"cd ~/mbps_panoptic_segmentation && mkdir -p logs && "
        f"nohup python3 scripts/coordinate.py "
        f"--config configs/{config} --experiment {experiment} "
        f"> logs/coordinator_{experiment}.log 2>&1 &"
    )
    rc, _, err = gcloud_ssh(vm_name, zone, cmd, timeout=60, dry_run=dry_run)
    if rc == 0 or dry_run:
        log.info(f"Coordinator launched on {vm_name} for {experiment}")
        return True
    log.error(f"Failed to launch coordinator: {err[:200]}")
    return False


# ---------------------------------------------------------------------------
# Status Display
# ---------------------------------------------------------------------------

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_status_table(
    all_jobs: List[Job],
    vm_specs: List[VMSpec],
    dry_run: bool = False,
) -> None:
    """Print formatted status table for all jobs."""
    header = (
        f"{BOLD}{'Wave':>4}  {'VM':<12} {'Zone':<12} "
        f"{'Experiment':<35} {'Seed':>4}  {'Status':<10} {'Epoch':>5}{RESET}"
    )
    print(f"\n{'=' * 90}")
    print(f"{BOLD}  MBPS Experiment Status  ({datetime.now().strftime('%H:%M:%S')}){RESET}")
    print(f"{'=' * 90}")
    print(header)
    print("-" * 90)

    for wave_num in [1, 2, 3]:
        wave_jobs = get_wave_jobs(all_jobs, wave_num)
        if not wave_jobs:
            continue

        for job in wave_jobs:
            zone_short = "central2" if "v4" in job.vm_name else "central1"

            if dry_run:
                status = f"{DIM}  PEND{RESET}"
                epoch_str = "  -"
            else:
                vm_state = check_vm_state(job.vm_name, get_vm_zone(job.vm_name))
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

            exp_short = job.experiment_name[:35]
            print(
                f"{job.wave:>4}  {job.vm_name:<12} {zone_short:<12} "
                f"{exp_short:<35} {job.seed:>4}  {status:<10} {epoch_str}"
            )

        if wave_num < 3:
            print(f"{DIM}{'·' * 90}{RESET}")

    print(f"{'=' * 90}")

    # Summary counts
    if not dry_run:
        total = len(all_jobs)
        done = sum(1 for j in all_jobs if is_job_complete(j))
        print(f"\n  Total: {total} | Complete: {GREEN}{done}{RESET} | Remaining: {total - done}\n")


# ---------------------------------------------------------------------------
# Phase Executors
# ---------------------------------------------------------------------------

def phase_create(vm_specs: List[VMSpec], dry_run: bool = False) -> bool:
    """Create all VMs in parallel."""
    log.info(f"Creating {len(vm_specs)} VMs...")
    results: Dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(create_vm, spec, dry_run): spec
            for spec in vm_specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            results[spec.vm_name] = future.result()

    success = sum(results.values())
    log.info(f"VM creation: {success}/{len(vm_specs)} successful")
    return success == len(vm_specs)


def phase_setup(vm_specs: List[VMSpec], dry_run: bool = False) -> bool:
    """Setup all VMs in parallel (deps + code sync)."""
    log.info(f"Setting up {len(vm_specs)} VMs...")
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
                log.error(f"Setup failed for {spec.vm_name}: {e}")
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
    """Launch experiments in waves.

    Args:
        all_jobs: All 36 jobs.
        vm_specs: All VM specs.
        waves: Which waves to run (default: all).
        poll_interval_s: Poll interval for monitoring.
        max_wait_hours: Max wait per wave.
        dry_run: Print commands only.

    Returns:
        True if all waves completed successfully.
    """
    target_waves = waves or [1, 2, 3]
    all_success = True

    for wave_num in target_waves:
        wave_jobs = get_wave_jobs(all_jobs, wave_num)
        if not wave_jobs:
            continue

        log.info(f"\n{'=' * 60}")
        log.info(f"WAVE {wave_num}: Launching {len(wave_jobs)} jobs")
        log.info(f"{'=' * 60}")

        # Launch all jobs in this wave
        results = schedule_wave(wave_jobs, dry_run)
        launched = sum(results.values())
        log.info(f"Wave {wave_num}: {launched}/{len(wave_jobs)} jobs launched")

        # Launch coordinator alongside wave 1
        if wave_num == 1:
            launch_coordinator(
                "mbps-v4-0", "cityscapes_gcs.yaml", "cityscapes_full", dry_run,
            )

        # Wait for wave to complete
        success = wait_for_wave(
            wave_jobs, vm_specs, poll_interval_s, max_wait_hours, dry_run,
        )
        if not success:
            log.warning(f"Wave {wave_num} did not fully complete")
            all_success = False

    return all_success


def phase_monitor(
    all_jobs: List[Job],
    vm_specs: List[VMSpec],
    poll_interval_s: int = 60,
) -> None:
    """Live monitoring dashboard. Press Ctrl+C to exit."""
    log.info("Starting live monitor (Ctrl+C to exit)...")
    try:
        while True:
            print("\033[2J\033[H")  # Clear screen
            print_status_table(all_jobs, vm_specs)
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def phase_status(
    all_jobs: List[Job],
    vm_specs: List[VMSpec],
    dry_run: bool = False,
) -> None:
    """One-shot status check."""
    print_status_table(all_jobs, vm_specs, dry_run)


def phase_cleanup(
    vm_specs: List[VMSpec],
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Delete all VMs."""
    if not force and not dry_run:
        answer = input(
            f"Delete ALL {len(vm_specs)} VMs? This cannot be undone. [y/N]: "
        )
        if answer.lower() != "y":
            log.info("Cleanup cancelled.")
            return False

    log.info(f"Deleting {len(vm_specs)} VMs...")
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
# Smoke Test Phase
# ---------------------------------------------------------------------------

def phase_smoke_test(dry_run: bool = False) -> bool:
    """Run smoke test: create 2 VMs, run proxy jobs, validate.

    Creates 1 v4-8 + 1 v5e-8, runs proxy training (6 epochs + 1 self-training),
    validates checkpoints appear on GCS. VMs are kept for reuse in the full run.

    Returns:
        True if both smoke jobs pass validation.
    """
    smoke_specs = generate_smoke_vm_specs()
    smoke_jobs = build_smoke_jobs()

    log.info("=" * 60)
    log.info("SMOKE TEST: Validating TPU pipeline on 2 VMs")
    log.info("=" * 60)
    log.info(f"  VMs:     {', '.join(s.vm_name for s in smoke_specs)}")
    log.info(f"  Jobs:    {', '.join(j.job_id for j in smoke_jobs)}")
    log.info(f"  Target:  {SMOKE_TEST_TARGET_EPOCHS} epochs per job")
    log.info(f"  Timeout: {SMOKE_TEST_TIMEOUT_MINUTES} minutes per job")

    # Step 1: Create smoke test VMs
    log.info("\n--- Smoke: Creating VMs ---")
    if not phase_create(smoke_specs, dry_run):
        log.error("Smoke test: VM creation failed")
        return False

    # Wait for VMs to be READY
    if not dry_run:
        for spec in smoke_specs:
            for _ in range(30):
                state = check_vm_state(spec.vm_name, spec.zone)
                if state == "READY":
                    break
                time.sleep(10)
            else:
                log.error(f"Smoke test: {spec.vm_name} never became READY")
                return False

    # Step 2: Setup VMs
    log.info("\n--- Smoke: Setting up VMs ---")
    if not phase_setup(smoke_specs, dry_run):
        log.error("Smoke test: VM setup failed")
        return False

    # Step 3: Launch smoke jobs
    log.info("\n--- Smoke: Launching jobs ---")
    results = schedule_wave(smoke_jobs, dry_run)
    if not all(results.values()):
        failed = [jid for jid, ok in results.items() if not ok]
        log.error(f"Smoke test: Failed to launch jobs: {failed}")
        return False

    # Step 4: Validate each job
    log.info("\n--- Smoke: Validating jobs ---")
    all_passed = True
    for job in smoke_jobs:
        passed, reason = validate_smoke_job(job, dry_run=dry_run)
        if passed:
            log.info(f"  PASS: {job.job_id} -- {reason}")
        else:
            log.error(f"  FAIL: {job.job_id} -- {reason}")
            all_passed = False

    # Step 5: Report
    if all_passed:
        log.info("\n" + "=" * 60)
        log.info("SMOKE TEST PASSED -- pipeline validated")
        log.info("=" * 60)
    else:
        log.error("\n" + "=" * 60)
        log.error("SMOKE TEST FAILED -- aborting")
        log.error("=" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MBPS TPU Experiment Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/orchestrate.py --phase all --dry_run
  python scripts/orchestrate.py --phase all --skip_smoke
  python scripts/orchestrate.py --phase smoke
  python scripts/orchestrate.py --phase smoke --dry_run
  python scripts/orchestrate.py --phase create
  python scripts/orchestrate.py --phase setup
  python scripts/orchestrate.py --phase launch --waves 1
  python scripts/orchestrate.py --phase monitor
  python scripts/orchestrate.py --phase status
  python scripts/orchestrate.py --phase cleanup --force
""",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "smoke", "create", "setup", "launch", "monitor", "status", "cleanup"],
        default="all",
        help="Execution phase (default: all)",
    )
    parser.add_argument(
        "--skip_smoke", action="store_true",
        help="Skip smoke test in --phase all",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--waves", type=int, nargs="*", default=None,
        help="Specific wave numbers to launch (default: all)",
    )
    parser.add_argument(
        "--poll_interval", type=int, default=300,
        help="Job poll interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--max_wait_hours", type=float, default=96,
        help="Max wait per wave in hours (default: 96)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Also log to file
    log_file = f"logs/orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
        import os
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception:
        pass  # Non-fatal if log file can't be created

    # Generate VM specs and job matrix
    vm_specs = generate_vm_specs()
    all_jobs = build_job_matrix()

    log.info(f"MBPS Orchestrator: {len(all_jobs)} jobs across {len(vm_specs)} VMs")
    if args.dry_run:
        log.info("DRY RUN MODE — no commands will be executed")

    # Print job matrix summary
    for wave_num in [1, 2, 3]:
        wave_jobs = get_wave_jobs(all_jobs, wave_num)
        log.info(f"  Wave {wave_num}: {len(wave_jobs)} jobs")

    # Execute phase
    if args.phase == "all":
        # Smoke test gate
        if not args.skip_smoke:
            log.info("\n=== PHASE: SMOKE TEST ===")
            smoke_passed = phase_smoke_test(dry_run=args.dry_run)
            if not smoke_passed:
                log.error(
                    "Smoke test failed. Fix issues and retry, or use "
                    "--skip_smoke to bypass."
                )
                sys.exit(1)
            log.info("Smoke test passed. Proceeding to full run.\n")
        else:
            log.info("Smoke test skipped (--skip_smoke).")

        # Full run
        log.info("\n=== PHASE: CREATE VMs ===")
        if not phase_create(vm_specs, args.dry_run):
            log.error("VM creation had failures. Continuing with available VMs.")

        log.info("\n=== PHASE: SETUP VMs ===")
        if not phase_setup(vm_specs, args.dry_run):
            log.error("VM setup had failures. Continuing with available VMs.")

        log.info("\n=== PHASE: LAUNCH EXPERIMENTS ===")
        phase_launch(
            all_jobs, vm_specs,
            waves=args.waves,
            poll_interval_s=args.poll_interval,
            max_wait_hours=args.max_wait_hours,
            dry_run=args.dry_run,
        )

        log.info("\n=== PHASE: FINAL STATUS ===")
        phase_status(all_jobs, vm_specs, args.dry_run)

        if not args.dry_run:
            log.info("\nAll waves complete. Run --phase cleanup when ready to delete VMs.")

    elif args.phase == "smoke":
        success = phase_smoke_test(dry_run=args.dry_run)
        if not success:
            sys.exit(1)

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

    elif args.phase == "monitor":
        phase_monitor(all_jobs, vm_specs, poll_interval_s=args.poll_interval)

    elif args.phase == "status":
        phase_status(all_jobs, vm_specs, args.dry_run)

    elif args.phase == "cleanup":
        phase_cleanup(vm_specs, force=args.force, dry_run=args.dry_run)

    log.info("Orchestrator finished.")


if __name__ == "__main__":
    main()
