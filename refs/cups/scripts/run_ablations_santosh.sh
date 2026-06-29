#!/usr/bin/env bash
# Queue ABL-A → ABL-B → ABL-C sequentially on santosh after the current baseline run.
# Usage: nohup bash run_ablations_santosh.sh > /home/santosh/experiments/ablations_queue.log 2>&1 &
#
# Waits for PID of current training run to exit, then launches each ablation in sequence.
# Each run uses the same DDP config as the baseline (2xGPU, bs1, accum8).

set -euo pipefail

REPO="/home/santosh/mbps_panoptic_segmentation/refs/cups"
LOG_ROOT="/home/santosh/experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 1. Wait for baseline run (PID 30743) ───────────────────────────────────
BASELINE_PID=30743
log "Waiting for baseline PID ${BASELINE_PID} to finish..."
while kill -0 "${BASELINE_PID}" 2>/dev/null; do
    sleep 60
done
log "Baseline finished. Starting ablation queue."

# ── 2. Sync latest configs ─────────────────────────────────────────────────
cd "${REPO}"
git pull --ff-only origin feat/ga-uniap-geometric-pooling 2>&1 || log "WARN: git pull failed, using local configs"

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
CONDA_PREFIX="/home/santosh/anaconda3/envs/cups"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# ── helper: run one ablation ──────────────────────────────────────────────
run_ablation() {
    local name="$1"
    local cfg="$2"
    local logfile="${LOG_ROOT}/${name}/logs/train_${TIMESTAMP}.log"
    mkdir -p "${LOG_ROOT}/${name}/logs"
    log "=== Starting ${name} (config: ${cfg}) ==="
    log "Log: ${logfile}"
    python -u train_eomt.py \
        --experiment_config_file "configs/${cfg}" \
        --disable_wandb \
        > "${logfile}" 2>&1
    log "=== ${name} DONE ==="
}

# ── 3. ABL-A: No DropLoss + No CopyPaste ──────────────────────────────────
run_ablation "stage2_eomt_ablA" \
    "train_cityscapes_eomt_dinov2_causetr_anyup_santosh_ablA.yaml"

# ── 4. ABL-B: Low threshold + half LR + No CopyPaste ─────────────────────
run_ablation "stage2_eomt_ablB" \
    "train_cityscapes_eomt_dinov2_causetr_anyup_santosh_ablB.yaml"

# ── 5. ABL-C: Long training + No DropLoss + half LR + No CopyPaste ────────
run_ablation "stage2_eomt_ablC" \
    "train_cityscapes_eomt_dinov2_causetr_anyup_santosh_ablC.yaml"

log "All ablations complete."
