#!/usr/bin/env bash
# E1 — same-backbone CUPS Stage-1 pseudo-label regeneration on RTX A6000 48GB.
#
# Goal: re-run CUPS Stage-1 (DepthG + RAFT-SMURF + SF2SE3) end-to-end to produce
# the ~2975 train pseudo-label pairs (semantic + instance) that the original CUPS
# paper trained on. Stage-2/3 then trains our DINOv3 ViT-B/16 + Cascade Mask R-CNN
# on these labels — isolating backbone vs pseudo-label-source contribution.
#
# Inputs (downloaded from a public Hugging Face repo via wget — no auth):
#   - depthg.ckpt          (CUPS-released DepthG semantic+depth model, 352 MB)
#   - raft_smurf.pt        (CUPS-released SMURF optical-flow + disparity)
# Inputs assumed already on the A6000 box:
#   - Cityscapes leftImg8bit, rightImg8bit, leftImg8bit_sequence,
#     rightImg8bit_sequence, camera, gtFine.
#
# NOT used: cups.ckpt (that is the AFTER-Stage-3 final model — using it as
# Stage-1 input would be circular). dino_RN50_pretrain_d2_format.pkl is also
# unused for E1 because Stage-2 swaps in DINOv3 ViT-B/16.
#
# Usage:
#   HF_REPO_BASE="https://huggingface.co/<user>/mbps-cups-e1/resolve/main" \
#   CITYSCAPES_ROOT=/data/cityscapes \
#   WORK_DIR=$HOME/mbps_e1 \
#   bash scripts/e1_stage1_pseudolabel_gen_a6000.sh
#
# Resume-friendly: re-running skips already-downloaded checkpoints and
# already-generated pseudo-label pairs.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Configuration (override via env vars)
# ---------------------------------------------------------------------------
HF_REPO_BASE="${HF_REPO_BASE:-https://huggingface.co/qbit-glitch/mbps-cups-e1/resolve/main}"
WORK_DIR="${WORK_DIR:-$HOME/mbps_e1}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:?CITYSCAPES_ROOT must be set (path to Cityscapes root with leftImg8bit_sequence/, rightImg8bit_sequence/, camera/, gtFine/)}"
REPO_BRANCH="${REPO_BRANCH:-implement-dora-adapters}"
REPO_URL="${REPO_URL:-https://github.com/qbit-glitch/unsupervised_panoptic_segmentation.git}"
NUM_WORKERS="${NUM_WORKERS:-6}"
NUM_SUBSPLITS="${NUM_SUBSPLITS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-cups_e1}"
SKIP_DEPS="${SKIP_DEPS:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/cups_pseudo_labels_e1}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/e1_stage1_${TS}.log"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR" "$LOG_DIR"

log() { echo "[E1 $(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "================================================================"
log "E1 Stage-1 Pseudo-Label Regeneration"
log "================================================================"
log "WORK_DIR=$WORK_DIR"
log "OUTPUT_DIR=$OUTPUT_DIR"
log "CITYSCAPES_ROOT=$CITYSCAPES_ROOT"
log "HF_REPO_BASE=$HF_REPO_BASE"
log "GPU_ID=$GPU_ID  NUM_SUBSPLITS=$NUM_SUBSPLITS  NUM_WORKERS=$NUM_WORKERS"
log "----------------------------------------------------------------"

# ---------------------------------------------------------------------------
# 2. Pre-flight checks
# ---------------------------------------------------------------------------
log "Pre-flight checks..."

command -v wget >/dev/null || { log "ERROR: wget not found"; exit 1; }
command -v git  >/dev/null || { log "ERROR: git not found";  exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { log "ERROR: $PYTHON_BIN not found"; exit 1; }

if ! command -v nvidia-smi >/dev/null; then
    log "ERROR: nvidia-smi not found — is this an NVIDIA GPU box?"
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" | head -1)"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_ID" | head -1)"
log "GPU $GPU_ID: $GPU_NAME (${GPU_MEM_MB} MB)"

if (( GPU_MEM_MB < 16000 )); then
    log "ERROR: GPU has only ${GPU_MEM_MB} MB. CUPS Stage-1 needs >=16 GB. Use A6000/A100/V100."
    exit 1
fi

for sub in leftImg8bit_sequence rightImg8bit_sequence camera gtFine; do
    if [[ ! -d "$CITYSCAPES_ROOT/$sub" ]]; then
        log "ERROR: missing $CITYSCAPES_ROOT/$sub — Stage-1 needs stereo+sequence+camera."
        exit 1
    fi
done
log "Cityscapes layout looks complete."

DISK_FREE_GB="$(df -BG "$WORK_DIR" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
log "Free disk in $WORK_DIR: ${DISK_FREE_GB} GB"
if (( DISK_FREE_GB < 30 )); then
    log "WARNING: <30 GB free; pseudo-labels + temp may not fit comfortably."
fi

# ---------------------------------------------------------------------------
# 3. Clone repo + reset to known branch
# ---------------------------------------------------------------------------
REPO_ROOT="$WORK_DIR/unsupervised_panoptic_segmentation"
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    log "Cloning $REPO_URL (branch=$REPO_BRANCH) into $REPO_ROOT ..."
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_ROOT"
else
    log "Repo already cloned at $REPO_ROOT — fetching latest $REPO_BRANCH"
    git -C "$REPO_ROOT" fetch origin "$REPO_BRANCH"
    git -C "$REPO_ROOT" checkout "$REPO_BRANCH"
    git -C "$REPO_ROOT" pull --ff-only origin "$REPO_BRANCH"
fi
COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
log "Repo at commit $COMMIT"

# ---------------------------------------------------------------------------
# 4. Conda env setup (idempotent)
# ---------------------------------------------------------------------------
if [[ "$SKIP_DEPS" != "1" ]]; then
    if command -v conda >/dev/null; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
            log "Creating conda env $CONDA_ENV (Python 3.10)..."
            conda create -y -n "$CONDA_ENV" python=3.10
        fi
        conda activate "$CONDA_ENV"

        log "Installing PyTorch 2.5.1+cu121 + CUPS Stage-1 deps ..."
        pip install --quiet --upgrade pip
        pip install --quiet \
            torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cu121
        pip install --quiet -r "$REPO_ROOT/refs/cups/requirements.txt"
    else
        log "WARNING: conda not found — assuming current Python env already has CUPS deps."
    fi
else
    log "SKIP_DEPS=1 — skipping env setup."
fi

# ---------------------------------------------------------------------------
# 5. Download checkpoints from public HF repo (no auth)
# ---------------------------------------------------------------------------
WEIGHTS_DIR="$REPO_ROOT/weights"
SMURF_DIR="$REPO_ROOT/refs/cups/cups/optical_flow/checkpoints"
mkdir -p "$WEIGHTS_DIR" "$SMURF_DIR"

fetch_if_missing() {
    local url="$1"
    local dest="$2"
    local label="$3"
    if [[ -s "$dest" ]]; then
        log "[skip] $label already at $dest ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    log "[fetch] $label  <-  $url"
    wget -q --show-progress -O "$dest.partial" "$url"
    mv "$dest.partial" "$dest"
    log "[done]  $label  ($(du -h "$dest" | cut -f1))"
}

fetch_if_missing "$HF_REPO_BASE/depthg.ckpt"      "$WEIGHTS_DIR/depthg.ckpt"        "depthg.ckpt"
fetch_if_missing "$HF_REPO_BASE/raft_smurf.pt"    "$SMURF_DIR/raft_smurf.pt"        "raft_smurf.pt"

# ---------------------------------------------------------------------------
# 6. Patch the 3 config drifts CUPS-original vs our prior run
# ---------------------------------------------------------------------------
PSEUDO_CFG_DIR="$REPO_ROOT/refs/cups/cups/pseudo_labels"
PSEUDO_CFG="$PSEUDO_CFG_DIR/config_pseudo_labels.yaml"
PSEUDO_CFG_E1="$PSEUDO_CFG_DIR/config_pseudo_labels_e1.yaml"

cat > "$PSEUDO_CFG_E1" <<EOF
SYSTEM:
  NUM_WORKERS: $NUM_WORKERS
  EXPERIMENT_NAME: "e1_cups_a6000"
DATA:
  CROP_RESOLUTION: (1024, 2048)
  VAL_SCALE: 1.0
  DATASET: "cityscapes"
  ROOT: "$CITYSCAPES_ROOT"
  PSEUDO_ROOT: "$OUTPUT_DIR"
MODEL:
  CHECKPOINT: "$WEIGHTS_DIR/depthg.ckpt"
  MAX_NUM_PASTED_OBJECTS: 8
  NUM_STEPS_STARTUP: 1000
  IGNORE_UNKNOWN_THING_REGIONS: False
EOF
log "Wrote E1 config: $PSEUDO_CFG_E1"

# ---------------------------------------------------------------------------
# 7. Generate pseudo-labels (NUM_SUBSPLITS parallel workers on one A6000)
# ---------------------------------------------------------------------------
cd "$REPO_ROOT/refs/cups"

log "Launching pseudo-label generation:"
log "  GPU=$GPU_ID  subsplits=$NUM_SUBSPLITS  workers=$NUM_WORKERS"
log "  output dir: $OUTPUT_DIR"
log "  expect ~2975 *_semantic.png + ~2975 *_instance.png + pseudo_classes_split_N.pt"

PIDS=()
for ((i=1; i<=NUM_SUBSPLITS; i++)); do
    SPLIT_LOG="$LOG_DIR/e1_split_${i}_${TS}.log"
    log "  -> subsplit $i/$NUM_SUBSPLITS  log=$SPLIT_LOG"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTHONHASHSEED="$SEED" \
    "$PYTHON_BIN" -u cups/pseudo_labels/gen_pseudo_labels.py \
        --DATA.DATASET cityscapes \
        --MODEL.CHECKPOINT "$WEIGHTS_DIR/depthg.ckpt" \
        --DATA.NUM_PREPROCESSING_SUBSPLITS "$NUM_SUBSPLITS" \
        --DATA.PREPROCESSING_SUBSPLIT "$i" \
        --DATA.PSEUDO_ROOT "$OUTPUT_DIR" \
        --SYSTEM.NUM_WORKERS "$NUM_WORKERS" \
        > "$SPLIT_LOG" 2>&1 &
    PIDS+=($!)
    sleep 3   # stagger model init to avoid simultaneous CUDA context creation
done

log "All subsplit PIDs: ${PIDS[*]}"
log "Tail any of the split logs to follow progress."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        log "ERROR: subsplit pid=$pid failed."
        FAILED=1
    fi
done
if (( FAILED )); then
    log "One or more subsplits failed — see $LOG_DIR/e1_split_*_${TS}.log."
    exit 2
fi

# ---------------------------------------------------------------------------
# 8. Sanity gates
# ---------------------------------------------------------------------------
log "Running sanity gates ..."
"$PYTHON_BIN" "$REPO_ROOT/scripts/e1_verify_pseudolabels.py" \
    --pseudo_root "$OUTPUT_DIR" \
    --expected_count 2975 \
    --max_empty_frac 0.01 \
    --report "$LOG_DIR/e1_verify_${TS}.json" \
    | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
log "================================================================"
log "E1 Stage-1 generation COMPLETE."
log "Pseudo-labels in: $OUTPUT_DIR"
log "Verify report:    $LOG_DIR/e1_verify_${TS}.json"
log "Repo commit:      $COMMIT"
log ""
log "Next: launch Stage-2 with DINOv3 ViT-B/16 + Cascade Mask R-CNN, e.g."
log "  python refs/cups/train.py \\"
log "    --config refs/cups/configs/train_cityscapes_dinov3_vitb_cups_official_1gpu.yaml \\"
log "    DATA.ROOT_PSEUDO $OUTPUT_DIR DATA.ROOT $CITYSCAPES_ROOT"
log "  (BEFORE Stage-2: apply the scale_factor=0.625 fix in"
log "   refs/cups/cups/data/pseudo_label_dataset.py — see E1_HF_UPLOAD_README.md.)"
log "================================================================"
