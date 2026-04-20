#!/usr/bin/env bash
# Stage-2 CUPS training on DCFA+DepthPro+SIMCF-ABC pseudo-labels (PQ=25.85)
# Target: santosh@100.93.203.100, 2x GTX 1080 Ti
#
# Usage:
#   bash scripts/run_dcfa_simcf_abc_stage2_santosh.sh transfer   # scp labels (101 MB)
#   bash scripts/run_dcfa_simcf_abc_stage2_santosh.sh verify     # check labels on remote
#   bash scripts/run_dcfa_simcf_abc_stage2_santosh.sh train      # launch training
#   bash scripts/run_dcfa_simcf_abc_stage2_santosh.sh status     # check training status
#   bash scripts/run_dcfa_simcf_abc_stage2_santosh.sh all        # transfer + verify + train

set -euo pipefail

REMOTE="santosh@100.93.203.100"
REMOTE_DATA="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes"
REMOTE_CUPS="/home/santosh/cups"
REMOTE_LABELS="${REMOTE_DATA}/cups_pseudo_labels_dcfa_simcf_abc"
LOCAL_LABELS="/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc"
LOCAL_CONFIG="refs/cups/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
REMOTE_CONFIG="${REMOTE_CUPS}/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
CONFIG="configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
LOG_DIR="/home/santosh/experiments/stage2_dcfa_simcf_abc"
LOG_FILE="${LOG_DIR}/train.log"

step_transfer() {
    echo "=== Transferring DCFA+SIMCF-ABC pseudo-labels to santosh (101 MB) ==="

    # Tar locally, scp, extract remotely — faster than 8925 individual files
    local tarball="/tmp/cups_pseudo_labels_dcfa_simcf_abc.tar.gz"

    echo "[1/3] Creating tarball..."
    tar -czf "$tarball" -C "$(dirname "$LOCAL_LABELS")" "$(basename "$LOCAL_LABELS")"
    echo "       $(du -h "$tarball" | cut -f1) compressed"

    echo "[2/3] Uploading to santosh..."
    scp "$tarball" "${REMOTE}:/tmp/"

    echo "[3/3] Extracting on remote..."
    ssh "$REMOTE" "mkdir -p ${REMOTE_DATA} && tar -xzf /tmp/$(basename "$tarball") -C ${REMOTE_DATA}/ && rm /tmp/$(basename "$tarball")"

    rm -f "$tarball"
    echo "=== Transfer complete ==="
}

step_verify() {
    echo "=== Verifying pseudo-labels on santosh ==="
    ssh "$REMOTE" bash -s <<'VERIFY_EOF'
LABEL_DIR="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc"

if [ ! -d "$LABEL_DIR" ]; then
    echo "ERROR: Label directory not found: $LABEL_DIR"
    exit 1
fi

n_sem=$(find "$LABEL_DIR" -name '*_semantic.png' | wc -l)
n_inst=$(find "$LABEL_DIR" -name '*_instance.png' | wc -l)
n_pt=$(find "$LABEL_DIR" -name '*.pt' | wc -l)
n_total=$(ls "$LABEL_DIR" | wc -l)

echo "  semantic.png : $n_sem"
echo "  instance.png : $n_inst"
echo "  .pt files    : $n_pt"
echo "  total files  : $n_total"
echo "  disk usage   : $(du -sh "$LABEL_DIR" | cut -f1)"

if [ "$n_sem" -eq 2975 ] && [ "$n_inst" -eq 2975 ] && [ "$n_pt" -eq 2975 ]; then
    echo "=== PASS: All 8925 files present ==="
else
    echo "=== FAIL: Expected 2975 each (semantic, instance, .pt) ==="
    exit 1
fi
VERIFY_EOF
}

step_train() {
    echo "=== Launching Stage-2 training on santosh ==="

    # Upload config (refs/cups/ is gitignored, so scp it directly)
    echo "[1/5] Uploading config to santosh..."
    scp "$LOCAL_CONFIG" "${REMOTE}:${REMOTE_CONFIG}"
    echo "       Config uploaded to ${REMOTE_CONFIG}"

    # Pull latest code
    echo "[2/5] Pulling latest code..."
    ssh "$REMOTE" "cd ${REMOTE_CUPS} && git pull"

    # Pre-flight: check GPUs
    echo "[3/5] Checking GPUs..."
    ssh "$REMOTE" "nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader"

    # Create log dir
    echo "[4/5] Creating log directory..."
    ssh "$REMOTE" "mkdir -p ${LOG_DIR}"

    # Launch training with nohup
    echo "[5/5] Launching training (8000 steps, ~90 min)..."
    ssh "$REMOTE" bash -s <<TRAIN_EOF
cd ${REMOTE_CUPS}
conda activate cups

nohup python -u train.py \\
    --experiment_config_file ${CONFIG} \\
    --disable_wandb \\
    > ${LOG_FILE} 2>&1 &

PID=\$!
echo \$PID > ${LOG_DIR}/train.pid
echo ""
echo "============================================"
echo "  Training launched!"
echo "  PID:     \$PID"
echo "  Config:  ${CONFIG}"
echo "  Log:     ${LOG_FILE}"
echo "  Monitor: ssh ${REMOTE} 'tail -f ${LOG_FILE}'"
echo "  Kill:    ssh ${REMOTE} 'kill \$PID'"
echo "============================================"
TRAIN_EOF
}

step_status() {
    echo "=== Training status on santosh ==="
    ssh "$REMOTE" bash -s <<'STATUS_EOF'
LOG_DIR="/home/santosh/experiments/stage2_dcfa_simcf_abc"
PID_FILE="${LOG_DIR}/train.pid"
LOG_FILE="${LOG_DIR}/train.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "  Status: RUNNING (PID $PID)"
    else
        echo "  Status: FINISHED/CRASHED (PID $PID no longer active)"
    fi
else
    echo "  Status: NOT STARTED (no PID file)"
fi

if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "  Last 20 lines of log:"
    echo "  ----------------------"
    tail -20 "$LOG_FILE" | sed 's/^/  /'
else
    echo "  No log file found."
fi

# Check for val PQ results
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "  Validation PQ results:"
    echo "  ----------------------"
    grep -i "panoptic_quality\|PQ\|val/" "$LOG_FILE" | tail -10 | sed 's/^/  /' || echo "  (none yet)"
fi
STATUS_EOF
}

case "${1:-help}" in
    transfer)
        step_transfer
        ;;
    verify)
        step_verify
        ;;
    train)
        step_train
        ;;
    status)
        step_status
        ;;
    all)
        step_transfer
        step_verify
        step_train
        ;;
    *)
        echo "Usage: $0 {transfer|verify|train|status|all}"
        echo ""
        echo "  transfer  - Upload 101MB pseudo-labels to santosh via scp"
        echo "  verify    - Check all 8925 files are present on santosh"
        echo "  train     - Pull code + launch nohup training (2x 1080 Ti)"
        echo "  status    - Check if training is running + show last log lines"
        echo "  all       - transfer + verify + train (full pipeline)"
        exit 1
        ;;
esac
