#!/usr/bin/env bash
# Stage-3 self-training on DCFA+DepthPro+SIMCF-ABC (from Stage-2 best PQ~28.1)
# Target: santosh@172.17.254.146, 2x GTX 1080 Ti
#
# Usage:
#   bash scripts/run_stage3_dcfa_simcf_abc_santosh.sh train    # launch Stage-3
#   bash scripts/run_stage3_dcfa_simcf_abc_santosh.sh status   # check progress

set -euo pipefail

REMOTE="santosh@172.17.254.146"
REMOTE_CUPS="/home/santosh/cups"
LOCAL_CONFIG="refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
REMOTE_CONFIG="${REMOTE_CUPS}/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
CONFIG="configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
LOG_DIR="/home/santosh/experiments/stage3_dcfa_simcf_abc"
LOG_FILE="${LOG_DIR}/train.log"

step_train() {
    echo "=== Launching Stage-3 self-training on santosh ==="

    echo "[1/4] Uploading config..."
    scp "$LOCAL_CONFIG" "${REMOTE}:${REMOTE_CONFIG}"

    echo "[2/4] Checking GPUs..."
    ssh "$REMOTE" "nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader"

    echo "[3/4] Creating log directory..."
    ssh "$REMOTE" "mkdir -p ${LOG_DIR}"

    echo "[4/4] Launching training (3 rounds x 4000 steps, ~5-6 hrs)..."
    ssh "$REMOTE" bash -s <<TRAIN_EOF
cd ${REMOTE_CUPS}
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:\${LD_LIBRARY_PATH:-}"
PYTHON="/home/santosh/anaconda3/envs/cups/bin/python"

nohup \$PYTHON -u train_self.py \\
    --experiment_config_file ${CONFIG} \\
    --disable_wandb \\
    > ${LOG_FILE} 2>&1 &

PID=\$!
echo \$PID > ${LOG_DIR}/train.pid
echo ""
echo "============================================"
echo "  Stage-3 Self-Training launched!"
echo "  PID:     \$PID"
echo "  Config:  ${CONFIG}"
echo "  Log:     ${LOG_FILE}"
echo "  Monitor: ssh ${REMOTE} 'tail -f ${LOG_FILE}'"
echo "============================================"
TRAIN_EOF
}

step_status() {
    echo "=== Stage-3 status on santosh ==="
    ssh "$REMOTE" bash -s <<'STATUS_EOF'
LOG_DIR="/home/santosh/experiments/stage3_dcfa_simcf_abc"
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
    echo ""
    echo "  Validation PQ results:"
    echo "  ----------------------"
    grep -E "^[0-9]+\.[0-9]+" "$LOG_FILE" | awk -F';' '{gsub(/ /,""); printf "  PQ=%.4f PQ_th=%.4f PQ_st=%.4f mIoU=%.4f\n", $1, $4, $7, $11}' | tail -10
fi
STATUS_EOF
}

case "${1:-help}" in
    train)  step_train ;;
    status) step_status ;;
    *)
        echo "Usage: $0 {train|status}"
        exit 1
        ;;
esac
