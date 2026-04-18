#!/bin/bash
# M5 Conv-DoRA + DepthPro k=80 pipeline on santosh (2x GTX 1080 Ti)
# Stage-2: Conv-DoRA fine-tune (8000 steps, ~90 min on 2 GPUs)
# Stage-3: Progressive Conv-DoRA + M5 self-training (3 rounds x 500 steps, ~60 min)
#
# Usage:
#   bash scripts/run_m5_depthpro_santosh.sh verify   # pre-flight checks
#   bash scripts/run_m5_depthpro_santosh.sh smoke    # mitigation unit tests
#   bash scripts/run_m5_depthpro_santosh.sh stage2   # launch Stage-2 Conv-DoRA
#   bash scripts/run_m5_depthpro_santosh.sh stage3   # launch Stage-3 M5 (auto-finds Stage-2 ckpt)
#   bash scripts/run_m5_depthpro_santosh.sh status   # latest log tail + running PIDs

set -euo pipefail

CUPS_ROOT="/home/santosh/cups"
DATASET_ROOT="/home/santosh/datasets/cityscapes"
PSEUDO_LABELS="${DATASET_ROOT}/cups_pseudo_labels_depthpro_tau020"
WEIGHTS="${CUPS_ROOT}/weights/dinov3_vitb16_official.pth"
LOG_DIR="${CUPS_ROOT}/logs"
EXP_DIR="${CUPS_ROOT}/experiments"
STAGE2_CONFIG="${CUPS_ROOT}/configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_2gpu.yaml"
STAGE3_CONFIG="${CUPS_ROOT}/configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml"
STAGE2_RUN_NAME="e2_depthpro_conv_dora_r4_2gpu"
STAGE3_RUN_NAME="stage3_depthpro_m5_conv_dora_progressive_2gpu"

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/santosh/anaconda3/envs/cups/bin/python"

mkdir -p "${LOG_DIR}"

verify() {
    echo "=== M5 DepthPro Pre-flight ==="
    local ok=0 fail=0

    local sem inst pts
    sem=$(find "${PSEUDO_LABELS}" -maxdepth 1 -name "*_semantic.png" 2>/dev/null | wc -l)
    inst=$(find "${PSEUDO_LABELS}" -maxdepth 1 -name "*_instance.png" 2>/dev/null | wc -l)
    pts=$(find "${PSEUDO_LABELS}" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
    echo "[labels] semantic=${sem} instance=${inst} pt=${pts}"
    if [ "${sem}" -ge 2900 ] && [ "${inst}" -ge 2900 ] && [ "${pts}" -ge 2900 ]; then
        ((ok++))
    else
        echo "  FAIL: expected ~2975 of each"; ((fail++))
    fi

    if [ -f "${WEIGHTS}" ]; then
        echo "[weights] ${WEIGHTS}"; ((ok++))
    else
        echo "  FAIL: weights missing"; ((fail++))
    fi

    for cfg in "${STAGE2_CONFIG}" "${STAGE3_CONFIG}"; do
        if [ -f "${cfg}" ]; then
            echo "[cfg] ${cfg}"; ((ok++))
        else
            echo "  FAIL: ${cfg} missing"; ((fail++))
        fi
    done

    local gpus
    gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "[gpu] ${gpus} GPUs"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    if [ "${gpus}" -ge 2 ]; then ((ok++)); else echo "  FAIL: need 2 GPUs"; ((fail++)); fi

    echo "=== ${ok} ok, ${fail} fail ==="
    [ "${fail}" -eq 0 ] || return 1
}

smoke() {
    echo "=== M5 smoke test: mitigation unit tests ==="
    cd "${CUPS_ROOT}"
    ${PYTHON} -m pytest tests/test_mitigations.py -v --no-header 2>&1 | tail -30
}

stage2() {
    verify || exit 1
    local logfile="${LOG_DIR}/stage2_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Stage-2 Conv-DoRA launch ==="
    echo "Config:  ${STAGE2_CONFIG}"
    echo "Log:     ${logfile}"
    echo "Monitor: tail -f ${logfile}"
    cd "${CUPS_ROOT}"
    nohup ${PYTHON} -u train.py \
        --experiment_config_file "${STAGE2_CONFIG}" \
        --disable_wandb \
        > "${logfile}" 2>&1 &
    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/stage2.pid"
}

stage3() {
    local s2_dir="${EXP_DIR}/${STAGE2_RUN_NAME}"
    if [ ! -d "${s2_dir}" ]; then
        echo "ERROR: Stage-2 experiment dir not found: ${s2_dir}"
        echo "Run stage2 first."
        exit 1
    fi
    local ckpt
    ckpt=$(find "${s2_dir}" -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    if [ -z "${ckpt}" ]; then
        echo "ERROR: No .ckpt found under ${s2_dir}"
        exit 1
    fi
    echo "Using Stage-2 checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/stage3_m5_$(date +%Y%m%d_%H%M%S).log"
    echo "Config:  ${STAGE3_CONFIG}"
    echo "Log:     ${logfile}"
    echo "Monitor: tail -f ${logfile}"
    cd "${CUPS_ROOT}"
    nohup ${PYTHON} -u train_self.py \
        --experiment_config_file "${STAGE3_CONFIG}" \
        --ckpt_path "${ckpt}" \
        --disable_wandb \
        > "${logfile}" 2>&1 &
    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/stage3.pid"
}

status() {
    echo "=== PIDs ==="
    for stage in stage2 stage3; do
        local pidfile="${LOG_DIR}/${stage}.pid"
        if [ -f "${pidfile}" ]; then
            local pid
            pid=$(cat "${pidfile}")
            if kill -0 "${pid}" 2>/dev/null; then
                echo "  ${stage}: RUNNING pid=${pid}"
            else
                echo "  ${stage}: DEAD pid=${pid} (stale pidfile)"
            fi
        fi
    done

    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

    echo "=== Latest logs (last 20 lines each) ==="
    for f in $(ls -t "${LOG_DIR}"/stage2_*.log "${LOG_DIR}"/stage3_m5_*.log 2>/dev/null | head -2); do
        echo "--- ${f} ---"
        tail -20 "${f}"
    done
}

case "${1:-help}" in
    verify) verify ;;
    smoke)  smoke ;;
    stage2) stage2 ;;
    stage3) stage3 ;;
    status) status ;;
    *)
        echo "Usage: $0 {verify|smoke|stage2|stage3|status}"
        exit 1
        ;;
esac
