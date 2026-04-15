#!/bin/bash
# E1 Control Experiment: CUPS official pseudo-labels + DINOv3 ViT-B/16
# Target: santosh@172.17.254.146, GTX 1080 Ti GPU 0
#
# Usage:
#   Stage-2:  bash scripts/run_e1_santosh.sh stage2
#   Stage-3:  bash scripts/run_e1_santosh.sh stage3 /path/to/stage2/best_checkpoint.ckpt
#   Verify:   bash scripts/run_e1_santosh.sh verify

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CUPS_ROOT="${REPO_ROOT}/refs/cups"
DATASET_ROOT="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes"
PSEUDO_LABELS="${DATASET_ROOT}/cups_pseudo_labels"
WEIGHTS="${CUPS_ROOT}/weights/dinov3_vitb16_official.pth"
LOG_DIR="${REPO_ROOT}/logs"
EXPERIMENT_DIR="/media/santosh/Kuldeep/panoptic_segmentation/experiments"
GPU_ID=0

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"

verify() {
    echo "=== E1 Pre-flight Checks ==="
    local ok=0
    local fail=0

    # Check pseudo-labels
    if [ -d "${PSEUDO_LABELS}" ]; then
        local count
        count=$(find "${PSEUDO_LABELS}" -name "*.pt" -o -name "*.png" 2>/dev/null | head -3000 | wc -l)
        echo "[OK] Pseudo-labels dir exists: ${PSEUDO_LABELS} (${count} files found)"
        ((ok++))
    else
        echo "[FAIL] Pseudo-labels NOT found: ${PSEUDO_LABELS}"
        echo "  Check if pipeline output is at /home/santosh/datasets/cityscapes/cups_pseudo_labels_pipeline/"
        echo "  If so, symlink: ln -s /home/santosh/datasets/cityscapes/cups_pseudo_labels_pipeline/ ${PSEUDO_LABELS}"
        ((fail++))
    fi

    # Check DINOv3 weights
    if [ -f "${WEIGHTS}" ]; then
        echo "[OK] DINOv3 weights: ${WEIGHTS}"
        ((ok++))
    else
        echo "[FAIL] DINOv3 weights NOT found: ${WEIGHTS}"
        echo "  Copy from local: scp weights/dinov3_vitb16_official.pth santosh@172.17.254.146:${WEIGHTS}"
        ((fail++))
    fi

    # Check Cityscapes images
    if [ -d "${DATASET_ROOT}/leftImg8bit/train" ]; then
        echo "[OK] Cityscapes images: ${DATASET_ROOT}/leftImg8bit/train"
        ((ok++))
    else
        echo "[FAIL] Cityscapes images NOT found: ${DATASET_ROOT}/leftImg8bit/train"
        ((fail++))
    fi

    # Check Cityscapes GT (for validation)
    if [ -d "${DATASET_ROOT}/gtFine/val" ]; then
        echo "[OK] Cityscapes GT val: ${DATASET_ROOT}/gtFine/val"
        ((ok++))
    else
        echo "[FAIL] Cityscapes GT val NOT found: ${DATASET_ROOT}/gtFine/val"
        ((fail++))
    fi

    # Check CUDA
    if command -v nvidia-smi &>/dev/null; then
        echo "[OK] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i ${GPU_ID} 2>/dev/null || echo 'unknown')"
        ((ok++))
    else
        echo "[FAIL] nvidia-smi not found"
        ((fail++))
    fi

    echo ""
    echo "=== ${ok} passed, ${fail} failed ==="
    if [ "${fail}" -gt 0 ]; then
        echo "Fix failures before running. Exiting."
        return 1
    fi
    echo "Ready to run E1."
    return 0
}

run_stage2() {
    echo "=== E1 Stage-2: CUPS official PLs + DINOv3 ViT-B/16 ==="
    local config="${CUPS_ROOT}/configs/train_cityscapes_dinov3_vitb_cups_official_1gpu.yaml"
    local logfile="${LOG_DIR}/e1_stage2_$(date +%Y%m%d_%H%M%S).log"

    if [ ! -f "${config}" ]; then
        echo "ERROR: Config not found: ${config}"
        exit 1
    fi

    verify || exit 1

    echo ""
    echo "Config:  ${config}"
    echo "Log:     ${logfile}"
    echo "GPU:     ${GPU_ID}"
    echo ""
    echo "Starting Stage-2 training (8000 steps, bs=1, accum=16)..."
    echo "Monitor: tail -f ${logfile}"
    echo ""

    cd "${CUPS_ROOT}"
    nohup python -u train.py \
        --experiment_config_file "${config}" \
        > "${logfile}" 2>&1 &

    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/e1_stage2.pid"
    echo ""
    echo "Stage-2 launched. When done, find best checkpoint in:"
    echo "  ${EXPERIMENT_DIR}/e1_cups_official_dinov3_vitb_8k_stage2_gpu0/"
    echo ""
    echo "Then run Stage-3:"
    echo "  bash scripts/run_e1_santosh.sh stage3 /path/to/best_checkpoint.ckpt"
}

run_stage3() {
    local ckpt_path="${1:-}"
    if [ -z "${ckpt_path}" ]; then
        echo "ERROR: Stage-3 requires a Stage-2 checkpoint path."
        echo "Usage: bash scripts/run_e1_santosh.sh stage3 /path/to/checkpoint.ckpt"
        echo ""
        echo "Looking for checkpoints in ${EXPERIMENT_DIR}/e1_cups_official_dinov3_vitb_8k_stage2_gpu0/..."
        if [ -d "${EXPERIMENT_DIR}/e1_cups_official_dinov3_vitb_8k_stage2_gpu0" ]; then
            find "${EXPERIMENT_DIR}/e1_cups_official_dinov3_vitb_8k_stage2_gpu0" -name "*.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -5 | cut -d' ' -f2-
        fi
        exit 1
    fi

    if [ ! -f "${ckpt_path}" ]; then
        echo "ERROR: Checkpoint not found: ${ckpt_path}"
        exit 1
    fi

    echo "=== E1 Stage-3: Self-Training with EMA Teacher ==="
    local config="${CUPS_ROOT}/configs/train_self_cityscapes_dinov3_vitb_cups_official_1gpu.yaml"
    local logfile="${LOG_DIR}/e1_stage3_$(date +%Y%m%d_%H%M%S).log"

    echo "Config:     ${config}"
    echo "Checkpoint: ${ckpt_path}"
    echo "Log:        ${logfile}"
    echo "GPU:        ${GPU_ID}"
    echo ""
    echo "Starting Stage-3 self-training (3 rounds x 500 steps)..."
    echo "Monitor: tail -f ${logfile}"
    echo ""

    cd "${CUPS_ROOT}"
    nohup python -u train_self.py \
        --experiment_config_file "${config}" \
        --ckpt_path "${ckpt_path}" \
        > "${logfile}" 2>&1 &

    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/e1_stage3.pid"
    echo ""
    echo "Stage-3 launched. Results in:"
    echo "  ${EXPERIMENT_DIR}/e1_cups_official_dinov3_vitb_stage3_gpu0/"
}

case "${1:-}" in
    verify)
        verify
        ;;
    stage2)
        run_stage2
        ;;
    stage3)
        run_stage3 "${2:-}"
        ;;
    *)
        echo "Usage: bash scripts/run_e1_santosh.sh {verify|stage2|stage3} [checkpoint_path]"
        echo ""
        echo "  verify  — Check pseudo-labels, weights, GPU before running"
        echo "  stage2  — Launch Stage-2 training (8000 steps, ~12-16h on 1080 Ti)"
        echo "  stage3  — Launch Stage-3 self-training (needs Stage-2 checkpoint)"
        exit 1
        ;;
esac
