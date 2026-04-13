#!/bin/bash
# CUPS Stage-2: DINOv3 ViT-B/16 — DepthPro v2 depth-guided CC pseudo-labels
# v2: instances split at DepthPro depth edges (65% more instances than plain CC)
# 2 GPUs, NUM_WORKERS=2 (reduced from 4 to avoid OOM on 47GB RAM)
# Auto-retry on crash (up to 5 attempts)
#
# ─── Deploy ───────────────────────────────────────────────────────────────
#   scp refs/cups/configs/train_cityscapes_dinov3_vitb_depthpro_v2_8k_2gpu.yaml \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/configs/
#   scp scripts/run_cups_dinov3_vitb_depthpro_v2_stage2.sh \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/scripts/
#
# ─── Launch ───────────────────────────────────────────────────────────────
#   nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_dinov3_vitb_depthpro_v2_stage2.sh \
#     > /dev/null 2>&1 &
#
# ─── Monitor ──────────────────────────────────────────────────────────────
#   tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_dinov3_vitb_depthpro_v2_stage2.log

set -uo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_dinov3_vitb_depthpro_v2_8k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_dinov3_vitb_depthpro_v2_stage2.log"
PSEUDO_DIR="${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_depthpro_v2"
MAX_RETRIES=5

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2: DINOv3 ViT-B/16 + DepthPro v2 pseudo-labels ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Pseudo-labels: ${PSEUDO_DIR}" | tee -a "${LOG_FILE}"
echo "eff_bs: 2 GPUs × bs=1 × accum=8 = 16" | tee -a "${LOG_FILE}"
echo "NUM_WORKERS: 2 (reduced to avoid OOM on 47GB RAM)" | tee -a "${LOG_FILE}"
echo "Max retries on crash: ${MAX_RETRIES}" | tee -a "${LOG_FILE}"

# GPU info
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# Verify pseudo-labels
if [ ! -d "${PSEUDO_DIR}" ]; then
    echo "ERROR: Pseudo-labels not found at ${PSEUDO_DIR}" | tee -a "${LOG_FILE}"
    exit 1
fi
NUM_SEM=$(find "${PSEUDO_DIR}" -name "*_semantic.png" 2>/dev/null | wc -l)
NUM_INST=$(find "${PSEUDO_DIR}" -name "*_instance.png" 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"

# ─── Auto-retry loop ───
ATTEMPT=0
while [ ${ATTEMPT} -lt ${MAX_RETRIES} ]; do
    ATTEMPT=$((ATTEMPT + 1))

    # Find latest checkpoint for resume
    LATEST_CKPT=""
    CKPT_DIR=$(find "${WORK_DIR}/experiments/experiments" -path "*depthpro_v2_8k_stage2*" -name "last.ckpt" 2>/dev/null | head -1)
    if [ -n "${CKPT_DIR}" ]; then
        LATEST_CKPT="${CKPT_DIR}"
    fi

    if [ -n "${LATEST_CKPT}" ]; then
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Attempt ${ATTEMPT}/${MAX_RETRIES}: Resuming from ${LATEST_CKPT} ===" | tee -a "${LOG_FILE}"
        echo "Time: $(date)" | tee -a "${LOG_FILE}"
        ${PYTHON} -u train.py \
            --experiment_config_file "${CONFIG}" \
            --disable_wandb \
            --ckpt_path "${LATEST_CKPT}" \
            2>&1 | tee -a "${LOG_FILE}"
    else
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Attempt ${ATTEMPT}/${MAX_RETRIES}: Starting fresh ===" | tee -a "${LOG_FILE}"
        echo "Time: $(date)" | tee -a "${LOG_FILE}"
        ${PYTHON} -u train.py \
            --experiment_config_file "${CONFIG}" \
            --disable_wandb \
            2>&1 | tee -a "${LOG_FILE}"
    fi

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "=== Training completed successfully at $(date) ===" | tee -a "${LOG_FILE}"
        break
    else
        echo "" | tee -a "${LOG_FILE}"
        echo "=== CRASHED (exit code ${EXIT_CODE}) at $(date). Waiting 30s before retry... ===" | tee -a "${LOG_FILE}"
        sleep 30
    fi
done

if [ ${ATTEMPT} -ge ${MAX_RETRIES} ] && [ ${EXIT_CODE} -ne 0 ]; then
    echo "=== FAILED after ${MAX_RETRIES} attempts ===" | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "Best checkpoint(s):" | tee -a "${LOG_FILE}"
find "${WORK_DIR}/experiments" -path "*depthpro_v2*best_pq*" -name "*.ckpt" \
    2>/dev/null | sort | tee -a "${LOG_FILE}" || true
