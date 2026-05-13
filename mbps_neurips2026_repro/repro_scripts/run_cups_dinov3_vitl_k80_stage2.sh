#!/bin/bash
# CUPS Stage-2: DINOv3 ViT-L/16 — k=80 pseudo-labels (clip+precision ablation)
# Changes from failing run (cups_dinov3_vitl_da3_causetr_16k_bs32_stage2):
#   1. ROOT_PSEUDO → cups_pseudo_labels_k80/ (CAUSE k=80 STEGO DINOv2)
#   2. GRADIENT_CLIP_VAL: 1.0 → 0.1
#   3. PRECISION: 16-mixed → 32-true  (NOTE: may OOM at 640×1280 — monitor GPU mem)
# Everything else identical: 8 objects, 9 resolutions, accum=16, LR=0.0002, 16K steps
#
# ─── Deploy (run locally) ─────────────────────────────────────────────────
#   scp refs/cups/configs/train_cityscapes_dinov3_vitl_k80_16k_2gpu.yaml \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/configs/
#   scp scripts/run_cups_dinov3_vitl_k80_stage2.sh \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/scripts/
#
# ─── Launch (on remote, after original run finishes) ──────────────────────
#   nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_dinov3_vitl_k80_stage2.sh \
#     > /dev/null 2>&1 &
#   echo "PID: $!"
#
# ─── Monitor ──────────────────────────────────────────────────────────────
#   tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_dinov3_vitl_k80_16k_bs32_stage2.log
#
# ─── OOM fallback ─────────────────────────────────────────────────────────
#   If 32-true OOMs (>11264 MiB), edit the config:
#   PRECISION: "16-mixed"  and re-run. Expected peak at 640×1280 fp32 ~9-10 GB.

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_dinov3_vitl_k80_16k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_dinov3_vitl_k80_8k_bs16_stage2.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2: DINOv3 ViT-L/16 + k80 (ViT-B-matched config) ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "ViT-B-matched: 3 objects, 5 resolutions, accum=8, LR=0.0001, 8K steps, clip=0.1, 32-true" | tee -a "${LOG_FILE}"
echo "Only difference from ViT-B (27.9%): backbone=ViT-L/16 (1024-dim vs 768-dim)" | tee -a "${LOG_FILE}"

${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    total = torch.cuda.get_device_properties(i).total_memory / 1024**2
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {total:.0f} MiB total')
" 2>&1 | tee -a "${LOG_FILE}"

# Verify k=80 pseudo-labels
PSEUDO_DIR="${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k80"
if [ ! -d "${PSEUDO_DIR}" ]; then
    echo "ERROR: Pseudo-labels not found at ${PSEUDO_DIR}" | tee -a "${LOG_FILE}"
    exit 1
fi
NUM_SEM=$(find "${PSEUDO_DIR}" -name "*_semantic.png" 2>/dev/null | wc -l)
NUM_INST=$(find "${PSEUDO_DIR}" -name "*_instance.png" 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"
if [ "${NUM_SEM}" -lt 2975 ] || [ "${NUM_INST}" -lt 2975 ]; then
    echo "ERROR: Expected >=2975 train images. Directory incomplete." | tee -a "${LOG_FILE}"
    exit 1
fi

# Verify ViT-L weights
VITL_CKPT="${WORK_DIR}/weights/dinov3_vitl16_official.pth"
if [ -f "${VITL_CKPT}" ]; then
    echo "DINOv3 ViT-L/16 weights: ${VITL_CKPT}" | tee -a "${LOG_FILE}"
else
    echo "WARNING: ViT-L weights not at ${VITL_CKPT} — will auto-download" | tee -a "${LOG_FILE}"
fi

echo "Launching Stage-2 DINOv3 ViT-L/16 k80 (2x GPU DDP, 32-true)..." | tee -a "${LOG_FILE}"

CKPT_PATH="${1:-}"
if [ -n "${CKPT_PATH}" ]; then
    echo "Resuming from: ${CKPT_PATH}" | tee -a "${LOG_FILE}"
    ${PYTHON} -u train.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        --ckpt_path "${CKPT_PATH}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    ${PYTHON} -u train.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "=== Stage-2 DINOv3 ViT-L/16 k80 complete: $(date) ===" | tee -a "${LOG_FILE}"
