#!/bin/bash
# Launch CUPS Stage-3 self-training: DINO ResNet-50 + k=80 pseudo-labels
# Loads best Stage-2 checkpoint (PQ=24.82) and refines via EMA self-training.
# Config: bs=1 per GPU x 2 GPUs x accumulate=8 = effective BS=16
# Steps: 3 rounds x 4000 = 12000 total self-training steps
# Precision: 16-mixed (1080 Ti doesn't support bf16)
#
# Usage:
#   ssh santosh@172.17.254.146 "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_resnet50_k80_stage3.sh > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_resnet50_k80_stage3.log"

set -euo pipefail

# --- Environment ---
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_self_cityscapes_resnet50_k80_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_resnet50_k80_stage3.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-3: Self-Training DINO ResNet-50 + k=80 (3 rounds x 4000 steps) ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Entry point: train_self.py" | tee -a "${LOG_FILE}"
echo "Effective batch size: 1 x 2 GPUs x 8 accumulate = 16" | tee -a "${LOG_FILE}"

# Verify GPU availability
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# Verify Stage-2 checkpoint exists
STAGE2_CKPT="${WORK_DIR}/experiments/experiments/cups_resnet50_k80_12k_stage2/Unsupervised Panoptic Segmentation/x5t7nxo9/checkpoints/best_pq_step=002000.ckpt"
if [ -f "${STAGE2_CKPT}" ]; then
    echo "Stage-2 checkpoint: OK ($(du -h "${STAGE2_CKPT}" | cut -f1))" | tee -a "${LOG_FILE}"
else
    echo "ERROR: Stage-2 checkpoint not found at ${STAGE2_CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi

# Verify DINO backbone checkpoint
DINO_CKPT="${WORK_DIR}/refs/cups/cups/model/backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl"
if [ -f "${DINO_CKPT}" ]; then
    echo "DINO backbone checkpoint: OK" | tee -a "${LOG_FILE}"
else
    echo "ERROR: DINO backbone checkpoint not found at ${DINO_CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi

# Verify leftImg8bit_sequence exists (needed for self-training data loader)
SEQ_DIR="${WORK_DIR}/datasets/cityscapes/leftImg8bit_sequence"
if [ -d "${SEQ_DIR}" ]; then
    NUM_SEQ=$(find "${SEQ_DIR}" -name "*.png" | head -100 | wc -l)
    echo "leftImg8bit_sequence: OK (${NUM_SEQ}+ images)" | tee -a "${LOG_FILE}"
else
    echo "WARNING: leftImg8bit_sequence not found, checking leftImg8bit..." | tee -a "${LOG_FILE}"
    ALT_DIR="${WORK_DIR}/datasets/cityscapes/leftImg8bit"
    if [ -d "${ALT_DIR}" ]; then
        echo "leftImg8bit found (self-training may use this instead)" | tee -a "${LOG_FILE}"
    else
        echo "ERROR: No image directory found" | tee -a "${LOG_FILE}"
        exit 1
    fi
fi

CKPT_PATH="${1:-}"  # Optional: pass checkpoint path as first argument to resume

echo "Launching self-training (2x GPU DDP)..." | tee -a "${LOG_FILE}"

# Launch self-training
if [ -n "${CKPT_PATH}" ]; then
    echo "Resuming from checkpoint: ${CKPT_PATH}" | tee -a "${LOG_FILE}"
    ${PYTHON} -u train_self.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        --ckpt_path "${CKPT_PATH}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    ${PYTHON} -u train_self.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "=== Self-training complete: $(date) ===" | tee -a "${LOG_FILE}"
