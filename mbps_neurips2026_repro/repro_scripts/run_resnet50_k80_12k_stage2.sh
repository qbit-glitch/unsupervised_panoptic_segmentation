#!/bin/bash
# Launch CUPS Stage-2 training: DINO ResNet-50 + k=80 pseudo-labels, 12000 steps
# Extended from previous 8000-step run (PQ=24.68 at step 6500).
# Config: bs=2 per GPU x 2 GPUs x accumulate=4 = effective BS=16
# Steps: 12000 (24 validation checkpoints every 500 steps)
# Precision: 16-mixed (1080 Ti doesn't support bf16)
#
# Usage:
#   ssh santosh@172.17.254.146 "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_resnet50_k80_12k_stage2.sh > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_resnet50_k80_12k_stage2.log"

set -euo pipefail

# --- Environment ---
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_resnet50_k80_12k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_resnet50_k80_12k_stage2.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

CKPT_PATH="${1:-}"  # Optional: pass checkpoint path as first argument to resume

echo "=== CUPS Stage-2: DINO ResNet-50 + k=80 Pseudo-Labels (12000 steps) ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Backbone: DINO ResNet-50 (CutLER pretrained)" | tee -a "${LOG_FILE}"
echo "Pseudo-labels: cups_pseudo_labels_k80" | tee -a "${LOG_FILE}"
echo "Effective batch size: 2 x 2 GPUs x 4 accumulate = 16" | tee -a "${LOG_FILE}"

# Verify GPU availability
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# Verify pseudo-labels exist
NUM_SEM=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k80/*_semantic.png 2>/dev/null | wc -l)
NUM_INST=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k80/*_instance.png 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"

# Verify DINO backbone checkpoint
DINO_CKPT="${WORK_DIR}/refs/cups/cups/model/backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl"
if [ -f "${DINO_CKPT}" ]; then
    echo "DINO backbone checkpoint: OK" | tee -a "${LOG_FILE}"
else
    echo "ERROR: DINO backbone checkpoint not found at ${DINO_CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "Launching training (2x GPU DDP)..." | tee -a "${LOG_FILE}"

# Launch training
if [ -n "${CKPT_PATH}" ]; then
    echo "Resuming from checkpoint: ${CKPT_PATH}" | tee -a "${LOG_FILE}"
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

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
