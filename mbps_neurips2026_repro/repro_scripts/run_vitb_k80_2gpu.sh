#!/bin/bash
# Launch CUPS Stage-2+3 training: DINOv2 ViT-B/14 + k=80 pseudo-labels on 2x GTX 1080 Ti
#
# Config: bs=2 per GPU × 2 GPUs × accumulate=4 = effective BS=16
# Steps: 4000 (Stage-2) + 3×500 (Stage-3 self-training) = 5500 total
# Precision: 16-mixed (1080 Ti doesn't support bf16)
# Expected memory: ~7-8 GB per GPU (safe for 11GB)
# Expected time: ~6-8 hours
#
# Usage:
#   ssh santosh@172.17.254.146 "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_vitb_k80_2gpu.sh > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_vitb_k80_2gpu.log"

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

PYTHON="/home/santosh/anaconda3/envs/cups/bin/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_vitb_k80_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_vitb_k80_2gpu.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2+3: DINOv2 ViT-B/14 + k=80 pseudo-labels ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Effective batch size: 2 × 2 GPUs × 4 accumulate = 16" | tee -a "${LOG_FILE}"

# Verify GPUs
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# Verify pseudo-labels
NUM_SEM=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k80/*_semantic.png 2>/dev/null | wc -l)
NUM_INST=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k80/*_instance.png 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"

echo "Launching training (2x GPU DDP, train.py)..." | tee -a "${LOG_FILE}"

${PYTHON} train.py \
    --experiment_config_file "${CONFIG}" \
    --disable_wandb \
    2>&1 | tee -a "${LOG_FILE}"

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
