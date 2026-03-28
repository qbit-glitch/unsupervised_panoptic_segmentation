#!/bin/bash
# Launch CUPS stage-2 training with DINOv2 ViT-B/14 backbone + v5 config.
# v5 FIX: batch_size=2, 16000 steps (64K total images, matching CUPS default)
# v3 bug: batch_size=1 × 8000 steps = only 16K images (4× less than needed)
#
# v5 pseudo-labels: same as v3 (overclustered k=300 semantics + SPIdepth instances)
#
# Expected training time: ~40-50 hours on 2x GTX 1080 Ti
#
# Usage:
#   ssh santosh@172.17.254.146 "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_vitb_v5_stage2.sh > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_vitb_v5_stage2.log"
#
# Check PQ progress:
#   ssh santosh@172.17.254.146 "grep 'PQ\|pq' /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_vitb_v5_stage2.log"

set -euo pipefail

# --- Environment ---
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
LOG_FILE="${WORK_DIR}/experiments/cups_vitb_v5_stage2.log"

cd "${WORK_DIR}/refs/cups"

mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2 Training v5: DINOv2 ViT-B/14 + Fixed Batch Size ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "KEY FIX: batch_size=2, steps=16000 (total=64K images, was 16K in v3)" | tee -a "${LOG_FILE}"
echo "Pseudo-labels: cups_pseudo_labels_v3 (overclustered k=300 + SPIdepth gt=0.6/ma=500)" | tee -a "${LOG_FILE}"

# Verify GPU availability
${PYTHON} -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "${LOG_FILE}"

# Verify pseudo-labels exist
NUM_SEM=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_v3/*_semantic.png 2>/dev/null | wc -l)
NUM_INST=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_v3/*_instance.png 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"

# Check GPU memory
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

echo "Launching training (2x GPU DDP, batch_size=2)..." | tee -a "${LOG_FILE}"

# Launch training with ViT-B v5 config
${PYTHON} train_hybrid.py \
    --experiment_config_file "${WORK_DIR}/refs/cups/configs/train_cityscapes_vitb_v5.yaml" \
    --disable_wandb \
    2>&1 | tee -a "${LOG_FILE}"

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
