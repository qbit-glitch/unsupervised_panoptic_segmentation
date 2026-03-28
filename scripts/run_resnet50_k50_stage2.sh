#!/bin/bash
# Launch CUPS stage-2 training with DINOv2-ResNet50 backbone + k=50 raw overcluster pseudo-labels.
# k=50 raw overclusters: all 50 pseudo-classes active, CC instances from thing clusters.
# Based on CUPS Table 7b: more pseudo-classes = better PQ (27→54 gives +2.8 PQ).
#
# Usage:
#   ssh santosh@172.17.254.146 "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_resnet50_k50_stage2.sh > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_resnet50_k50_stage2.log"

set -euo pipefail

# ─── Environment ───
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
LOG_FILE="${WORK_DIR}/experiments/cups_resnet50_k50_stage2.log"

cd "${WORK_DIR}/refs/cups"

mkdir -p "${WORK_DIR}/experiments"

CKPT_PATH="${1:-}"  # Optional: pass checkpoint path as first argument to resume

echo "=== CUPS Stage-2 Training: ResNet50 + k=50 Raw Overcluster Pseudo-Labels ===" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "Backbone: DINOv2-ResNet50 (original CUPS backbone)" | tee -a "${LOG_FILE}"
echo "Pseudo-labels: cups_pseudo_labels_k50 (k=50 raw overclusters, CC instances)" | tee -a "${LOG_FILE}"

# Verify GPU availability
${PYTHON} -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "${LOG_FILE}"

# Verify pseudo-labels exist
NUM_SEM=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k50/*_semantic.png 2>/dev/null | wc -l)
NUM_INST=$(ls ${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k50/*_instance.png 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"

echo "Launching training (2x GPU DDP)..." | tee -a "${LOG_FILE}"

# Launch training with k50 config
if [ -n "${CKPT_PATH}" ]; then
    echo "Resuming from checkpoint: ${CKPT_PATH}" | tee -a "${LOG_FILE}"
    ${PYTHON} train.py \
        --experiment_config_file "${WORK_DIR}/refs/cups/configs/train_cityscapes_resnet50_k50.yaml" \
        --disable_wandb \
        --ckpt_path "${CKPT_PATH}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    ${PYTHON} train.py \
        --experiment_config_file "${WORK_DIR}/refs/cups/configs/train_cityscapes_resnet50_k50.yaml" \
        --disable_wandb \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
