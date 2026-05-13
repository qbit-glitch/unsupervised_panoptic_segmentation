#!/bin/bash
# Launch CUPS stage-2 training with DINOv2 ViT-B/14 backbone on remote GPU machine.
#
# Usage:
#   ssh santosh@172.17.254.146 "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_vitb_stage2.sh &"
#
# Monitor:
#   ssh santosh@172.17.254.146 "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_vitb_stage2.log"

set -euo pipefail

# ─── Environment ───
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
LOG_FILE="${WORK_DIR}/experiments/cups_vitb_stage2.log"

cd "${WORK_DIR}/refs/cups"

mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2 Training: DINOv2 ViT-B/14 ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"

# Verify GPU availability
${PYTHON} -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "${LOG_FILE}"

# Verify DINOv2 is cached
${PYTHON} -c "
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', verbose=False)
print(f'DINOv2 loaded: embed_dim={model.embed_dim}, patch_size={model.patch_size}')
del model
" 2>&1 | tee -a "${LOG_FILE}"

echo "Launching training..." | tee -a "${LOG_FILE}"

# Launch training with ViT-B config
${PYTHON} train.py \
    --experiment_config_file "${WORK_DIR}/refs/cups/configs/train_cityscapes_vitb.yaml" \
    --disable_wandb \
    2>&1 | tee -a "${LOG_FILE}"

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
