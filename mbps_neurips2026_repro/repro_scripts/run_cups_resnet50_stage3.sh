#!/bin/bash
# Launch CUPS Stage-3: DINOv2 ResNet-50 EMA self-training, 3 rounds x 4000 = 12K steps.
# Effective batch: 2 GPUs x bs=2 x 8 accum = 32 (LR=0.0002)
# Loads best Stage-2 checkpoint from cups_resnet50_k80_16k_bs32_stage2.
# Precision: 16-mixed (GTX 1080 Ti does not support bf16)
#
# PREREQUISITE: Update MODEL.CHECKPOINT in train_self_resnet50_k80_12k_2gpu.yaml
#   with the best Stage-2 checkpoint path before running this script.
#   Find it with:
#     find /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_resnet50_k80_16k_bs32_stage2 \
#          -name "best_pq*.ckpt"
#
# Usage (run on remote or via ssh):
#   ssh santosh@172.17.254.146 \
#     "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_resnet50_stage3.sh \
#      > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 \
#     "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_resnet50_k80_bs32_stage3.log"

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_self_resnet50_k80_12k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_resnet50_k80_bs32_stage3.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-3: DINOv2 ResNet-50 EMA self-training (3 x 4000 steps, eff_bs=32) ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Effective batch: 2 GPUs x bs=2 x 8 accum = 32 | LR=0.0002" | tee -a "${LOG_FILE}"

# GPU info
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# Check Stage-2 checkpoint is configured
if grep -q "__FILL_AFTER_STAGE2__" "${CONFIG}"; then
    echo "ERROR: MODEL.CHECKPOINT still set to placeholder in ${CONFIG}" | tee -a "${LOG_FILE}"
    echo "       Run the following on the remote machine to find the best Stage-2 checkpoint:" | tee -a "${LOG_FILE}"
    echo "       find ${WORK_DIR}/experiments/cups_resnet50_k80_16k_bs32_stage2 -name 'best_pq*.ckpt'" | tee -a "${LOG_FILE}"
    echo "       Then update MODEL.CHECKPOINT in ${CONFIG} before running Stage-3." | tee -a "${LOG_FILE}"
    exit 1
fi

# Extract and verify checkpoint path from config
STAGE2_CKPT=$(grep "CHECKPOINT:" "${CONFIG}" | awk '{print $2}' | tr -d '"')
if [ -f "${STAGE2_CKPT}" ]; then
    echo "Stage-2 checkpoint: OK ($(du -h "${STAGE2_CKPT}" | cut -f1))" | tee -a "${LOG_FILE}"
else
    echo "ERROR: Stage-2 checkpoint not found: ${STAGE2_CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi

# Verify DINO ResNet-50 backbone checkpoint
DINO_CKPT="${WORK_DIR}/refs/cups/cups/model/backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl"
if [ -f "${DINO_CKPT}" ]; then
    echo "DINO ResNet-50 backbone: OK" | tee -a "${LOG_FILE}"
else
    echo "ERROR: DINO ResNet-50 backbone not found at ${DINO_CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi

# Verify leftImg8bit_sequence for self-training optical flow
SEQ_DIR="${WORK_DIR}/datasets/cityscapes/leftImg8bit_sequence"
ALT_DIR="${WORK_DIR}/datasets/cityscapes/leftImg8bit"
if [ -d "${SEQ_DIR}" ]; then
    echo "leftImg8bit_sequence: OK" | tee -a "${LOG_FILE}"
elif [ -d "${ALT_DIR}" ]; then
    echo "leftImg8bit (fallback): OK" | tee -a "${LOG_FILE}"
else
    echo "ERROR: Neither leftImg8bit_sequence nor leftImg8bit found" | tee -a "${LOG_FILE}"
    exit 1
fi

CKPT_PATH="${1:-}"

echo "Launching Stage-3 self-training (2x GPU DDP)..." | tee -a "${LOG_FILE}"

if [ -n "${CKPT_PATH}" ]; then
    echo "Resuming from: ${CKPT_PATH}" | tee -a "${LOG_FILE}"
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

echo "" | tee -a "${LOG_FILE}"
echo "=== Stage-3 complete: $(date) ===" | tee -a "${LOG_FILE}"
