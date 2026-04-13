#!/bin/bash
# Launch CUPS Stage-2: DINOv2 ResNet-50 Cascade Mask R-CNN, k=80 pseudo-labels, 16K steps.
# Effective batch: 2 GPUs x bs=2 x 8 accum = 32 (LR=0.0002, linear-scaled)
# Pseudo-labels: cups_pseudo_labels_k80 (CAUSE-TR k=80 semantics + SPIdepth instances)
# Precision: 16-mixed (GTX 1080 Ti does not support bf16)
#
# Usage (run on remote or via ssh):
#   ssh santosh@172.17.254.146 \
#     "nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_resnet50_stage2.sh \
#      > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 \
#     "tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_resnet50_k80_16k_bs32_stage2.log"
#
# Resume from checkpoint (pass as first arg):
#   bash run_cups_resnet50_stage2.sh /path/to/checkpoint.ckpt

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_resnet50_k80_16k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_resnet50_k80_16k_bs32_stage2.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2: DINOv2 ResNet-50 + k=80 (16K steps, eff_bs=32) ===" | tee "${LOG_FILE}"
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

# Verify pseudo-labels
PSEUDO_DIR="${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k80"
if [ ! -d "${PSEUDO_DIR}" ]; then
    echo "ERROR: Pseudo-labels not found at ${PSEUDO_DIR}" | tee -a "${LOG_FILE}"
    exit 1
fi
NUM_SEM=$(find "${PSEUDO_DIR}" -name "*_semantic.png" 2>/dev/null | wc -l)
NUM_INST=$(find "${PSEUDO_DIR}" -name "*_instance.png" 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"

# Verify DINO ResNet-50 backbone checkpoint
DINO_CKPT="${WORK_DIR}/refs/cups/cups/model/backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl"
if [ -f "${DINO_CKPT}" ]; then
    echo "DINO ResNet-50 backbone: OK ($(du -h "${DINO_CKPT}" | cut -f1))" | tee -a "${LOG_FILE}"
else
    echo "ERROR: DINO ResNet-50 backbone not found at ${DINO_CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi

CKPT_PATH="${1:-}"

echo "Launching Stage-2 (2x GPU DDP)..." | tee -a "${LOG_FILE}"

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

echo "" | tee -a "${LOG_FILE}"
echo "=== Stage-2 complete: $(date) ===" | tee -a "${LOG_FILE}"
echo "Next: find best checkpoint and update train_self_resnet50_k80_12k_2gpu.yaml" | tee -a "${LOG_FILE}"
echo "  find ${WORK_DIR}/experiments/cups_resnet50_k80_16k_bs32_stage2 -name 'best_pq*.ckpt'" | tee -a "${LOG_FILE}"
