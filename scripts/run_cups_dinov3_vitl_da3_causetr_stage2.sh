#!/bin/bash
# CUPS Stage-2: DINOv3 ViT-L/16 — DA3 + CAUSE-TR k=80
# Backbone: DINOv3 ViT-L/16 frozen (embed_dim=1024, 307M params)
# Memory: ~3-4GB per 11GB GPU (frozen backbone = no grad through 307M params)
# eff_bs: 2 GPUs × bs=1 × accum=16 = 32
# LR: 0.0002 (same eff_bs=32 as ResNet-50 baseline)
# Pseudo-labels: cups_pseudo_labels_da3_causetr/ (same as ResNet-50 run)
#
# ─── Deploy (run locally) ─────────────────────────────────────────────────
#   rsync -av refs/cups/cups/model/backbone_dinov3_vit.py \
#             refs/cups/cups/model/model_vitb.py \
#             refs/cups/cups/pl_model_pseudo.py \
#             santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/cups/model/
#   scp refs/cups/configs/train_cityscapes_dinov3_vitl_da3_causetr_16k_2gpu.yaml \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/configs/
#   scp scripts/run_cups_dinov3_vitl_da3_causetr_stage2.sh \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/scripts/
#
# ─── Launch (on remote) ───────────────────────────────────────────────────
#   nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_dinov3_vitl_da3_causetr_stage2.sh \
#     > /dev/null 2>&1 &
#
# ─── Monitor ──────────────────────────────────────────────────────────────
#   tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_dinov3_vitl_da3_causetr_16k_bs32_stage2.log

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_dinov3_vitl_da3_causetr_16k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_dinov3_vitl_da3_causetr_16k_bs32_stage2.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2: DINOv3 ViT-L/16 + DA3 (τ=0.03, A=1000) + CAUSE-TR k=80 ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "eff_bs: 2 GPUs × bs=1 × accum=16 = 32" | tee -a "${LOG_FILE}"

${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# Check pseudo-labels
PSEUDO_DIR="${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_da3_causetr"
if [ ! -d "${PSEUDO_DIR}" ]; then
    echo "ERROR: Pseudo-labels not found at ${PSEUDO_DIR}" | tee -a "${LOG_FILE}"
    exit 1
fi
NUM_SEM=$(find "${PSEUDO_DIR}" -name "*_semantic.png" 2>/dev/null | wc -l)
NUM_INST=$(find "${PSEUDO_DIR}" -name "*_instance.png" 2>/dev/null | wc -l)
NUM_PT=$(find "${PSEUDO_DIR}" -name "*.pt" 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance, ${NUM_PT} .pt" | tee -a "${LOG_FILE}"
if [ "${NUM_SEM}" -lt 2975 ] || [ "${NUM_INST}" -lt 2975 ]; then
    echo "ERROR: Expected 2975 train images. Assembly incomplete." | tee -a "${LOG_FILE}"
    exit 1
fi

# Check ViT-L weights (optional — will auto-download from Meta CDN if absent)
VITL_CKPT="${WORK_DIR}/weights/dinov3_vitl16_official.pth"
if [ -f "${VITL_CKPT}" ]; then
    echo "DINOv3 ViT-L/16 weights: local (${VITL_CKPT})" | tee -a "${LOG_FILE}"
else
    echo "DINOv3 ViT-L/16 weights: not found locally — will download from Meta CDN on first run" | tee -a "${LOG_FILE}"
fi

CKPT_PATH="${1:-}"
echo "Launching Stage-2 DINOv3 ViT-L/16 (2x GPU DDP)..." | tee -a "${LOG_FILE}"

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

echo "=== Stage-2 DINOv3 ViT-L/16 complete: $(date) ===" | tee -a "${LOG_FILE}"
