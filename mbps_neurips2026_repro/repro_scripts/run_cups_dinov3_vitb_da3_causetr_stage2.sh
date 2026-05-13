#!/bin/bash
# CUPS Stage-2: DINOv3 ViT-B/16 — DA3+CAUSE-TR pseudo-labels
# Backbone: DINOv3 ViT-B/16 (same as successful k80 run: Stage-2 PQ=27.865%)
# Pseudo-labels: cups_pseudo_labels_da3_causetr/ (DA3 τ=0.03 + CAUSE-TR k80)
#   Pseudo-label PQ: 27.37 (+0.63 vs k80+SPIdepth 26.74), PQ_things: 20.90 (+1.49)
# Target: > 28% PQ at Stage-2 (vs 27.865% with k80+SPIdepth)
# eff_bs: 2 GPUs × bs=1 × accum=8 = 16 (same as k80 ViT-B run)
#
# ─── Deploy (run locally) ─────────────────────────────────────────────────────
#   scp refs/cups/configs/train_cityscapes_dinov3_vitb_da3_causetr_8k_2gpu.yaml \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/configs/
#   scp scripts/run_cups_dinov3_vitb_da3_causetr_stage2.sh \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/scripts/
#
# ─── Launch (on remote) ───────────────────────────────────────────────────────
#   nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_dinov3_vitb_da3_causetr_stage2.sh \
#     > /dev/null 2>&1 &
#   echo "PID: $!"
#
# ─── Monitor ──────────────────────────────────────────────────────────────────
#   tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_dinov3_vitb_da3_causetr_8k_bs16_stage2.log
#
# ─── Resume (if interrupted) ──────────────────────────────────────────────────
#   bash run_cups_dinov3_vitb_da3_causetr_stage2.sh "/path/to/.../checkpoints/last.ckpt"
#
# ─── After Stage-2 completes ──────────────────────────────────────────────────
#   1. Find best checkpoint:
#      ls /media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/
#         cups_dinov3_vitb_da3_causetr_8k_bs16_stage2/*/checkpoints/best_pq_step=*.ckpt
#   2. Update train_self_cityscapes_dinov3_vitb_da3_causetr_2gpu.yaml MODEL.CHECKPOINT
#   3. Launch Stage-3: run_cups_dinov3_vitb_da3_causetr_stage3.sh

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_dinov3_vitb_da3_causetr_8k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_dinov3_vitb_da3_causetr_8k_bs16_stage2.log"
PSEUDO_DIR="${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_da3_causetr"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2: DINOv3 ViT-B/16 + DA3+CAUSE-TR pseudo-labels ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Pseudo-labels: ${PSEUDO_DIR}" | tee -a "${LOG_FILE}"
echo "eff_bs: 2 GPUs × bs=1 × accum=8 = 16" | tee -a "${LOG_FILE}"

# GPU info
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# Verify pseudo-labels
if [ ! -d "${PSEUDO_DIR}" ]; then
    echo "ERROR: Pseudo-labels not found at ${PSEUDO_DIR}" | tee -a "${LOG_FILE}"
    exit 1
fi
NUM_SEM=$(find "${PSEUDO_DIR}" -name "*_semantic.png" 2>/dev/null | wc -l)
NUM_INST=$(find "${PSEUDO_DIR}" -name "*_instance.png" 2>/dev/null | wc -l)
NUM_PT=$(find "${PSEUDO_DIR}" -name "*.pt" 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance, ${NUM_PT} .pt" | tee -a "${LOG_FILE}"
if [ "${NUM_SEM}" -lt 2975 ] || [ "${NUM_INST}" -lt 2975 ]; then
    echo "ERROR: Expected ≥2975 training images. Assembly incomplete." | tee -a "${LOG_FILE}"
    exit 1
fi

echo "Launching Stage-2 (8000 steps, ViT-B/16, DA3+CAUSE-TR)..." | tee -a "${LOG_FILE}"

RESUME_CKPT="${1:-}"
if [ -n "${RESUME_CKPT}" ]; then
    echo "Resuming from Lightning checkpoint: ${RESUME_CKPT}" | tee -a "${LOG_FILE}"
    ${PYTHON} -u train.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        --ckpt_path "${RESUME_CKPT}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    ${PYTHON} -u train.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "=== Stage-2 ViT-B DA3+CAUSE-TR complete: $(date) ===" | tee -a "${LOG_FILE}"

# Print best checkpoint path for Stage-3
echo "" | tee -a "${LOG_FILE}"
echo "Best checkpoint(s):" | tee -a "${LOG_FILE}"
find "${WORK_DIR}/experiments" -path "*/cups_dinov3_vitb_da3_causetr_8k_bs16_stage2/*/checkpoints/best_pq_step=*.ckpt" \
    2>/dev/null | sort | tee -a "${LOG_FILE}" || true
