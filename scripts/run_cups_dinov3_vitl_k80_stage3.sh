#!/bin/bash
# CUPS Stage-3: DINOv3 ViT-L/16 — k=80 self-training (EMA teacher)
# Starts from: cups_dinov3_vitl_k80_8k_bs16_stage2 best_pq_step=006510.ckpt (PQ=25.45%)
# Target: > 27% PQ (ViT-B Stage-3 gained +2.35 PQ; expect similar gain here)
#
# ─── Deploy (run locally) ─────────────────────────────────────────────────────
#   scp refs/cups/configs/train_self_cityscapes_dinov3_vitl_k80_2gpu.yaml \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/configs/
#   scp scripts/run_cups_dinov3_vitl_k80_stage3.sh \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/scripts/
#
# ─── Launch (on remote) ───────────────────────────────────────────────────────
#   nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_dinov3_vitl_k80_stage3.sh \
#     > /dev/null 2>&1 &
#   echo "PID: $!"
#
# ─── Monitor ──────────────────────────────────────────────────────────────────
#   tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_dinov3_vitl_k80_stage3.log
#
# ─── Resume (if interrupted) ──────────────────────────────────────────────────
#   Pass the Lightning last.ckpt path as $1:
#   bash run_cups_dinov3_vitl_k80_stage3.sh "/path/to/.../checkpoints/last.ckpt"
#
# ─── OOM fallback ─────────────────────────────────────────────────────────────
#   If OOM: edit config TTA_SCALES to just [1.0] and relaunch.

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_self_cityscapes_dinov3_vitl_k80_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_dinov3_vitl_k80_stage3.log"
CKPT="${WORK_DIR}/experiments/experiments/cups_dinov3_vitl_k80_8k_bs16_stage2/Unsupervised Panoptic Segmentation/520z0j49/checkpoints/best_pq_step=006510.ckpt"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-3: DINOv3 ViT-L/16 k80 EMA self-training ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Stage-2 checkpoint: ${CKPT}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"

# Verify Stage-2 checkpoint exists
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Stage-2 checkpoint not found: ${CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi

${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    total = torch.cuda.get_device_properties(i).total_memory / 1024**2
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {total:.0f} MiB total')
" 2>&1 | tee -a "${LOG_FILE}"

echo "Launching Stage-3 (3 rounds x 4000 steps = 12000 steps total)..." | tee -a "${LOG_FILE}"

RESUME_CKPT="${1:-}"
if [ -n "${RESUME_CKPT}" ]; then
    echo "Resuming from Lightning checkpoint: ${RESUME_CKPT}" | tee -a "${LOG_FILE}"
    ${PYTHON} -u train_self.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        --ckpt_path "${RESUME_CKPT}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    ${PYTHON} -u train_self.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "=== Stage-3 ViT-L k80 complete: $(date) ===" | tee -a "${LOG_FILE}"
