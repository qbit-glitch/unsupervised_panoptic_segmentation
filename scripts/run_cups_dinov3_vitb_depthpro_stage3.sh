#!/bin/bash
# CUPS Stage-3: DINOv3 ViT-B/16 — Self-training on DepthPro pseudo-labels
# Loads Stage-2 best checkpoint (best_pq_step=001364, full-val PQ=27.03%)
# 3 rounds x 4000 steps = 12000 total self-training steps
# eff_bs: 2 GPUs × bs=1 × accum=8 = 16
#
# ─── Deploy ───────────────────────────────────────────────────────────────
#   scp refs/cups/configs/train_self_cityscapes_dinov3_vitb_depthpro_2gpu.yaml \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/configs/
#   scp scripts/run_cups_dinov3_vitb_depthpro_stage3.sh \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/scripts/
#
# ─── Launch ───────────────────────────────────────────────────────────────
#   nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_dinov3_vitb_depthpro_stage3.sh \
#     > /dev/null 2>&1 &
#
# ─── Monitor ──────────────────────────────────────────────────────────────
#   tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_dinov3_vitb_depthpro_stage3.log

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_self_cityscapes_dinov3_vitb_depthpro_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_dinov3_vitb_depthpro_stage3.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-3: DINOv3 ViT-B/16 + DepthPro self-training ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Stage-2 checkpoint: best_pq_step=001364 (full-val PQ=27.03%)" | tee -a "${LOG_FILE}"
echo "Self-training: 3 rounds x 4000 steps = 12000 total" | tee -a "${LOG_FILE}"

# GPU info
${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

echo "Launching Stage-3 self-training..." | tee -a "${LOG_FILE}"

${PYTHON} -u train_self.py \
    --experiment_config_file "${CONFIG}" \
    --disable_wandb \
    2>&1 | tee -a "${LOG_FILE}"

echo "=== Stage-3 DINOv3 ViT-B + DepthPro complete: $(date) ===" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Best checkpoint(s):" | tee -a "${LOG_FILE}"
find "${WORK_DIR}/experiments" -path "*depthpro*stage3*best_pq*" -name "*.ckpt" \
    2>/dev/null | sort | tee -a "${LOG_FILE}" || true
