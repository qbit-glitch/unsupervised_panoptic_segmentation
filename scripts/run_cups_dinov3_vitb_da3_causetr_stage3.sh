#!/bin/bash
# CUPS Stage-3: DINOv3 ViT-B/16 — DA3+CAUSE-TR self-training (EMA teacher)
# Starts from: cups_dinov3_vitb_da3_causetr_8k_bs16_stage2 best_pq_step checkpoint
# Expected: Stage-3 gain ≥ +2.35 PQ (same as k80 run: 27.865% → 30.255%)
# Target: PQ > 30.5%
#
# ⚠️  Before launching: update MODEL.CHECKPOINT in the Stage-3 config with
#     the actual Stage-2 best checkpoint path. Find it with:
#       find /media/santosh/Kuldeep/panoptic_segmentation/experiments \
#            -path "*/cups_dinov3_vitb_da3_causetr_8k_bs16_stage2/*/checkpoints/best_pq_step=*.ckpt"
#
# ─── Deploy (run locally) ─────────────────────────────────────────────────────
#   scp refs/cups/configs/train_self_cityscapes_dinov3_vitb_da3_causetr_2gpu.yaml \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/refs/cups/configs/
#   scp scripts/run_cups_dinov3_vitb_da3_causetr_stage3.sh \
#       santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/scripts/
#
# ─── Launch (on remote) ───────────────────────────────────────────────────────
#   nohup bash /media/santosh/Kuldeep/panoptic_segmentation/scripts/run_cups_dinov3_vitb_da3_causetr_stage3.sh \
#     > /dev/null 2>&1 &
#   echo "PID: $!"
#
# ─── Monitor ──────────────────────────────────────────────────────────────────
#   tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_dinov3_vitb_da3_causetr_stage3.log
#
# ─── Resume (if interrupted) ──────────────────────────────────────────────────
#   Pass the Lightning last.ckpt path as $1:
#   bash run_cups_dinov3_vitb_da3_causetr_stage3.sh "/path/to/.../checkpoints/last.ckpt"

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_self_cityscapes_dinov3_vitb_da3_causetr_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_dinov3_vitb_da3_causetr_stage3.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

# Verify Stage-3 config has a valid checkpoint (not placeholder)
CKPT_IN_CONFIG=$(grep "CHECKPOINT:" "${CONFIG}" | sed 's/.*CHECKPOINT: *//; s/"//g; s/[[:space:]]*$//')
if [ "${CKPT_IN_CONFIG}" = "FILL_IN_AFTER_STAGE2" ]; then
    echo "ERROR: MODEL.CHECKPOINT in ${CONFIG} is still a placeholder." >&2
    echo "       Find Stage-2 best checkpoint with:" >&2
    echo "       find ${WORK_DIR}/experiments -path '*/cups_dinov3_vitb_da3_causetr_8k_bs16_stage2/*/checkpoints/best_pq_step=*.ckpt'" >&2
    exit 1
fi

echo "=== CUPS Stage-3: DINOv3 ViT-B/16 DA3+CAUSE-TR EMA self-training ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Stage-2 checkpoint: ${CKPT_IN_CONFIG}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"

# Verify Stage-2 checkpoint exists
if [ ! -f "${CKPT_IN_CONFIG}" ]; then
    echo "ERROR: Stage-2 checkpoint not found: ${CKPT_IN_CONFIG}" | tee -a "${LOG_FILE}"
    exit 1
fi

${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

echo "Launching Stage-3 (3 rounds × 4000 steps = 12000 steps total)..." | tee -a "${LOG_FILE}"

RESUME_CKPT="${1:-}"
if [ -n "${RESUME_CKPT}" ]; then
    echo "Resuming from Lightning checkpoint: ${RESUME_CKPT}" | tee -a "${LOG_FILE}"
    ${PYTHON} -u train_self.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        --ckpt_path "${RESUME_CKPT}" \
        >> "${LOG_FILE}" 2>&1
else
    ${PYTHON} -u train_self.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        >> "${LOG_FILE}" 2>&1
fi

echo "=== Stage-3 ViT-B DA3+CAUSE-TR complete: $(date) ===" | tee -a "${LOG_FILE}"
