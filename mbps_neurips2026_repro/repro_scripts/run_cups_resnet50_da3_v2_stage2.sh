#!/bin/bash
# CUPS Stage-2 v2: DINOv2 ResNet-50 — FIXED pseudo-labels
# Fixes from v1:
#   - Thing cluster IDs: wrong {11-18} → correct DINOv3 k=80 {15,20,33,35,43,49,70}
#   - min_area: 1000 → 100 (adequate instance density, ~10-25/image)
# Pseudo-labels: cups_pseudo_labels_da3_v2/
# Baseline to beat: cups_resnet50_k80_16k_bs32_stage2 (original CAUSE-TR k=80 + SPIdepth)
#
# ─── STEP 0 (local, one-time): generate corrected DA3 instances ───────────────
#   python mbps_pytorch/generate_depth_guided_instances.py \
#     --semantic_dir .../pseudo_semantic_raw_dinov3_k80/train \
#     --depth_dir    .../depth_dav3/train \
#     --output_dir   .../pseudo_instance_dav3_v2/train \
#     --thing_ids 15 20 33 35 43 49 70 \
#     --grad_threshold 0.03 --min_area 100
#
# ─── STEP 1 (local, one-time): assemble CUPS pseudo-label dir ─────────────────
#   python scripts/assemble_cups_pseudo_da3.py \
#     --semantic_root .../pseudo_semantic_raw_dinov3_k80/train \
#     --instance_root .../pseudo_instance_dav3_v2/train \
#     --output_dir    .../cups_pseudo_labels_da3_v2
#
# ─── STEP 2 (local, one-time): rsync to remote ────────────────────────────────
#   rsync -av .../cups_pseudo_labels_da3_v2/ santosh@172.17.254.146:.../cups_pseudo_labels_da3_v2/
#   rsync -av refs/cups/configs/train_cityscapes_resnet50_da3_v2_16k_2gpu.yaml santosh:...configs/
#   rsync -av scripts/run_cups_resnet50_da3_v2_stage2.sh santosh:...scripts/
#
# ─── STEP 3 (remote): launch training ─────────────────────────────────────────
#   ssh santosh@172.17.254.146 \
#     "nohup bash .../scripts/run_cups_resnet50_da3_v2_stage2.sh > /dev/null 2>&1 &"
#
# ─── Monitor ──────────────────────────────────────────────────────────────────
#   ssh santosh@172.17.254.146 \
#     "tail -f .../experiments/cups_resnet50_da3_v2_16k_bs32_stage2.log"

set -euo pipefail

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/media/santosh/Kuldeep/panoptic_segmentation/refs/cups:${PYTHONPATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/panoptic_segmentation"
CONFIG="${WORK_DIR}/refs/cups/configs/train_cityscapes_resnet50_da3_v2_16k_2gpu.yaml"
LOG_FILE="${WORK_DIR}/experiments/cups_resnet50_da3_v2_16k_bs32_stage2.log"

cd "${WORK_DIR}/refs/cups"
mkdir -p "${WORK_DIR}/experiments"

echo "=== CUPS Stage-2 v2: DINOv2 ResNet-50 + DA3 instances (FIXED) + DINOv3 k=80 ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Fix: thing_ids={15,20,33,35,43,49,70} (DINOv3 k=80 GT overlap), min_area=100" | tee -a "${LOG_FILE}"

${PYTHON} -c "
import torch
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

PSEUDO_DIR="${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_da3_v2"
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

DINO_CKPT="${WORK_DIR}/refs/cups/cups/model/backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl"
if [ ! -f "${DINO_CKPT}" ]; then
    echo "ERROR: DINO ResNet-50 backbone not found at ${DINO_CKPT}" | tee -a "${LOG_FILE}"
    exit 1
fi
echo "DINO ResNet-50 backbone: OK" | tee -a "${LOG_FILE}"

CKPT_PATH="${1:-}"
echo "Launching Stage-2 v2 (2x GPU DDP)..." | tee -a "${LOG_FILE}"

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

echo "=== Stage-2 v2 complete: $(date) ===" | tee -a "${LOG_FILE}"
