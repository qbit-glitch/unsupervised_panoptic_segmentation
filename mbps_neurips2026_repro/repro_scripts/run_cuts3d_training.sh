#!/bin/bash
# CutS3D Full Pipeline — Paper Reproduction on TPU v4
# Sick et al., "CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation", ICCV 2025
#
# Runs on panoptic-tpu-cuts3d (4x TPU v4 cores)
# Data: COCO train2017 TFRecords on gs://mbps-panoptic/
#
# Usage:
#   bash scripts/run_cuts3d_training.sh [phase]
#   phase: extract | train | self-train | full (default: full)

set -euo pipefail

cd ~/mbps_panoptic_segmentation

PHASE="${1:-full}"
CONFIG="configs/cuts3d_coco.yaml"
OUTPUT="checkpoints/cuts3d/cuts3d_cad.npz"
PSEUDO_DIR="data/pseudo_masks"
TFRECORD_DIR="gs://mbps-panoptic/datasets/coco/tfrecords/train"
STORED_SIZE="512 512"

# Paper hyperparameters
EPOCHS=30
SELF_TRAIN_ROUNDS=3

# Create directories
mkdir -p checkpoints/cuts3d data/pseudo_masks logs

echo "=============================================="
echo "  CutS3D Paper Reproduction Pipeline"
echo "  Phase: ${PHASE}"
echo "  Config: ${CONFIG}"
echo "  TPU devices: $(python3 -c 'import jax; print(jax.device_count())')"
echo "=============================================="

# Redirect output to log file as well
LOG_FILE="logs/cuts3d_${PHASE}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

python3 scripts/train_cuts3d.py \
    --config "${CONFIG}" \
    --phase "${PHASE}" \
    --epochs ${EPOCHS} \
    --output "${OUTPUT}" \
    --pseudo-mask-dir "${PSEUDO_DIR}" \
    --tfrecord-dir "${TFRECORD_DIR}" \
    --stored-size ${STORED_SIZE} \
    --self-train-rounds ${SELF_TRAIN_ROUNDS} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=============================================="
echo "  Pipeline complete: ${PHASE}"
echo "  Log: ${LOG_FILE}"
echo "  Checkpoints: checkpoints/cuts3d/"
echo "=============================================="
