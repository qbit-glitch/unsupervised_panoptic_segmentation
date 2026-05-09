#!/bin/bash
# MSDA training on santosh remote (2x GTX 1080 Ti)
# Usage:
#   bash scripts/run_msda_remote.sh transformer hybrid
#   bash scripts/run_msda_remote.sh slot hybrid
#   bash scripts/run_msda_remote.sh conv_transformer hybrid
set -euo pipefail

ARCH="${1:?Usage: $0 <arch> <loss>}"
LOSS="${2:?Usage: $0 <arch> <loss>}"
RUN_NAME="${ARCH}_${LOSS}"

FEATURE_DIR="/home/santosh/datasets/cityscapes/dinov3_features_vitl16"
DEPTH_DIR="/home/santosh/datasets/cityscapes/depth_depthpro"
OUTPUT_DIR="/home/santosh/mbps_panoptic_segmentation/checkpoints/msda"
LOG_DIR="/home/santosh/mbps_panoptic_segmentation/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/$RUN_NAME"

if [ ! -d "$FEATURE_DIR" ]; then
    echo "ERROR: DINOv3 features not found at $FEATURE_DIR"
    echo "Run: rsync from local machine first"
    exit 1
fi

if [ ! -d "$DEPTH_DIR" ]; then
    echo "ERROR: DepthPro depth not found at $DEPTH_DIR"
    exit 1
fi

LOG_FILE="$LOG_DIR/msda_${RUN_NAME}.log"

echo "========================================="
echo "MSDA Training: arch=${ARCH}, loss=${LOSS}"
echo "Device: CUDA (GTX 1080 Ti)"
echo "Log: ${LOG_FILE}"
echo "========================================="

cd /home/santosh/mbps_panoptic_segmentation

nohup python3 -u -m mbps_pytorch.msda.train \
    --arch "$ARCH" \
    --loss "$LOSS" \
    --feature_dir "$FEATURE_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --seed 42 \
    --num_workers 4 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
echo "$PID" > "$OUTPUT_DIR/${RUN_NAME}/pid.txt"
