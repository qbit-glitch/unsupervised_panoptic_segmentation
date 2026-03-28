#!/bin/bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/mbps_panoptic_segmentation

DATA_DIR="$HOME/mbps_panoptic_segmentation/data/cityscapes"
DEPTH_DIR="$DATA_DIR/depth_zoedepth"
GCS_OUT="gs://mbps-panoptic/datasets/cityscapes/tfrecords"

echo "[$(date)] Creating Cityscapes TFRecords..."

echo "[$(date)] Train split..."
PYTHONUNBUFFERED=1 python3 scripts/create_tfrecords.py \
    --config configs/cityscapes.yaml \
    --output_dir "${GCS_OUT}/train" \
    --split train \
    --data_dir "$DATA_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --shard_size 256

echo "[$(date)] Val split..."
PYTHONUNBUFFERED=1 python3 scripts/create_tfrecords.py \
    --config configs/cityscapes.yaml \
    --output_dir "${GCS_OUT}/val" \
    --split val \
    --data_dir "$DATA_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --shard_size 256

echo "[$(date)] Cityscapes TFRecords complete."
