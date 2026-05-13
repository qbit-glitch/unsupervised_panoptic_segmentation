#!/bin/bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/mbps_panoptic_segmentation

DATA_DIR="$HOME/mbps_panoptic_segmentation/data/coco"
DEPTH_DIR="gs://mbps-panoptic/datasets/coco/depth_zoedepth"
GCS_OUT="gs://mbps-panoptic/datasets/coco/tfrecords"

# Step 1: Download COCO-Stuff annotations if not present
ANN_DIR="$DATA_DIR/annotations"
if [ ! -d "$ANN_DIR/stuff_train2017_pixelmaps" ]; then
    echo "[$(date)] Downloading COCO-Stuff annotations (~600MB)..."
    wget -q --show-progress -O "$DATA_DIR/stuffthingmaps_trainval2017.zip" \
        "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
    echo "[$(date)] Extracting..."
    cd "$DATA_DIR" && unzip -q stuffthingmaps_trainval2017.zip && rm -f stuffthingmaps_trainval2017.zip
    cd ~/mbps_panoptic_segmentation
    echo "[$(date)] Annotations ready."
else
    echo "[$(date)] Annotations already present."
fi

echo "[$(date)] Creating COCO TFRecords (depth from GCS — this takes a while)..."

echo "[$(date)] Train split (118k images)..."
PYTHONUNBUFFERED=1 python3 scripts/create_tfrecords.py \
    --config configs/coco_stuff27.yaml \
    --output_dir "${GCS_OUT}/train" \
    --split train \
    --data_dir "$DATA_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --shard_size 500

echo "[$(date)] Val split (5k images)..."
PYTHONUNBUFFERED=1 python3 scripts/create_tfrecords.py \
    --config configs/coco_stuff27.yaml \
    --output_dir "${GCS_OUT}/val" \
    --split val \
    --data_dir "$DATA_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --shard_size 500

echo "[$(date)] COCO TFRecords complete."
