#!/bin/bash
# Full Cityscapes pipeline: download, extract, depth, upload to GCS.
# Run with nohup on the VM:
#   nohup bash scripts/run_cityscapes_pipeline.sh > logs/cityscapes_pipeline.log 2>&1 &
set -euo pipefail

export PATH=$HOME/.local/bin:$PATH
export CITYSCAPES_USERNAME="2024pcp5302@mnit.ac.in"
export CITYSCAPES_PASSWORD="Umesh@QSB25"

PROJ=$HOME/mbps_panoptic_segmentation
CS=$PROJ/data/cityscapes
BUCKET=gs://mbps-panoptic

cd "$PROJ"
mkdir -p "$CS" logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Cityscapes Pipeline ==="
log "Disk free: $(df -h / | awk 'NR==2{print $4}')"

# Step 1: Download
log "Downloading leftImg8bit_trainvaltest.zip (~11GB)..."
rm -f "$CS/leftImg8bit_trainvaltest.zip"
PYTHONUNBUFFERED=1 python3 scripts/download_cityscapes.py \
    --output_dir "$CS" --packages leftImg8bit_trainvaltest.zip
log "leftImg8bit done."

log "Downloading gtFine_trainvaltest.zip (~250MB)..."
rm -f "$CS/gtFine_trainvaltest.zip"
PYTHONUNBUFFERED=1 python3 scripts/download_cityscapes.py \
    --output_dir "$CS" --packages gtFine_trainvaltest.zip
log "gtFine done."

log "Disk after downloads: $(df -h / | awk 'NR==2{print $4}') free"

# Step 2: Extract + delete zips
log "Extracting leftImg8bit..."
cd "$CS"
unzip -q leftImg8bit_trainvaltest.zip && rm -f leftImg8bit_trainvaltest.zip
log "Extracting gtFine..."
unzip -q gtFine_trainvaltest.zip && rm -f gtFine_trainvaltest.zip
cd "$PROJ"
log "Extracted. Disk: $(df -h / | awk 'NR==2{print $4}') free"

# Step 3: Upload raw images to GCS
log "Uploading raw images to GCS..."
gsutil -m rsync -r "$CS/leftImg8bit/" "$BUCKET/datasets/cityscapes/leftImg8bit/"
gsutil -m rsync -r "$CS/gtFine/" "$BUCKET/datasets/cityscapes/gtFine/"
log "Raw images uploaded."

# Step 4: Depth maps (ZoeDepth on CPU, ~5K images, ~4-6 hrs)
log "Computing depth maps (ZoeDepth CPU, ~5K images)..."
PYTHONUNBUFFERED=1 python3 scripts/precompute_depth.py \
    --data_dir "$CS/leftImg8bit" \
    --output_dir "$CS/depth_zoedepth" \
    --dataset cityscapes \
    --image_size 512 1024

log "Uploading depth maps to GCS..."
gsutil -m rsync -r "$CS/depth_zoedepth/" "$BUCKET/datasets/cityscapes/depth_zoedepth/"
log "Depth maps uploaded."

# Step 5: Done
log "=== Cityscapes pipeline complete ==="
log "Disk: $(df -h / | awk 'NR==2{print $4}') free"
log "Next: create TFRecords after depth maps are ready."
