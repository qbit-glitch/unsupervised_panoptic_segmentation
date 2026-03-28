#!/bin/bash
# Launch Cityscapes depth computation (run with nohup on VM)
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/mbps_panoptic_segmentation
mkdir -p logs

pkill -f precompute_depth 2>/dev/null || true
sleep 1

echo "Starting Cityscapes depth computation..."
PYTHONUNBUFFERED=1 python3 scripts/precompute_depth.py \
    --data_dir data/cityscapes/leftImg8bit \
    --output_dir data/cityscapes/depth_zoedepth \
    --dataset cityscapes \
    --image_size 512 1024

echo "Depth computation complete."
echo "Uploading to GCS..."
gsutil -m rsync -r data/cityscapes/depth_zoedepth/ gs://mbps-panoptic/datasets/cityscapes/depth_zoedepth/
echo "Upload complete."
