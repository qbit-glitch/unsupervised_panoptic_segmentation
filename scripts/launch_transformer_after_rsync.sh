#!/bin/bash
# Wait for rsync PID to finish, verify data, launch transformer training
# Usage: nohup bash scripts/launch_transformer_after_rsync.sh <rsync_pid> > logs/launch_transformer.log 2>&1 &
set -euo pipefail

RSYNC_PID="${1:?Usage: $0 <rsync_pid>}"
REMOTE="santosh@172.17.254.146"
FEATURE_DIR="/home/santosh/datasets/cityscapes/dinov3_features_vitl16"

echo "[$(date)] Waiting for rsync PID $RSYNC_PID to finish..."
while kill -0 "$RSYNC_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Rsync PID $RSYNC_PID finished."

echo "[$(date)] Verifying data on remote..."
TRAIN_COUNT=$(ssh "$REMOTE" "find $FEATURE_DIR/train -name '*.npy' 2>/dev/null | wc -l")
VAL_COUNT=$(ssh "$REMOTE" "find $FEATURE_DIR/val -name '*.npy' 2>/dev/null | wc -l")
echo "Features: train=$TRAIN_COUNT, val=$VAL_COUNT"

if [ "$TRAIN_COUNT" -lt 2900 ] || [ "$VAL_COUNT" -lt 400 ]; then
    echo "ERROR: Incomplete transfer (expected 2975 train, 500 val). Got $TRAIN_COUNT train, $VAL_COUNT val."
    exit 1
fi

echo "[$(date)] Data OK. Launching transformer+hybrid on remote..."
ssh "$REMOTE" "cd /home/santosh/mbps_panoptic_segmentation && bash scripts/run_msda_remote.sh transformer hybrid"
echo "[$(date)] Done!"
