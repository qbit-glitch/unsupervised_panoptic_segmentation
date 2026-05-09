#!/bin/bash
# Wait for rsync to finish, then launch MSDA transformer+hybrid on santosh
# Run locally: nohup bash scripts/wait_and_launch_msda_remote.sh > logs/wait_launch_msda.log 2>&1 &
set -euo pipefail

REMOTE="santosh@172.17.254.146"
FEATURE_DIR="/home/santosh/datasets/cityscapes/dinov3_features_vitl16"
DEPTH_DIR="/home/santosh/datasets/cityscapes/depth_depthpro"

echo "[$(date)] Waiting for rsync to complete..."

while true; do
    RSYNC_COUNT=$(pgrep -f "rsync.*cityscapes" | wc -l)
    if [ "$RSYNC_COUNT" -eq 0 ]; then
        echo "[$(date)] All rsync processes finished."
        break
    fi
    echo "[$(date)] Still rsyncing... ($RSYNC_COUNT processes)"
    sleep 60
done

echo "[$(date)] Verifying data on remote..."
TRAIN_COUNT=$(ssh "$REMOTE" "find $FEATURE_DIR/train -name '*.npy' 2>/dev/null | wc -l")
VAL_COUNT=$(ssh "$REMOTE" "find $FEATURE_DIR/val -name '*.npy' 2>/dev/null | wc -l")
DEPTH_COUNT=$(ssh "$REMOTE" "find $DEPTH_DIR -name '*.npy' 2>/dev/null | wc -l")

echo "Features: train=$TRAIN_COUNT, val=$VAL_COUNT"
echo "Depth: $DEPTH_COUNT"

if [ "$TRAIN_COUNT" -lt 2900 ] || [ "$VAL_COUNT" -lt 400 ]; then
    echo "ERROR: Incomplete feature transfer (expected 2975 train, 500 val)"
    exit 1
fi

echo "[$(date)] Data verified. Launching transformer+hybrid training..."
ssh "$REMOTE" "cd /home/santosh/mbps_panoptic_segmentation && bash scripts/run_msda_remote.sh transformer hybrid"

echo "[$(date)] Done! Training launched on remote."
echo "Monitor: ssh $REMOTE 'tail -f /home/santosh/mbps_panoptic_segmentation/logs/msda_transformer_hybrid.log'"
