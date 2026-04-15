#!/bin/bash
# Kill old training, clean broken checkpoints, restart with fixed weights.
# Run: nohup bash scripts/restart_cups_train.sh > cups_dinov3_train_v2.log 2>&1 &
set -e

echo "=== Finding old training processes ==="
ps aux | grep -E "train\.py|run_cups_train" | grep -v grep || echo "No training processes found"

echo "=== Killing old training ==="
pkill -f "run_cups_train" || true
pkill -f "train\.py.*experiment_config_file" || true
pkill -f "train\.py.*dinov3" || true
sleep 3

echo "=== Verify killed ==="
ps aux | grep -E "train\.py|run_cups_train" | grep -v grep && echo "WARNING: process still alive!" || echo "All killed."

echo "=== Cleaning broken checkpoints (trained with random backbone) ==="
OLD_CKPT_DIR="$HOME/umesh/experiments/cups_dinov3_vitb_k80_anydesk"
if [ -d "$OLD_CKPT_DIR" ]; then
    echo "Removing: $OLD_CKPT_DIR"
    rm -rf "$OLD_CKPT_DIR"
fi

echo "=== Starting training with correctly converted DINOv3 weights ==="
cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups
export WANDB_MODE=disabled
python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml TRAINING.BATCH_SIZE 4 TRAINING.ACCUMULATE_GRAD_BATCHES 4
