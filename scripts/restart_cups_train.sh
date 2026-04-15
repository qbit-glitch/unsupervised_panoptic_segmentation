#!/bin/bash
# Kill old training, clean broken checkpoints, restart with fixed weights.
# Run: nohup bash scripts/restart_cups_train.sh > cups_dinov3_train_v2.log 2>&1 &
set -e

echo "=== Killing old training process ==="
pkill -f "train.py.*train_cityscapes_dinov3_vitb_k80_anydesk" || echo "No old process found"
sleep 2

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
