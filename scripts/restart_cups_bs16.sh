#!/bin/bash
# Kill old training, start CUPS with bs=16 config (full batch, no accumulation).
# Run on A6000:
#   cd ~/umesh/unsupervised_panoptic_segmentation
#   nohup bash scripts/restart_cups_bs16.sh > ~/cups_bs16.log 2>&1 &
#   tail -f ~/cups_bs16.log
set -e

echo "=== Killing old training ==="
pkill -f "train\.py.*experiment_config_file" || true
pkill -f "train\.py.*dinov3" || true
sleep 3
ps aux | grep -E "train\.py" | grep -v grep && echo "WARNING: process still alive!" || echo "All killed."

echo "=== Starting training (bs=16, accum=1, effective=16) ==="
cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups
export WANDB_MODE=disabled
python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk_bs16.yaml
