#!/bin/bash
# Kill old training, clean checkpoints, restart with CORRECT pseudo-labels.
# Run on A6000: nohup bash scripts/restart_cups_fixed_labels.sh > ~/cups_fixed_labels.log 2>&1 &
set -e

echo "=== Killing old training ==="
pkill -f "train\.py.*experiment_config_file" || true
pkill -f "train\.py.*dinov3" || true
sleep 3
ps aux | grep -E "train\.py" | grep -v grep && echo "WARNING: process still alive!" || echo "All killed."

echo "=== Cleaning old checkpoints ==="
for d in "$HOME/umesh/experiments/cups_dinov3_vitb_depthpro_tau020_anydesk"*; do
    if [ -d "$d" ]; then
        echo "Removing: $d"
        rm -rf "$d"
    fi
done

echo "=== Pulling latest code ==="
cd ~/umesh/unsupervised_panoptic_segmentation
git pull origin main

echo "=== Starting training ==="
cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups
export WANDB_MODE=disabled
python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml
