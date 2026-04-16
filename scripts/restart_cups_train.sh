#!/bin/bash
# Kill old training, clean checkpoints, restart with bs=1 accum=16 (matches santosh PQ=28.4 config).
# Run: nohup bash scripts/restart_cups_train.sh > cups_dinov3_depthpro_tau020_bs1.log 2>&1 &
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

echo "=== Cleaning old checkpoints ==="
for d in "$HOME/umesh/experiments/cups_dinov3_vitb_depthpro_tau020_anydesk"*; do
    if [ -d "$d" ]; then
        echo "Removing: $d"
        rm -rf "$d"
    fi
done

echo "=== Starting training with bs=1 accum=16 (matches santosh PQ=28.4 config) ==="
cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups
export WANDB_MODE=disabled
python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml
