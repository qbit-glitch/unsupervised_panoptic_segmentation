#!/bin/bash
# Kill old training, clean checkpoints, restart with FIXED thing/stuff split.
# This forces santosh's proven 15 thing classes instead of A6000's wrong 12.
#
# Run on A6000: nohup bash scripts/restart_cups_fixed_split.sh > ~/cups_fixed_split.log 2>&1 &
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

echo "=== Pulling latest code (thing/stuff override) ==="
cd ~/umesh/unsupervised_panoptic_segmentation
git fetch origin main
git checkout origin/main -- refs/cups/cups/config.py refs/cups/cups/data/pseudo_label_dataset.py refs/cups/train.py refs/cups/configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml

echo "=== Cleaning old checkpoints ==="
for d in "$HOME/umesh/experiments/cups_dinov3_vitb_depthpro_tau020_anydesk"*; do
    if [ -d "$d" ]; then
        echo "Removing: $d"
        rm -rf "$d"
    fi
done

echo "=== Starting training with FIXED thing/stuff split ==="
echo "Expected: Thing classes (OVERRIDE): (3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75)"
cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups
export WANDB_MODE=disabled
python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml
