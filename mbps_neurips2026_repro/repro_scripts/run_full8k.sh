#!/bin/bash
# Launch 8000-step Stage 2 training on remote machine
# Usage: bash scripts/run_full8k.sh

set -e

cd /media/santosh/Kuldeep/panoptic_segmentation

# Activate cups environment
source /home/santosh/anaconda3/etc/profile.d/conda.sh
conda activate cups

# Fix potential LD_LIBRARY_PATH for GLIBCXX
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH

echo "Starting 8000-step Stage 2 full training..."
echo "Config: configs/cups_cityscapes_full8k.yaml"
echo "GPUs: 2x GTX 1080 Ti"
echo "Strategy: DDP with gloo backend (CPU validation compatible)"
echo "Validation every 500 steps"
echo "---"

python refs/cups/train.py \
    --experiment_config_file configs/cups_cityscapes_full8k.yaml \
    --disable_wandb \
    2>&1 | tee training_full8k.log

echo "---"
echo "Training complete. Log saved to training_full8k.log"
