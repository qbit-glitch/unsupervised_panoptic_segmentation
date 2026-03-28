#!/bin/bash
# Launch 100-step Stage 2 training test on remote machine
# Usage: bash scripts/run_test100.sh

set -e

cd /media/santosh/Kuldeep/panoptic_segmentation

# Activate cups environment
source /home/santosh/anaconda3/etc/profile.d/conda.sh
conda activate cups

# Fix potential LD_LIBRARY_PATH for GLIBCXX
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH

echo "Starting 100-step Stage 2 test training..."
echo "Config: configs/cups_cityscapes_test100.yaml"
echo "GPUs: 2x GTX 1080 Ti"
echo "Strategy: DDP with gloo backend (CPU validation compatible)"
echo "---"

python refs/cups/train.py \
    --experiment_config_file configs/cups_cityscapes_test100.yaml \
    --disable_wandb \
    2>&1 | tee training_test100.log

echo "---"
echo "Training complete. Log saved to training_test100.log"
