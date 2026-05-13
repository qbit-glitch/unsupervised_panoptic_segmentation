#!/usr/bin/env bash
# Train DINOv2 backbone adapter with feature distillation across 2 GPUs via DDP.
# CAUSE-TR head stays FROZEN.
#
# Run:
#   nohup bash scripts/train_dino_adapter_distill.sh > logs/train_dino_adapter_distill.log 2>&1 &
#   echo "PID: $!"

set -euo pipefail

# Activate conda environment
source /home/santosh/anaconda3/etc/profile.d/conda.sh
conda activate cups
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}

REPO="/home/santosh/mbps_panoptic_segmentation"
DATA_DIR="/home/santosh/datasets"
OUTPUT_DIR="$REPO/checkpoints/dino_adapter_distill_r4"

cd "$REPO"
mkdir -p logs "$OUTPUT_DIR"

# DDP: 2 GPUs, batch_size=16 per GPU → effective batch_size=32
torchrun --nproc_per_node=2 \
    mbps_pytorch/train_semantic_adapter.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir refs/cause \
    --output_dir "$OUTPUT_DIR" \
    --variant dora \
    --rank 4 \
    --alpha 4.0 \
    --dropout 0.05 \
    --late_block_start 6 \
    --losses distillation \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_every 5 \
    --num_workers 4 \
    --seed 42

echo "Training complete. Best checkpoint: $OUTPUT_DIR/best.pt"
