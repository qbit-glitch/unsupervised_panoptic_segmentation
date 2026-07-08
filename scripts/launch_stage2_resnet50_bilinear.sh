#!/usr/bin/env bash
# Stage-2 ResNet-50 + Cascade CMRCNN on bilinear labels (ablA, seed 42).
# Use GPU0 (same GPU that ran bilinear DINOv2 arm).
# Eval with eval_cascade_checkpoint_19.py after training.
set -e

REPO="/home/santosh/mbps_panoptic_segmentation"
CFG="$REPO/refs/cups/configs/stage2_ab_resnet50_bilinear.yaml"
LOG="$REPO/logs/stage2_resnet50_bilinear.log"
CONDA="/home/santosh/anaconda3/envs/cups/bin/python"

mkdir -p "$REPO/logs"

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
  nohup setsid "$CONDA" -u \
    "$REPO/refs/cups/train_eomt.py" \
    --experiment_config_file "$CFG" \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "Log: $LOG"
echo "tail -f $LOG"
