#!/usr/bin/env bash
# Stage-2 ResNet-50 + Cascade CMRCNN on bilinear labels (ablA, seed 42).
# DDP across both GPUs: effective batch = 2 GPUs x bs2 x accum4 = 16.
# NOTE: gloo DDP can die headless under setsid (same issue as DINOv2 arm).
# If it crashes with "Connection closed by peer", switch to single GPU:
#   CUDA_VISIBLE_DEVICES=0 + SYSTEM.NUM_GPUS 1 + TRAINING.ACCUMULATE_GRAD_BATCHES 8
# Eval with eval_cascade_checkpoint_19.py (NOT eval_eomt_checkpoint_19.py).
set -e

REPO="/home/santosh/mbps_panoptic_segmentation"
CFG="$REPO/refs/cups/configs/stage2_ab_resnet50_bilinear.yaml"
LOG="$REPO/logs/stage2_resnet50_bilinear.log"
CONDA="/home/santosh/anaconda3/envs/cups/bin/python"

mkdir -p "$REPO/logs"

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 \
  nohup setsid "$CONDA" -u \
    "$REPO/refs/cups/train_eomt.py" \
    --experiment_config_file "$CFG" \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "Log: $LOG"
echo "tail -f $LOG"
