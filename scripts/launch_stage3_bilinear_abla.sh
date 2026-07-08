#!/usr/bin/env bash
# Stage-3 self-training: bilinear best (step 2656, PQ=32.01), ablA (no drop/paste).
# Run on santosh GPU0. Requires NVML healthy (reboot if "Can't initialize NVML").
set -e

REPO="/home/santosh/mbps_panoptic_segmentation"
CFG="$REPO/refs/cups/configs/stage3_bilinear_abla_santosh.yaml"
LOG="$REPO/logs/stage3_bilinear_abla.log"
CONDA="/home/santosh/anaconda3/envs/cups/bin/python"

mkdir -p "$REPO/logs"

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
  nohup setsid "$CONDA" -u \
    "$REPO/refs/cups/train_self_eomt_cause.py" \
    --experiment_config_file "$CFG" \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "Log: $LOG"
echo "tail -f $LOG"
