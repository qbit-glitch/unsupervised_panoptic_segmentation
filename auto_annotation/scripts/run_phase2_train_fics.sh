#!/bin/bash
# Phase-2 EoMT-mobile training launcher for fics-lab.
# Waits for the Phase-1 auto-label to finish, then trains EoMT (dinov2-small)
# on the auto-labels with gradient accumulation -> effective batch size 16.
set -u
COCO=/mnt/HDD_16TB/coco
ROOT=/mnt/HDD_16TB/mbps_mobile
LOG=$COCO/phase2_train.log
EPOCHS=${1:-40}

echo "[$(date)] waiting for Phase-1 auto-label to finish (phase1_autolabel.log DONE)..." >> "$LOG"
while ! grep -q "DONE" "$COCO/phase1_autolabel.log" 2>/dev/null; do sleep 300; done
N=$(ls "$COCO/autolabels_train"/*.png 2>/dev/null | wc -l)
echo "[$(date)] $N labels ready, starting EoMT-mobile training (eff_batch=16)" >> "$LOG"

export COCO_ROOT=$COCO PYTHONPATH=$ROOT
export HF_HOME=/mnt/HDD_16TB/jaipur_work/hf_cache HF_HUB_OFFLINE=1
/mnt/HDD_16TB/jaipur_work/env/bin/python "$ROOT/mbps_pytorch/mobile_panoptic_sup/train_eomt_mobile.py" \
    --data_dir "$COCO/autolabels_train" --img_dir "$COCO/train2017" \
    --device cuda --img 640 --eff_batch 16 --micro_batch 4 --epochs "$EPOCHS" \
    --backbone facebook/dinov2-small --out "$COCO/ckpt_eomt_dinov2s" >> "$LOG" 2>&1

echo "[$(date)] TRAIN DONE -> $COCO/ckpt_eomt_dinov2s" >> "$LOG"
