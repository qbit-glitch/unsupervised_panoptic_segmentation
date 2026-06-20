#!/bin/bash
# Phase-1 label-free COCO auto-label launcher for fics-lab.
# Waits for train2017 to finish extracting, then auto-labels LIMIT train images
# with the full label-free set (80 SAM3 thing prompts + 53 INSID3 stuff concepts).
# Resumable: re-run with a larger LIMIT to extend (skips already-labeled images).
set -u
ROOT=/mnt/HDD_16TB/mbps_mobile
COCO=/mnt/HDD_16TB/coco
LOG=$COCO/phase1_autolabel.log
LIMIT=${1:-3000}

echo "[$(date)] waiting for train2017 (limit=$LIMIT) ..." >> "$LOG"
while [ ! -d "$COCO/train2017" ] || [ "$(ls "$COCO/train2017" 2>/dev/null | wc -l)" -lt 118000 ]; do
    sleep 60
done
echo "[$(date)] train2017 ready ($(ls "$COCO/train2017" | wc -l) imgs), starting auto-label" >> "$LOG"

export COCO_ROOT=$COCO PYTHONPATH=$ROOT
export HF_HOME=/mnt/HDD_16TB/jaipur_work/hf_cache HF_HUB_OFFLINE=1 SAM3_OFFLINE=1
/mnt/HDD_16TB/jaipur_work/env/bin/python "$ROOT/auto_annotation/scripts/autolabel_coco_full.py" \
    --img_dir "$COCO/train2017" --device cuda \
    --n_things 80 --n_stuff 53 --limit "$LIMIT" \
    --out "$COCO/autolabels_train" >> "$LOG" 2>&1

echo "[$(date)] DONE ($(ls "$COCO/autolabels_train"/*.png 2>/dev/null | wc -l) images labeled)" >> "$LOG"
