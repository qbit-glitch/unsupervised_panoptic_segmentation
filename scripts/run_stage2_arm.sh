#!/usr/bin/env bash
# ONE Stage-2 EoMT arm, single-GPU (no DDP -> avoids the headless gloo-barrier crash).
# Usage: run_stage2_arm.sh <name> <config> <gpu_id>
# Effective batch kept at 16 via ACCUMULATE_GRAD_BATCHES=16 (matches the 2-GPU recipe: 1x2x8).
# Launch both arms in parallel (one per GPU):
#   cd ~/mbps_panoptic_segmentation
#   setsid bash scripts/run_stage2_arm.sh bilinear refs/cups/configs/stage2_ab_bilinear.yaml 0 > logs/arm_bilinear.log 2>&1 < /dev/null &
#   setsid bash scripts/run_stage2_arm.sh anyup    refs/cups/configs/stage2_ab_anyup.yaml    1 > logs/arm_anyup.log    2>&1 < /dev/null &
set -u
NAME="$1"; CFG="$2"; GPU="$3"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA="${CONDA:-$HOME/anaconda3/envs/cups}"
PY="$CONDA/bin/python"
export LD_LIBRARY_PATH="$CONDA/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/refs/cups:$REPO/refs/eomt"
# WANDB_MODE=disabled: headless SSH lacks WANDB_API_KEY, and offline mode enumerates packages where
# packaging-26.2 / numpy-2.2.6 have broken dist-info (metadata["Name"]=None) -> wandb crash. disabled
# skips both; we read PQ from eval logs, not W&B.
export WANDB_MODE="${WANDB_MODE:-disabled}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
mkdir -p logs
LOGP="/home/santosh/experiments/anyup_vs_bilinear/$NAME"

echo "########## [$(date)] TRAIN arm=$NAME gpu=$GPU (single-GPU, accum=16) ##########"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u refs/cups/train_eomt.py \
    --experiment_config_file "$CFG" --disable_wandb SYSTEM.NUM_GPUS 1 TRAINING.ACCUMULATE_GRAD_BATCHES 16 \
  || { echo "!!! TRAIN $NAME FAILED"; exit 1; }

echo "########## [$(date)] EVAL arm=$NAME (every best_pq ckpt) ##########"
found=0
while IFS= read -r ck; do
  [ -z "$ck" ] && continue
  found=1
  echo "---- eval $NAME : $ck ----"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u refs/cups/eval_eomt_checkpoint.py \
      --experiment_config_file "$CFG" --ckpt "$ck" SYSTEM.NUM_GPUS 1 \
    2>&1 | tee "logs/eval_${NAME}_$(basename "$ck" .ckpt).txt"
done < <(find "$LOGP" -name 'best_pq_*.ckpt' 2>/dev/null | sort -V)
[ "$found" -eq 0 ] && echo "!!! no best_pq ckpt under $LOGP for arm=$NAME"

echo "########## [$(date)] DONE arm=$NAME -- PQ summary ##########"
grep -H -E '^[[:space:]]+(PQ|PQ_things|PQ_stuff|SQ|RQ)[[:space:]]*=' logs/eval_${NAME}_*.txt 2>/dev/null \
  || echo "read logs/eval_${NAME}_*.txt directly"
