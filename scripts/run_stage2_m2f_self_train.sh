#!/usr/bin/env bash
# Stage-3 self-training orchestrator.
# Generates pseudo-labels using EMA teacher, thresholds at tau, fine-tunes
# one round, then repeats. Each round produces its own label set and
# checkpoint for ablation auditing.
set -euo pipefail

CFG="configs/stage2_m2f/Stage3_self_train.yaml"
RUN_NAME="Stage3_self_train"
RESULTS="results/stage2_m2f/${RUN_NAME}"
LOGDIR="logs/stage2_m2f"
mkdir -p "$RESULTS" "$LOGDIR"

LOGFILE="${LOGDIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
nohup python -u refs/cups/self_train_pseudo.py \
  --config "$CFG" \
  --output "$RESULTS" \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "PID=$PID"
echo "LOG=$LOGFILE"
