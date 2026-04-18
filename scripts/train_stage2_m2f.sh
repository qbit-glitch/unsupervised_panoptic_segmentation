#!/usr/bin/env bash
# Usage: scripts/train_stage2_m2f.sh <config-relative-path>
# Example: scripts/train_stage2_m2f.sh configs/stage2_m2f/M0_baseline_dinov3_vitb_k80.yaml
set -euo pipefail

CFG="${1:?config path required}"
RUN_NAME="$(basename "$CFG" .yaml)"
LOGDIR="logs/stage2_m2f"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Training $CFG"
echo "Log -> $LOGFILE"

nohup python -u refs/cups/train_pseudo.py --config "$CFG" \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "PID=$PID"
echo "Tail with: tail -f $LOGFILE"
