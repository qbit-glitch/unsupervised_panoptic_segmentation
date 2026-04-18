#!/usr/bin/env bash
# Sweeps TTA scale combinations over a trained Stage-2 M2F checkpoint.
# Usage: scripts/eval_stage2_m2f_tta.sh <results-dir>
# Assumes <results-dir>/config.yaml and <results-dir>/checkpoints/*.ckpt exist.
set -euo pipefail

RESULTS="${1:?results directory required}"
CKPT="$(ls "$RESULTS"/checkpoints/*.ckpt | head -n1)"
CFG="$RESULTS/config.yaml"

declare -a SCALE_SETS=(
  "1.0"
  "0.75 1.0 1.25"
  "0.5 0.75 1.0 1.25 1.5"
  "0.75 1.0 1.25 1.5"
)

for SCALES in "${SCALE_SETS[@]}"; do
  TAG="$(echo "$SCALES" | tr ' ' '_')"
  OUT="$RESULTS/eval_tta_${TAG}.json"
  python refs/cups/eval_pseudo.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --output "$OUT" \
    --override MODEL.TTA_SCALES "($(echo $SCALES | tr ' ' ','))" VALIDATION.USE_TTA True
  echo "Wrote $OUT"
done
