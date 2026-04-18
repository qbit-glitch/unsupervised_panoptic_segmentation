#!/usr/bin/env bash
# Usage: scripts/eval_stage2_m2f.sh <results-dir>
set -euo pipefail

RESULTS="${1:?results directory required}"
CKPT="$(ls "$RESULTS"/checkpoints/*.ckpt | head -n1)"
CFG="$RESULTS/config.yaml"
OUT="$RESULTS/eval.json"

python refs/cups/eval_pseudo.py \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --output "$OUT"

echo "Wrote $OUT"
