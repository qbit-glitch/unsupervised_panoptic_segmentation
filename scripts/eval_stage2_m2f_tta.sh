#!/usr/bin/env bash
# Sweeps TTA scale combinations over a trained Stage-2 M2F checkpoint.
# Usage: scripts/eval_stage2_m2f_tta.sh <results-dir>
#
# Expects:
#   <results-dir>/config.yaml             — training config snapshot
#   <results-dir>/checkpoints/*.ckpt      — Lightning checkpoints (picks first)
#
# Writes:
#   <results-dir>/eval_tta_<scales>.json  — per-scale-set PQ/mIoU JSON
#
# Implementation: bakes TTA overrides into temp configs per scale-set and
# invokes refs/cups/evaluate_cityscapes.py (the canonical eval entrypoint).
# The ref evaluator does not expose --override, so we materialise overrides
# as yaml fragments under a temp-dir and delete them after the run.
set -euo pipefail

RESULTS="${1:?results directory required}"
CKPT="$(ls "$RESULTS"/checkpoints/*.ckpt 2>/dev/null | head -n1)"
BASE_CFG="$RESULTS/config.yaml"
[[ -f "$BASE_CFG" ]] || { echo "FATAL: $BASE_CFG missing" >&2; exit 1; }
[[ -n "$CKPT"     ]] || { echo "FATAL: no .ckpt under $RESULTS/checkpoints/" >&2; exit 1; }

declare -a SCALE_SETS=(
  "1.0"
  "0.75 1.0 1.25"
  "0.5 0.75 1.0 1.25 1.5"
  "0.75 1.0 1.25 1.5"
)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for SCALES in "${SCALE_SETS[@]}"; do
  TAG="$(echo "$SCALES" | tr ' ' '_')"
  OUT="$RESULTS/eval_tta_${TAG}.json"
  # Build a yaml override that _BASE_s the training config and flips TTA on
  # with the target scale tuple. yacs tuples in yaml: [v1, v2, ...].
  SCALE_YAML="$(echo "$SCALES" | awk '{for(i=1;i<=NF;i++)printf "%s%s",$i,(i<NF?", ":"")}')"
  OVERRIDE="$TMPDIR/override_${TAG}.yaml"
  cat > "$OVERRIDE" <<EOF
_BASE_: "$BASE_CFG"
MODEL:
  TTA_SCALES: [${SCALE_YAML}]
VALIDATION:
  USE_TTA: True
EOF
  python refs/cups/evaluate_cityscapes.py \
    --experiment_config_file "$OVERRIDE" \
    --checkpoint "$CKPT" \
    --output_json "$OUT"
  echo "Wrote $OUT"
done
