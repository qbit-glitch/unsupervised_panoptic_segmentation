#!/usr/bin/env bash
# Usage: scripts/run_stage2_m2f_ablations.sh <glob> [machine]
# Example:
#   scripts/run_stage2_m2f_ablations.sh 'configs/stage2_m2f/G*.yaml' anydesk
#   STAGE2_MACHINE=anydesk scripts/run_stage2_m2f_ablations.sh 'configs/stage2_m2f/N*.yaml'
#
# Launches each matching config sequentially via scripts/train_stage2_m2f.sh.
# The per-config launcher backgrounds the training job; this script waits
# 2s between kicks so log files get unique timestamps. Coordinate GPU
# availability yourself — this does NOT wait for the previous run.
set -euo pipefail

GLOB="${1:?glob required (e.g. 'configs/stage2_m2f/G*.yaml')}"
MACHINE="${2:-${STAGE2_MACHINE:-anydesk}}"

# Sort for deterministic launch order; filter out _machines/ and Stage3_* by default.
FILES="$(ls $GLOB 2>/dev/null | sort || true)"
[[ -n "$FILES" ]] || { echo "FATAL: glob '$GLOB' matched nothing" >&2; exit 1; }

for CFG in $FILES; do
  # Skip machine override files and the self-train stage-3 config (use run_stage3_self_train.sh)
  case "$CFG" in
    *_machines/*|*Stage3_*) echo "-- skipping $CFG"; continue;;
  esac
  echo "=== Launching $CFG on $MACHINE ==="
  scripts/train_stage2_m2f.sh "$CFG" "$MACHINE"
  sleep 2
done
