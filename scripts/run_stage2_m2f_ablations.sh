#!/usr/bin/env bash
# Usage: scripts/run_stage2_m2f_ablations.sh <glob>
# Example: scripts/run_stage2_m2f_ablations.sh 'configs/stage2_m2f/G*.yaml'
set -euo pipefail

GLOB="${1:?glob required (e.g. 'configs/stage2_m2f/G*.yaml')}"
for CFG in $GLOB; do
  echo "=== Launching $CFG ==="
  scripts/train_stage2_m2f.sh "$CFG"
  # Wait for foreground completion? No — launcher backgrounds.
  # This script kicks off all 10 in sequence; user coordinates GPU availability.
  sleep 2
done
