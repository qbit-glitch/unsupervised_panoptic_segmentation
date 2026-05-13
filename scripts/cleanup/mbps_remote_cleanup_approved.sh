#!/usr/bin/env bash
# Approved deletion list from user (8 paths, ~200 GB).
# Usage: bash mbps_remote_cleanup_approved.sh [dry|run]
set -u
MODE="${1:-dry}"

PATHS=(
  "$HOME/cups/experiments/experiments/e2_clean_conv_dora_r4"
  "$HOME/cups/experiments/experiments/e2_depthpro_conv_dora_r4_1gpu"
  "$HOME/cups/experiments/experiments/e2_depthpro_conv_dora_r4_2gpu"
  "$HOME/cups/experiments/experiments/e2_dinov3_vitb_k80_conv_dora_r4"
  "$HOME/.cache/pip"
  "$HOME/datasets/cityscapes/rightImg8bit"
  "$HOME/experiments/stage2_dcfa_simcf_abc_dora_r32"
  "$HOME/experiments/stage3_dcfa_simcf_abc_dora_r32"
)

echo "=== BEFORE ==="
df -h /home / 2>/dev/null | head -3

for p in "${PATHS[@]}"; do
  echo "----"
  if [[ ! -e "$p" ]]; then echo "(skip, not present) $p"; continue; fi
  du -sh -- "$p" 2>/dev/null
  if [[ "$MODE" == "run" ]]; then
    rm -rf -- "$p" && echo "[DONE] removed $p" || echo "[FAIL] $p"
  else
    echo "[DRY] would: rm -rf -- $p"
  fi
done

echo "=== AFTER ==="
df -h /home / 2>/dev/null | head -3
echo "MODE=$MODE"
