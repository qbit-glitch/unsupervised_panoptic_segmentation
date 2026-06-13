#!/usr/bin/env bash
# Fusion adapter training matrix (plan T9 side A + T11 side B).
# Phase 0b PASSED (headroom 8.63 mono / 11.54 official, 2026-06-13).
# ~7 min/epoch on MPS => ~6h per 50-epoch run. Sequential to avoid MPS contention.
#
# Usage:
#   bash scripts/run_fusion_adapter_training.sh primaries   # A2 + B1 (headline configs)
#   bash scripts/run_fusion_adapter_training.sh ablations    # A1 A3 A4 B2
#   bash scripts/run_fusion_adapter_training.sh all
# Launch in background:
#   nohup bash scripts/run_fusion_adapter_training.sh primaries \
#     > logs/fusion_train_primaries_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -u
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
PY=.venv_cups_cpu/bin/python
TS() { date +%Y%m%d_%H%M%S; }
SET="${1:-primaries}"

run() {  # name side teacher_mode proj_width
  local name="$1" side="$2" mode="$3" w="$4"
  echo "=== [$name] side=$side mode=$mode width=$w START $(date) ==="
  $PY mbps_pytorch/train_fusion_adapter.py \
      --side "$side" --teacher_mode "$mode" --proj_width "$w" \
      --epochs 50 --batch_size 32 --lambda_preserve 20.0 --device mps \
      --output_dir "results/fusion_adapter/${name}" \
      > "logs/fusion_${name}_$(TS).log" 2>&1
  echo "=== [$name] DONE $(date) rc=$? ==="
}

if [ "$SET" = "primaries" ] || [ "$SET" = "all" ]; then
  run A2_strat_w16 A strat 16   # side-A primary
  run B1_strat_w16 B strat 16   # side-B primary
fi
if [ "$SET" = "ablations" ] || [ "$SET" = "all" ]; then
  run A1_plain_w16 A plain 16
  run A3_dual_w16  A dual  16
  run A4_strat_w32 A strat 32
  run B2_strat_w64 B strat 64
fi
echo "=== FUSION TRAINING ($SET) COMPLETE $(date) ==="
