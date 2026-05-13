#!/bin/zsh
# Evaluate Original adapter and V3 adapter with k=80 overclustering
set -euo pipefail

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
CS_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
LOG="logs/adapter_k80_eval.log"
mkdir -p logs

echo "=== Adapter k=80 Evaluation ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

# --- Original Adapter (1D raw, h=128, 2L, lp=20) ---
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  ORIGINAL: 1D raw, h=128, 2L, lp=20" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "[1/4] Generating pseudo-labels (Original, k=80)..." | tee -a "$LOG"
$PYTHON -u mbps_pytorch/generate_depth_overclustered_semantics.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --adapter_checkpoint results/depth_adapter/lp20.0/best.pt \
    --variant sinusoidal --alpha 0.1 --k 80 \
    --output_subdir "pseudo_semantic_adapter_orig_k80" \
    --skip_crf \
    2>&1 | tee -a "$LOG"

echo "[2/4] Evaluating (Original, k=80)..." | tee -a "$LOG"
echo "--- ORIGINAL k=80 Results ---" | tee -a "$LOG"
$PYTHON -u mbps_pytorch/evaluate_semantic_pseudolabels.py \
    --pred_dir "${CS_ROOT}/pseudo_semantic_adapter_orig_k80/val" \
    --gt_dir "${CS_ROOT}/gtFine/val" \
    --output "results/depth_adapter/orig_k80_eval.json" \
    2>&1 | tee -a "$LOG"

# --- V3 Adapter (16D sinusoidal, h=384, 2L, lp=20) ---
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  V3: 16D sinusoidal, h=384, 2L, lp=20" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "[3/4] Generating pseudo-labels (V3, k=80)..." | tee -a "$LOG"
$PYTHON -u mbps_pytorch/generate_depth_overclustered_semantics.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --adapter_checkpoint results/depth_adapter/V3_dd16_h384_l2/best.pt \
    --variant sinusoidal --alpha 0.1 --k 80 \
    --output_subdir "pseudo_semantic_adapter_V3_k80" \
    --skip_crf \
    2>&1 | tee -a "$LOG"

echo "[4/4] Evaluating (V3, k=80)..." | tee -a "$LOG"
echo "--- V3 k=80 Results ---" | tee -a "$LOG"
$PYTHON -u mbps_pytorch/evaluate_semantic_pseudolabels.py \
    --pred_dir "${CS_ROOT}/pseudo_semantic_adapter_V3_k80/val" \
    --gt_dir "${CS_ROOT}/gtFine/val" \
    --output "results/depth_adapter/V3_k80_eval.json" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  BOTH EVALUATIONS COMPLETE" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
