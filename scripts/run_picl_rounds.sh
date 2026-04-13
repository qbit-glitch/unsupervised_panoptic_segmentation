#!/bin/bash
# PICL Iterative Refinement Pipeline
#
# Usage:
#   bash scripts/run_picl_rounds.sh 1   # Run Round 1 (uses spidepth instances)
#   bash scripts/run_picl_rounds.sh 2   # Run Round 2 (uses Round 1 instances)
#
# Each round: train → eval → generate train-split instances for next round.

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
CS_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
CKPT_BASE="checkpoints/picl"
RESULTS_DIR="results/ablation_picl"
LOG_DIR="logs"
ROUND=${1:-1}

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Determine instance subdir for this round
if [ "$ROUND" -eq 1 ]; then
    INST_SUBDIR="pseudo_instance_spidepth"
    PREV_PQ="19.41"
else
    PREV_ROUND=$((ROUND - 1))
    INST_SUBDIR="pseudo_instance_picl_r${PREV_ROUND}"
    PREV_PQ=$(cat "$RESULTS_DIR/round${PREV_ROUND}_best_pq_things.txt" 2>/dev/null || echo "0.0")
fi

CKPT_DIR="$CKPT_BASE/round${ROUND}"

echo "========================================"
echo "  PICL Round $ROUND"
echo "  Instance subdir: $INST_SUBDIR"
echo "  Previous best PQ_things: $PREV_PQ"
echo "========================================"

# ── Stage 1: Train PICL ──────────────────────────────────────────────────────
echo ""
echo "--- Stage 1: Training PICL (Round $ROUND) ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/train_picl.py \
    --cityscapes_root "$CS_ROOT" \
    --instance_subdir "$INST_SUBDIR" \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --depth_subdir depth_spidepth \
    --feature_subdir dinov2_features \
    --picl_config 1 \
    --epochs 20 \
    --batch_size 4 \
    --num_workers 4 \
    --val_images 50 \
    --output_dir "$CKPT_DIR" \
    2>&1 | tee "$LOG_DIR/picl_train_round${ROUND}.log"

echo "Training complete. Checkpoint: $CKPT_DIR/best.pth"

# ── Stage 2: Evaluate on val (500 images, sweep) ─────────────────────────────
echo ""
echo "--- Stage 2: Evaluating PICL (Round $ROUND, 500 val images) ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/eval_picl.py \
    --cityscapes_root "$CS_ROOT" \
    --checkpoint "$CKPT_DIR/best.pth" \
    --split val \
    --max_images 500 \
    --sweep \
    --round_id "$ROUND" \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --depth_subdir depth_spidepth \
    --feature_subdir dinov2_features \
    --output_dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_DIR/picl_eval_round${ROUND}.log"

CURR_PQ=$(cat "$RESULTS_DIR/round${ROUND}_best_pq_things.txt" 2>/dev/null || echo "0.0")
echo ""
echo "  Round $ROUND PQ_things: $CURR_PQ  (was: $PREV_PQ)"

# ── Stage 3: Generate train-split instances for Round N+1 ────────────────────
echo ""
echo "--- Stage 3: Generating Round $ROUND instances on train split ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/generate_picl_instances.py \
    --cityscapes_root "$CS_ROOT" \
    --checkpoint "$CKPT_DIR/best.pth" \
    --split train \
    --output_subdir "pseudo_instance_picl_r${ROUND}" \
    --semantic_subdir pseudo_semantic_mapped_k80 \
    --depth_subdir depth_spidepth \
    --feature_subdir dinov2_features \
    2>&1 | tee "$LOG_DIR/picl_gen_round${ROUND}.log"

echo ""
echo "========================================"
echo "  Round $ROUND COMPLETE"
echo "  PQ_things: $CURR_PQ (prev: $PREV_PQ)"
echo "  Next round: bash scripts/run_picl_rounds.sh $((ROUND + 1))"
echo "========================================"
