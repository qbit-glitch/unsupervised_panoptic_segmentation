#!/bin/bash
# Evaluate DA3 DoRA adapter on panoptic metrics with tau=0.20, A_min=1000
# Run this on the remote machine (gpunode2) where training was done

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG: Adjust these paths to match your remote setup
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation}"
CS_ROOT="${CS_ROOT:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/da3_dora_adapter/best_val.pt}"
SEM_DIR="${SEM_DIR:-$CS_ROOT/pseudo_semantic_adapter_V3_k80/val}"
PYTHON="${PYTHON:-python}"

# Depth output directory (adapted DA3 depth NPYs)
DEPTH_SUBDIR="depth_da3_dora_adapter"
DEPTH_OUT="$CS_ROOT/$DEPTH_SUBDIR/val"

# Evaluation params
TAU="${TAU:-0.20}"
MIN_AREA="${MIN_AREA:-1000}"
SIGMA="${SIGMA:-0.0}"
DILATION="${DILATION:-3}"

RESULTS_JSON="$PROJECT_ROOT/results/depth_adapter/da3_dora_adapter_tau${TAU}_A${MIN_AREA}.json"
LOG="$PROJECT_ROOT/logs/da3_adapter_panoptic_eval.log"

mkdir -p "$PROJECT_ROOT/results/depth_adapter"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$DEPTH_OUT"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "  DA3 DoRA Adapter Panoptic Evaluation"
echo "========================================================================"
echo "Checkpoint:  $CHECKPOINT"
echo "Semantics:   $SEM_DIR"
echo "Depth out:   $DEPTH_OUT"
echo "Cityscapes:  $CS_ROOT"
echo "tau=$TAU  min_area=$MIN_AREA  sigma=$SIGMA  dilation=$DILATION"
echo "Started: $(date)"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Step 1: Generate adapted depth NPYs (skip if already exist)
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Generating adapted DA3 depth NPYs..."
if [ ! -f "$DEPTH_OUT/frankfurt/frankfurt_000000_000294.npy" ]; then
    $PYTHON -u mbps_pytorch/scripts/generate_da3_adapter_depth.py \
        --checkpoint "$CHECKPOINT" \
        --model_type dav3 \
        --image_dir "$CS_ROOT/leftImg8bit/val" \
        --output_dir "$DEPTH_OUT" \
        --device cuda \
        --image_size 512 1024 \
        2>&1 | tee -a "$LOG"
else
    echo "Depth NPYs already exist at $DEPTH_OUT, skipping generation."
fi

# ---------------------------------------------------------------------------
# Step 2: Evaluate panoptic quality
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Evaluating panoptic quality..."
$PYTHON -u mbps_pytorch/evaluate_panoptic_combined.py \
    --sem_dir "$SEM_DIR" \
    --cityscapes_root "$CS_ROOT" \
    --depth_subdir "$DEPTH_SUBDIR" \
    --tau "$TAU" \
    --min_area "$MIN_AREA" \
    --sigma "$SIGMA" \
    --dilation "$DILATION" \
    --output "$RESULTS_JSON" \
    2>&1 | tee -a "$LOG"

echo ""
echo "========================================================================"
echo "  Evaluation complete: $(date)"
echo "  Results: $RESULTS_JSON"
echo "  Log:     $LOG"
echo "========================================================================"
