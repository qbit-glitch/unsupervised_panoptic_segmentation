#!/bin/bash
# Depth-Guided Semantic Pseudo-Label Ablation Study
# Run all Approach A (training-free) and Approach B (training) variants.
#
# Phase 1: Approach A on MPS (local, ~1 hr total)
# Phase 2: Approach B on A6000 (remote, ~20 hrs total)
# Phase 3: Evaluation for all variants (~5 min each)
#
# Usage:
#   bash scripts/run_depth_semantic_ablations.sh phase1   # Approach A
#   bash scripts/run_depth_semantic_ablations.sh phase3   # Evaluate all
#   bash scripts/run_depth_semantic_ablations.sh all       # Everything

set -euo pipefail

# ─── Config ───
CS_ROOT="${CITYSCAPES_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PYTHON="${PYTHON:-/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python}"
SPLIT="val"
K=300
LOG_DIR="logs/depth_ablation"
RESULTS_DIR="reports"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# ─── Phase 1: Approach A (Training-Free, MPS) ───
phase1() {
    echo "=== Phase 1: Approach A — Depth-Conditioned Overclustering ==="

    # A0: Baseline (no depth)
    echo "[A0] Baseline (no depth, k=$K)"
    $PYTHON -u mbps_pytorch/generate_depth_overclustered_semantics.py \
        --cityscapes_root "$CS_ROOT" --split "$SPLIT" --k "$K" \
        --variant none --device auto \
        > "$LOG_DIR/A0_none.log" 2>&1
    echo "  Done. Log: $LOG_DIR/A0_none.log"

    # A1-A3: Each variant at alpha=1.0
    for VARIANT in raw sobel sinusoidal; do
        echo "[A1-A3] variant=$VARIANT, alpha=1.0"
        $PYTHON -u mbps_pytorch/generate_depth_overclustered_semantics.py \
            --cityscapes_root "$CS_ROOT" --split "$SPLIT" --k "$K" \
            --variant "$VARIANT" --alpha 1.0 --device auto \
            > "$LOG_DIR/A_${VARIANT}_a1.0.log" 2>&1
        echo "  Done. Log: $LOG_DIR/A_${VARIANT}_a1.0.log"
    done

    # A4: Alpha sweep for each variant
    for VARIANT in raw sobel sinusoidal; do
        for ALPHA in 0.1 0.5 2.0; do
            echo "[A4] variant=$VARIANT, alpha=$ALPHA"
            $PYTHON -u mbps_pytorch/generate_depth_overclustered_semantics.py \
                --cityscapes_root "$CS_ROOT" --split "$SPLIT" --k "$K" \
                --variant "$VARIANT" --alpha "$ALPHA" --device auto \
                > "$LOG_DIR/A_${VARIANT}_a${ALPHA}.log" 2>&1
            echo "  Done. Log: $LOG_DIR/A_${VARIANT}_a${ALPHA}.log"
        done
    done

    echo "Phase 1 complete. All Approach A pseudo-labels generated."
}

# ─── Phase 2: Approach B (Training, A6000) ───
phase2() {
    echo "=== Phase 2: Approach B — CAUSE + Depth Contrastive Fine-Tuning ==="
    echo "Run these on the A6000 machine:"
    echo ""

    for LAMBDA in 0.0 0.01 0.05 0.1; do
        echo "nohup python -u mbps_pytorch/train_cause_depth_finetune.py \\"
        echo "    --data_dir /path/to/datasets \\"
        echo "    --output_dir results/cause_depth_ft_lambda${LAMBDA} \\"
        echo "    --lambda_depth $LAMBDA --epochs 20 --device cuda \\"
        echo "    > logs/B_lambda${LAMBDA}.log 2>&1 &"
        echo ""
    done

    echo "After training, generate pseudo-labels for each checkpoint:"
    echo ""
    for LAMBDA in 0.0 0.01 0.05 0.1; do
        echo "python mbps_pytorch/generate_depth_overclustered_semantics.py \\"
        echo "    --cityscapes_root /path/to/cityscapes --split val --k $K \\"
        echo "    --checkpoint_dir results/cause_depth_ft_lambda${LAMBDA}/epoch_000 \\"
        echo "    --variant none \\"
        echo "    --output_subdir pseudo_semantic_depth_ft_lambda${LAMBDA}_k${K}"
        echo ""
    done
}

# ─── Phase 3: Evaluation ───
phase3() {
    echo "=== Phase 3: Evaluating All Variants ==="

    # Find all pseudo-label directories
    for DIR in "$CS_ROOT"/pseudo_semantic_depth_*_k"$K" "$CS_ROOT"/pseudo_semantic_overclustered_k"$K" "$CS_ROOT"/pseudo_semantic_depth_ft_*_k"$K"; do
        if [ ! -d "$DIR/$SPLIT" ]; then
            continue
        fi
        SUBDIR=$(basename "$DIR")
        EVAL_OUT="eval_depth_ablation_${SUBDIR}.json"

        if [ -f "$EVAL_OUT" ]; then
            echo "[SKIP] $SUBDIR (already evaluated)"
            continue
        fi

        echo "[EVAL] $SUBDIR"
        $PYTHON -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
            --cityscapes_root "$CS_ROOT" --split "$SPLIT" \
            --semantic_subdir "$SUBDIR" \
            --num_clusters "$K" --skip_instance \
            --output "$EVAL_OUT" \
            > "$LOG_DIR/eval_${SUBDIR}.log" 2>&1
        echo "  Done -> $EVAL_OUT"
    done

    echo ""
    echo "Phase 3 complete. Collecting results..."

    # Print summary table
    echo ""
    echo "| Variant | mIoU (%) | Pixel Acc (%) |"
    echo "|---------|----------|---------------|"
    for JSON in eval_depth_ablation_*.json; do
        if [ ! -f "$JSON" ]; then continue; fi
        VARIANT=$(echo "$JSON" | sed 's/eval_depth_ablation_//;s/\.json//')
        MIOU=$($PYTHON -c "import json; d=json.load(open('$JSON')); print(f\"{d['semantic']['miou']:.2f}\")")
        PACC=$($PYTHON -c "import json; d=json.load(open('$JSON')); print(f\"{d['semantic']['pixel_accuracy']:.2f}\")")
        echo "| $VARIANT | $MIOU | $PACC |"
    done
}

# ─── Dispatch ───
case "${1:-all}" in
    phase1) phase1 ;;
    phase2) phase2 ;;
    phase3) phase3 ;;
    all)
        phase1
        phase3
        echo ""
        echo "Phase 2 (training) must be run manually on A6000. See output above."
        phase2
        ;;
    *)
        echo "Usage: $0 {phase1|phase2|phase3|all}"
        exit 1
        ;;
esac
