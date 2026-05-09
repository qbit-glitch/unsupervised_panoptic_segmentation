#!/bin/bash
# MSDA Ablation: 2-phase architecture/loss sweep
# Phase 1: 4 architectures × hybrid loss → find best architecture
# Phase 2: best architecture × 4 losses → find best loss
#
# Usage:
#   bash scripts/run_msda_ablation.sh phase1     # Architecture sweep
#   bash scripts/run_msda_ablation.sh phase2 conv # Loss sweep with best arch
#   bash scripts/run_msda_ablation.sh all         # Both phases sequentially

set -euo pipefail

FEATURE_DIR="/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features_vitl16"
DEPTH_DIR="/Users/qbit-glitch/Desktop/datasets/cityscapes/depth_depthpro"
GT_DIR="/Users/qbit-glitch/Desktop/datasets/cityscapes/gtFine"
OUTPUT_DIR="checkpoints/msda"
EPOCHS=50
BATCH_SIZE=8
LR=1e-4
SEED=42

ARCHS=("conv" "transformer" "slot" "conv_transformer")
LOSSES=("depth_contrastive" "stego" "swav_sinkhorn" "hybrid")

run_training() {
    local arch=$1
    local loss=$2
    local run_name="${arch}_${loss}"
    local log_file="logs/msda_${run_name}.log"

    mkdir -p logs "$OUTPUT_DIR/$run_name"

    echo "========================================="
    echo "Training: arch=${arch}, loss=${loss}"
    echo "Log: ${log_file}"
    echo "========================================="

    python -u -m mbps_pytorch.msda.train \
        --arch "$arch" \
        --loss "$loss" \
        --feature_dir "$FEATURE_DIR" \
        --depth_dir "$DEPTH_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$log_file"

    echo "Training complete: ${run_name}"
}

run_evaluation() {
    local arch=$1
    local loss=$2
    local run_name="${arch}_${loss}"
    local ckpt="$OUTPUT_DIR/$run_name/best.pt"

    if [ ! -f "$ckpt" ]; then
        echo "WARNING: No checkpoint found at $ckpt, skipping eval"
        return
    fi

    echo "Evaluating: ${run_name}"

    python -u -m mbps_pytorch.msda.evaluate \
        --checkpoint "$ckpt" \
        --feature_dir "$FEATURE_DIR" \
        --depth_dir "$DEPTH_DIR" \
        --gt_dir "$GT_DIR" \
        --k 100 \
        --batch_size 8 \
        2>&1 | tee "logs/msda_eval_${run_name}.log"
}

phase1() {
    echo "===== PHASE 1: Architecture Sweep (loss=hybrid) ====="
    for arch in "${ARCHS[@]}"; do
        run_training "$arch" "hybrid"
        run_evaluation "$arch" "hybrid"
    done

    echo ""
    echo "===== PHASE 1 RESULTS ====="
    for arch in "${ARCHS[@]}"; do
        local result="$OUTPUT_DIR/${arch}_hybrid/eval_results.json"
        if [ -f "$result" ]; then
            local miou
            miou=$(python3 -c "import json; r=json.load(open('$result')); print(f'{r[\"mIoU\"]:.2f}')")
            local recovered
            recovered=$(python3 -c "import json; r=json.load(open('$result')); print(r['dead_classes_recovered'])")
            echo "  ${arch}: mIoU=${miou}%, dead_recovered=${recovered}/7"
        fi
    done
}

phase2() {
    local best_arch="${1:-conv}"
    echo "===== PHASE 2: Loss Sweep (arch=${best_arch}) ====="
    for loss in "${LOSSES[@]}"; do
        if [ "$loss" = "hybrid" ] && [ -f "$OUTPUT_DIR/${best_arch}_hybrid/best.pt" ]; then
            echo "Skipping ${best_arch}_hybrid (already trained in Phase 1)"
            continue
        fi
        run_training "$best_arch" "$loss"
        run_evaluation "$best_arch" "$loss"
    done

    echo ""
    echo "===== PHASE 2 RESULTS ====="
    for loss in "${LOSSES[@]}"; do
        local result="$OUTPUT_DIR/${best_arch}_${loss}/eval_results.json"
        if [ -f "$result" ]; then
            local miou
            miou=$(python3 -c "import json; r=json.load(open('$result')); print(f'{r[\"mIoU\"]:.2f}')")
            local recovered
            recovered=$(python3 -c "import json; r=json.load(open('$result')); print(r['dead_classes_recovered'])")
            echo "  ${loss}: mIoU=${miou}%, dead_recovered=${recovered}/7"
        fi
    done
}

case "${1:-}" in
    phase1)
        phase1
        ;;
    phase2)
        phase2 "${2:-conv}"
        ;;
    all)
        phase1
        echo ""
        echo "Phase 1 complete. Review results above and run:"
        echo "  bash scripts/run_msda_ablation.sh phase2 <best_arch>"
        ;;
    *)
        echo "Usage: $0 {phase1|phase2 <arch>|all}"
        exit 1
        ;;
esac
