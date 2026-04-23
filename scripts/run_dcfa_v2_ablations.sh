#!/bin/bash
# DCFA v2 Ablation Sweep — 8 individual experiments + DCFA-X combined.
# Each changes ONE thing from V3 baseline. All on MPS (~30 min/run).
#
# Prerequisites:
#   1. Pre-extract DINOv2 768D: python scripts/extract_cause_codes.py \
#        --cityscapes_root $CS_ROOT --save_dino768
#   2. Pre-extract normals: python scripts/extract_surface_normals.py \
#        --cityscapes_root $CS_ROOT
#
# Run all:   bash scripts/run_dcfa_v2_ablations.sh
# Run one:   bash scripts/run_dcfa_v2_ablations.sh A1
set -euo pipefail

CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
CENTROIDS="$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz"
BASE_OUT="results/depth_adapter/DCFA_v2"
COMMON="--cityscapes_root $CS_ROOT --epochs 20 --lambda_preserve 20.0 --lr 1e-3 --seed 42 --depth_dim 16 --batch_size 8"

run_experiment() {
    local name="$1"
    shift
    local out_dir="$BASE_OUT/${name}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━���━"
    echo "  DCFA v2 Ablation: ${name}"
    echo "  Output: ${out_dir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 -u mbps_pytorch/train_depth_adapter.py \
        $COMMON --output_dir "$out_dir" "$@"
    echo "  Done: ${name}"
}

# Parse optional single-experiment argument
TARGET="${1:-all}"

# ─── Group A: Input Enrichment ────────────────────────────────────────────────

if [[ "$TARGET" == "all" || "$TARGET" == "A1" ]]; then
    # A1: DINOv2 768D cross-attention (requires --save_dino768 pre-extraction)
    run_experiment "A1_cross_attn" \
        --adapter_type cross_attn --use_dino768
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A2" ]]; then
    # A2: Surface normals (requires extract_surface_normals.py)
    run_experiment "A2_normals" \
        --adapter_type v3 --use_normals --hidden_dim 384 --num_layers 2
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A3" ]]; then
    # A3: Depth Sobel gradients (computed on-the-fly)
    run_experiment "A3_gradients" \
        --adapter_type v3 --use_gradients --hidden_dim 384 --num_layers 2
fi

# ─── Group B: Architecture Enhancement ────────────────────────────────────────

if [[ "$TARGET" == "all" || "$TARGET" == "B1" ]]; then
    # B1: FiLM conditioning
    run_experiment "B1_film" \
        --adapter_type film --hidden_dim 384 --num_layers 2
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B2" ]]; then
    # B2: Deeper bottleneck MLP (4 layers)
    run_experiment "B2_deep" \
        --adapter_type deep --hidden_dim 384
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B3" ]]; then
    # B3: Local 3x3 window attention
    run_experiment "B3_window_attn" \
        --adapter_type window_attn --hidden_dim 384 --num_layers 2
fi

# ─── Group C: Loss Improvements ──────────────────────────────────────────────

if [[ "$TARGET" == "all" || "$TARGET" == "C1" ]]; then
    # C1: Contrastive cluster loss (needs centroids)
    run_experiment "C1_contrastive" \
        --adapter_type v3 --hidden_dim 384 --num_layers 2 \
        --lambda_cluster 0.1 --centroids_path "$CENTROIDS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "C2" ]]; then
    # C2: Cross-image hard negative mining
    run_experiment "C2_cross_image" \
        --adapter_type v3 --hidden_dim 384 --num_layers 2 \
        --cross_image_mining
fi

# ─── DCFA-X: Combined Architecture ───────────────────────────────────────────

if [[ "$TARGET" == "all" || "$TARGET" == "X" ]]; then
    # Full DCFA-X: FiLM + cross-attention + normals + contrastive loss
    run_experiment "X_combined" \
        --adapter_type x --use_dino768 --use_normals \
        --lambda_cluster 0.1 --centroids_path "$CENTROIDS"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All requested DCFA v2 ablations complete."
echo "  Results: ${BASE_OUT}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
