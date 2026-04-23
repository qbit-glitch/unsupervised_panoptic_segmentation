#!/bin/bash
# DCFA v2 Resume: A1 already complete, run A2 onwards.
# Fixes: depth_dim now includes geo_dim for v3/deep/window_attn adapters.
set -euo pipefail

CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PROJ_ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
CENTROIDS="$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz"
BASE_OUT="$PROJ_ROOT/results/depth_adapter/DCFA_v2"
COMMON="--cityscapes_root $CS_ROOT --epochs 20 --lambda_preserve 20.0 --lr 1e-3 --seed 42 --depth_dim 16 --batch_size 8"

cd "$PROJ_ROOT"

run_experiment() {
    local name="$1"
    shift
    local out_dir="$BASE_OUT/${name}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$name] Starting at $(date '+%H:%M:%S')"
    echo "  Output: ${out_dir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local t0=$(date +%s)
    python3 -u mbps_pytorch/train_depth_adapter.py \
        $COMMON --output_dir "$out_dir" "$@"
    local t1=$(date +%s)
    echo "  [$name] Done in $((t1 - t0))s"
}

echo "================================================================"
echo "  DCFA v2 Resume from A2 (A1 already complete)"
echo "  Started: $(date)"
echo "================================================================"

# A2: Surface normals
run_experiment "A2_normals" \
    --adapter_type v3 --use_normals --hidden_dim 384 --num_layers 2

# A3: Depth gradients
run_experiment "A3_gradients" \
    --adapter_type v3 --use_gradients --hidden_dim 384 --num_layers 2

# B1: FiLM conditioning
run_experiment "B1_film" \
    --adapter_type film --hidden_dim 384 --num_layers 2

# B2: Deeper bottleneck MLP
run_experiment "B2_deep" \
    --adapter_type deep --hidden_dim 384

# B3: Local window attention
run_experiment "B3_window_attn" \
    --adapter_type window_attn --hidden_dim 384 --num_layers 2

# C1: Contrastive cluster loss
run_experiment "C1_contrastive" \
    --adapter_type v3 --hidden_dim 384 --num_layers 2 \
    --lambda_cluster 0.1 --centroids_path "$CENTROIDS"

# C2: Cross-image mining
run_experiment "C2_cross_image" \
    --adapter_type v3 --hidden_dim 384 --num_layers 2 \
    --cross_image_mining

# DCFA-X: Combined
run_experiment "X_combined" \
    --adapter_type x --use_dino768 --use_normals \
    --lambda_cluster 0.1 --centroids_path "$CENTROIDS"

echo ""
echo "================================================================"
echo "  DCFA v2 ALL experiments complete."
echo "  Finished: $(date)"
echo "  Results: $BASE_OUT/"
echo "================================================================"
