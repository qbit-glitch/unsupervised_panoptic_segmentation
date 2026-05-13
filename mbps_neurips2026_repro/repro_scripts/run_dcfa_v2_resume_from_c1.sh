#!/bin/bash
# DCFA v2 Resume: A1-B3 complete, run C1 onwards.
# Fix: centroids sliced to 90D code space (was 106D = codes+depth).
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
echo "  DCFA v2 Resume from C1 (A1-B3 complete)"
echo "  Started: $(date)"
echo "================================================================"

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
