#!/bin/bash
# Train DCFA V3 adapter with cluster-aware loss (cross-entropy toward centroids).
# Replaces depth correlation loss with direct cluster separability optimization.
set -euo pipefail

CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PROJ_ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
CENTROIDS="$PROJ_ROOT/weights/kmeans_centroids_k80_santosh.npz"

cd "$PROJ_ROOT"

echo "================================================================"
echo "  DCFA V3 — Cluster-Aware Training"
echo "  Loss: CE toward centroid assignments + MSE preservation"
echo "  Started: $(date)"
echo "================================================================"

# Experiment 1: cluster_aware with lp=20 (same preservation weight as V3 baseline)
OUT_DIR="results/depth_adapter/DCFA_v2/CA1_cluster_aware_lp20"
mkdir -p "$OUT_DIR"
echo ""
echo ">>> CA1: cluster_aware, lp=20"
python3 -u mbps_pytorch/train_depth_adapter.py \
    --cityscapes_root "$CS_ROOT" \
    --adapter_type v3 \
    --depth_dim 16 \
    --hidden_dim 384 \
    --num_layers 2 \
    --epochs 20 \
    --lr 1e-3 \
    --lambda_preserve 20.0 \
    --loss_type cluster_aware \
    --centroids_path "$CENTROIDS" \
    --output_dir "$OUT_DIR" \
    --seed 42

echo ""
echo ">>> CA1 eval at k=80"
python3 -u mbps_pytorch/generate_depth_overclustered_semantics.py \
    --cityscapes_root "$CS_ROOT" \
    --k 80 --variant sinusoidal --alpha 0.1 --skip_crf \
    --adapter_checkpoint "$OUT_DIR/best.pt" \
    --output_subdir "pseudo_semantic_dcfa_v2_CA1_k80"

python3 -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --semantic_subdir "pseudo_semantic_dcfa_v2_CA1_k80" \
    --num_clusters 80 \
    --skip_instance \
    --output "$OUT_DIR/eval_k80.json"

echo ""
echo "================================================================"

# Experiment 2: cluster_aware with lp=10 (lower preservation — allow more drift)
OUT_DIR="results/depth_adapter/DCFA_v2/CA2_cluster_aware_lp10"
mkdir -p "$OUT_DIR"
echo ""
echo ">>> CA2: cluster_aware, lp=10"
python3 -u mbps_pytorch/train_depth_adapter.py \
    --cityscapes_root "$CS_ROOT" \
    --adapter_type v3 \
    --depth_dim 16 \
    --hidden_dim 384 \
    --num_layers 2 \
    --epochs 20 \
    --lr 1e-3 \
    --lambda_preserve 10.0 \
    --loss_type cluster_aware \
    --centroids_path "$CENTROIDS" \
    --output_dir "$OUT_DIR" \
    --seed 42

echo ""
echo ">>> CA2 eval at k=80"
python3 -u mbps_pytorch/generate_depth_overclustered_semantics.py \
    --cityscapes_root "$CS_ROOT" \
    --k 80 --variant sinusoidal --alpha 0.1 --skip_crf \
    --adapter_checkpoint "$OUT_DIR/best.pt" \
    --output_subdir "pseudo_semantic_dcfa_v2_CA2_k80"

python3 -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --semantic_subdir "pseudo_semantic_dcfa_v2_CA2_k80" \
    --num_clusters 80 \
    --skip_instance \
    --output "$OUT_DIR/eval_k80.json"

echo ""
echo "================================================================"

# Experiment 3: hybrid — depth_corr + cluster_aware as auxiliary (lambda_cluster=1.0)
OUT_DIR="results/depth_adapter/DCFA_v2/CA3_hybrid_lc1"
mkdir -p "$OUT_DIR"
echo ""
echo ">>> CA3: depth_corr + cluster CE auxiliary (lambda_cluster=1.0)"
python3 -u mbps_pytorch/train_depth_adapter.py \
    --cityscapes_root "$CS_ROOT" \
    --adapter_type v3 \
    --depth_dim 16 \
    --hidden_dim 384 \
    --num_layers 2 \
    --epochs 20 \
    --lr 1e-3 \
    --lambda_preserve 20.0 \
    --loss_type depth_corr \
    --lambda_cluster 1.0 \
    --centroids_path "$CENTROIDS" \
    --output_dir "$OUT_DIR" \
    --seed 42

echo ""
echo ">>> CA3 eval at k=80"
python3 -u mbps_pytorch/generate_depth_overclustered_semantics.py \
    --cityscapes_root "$CS_ROOT" \
    --k 80 --variant sinusoidal --alpha 0.1 --skip_crf \
    --adapter_checkpoint "$OUT_DIR/best.pt" \
    --output_subdir "pseudo_semantic_dcfa_v2_CA3_k80"

python3 -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --semantic_subdir "pseudo_semantic_dcfa_v2_CA3_k80" \
    --num_clusters 80 \
    --skip_instance \
    --output "$OUT_DIR/eval_k80.json"

echo ""
echo "================================================================"
echo "  Summary"
echo "================================================================"
echo ""
echo "Baseline V3: mIoU=55.29%"
echo ""
for exp in CA1_cluster_aware_lp20 CA2_cluster_aware_lp10 CA3_hybrid_lc1; do
    ej="results/depth_adapter/DCFA_v2/${exp}/eval_k80.json"
    if [ -f "$ej" ]; then
        miou=$(python3 -c "import json; d=json.load(open('$ej')); print(f\"{d['semantic']['miou']:.2f}\")")
        echo "$exp: mIoU=${miou}%"
    fi
done
echo ""
echo "Done: $(date)"
echo "================================================================"
