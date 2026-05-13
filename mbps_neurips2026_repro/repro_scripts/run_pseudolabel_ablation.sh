#!/usr/bin/env bash
# Run pseudo-label ablation: PL-0 (baseline) through PL-5 (DepthG + depth-weighted).
#
# Prerequisites:
#   - DINOv3 features at: ${CS}/dinov3_features/{train,val}/
#   - DepthPro depth at: ${CS}/depth_depthpro/{train,val}/
#   - Baseline k=80 labels at: ${CS}/pseudo_semantic_raw_dinov3_k80/
set -euo pipefail

CS="${1:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="${PROJECT_ROOT}/results/depth_enhanced"
mkdir -p "$RESULTS"

echo "============================================"
echo "  Pseudo-Label Ablation (PL-0 to PL-5)"
echo "  Cityscapes root: ${CS}"
echo "  Results: ${RESULTS}"
echo "============================================"

# PL-0: Baseline (existing k=80, no depth)
echo ""
echo ">>> PL-0: Baseline k=80 (existing)"
$PYTHON -u "${PROJECT_ROOT}/mbps_pytorch/evaluate_cascade_pseudolabels.py" \
    --cityscapes_root "$CS" \
    --semantic_subdir pseudo_semantic_raw_dinov3_k80 \
    --split val --eval_size 512 1024 \
    --num_clusters 80 \
    --output "${RESULTS}/pl0_baseline.json" \
    2>&1 | tee "${RESULTS}/pl0_baseline.log"

# PL-1..PL-3: Depth-weighted k-means at different lambdas
for LAMBDA in 0.1 0.3 0.5; do
    TAG=$(echo $LAMBDA | tr -d '.')
    SUBDIR="pseudo_semantic_dw_k80_l${TAG}"
    echo ""
    echo ">>> PL-${TAG}: Depth-weighted k-means (lambda=$LAMBDA, sinusoidal)"
    $PYTHON -u "${PROJECT_ROOT}/mbps_pytorch/generate_depth_weighted_kmeans.py" \
        --cityscapes_root "$CS" \
        --depth_weight $LAMBDA --depth_encoding sinusoidal \
        --output_subdir "$SUBDIR" \
        2>&1 | tee "${RESULTS}/pl_l${TAG}_generate.log"
    $PYTHON -u "${PROJECT_ROOT}/mbps_pytorch/evaluate_cascade_pseudolabels.py" \
        --cityscapes_root "$CS" \
        --semantic_subdir "$SUBDIR" \
        --split val --eval_size 512 1024 \
        --output "${RESULTS}/pl_l${TAG}_eval.json" \
        2>&1 | tee "${RESULTS}/pl_l${TAG}_eval.log"
done

# PL-4: Trained DepthG projector
echo ""
echo ">>> PL-4: DepthG projector (20 epochs)"
$PYTHON -u "${PROJECT_ROOT}/mbps_pytorch/train_depthg_dinov3.py" \
    --cityscapes_root "$CS" \
    --epochs 20 --batch_size 8 --lr 1e-3 --lambda_depthg 0.3 \
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/depthg_dinov3" \
    2>&1 | tee "${RESULTS}/pl4_train.log"

echo ">>> PL-4: Generating pseudo-labels from best checkpoint"
$PYTHON -u "${PROJECT_ROOT}/mbps_pytorch/train_depthg_dinov3.py" \
    --cityscapes_root "$CS" \
    --generate_labels \
    --output_subdir pseudo_semantic_depthg_dinov3_k80 \
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/depthg_dinov3" \
    2>&1 | tee "${RESULTS}/pl4_generate.log"

$PYTHON -u "${PROJECT_ROOT}/mbps_pytorch/evaluate_cascade_pseudolabels.py" \
    --cityscapes_root "$CS" \
    --semantic_subdir pseudo_semantic_depthg_dinov3_k80 \
    --split val --eval_size 512 1024 \
    --output "${RESULTS}/pl4_eval.json" \
    2>&1 | tee "${RESULTS}/pl4_eval.log"

# PL-5: DepthG codes + depth-weighted k-means
echo ""
echo ">>> PL-5: DepthG codes + depth-weighted k-means (lambda=0.3)"
echo "NOTE: Requires extracting DepthG codes first, then running"
echo "      depth-weighted k-means on those codes. Manual step — see plan."

echo ""
echo "============================================"
echo "  Results saved to: ${RESULTS}/"
echo "============================================"
echo ""
echo "Experiments completed:"
ls -la "${RESULTS}"/*.log 2>/dev/null || echo "  (no log files yet)"
