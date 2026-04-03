#!/bin/bash
# Falcon K-way NCut sweep: configs F1-F10 from the plan
# Evaluates on 501 val images (--dino_only) for consistency with DiffCut ablations
# Usage: nohup bash scripts/run_falcon_sweep.sh > /tmp/falcon_sweep.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/falcon_pseudo_semantics.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"
DINO="--dino_only"

echo "=== Falcon K-way NCut Sweep (501 val images with DINOv3 features) ==="
echo "Start: $(date)"

# Verify features exist
N_DINO=$(ls "${COCO}/dinov3_features_64x64/val2017/" | wc -l | tr -d ' ')
N_FEATS=$(ls "${COCO}/sd_features_v14_s50/val2017/" | wc -l | tr -d ' ')
echo "DINOv3 features: ${N_DINO}, SD features: ${N_FEATS}"

# --- Config F1: baseline (K=27, α=4.5, β=0.5, no PAMR) ---
echo ""
echo "=== Config F1: K=27, alpha=4.5, beta=0.5, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F1: $(date)"

# --- Config F2: + PAMR ---
echo ""
echo "=== Config F2: K=27, alpha=4.5, beta=0.5, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F2: $(date)"

# --- Config F3: slight overclustering K=32 ---
echo ""
echo "=== Config F3: K=32, alpha=4.5, beta=0.5, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 32 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 32 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F3: $(date)"

# --- Config F4: heavy overclustering K=80 ---
echo ""
echo "=== Config F4: K=80, alpha=4.5, beta=0.5, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 80 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 80 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F4: $(date)"

# --- Config F5: lower alpha (α=3.0) ---
echo ""
echo "=== Config F5: K=27, alpha=3.0, beta=0.5, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 3.0 --beta 0.5 --n_iter 15 --init kmeans --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 3.0 --beta 0.5 --n_iter 15 --init kmeans --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F5: $(date)"

# --- Config F6: higher alpha (α=6.0) ---
echo ""
echo "=== Config F6: K=27, alpha=6.0, beta=0.5, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 6.0 --beta 0.5 --n_iter 15 --init kmeans --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 6.0 --beta 0.5 --n_iter 15 --init kmeans --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F6: $(date)"

# --- Config F7: sharper reweighting (β=0.1) ---
echo ""
echo "=== Config F7: K=27, alpha=4.5, beta=0.1, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.1 --n_iter 15 --init kmeans --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.1 --n_iter 15 --init kmeans --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F7: $(date)"

# --- Config F8: softer reweighting (β=1.0) ---
echo ""
echo "=== Config F8: K=27, alpha=4.5, beta=1.0, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 1.0 --n_iter 15 --init kmeans --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 1.0 --n_iter 15 --init kmeans --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F8: $(date)"

# --- Config F9: spectral init ---
echo ""
echo "=== Config F9: K=27, alpha=4.5, beta=0.5, spectral init, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init spectral --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init spectral --pamr \
    --K_global 27 --feature_source dinov3 $DINO
echo "Done F9: $(date)"

# --- Config F10: DINOv3 features for segmentation ---
echo ""
echo "=== Config F10: K=27, alpha=4.5, beta=0.5, DINOv3 seg features, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr \
    --seg_feature_source dinov3 $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --pamr \
    --seg_feature_source dinov3 --feature_source dinov3 --K_global 27 $DINO
echo "Done F10: $(date)"

echo ""
echo "=== All Falcon configs complete ==="
echo "End: $(date)"
echo "Results saved to ${COCO}/falcon_results.json"
