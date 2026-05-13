#!/bin/bash
# Quick evaluation of shift-avg features using baseline centroids.
# Skips train extraction — uses existing 32x64 centroids on 64x128 val features.
set -euo pipefail

CS_ROOT="${CITYSCAPES_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PYTHON="${PYTHON:-/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python}"
RESULTS_DIR="results/clustering_ablation"

echo "=== Step 1: Assign shift-avg val features with baseline centroids ==="
$PYTHON scripts/quick_assign_shiftavg.py

echo ""
echo "=== Step 2: Evaluate ==="
$PYTHON -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --semantic_subdir pseudo_semantic_raw_dinov3_k100_spherical_kmeans_vitl16_shiftavg64x128 \
    --instance_subdir pseudo_instance_spidepth \
    --num_clusters 100 \
    --cluster_mapping majority \
    --eval_size 512 1024 \
    --output "$RESULTS_DIR/eval_shiftavg_64x128_vitl16_k100.json"

echo ""
echo "=== Results ==="
$PYTHON -c "
import json
d = json.load(open('$RESULTS_DIR/eval_shiftavg_64x128_vitl16_k100.json'))
sem = d['semantic']
pan = d['panoptic']
print(f'mIoU: {sem[\"miou\"]:.1f}')
print(f'PQ: {pan[\"PQ\"]:.1f}, PQ_stuff: {pan[\"PQ_stuff\"]:.1f}, PQ_things: {pan[\"PQ_things\"]:.1f}')
names = ['wall','fence','traffic light','rider','truck','train','motorcycle']
iou = sem['per_class_iou']
dead = sum(1 for n in names if iou.get(n, 0.0) == 0.0)
print(f'Dead classes: {dead}/7')
for n in names:
    v = iou.get(n, 0.0)
    print(f'  {n}: {v:.1f}')
"
