#!/bin/bash
set -euo pipefail

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
CS="/Users/qbit-glitch/Desktop/datasets/cityscapes"
PROJECT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
LOGDIR="$PROJECT/logs"
RESULTS="$PROJECT/results/clustering_ablation"

mkdir -p "$LOGDIR" "$RESULTS"

echo "=== Step 1: Wait for train extraction (PID $1) ==="
if [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; then
    echo "Waiting for train extraction PID $1 to finish..."
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
    echo "Train extraction done."
fi

echo "=== Step 2: Extract val features ==="
$PYTHON -u "$PROJECT/mbps_pytorch/extract_dinov2_vitg14_features.py" \
    --data_dir "$CS/leftImg8bit/val" \
    --output_dir "$CS/dinov2g14_features/val" \
    --batch_size 1 \
    --device mps

TRAIN_COUNT=$(find "$CS/dinov2g14_features/train" -name "*.npy" | wc -l | tr -d ' ')
VAL_COUNT=$(find "$CS/dinov2g14_features/val" -name "*.npy" | wc -l | tr -d ' ')
echo "Feature files: train=$TRAIN_COUNT, val=$VAL_COUNT"

echo "=== Step 3: Spherical k-means (k=80) on ViT-g/14 features ==="
$PYTHON -u "$PROJECT/mbps_pytorch/generate_clustering_ablation.py" \
    --cityscapes_root "$CS" \
    --feat_subdir dinov2g14_features \
    --method spherical_kmeans \
    --k 80 --seed 42 \
    --splits train val \
    --output_suffix vitg14

echo "=== Step 4: Evaluate ==="
$PYTHON -u "$PROJECT/mbps_pytorch/evaluate_cascade_pseudolabels.py" \
    --cityscapes_root "$CS" \
    --split val \
    --semantic_subdir pseudo_semantic_raw_dinov3_k80_spherical_kmeans_vitg14 \
    --instance_subdir pseudo_instance_spidepth \
    --num_clusters 80 \
    --cluster_mapping majority \
    --eval_size 512 1024 \
    --output "$RESULTS/eval_spherical_kmeans_vitg14.json"

echo "=== Done ==="
$PYTHON -c "
import json
d = json.load(open('$RESULTS/eval_spherical_kmeans_vitg14.json'))
sem = d.get('semantic', {})
pan = d.get('panoptic', {})
print(f\"mIoU:      {sem.get('miou', 0):.2f}\")
print(f\"PQ:        {pan.get('PQ', 0):.2f}\")
print(f\"PQ_stuff:  {pan.get('PQ_stuff', 0):.2f}\")
print(f\"PQ_things: {pan.get('PQ_things', 0):.2f}\")
"
