#!/bin/bash
# Run the MBPS v2 pseudo-label pipeline locally on macOS (MPS-accelerated).
#
# Prerequisites:
#   source ~/Desktop/datasets/.venv_pytorch/bin/activate
#   Cityscapes extracted at ~/Desktop/datasets/cityscapes/
#
# Stages 1-5 run locally. Stage 6 (TFRecords) requires TensorFlow
# and should be run on the TPU VM or a Python <= 3.12 environment.

set -euo pipefail

DATA="${CITYSCAPES_DATA:-$HOME/Desktop/datasets/cityscapes}"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== MBPS v2 Local Pseudo-Label Pipeline ==="
echo "Data dir:    $DATA"
echo "Project dir: $PROJ"
echo ""

# Verify Cityscapes structure
if [ ! -d "$DATA/leftImg8bit/train" ]; then
    echo "ERROR: $DATA/leftImg8bit/train not found. Download Cityscapes first."
    exit 1
fi

TRAIN_COUNT=$(find "$DATA/leftImg8bit/train" -name "*_leftImg8bit.png" | wc -l | tr -d ' ')
VAL_COUNT=$(find "$DATA/leftImg8bit/val" -name "*_leftImg8bit.png" | wc -l | tr -d ' ')
echo "Found $TRAIN_COUNT train images, $VAL_COUNT val images"
echo ""

# ─── Stage 1: DINOv3 Feature Extraction (MPS-accelerated) ───
echo "=== Stage 1/5: DINOv3 Feature Extraction ==="

if [ -d "$DATA/dinov3_features/train" ] && [ "$(find "$DATA/dinov3_features/train" -name "*.npy" | wc -l | tr -d ' ')" -ge "$TRAIN_COUNT" ]; then
    echo "  Train features already extracted, skipping."
else
    python "$PROJ/mbps_pytorch/extract_dinov3_features.py" \
        --data_dir "$DATA/leftImg8bit/train" \
        --output_dir "$DATA/dinov3_features/train" \
        --device auto --batch_size 4
fi

if [ -d "$DATA/dinov3_features/val" ] && [ "$(find "$DATA/dinov3_features/val" -name "*.npy" | wc -l | tr -d ' ')" -ge "$VAL_COUNT" ]; then
    echo "  Val features already extracted, skipping."
else
    python "$PROJ/mbps_pytorch/extract_dinov3_features.py" \
        --data_dir "$DATA/leftImg8bit/val" \
        --output_dir "$DATA/dinov3_features/val" \
        --device auto --batch_size 4
fi

echo "  Done. Train: $(find "$DATA/dinov3_features/train" -name "*.npy" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  Done. Val:   $(find "$DATA/dinov3_features/val" -name "*.npy" 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

# ─── Stage 2: Semantic Pseudo-Labels (CPU, sklearn K-Means) ───
echo "=== Stage 2/5: Semantic Pseudo-Labels ==="

if [ -d "$DATA/pseudo_semantic/train" ] && [ "$(find "$DATA/pseudo_semantic/train" -name "*.png" | wc -l | tr -d ' ')" -ge "$TRAIN_COUNT" ]; then
    echo "  Train semantic labels already generated, skipping."
else
    python "$PROJ/mbps_pytorch/generate_semantic_pseudolabels.py" \
        --feature_dir "$DATA/dinov3_features/train" \
        --output_dir "$DATA/pseudo_semantic/train" \
        --num_classes 19 --image_size 512 1024
fi

if [ -d "$DATA/pseudo_semantic/val" ] && [ "$(find "$DATA/pseudo_semantic/val" -name "*.png" | wc -l | tr -d ' ')" -ge "$VAL_COUNT" ]; then
    echo "  Val semantic labels already generated, skipping."
else
    # Evaluate against GT on val set
    python "$PROJ/mbps_pytorch/generate_semantic_pseudolabels.py" \
        --feature_dir "$DATA/dinov3_features/val" \
        --output_dir "$DATA/pseudo_semantic/val" \
        --num_classes 19 --image_size 512 1024 \
        --kmeans_model "$DATA/pseudo_semantic/train/kmeans_model.pkl" \
        --gt_dir "$DATA/gtFine/val" --evaluate
fi

echo "  Done. Train: $(find "$DATA/pseudo_semantic/train" -name "*.png" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  Done. Val:   $(find "$DATA/pseudo_semantic/val" -name "*.png" 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

# ─── Stage 3: Instance Pseudo-Labels (CPU, scipy NCut) ───
echo "=== Stage 3/5: Instance Pseudo-Labels ==="

if [ -d "$DATA/pseudo_instance/train" ] && [ "$(find "$DATA/pseudo_instance/train" -name "*.npz" -o -name "*.npy" | wc -l | tr -d ' ')" -ge "$TRAIN_COUNT" ]; then
    echo "  Train instance labels already generated, skipping."
else
    python "$PROJ/mbps_pytorch/generate_instance_pseudolabels.py" \
        --feature_dir "$DATA/dinov3_features/train" \
        --output_dir "$DATA/pseudo_instance/train" \
        --image_size 512 1024
fi

if [ -d "$DATA/pseudo_instance/val" ] && [ "$(find "$DATA/pseudo_instance/val" -name "*.npz" -o -name "*.npy" | wc -l | tr -d ' ')" -ge "$VAL_COUNT" ]; then
    echo "  Val instance labels already generated, skipping."
else
    python "$PROJ/mbps_pytorch/generate_instance_pseudolabels.py" \
        --feature_dir "$DATA/dinov3_features/val" \
        --output_dir "$DATA/pseudo_instance/val" \
        --image_size 512 1024
fi

echo "  Done. Train: $(find "$DATA/pseudo_instance/train" -name "*.npz" -o -name "*.npy" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  Done. Val:   $(find "$DATA/pseudo_instance/val" -name "*.npz" -o -name "*.npy" 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

# ─── Stage 4: Depth Maps (MPS-accelerated, DA V2 fallback) ───
echo "=== Stage 4/5: Depth Maps (Depth Anything V2) ==="

if [ -d "$DATA/depth_dav3/train" ] && [ "$(find "$DATA/depth_dav3/train" -name "*.npy" | wc -l | tr -d ' ')" -ge "$TRAIN_COUNT" ]; then
    echo "  Train depth maps already generated, skipping."
else
    python "$PROJ/mbps_pytorch/generate_depth_maps.py" \
        --data_dir "$DATA/leftImg8bit/train" \
        --output_dir "$DATA/depth_dav3/train" \
        --device auto --use_dav2 --image_size 512 1024
fi

if [ -d "$DATA/depth_dav3/val" ] && [ "$(find "$DATA/depth_dav3/val" -name "*.npy" | wc -l | tr -d ' ')" -ge "$VAL_COUNT" ]; then
    echo "  Val depth maps already generated, skipping."
else
    python "$PROJ/mbps_pytorch/generate_depth_maps.py" \
        --data_dir "$DATA/leftImg8bit/val" \
        --output_dir "$DATA/depth_dav3/val" \
        --device auto --use_dav2 --image_size 512 1024
fi

echo "  Done. Train: $(find "$DATA/depth_dav3/train" -name "*.npy" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  Done. Val:   $(find "$DATA/depth_dav3/val" -name "*.npy" 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

# ─── Stage 5: Stuff/Things Classification (CPU, instant) ───
echo "=== Stage 5/5: Stuff/Things Classification ==="

if [ -f "$DATA/stuff_things_map.json" ]; then
    echo "  stuff_things_map.json already exists, skipping."
else
    python "$PROJ/mbps_pytorch/classify_stuff_things.py" \
        --semantic_dir "$DATA/pseudo_semantic/train" \
        --instance_dir "$DATA/pseudo_instance/train" \
        --output_path "$DATA/stuff_things_map.json" \
        --num_classes 19
fi

echo ""
echo "=== Pipeline Complete ==="
echo ""
echo "Output structure:"
echo "  $DATA/"
echo "  ├── dinov3_features/{train,val}/   DINOv3 768-dim patch features"
echo "  ├── pseudo_semantic/{train,val}/   K-means semantic labels (19 classes)"
echo "  ├── pseudo_instance/{train,val}/   NCut instance masks"
echo "  ├── depth_dav3/{train,val}/        Depth Anything V2 depth maps"
echo "  └── stuff_things_map.json          Stuff/things class mapping"
echo ""
echo "NOTE: Stage 6 (TFRecord generation) requires TensorFlow."
echo "  Run on TPU VM or Python <= 3.12 environment:"
echo "  python $PROJ/mbps_pytorch/generate_v2_tfrecords.py \\"
echo "      --image_dir $DATA/leftImg8bit/train \\"
echo "      --depth_dir $DATA/depth_dav3/train \\"
echo "      --semantic_dir $DATA/pseudo_semantic/train \\"
echo "      --instance_dir $DATA/pseudo_instance/train \\"
echo "      --output_dir $DATA/tfrecords_v2/train \\"
echo "      --image_size 512 1024"
