#!/bin/bash
# Full pipeline: Generate DepthPro tau=0.20 pseudo-labels + launch CUPS Stage-2 training.
# Target: Anydesk RTX A6000 48GB
#
# This replicates the santosh depthpro_tau020 pipeline that reached PQ=28.4.
#
# Prerequisites on A6000:
#   1. k=80 semantic pseudo-labels at: ~/umesh/datasets/cityscapes/pseudo_semantic_raw_k80/
#   2. k-means centroids at: ~/umesh/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz
#   3. DepthPro depth maps at: ~/umesh/datasets/cityscapes/depth_depthpro/
#   4. DINOv3 weights converted (188/188 keys): weights/dinov3_vitb16_official.pth
#   5. leftImg8bit_sequence symlink exists
set -euo pipefail

PROJ_ROOT="$HOME/umesh/unsupervised_panoptic_segmentation"
CS_ROOT="$HOME/umesh/datasets/cityscapes"

echo "=== Step 1: Verify prerequisites ==="

# Check k=80 semantic labels
SEM_DIR="$CS_ROOT/pseudo_semantic_raw_k80"
if [ ! -d "$SEM_DIR/train" ]; then
    echo "ERROR: k=80 semantic labels not found at $SEM_DIR/train"
    echo "Need: pseudo_semantic_raw_k80/{train,val}/ with PNG files"
    exit 1
fi
SEM_COUNT=$(find "$SEM_DIR/train" -name "*.png" | wc -l)
echo "  Semantic labels (train): $SEM_COUNT files"

# Check centroids
CENTROIDS="$SEM_DIR/kmeans_centroids.npz"
if [ ! -f "$CENTROIDS" ]; then
    echo "ERROR: Centroids not found at $CENTROIDS"
    exit 1
fi
echo "  Centroids: $CENTROIDS"

# Check DepthPro depth maps
DEPTH_DIR="$CS_ROOT/depth_depthpro"
if [ ! -d "$DEPTH_DIR" ]; then
    echo "ERROR: DepthPro depth maps not found at $DEPTH_DIR"
    echo "Need: depth_depthpro/{train,val}/ with .npy files"
    exit 1
fi
DEPTH_COUNT=$(find "$DEPTH_DIR" -name "*.npy" 2>/dev/null | wc -l)
echo "  Depth maps: $DEPTH_COUNT files"

echo ""
echo "=== Step 2: Generate CUPS pseudo-labels with depth-guided CC (tau=0.20) ==="
echo "  Output: $CS_ROOT/cups_pseudo_labels_depthpro_tau020/"

cd "$PROJ_ROOT"

# Generate for train split
echo ""
echo "--- Train split ---"
python -u mbps_pytorch/convert_to_cups_format.py \
    --cityscapes_root "$CS_ROOT" \
    --semantic_subdir pseudo_semantic_raw_k80 \
    --output_subdir cups_pseudo_labels_depthpro_tau020 \
    --split train \
    --num_classes 80 \
    --depth_cc_instances \
    --centroids_path "$CENTROIDS" \
    --depth_subdir depth_depthpro \
    --grad_threshold 0.20 \
    --depth_blur_sigma 0.0 \
    --dilation_iters 3 \
    --min_instance_area 1000

# Generate for val split
echo ""
echo "--- Val split ---"
python -u mbps_pytorch/convert_to_cups_format.py \
    --cityscapes_root "$CS_ROOT" \
    --semantic_subdir pseudo_semantic_raw_k80 \
    --output_subdir cups_pseudo_labels_depthpro_tau020 \
    --split val \
    --num_classes 80 \
    --depth_cc_instances \
    --centroids_path "$CENTROIDS" \
    --depth_subdir depth_depthpro \
    --grad_threshold 0.20 \
    --depth_blur_sigma 0.0 \
    --dilation_iters 3 \
    --min_instance_area 1000

# Verify output
echo ""
echo "=== Step 3: Verify pseudo-labels ==="
OUT_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020"
SEM_OUT=$(find "$OUT_DIR" -name "*_semantic.png" 2>/dev/null | wc -l)
INST_OUT=$(find "$OUT_DIR" -name "*_instance.png" 2>/dev/null | wc -l)
PT_OUT=$(find "$OUT_DIR" -name "*.pt" 2>/dev/null | wc -l)
echo "  Generated: $SEM_OUT semantic, $INST_OUT instance, $PT_OUT .pt files"

if [ "$SEM_OUT" -lt 2975 ] || [ "$INST_OUT" -lt 2975 ]; then
    echo "ERROR: Expected >= 2975 training files. Something went wrong."
    exit 1
fi
echo "  PASS: Pseudo-labels generated successfully!"

echo ""
echo "=== Step 4: Launch CUPS Stage-2 training ==="
echo "  Config: configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml"
echo "  Pseudo-labels: cups_pseudo_labels_depthpro_tau020"
echo "  Settings: bs=1, accum=16, fp32, 5 resolutions, 3 copy-paste"

cd "$PROJ_ROOT/refs/cups"
export WANDB_MODE=disabled
python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml
