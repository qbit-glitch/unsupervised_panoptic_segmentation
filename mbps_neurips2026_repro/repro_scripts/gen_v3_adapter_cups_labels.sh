#!/bin/bash
# Full pipeline: Generate V3 adapter k=80 pseudo-labels + convert to CUPS format (tau=0.20).
# Target: Local MPS (M4 Pro 48GB)
#
# This generates CUPS-format pseudo-labels using:
#   - V3 adapter k=80 semantics (mIoU=55.29% vs raw k=80 ~52%)
#   - DepthPro tau=0.20 depth-guided instances (fewer, larger instances for CUPS)
#
# Hypothesis: Better stuff semantics (+3% mIoU) lifts PQ_stuff ceiling (currently ~30%).
#
# Prerequisites:
#   1. CAUSE codes (train): cause_codes_90d/train/{city}/*_codes.npy
#   2. DepthPro depth (train): depth_depthpro/train/{city}/*.npy
#   3. V3 adapter checkpoint: results/depth_adapter/V3_dd16_h384_l2/best.pt
#   4. V3 k=80 centroids: pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz
set -euo pipefail

PROJ_ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
CS_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
ADAPTER="$PROJ_ROOT/results/depth_adapter/V3_dd16_h384_l2/best.pt"
CENTROIDS="$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz"

echo "=== V3 Adapter + DepthPro tau=0.20 CUPS Label Pipeline ==="
echo ""

# ─── Step 0: Verify prerequisites ───
echo "--- Step 0: Verify prerequisites ---"

if [ ! -d "$CS_ROOT/cause_codes_90d/train" ]; then
    echo "ERROR: CAUSE codes not found at $CS_ROOT/cause_codes_90d/train"
    exit 1
fi
CODES_COUNT=$(find "$CS_ROOT/cause_codes_90d/train" -name "*_codes.npy" | wc -l)
echo "  CAUSE codes (train): $CODES_COUNT files"

if [ ! -d "$CS_ROOT/depth_depthpro/train" ]; then
    echo "ERROR: DepthPro depth maps not found at $CS_ROOT/depth_depthpro/train"
    exit 1
fi
DEPTH_COUNT=$(find "$CS_ROOT/depth_depthpro/train" -name "*.npy" | wc -l)
echo "  DepthPro depth (train): $DEPTH_COUNT files"

if [ ! -f "$ADAPTER" ]; then
    echo "ERROR: V3 adapter checkpoint not found at $ADAPTER"
    exit 1
fi
echo "  V3 adapter: $ADAPTER"

if [ ! -f "$CENTROIDS" ]; then
    echo "ERROR: V3 k=80 centroids not found at $CENTROIDS"
    exit 1
fi
echo "  Centroids: $CENTROIDS"

echo ""
echo "--- Step 1: Generate V3 adapter k=80 raw cluster semantics (train split) ---"
echo "  Output: $CS_ROOT/pseudo_semantic_adapter_V3_k80/train/"
echo "  Mode: raw clusters (0-79), load pre-fit centroids, sinusoidal alpha=0.1"
echo ""

cd "$PROJ_ROOT"

python -u mbps_pytorch/generate_depth_overclustered_semantics.py \
    --cityscapes_root "$CS_ROOT" \
    --split train \
    --adapter_checkpoint "$ADAPTER" \
    --variant sinusoidal \
    --alpha 0.1 \
    --k 80 \
    --load_centroids "$CENTROIDS" \
    --output_subdir pseudo_semantic_adapter_V3_k80 \
    --skip_crf \
    --raw_clusters

# Verify raw cluster output (must have 50+ unique values, not 9-19 mapped classes)
echo ""
echo "--- Verifying raw cluster output ---"
FIRST_PNG=$(find "$CS_ROOT/pseudo_semantic_adapter_V3_k80/train" -name "*.png" | head -1)
if [ -n "$FIRST_PNG" ]; then
    UNIQUE=$(python3 -c "
import numpy as np; from PIL import Image
img = np.array(Image.open('$FIRST_PNG'))
print(len(np.unique(img)))
")
    echo "  First PNG unique values: $UNIQUE"
    if [ "$UNIQUE" -lt 30 ]; then
        echo "WARNING: Only $UNIQUE unique values — may be 19-class mapped, not raw clusters!"
        echo "         CUPS needs raw cluster IDs 0-79. Check --raw_clusters flag."
    else
        echo "  PASS: Raw cluster output confirmed ($UNIQUE unique values)"
    fi
fi

SEM_COUNT=$(find "$CS_ROOT/pseudo_semantic_adapter_V3_k80/train" -name "*.png" | wc -l)
echo "  Total train PNGs: $SEM_COUNT (expected ~2975)"

echo ""
echo "--- Step 2: Convert to CUPS format with DepthPro tau=0.20 ---"
echo "  Output: $CS_ROOT/cups_pseudo_labels_adapter_V3_tau020/"
echo "  Params: tau=0.20, A_min=1000, sigma=0.0, dil=3"
echo ""

python -u mbps_pytorch/convert_to_cups_format.py \
    --cityscapes_root "$CS_ROOT" \
    --semantic_subdir pseudo_semantic_adapter_V3_k80 \
    --output_subdir cups_pseudo_labels_adapter_V3_tau020 \
    --split train \
    --num_classes 80 \
    --depth_cc_instances \
    --centroids_path "$CENTROIDS" \
    --depth_subdir depth_depthpro \
    --grad_threshold 0.20 \
    --depth_blur_sigma 0.0 \
    --dilation_iters 3 \
    --min_instance_area 1000

echo ""
echo "--- Step 3: Verify CUPS pseudo-labels ---"
OUT_DIR="$CS_ROOT/cups_pseudo_labels_adapter_V3_tau020"
SEM_OUT=$(find "$OUT_DIR" -name "*_semantic.png" 2>/dev/null | wc -l)
INST_OUT=$(find "$OUT_DIR" -name "*_instance.png" 2>/dev/null | wc -l)
PT_OUT=$(find "$OUT_DIR" -name "*.pt" 2>/dev/null | wc -l)
echo "  Semantic PNGs: $SEM_OUT"
echo "  Instance PNGs: $INST_OUT"
echo "  Distribution .pt: $PT_OUT"

if [ "$SEM_OUT" -lt 2975 ] || [ "$INST_OUT" -lt 2975 ] || [ "$PT_OUT" -lt 2975 ]; then
    echo "ERROR: Expected >= 2975 files per type. Something went wrong."
    exit 1
fi
echo "  PASS: $SEM_OUT triplets generated successfully!"

echo ""
echo "=== Pipeline complete ==="
echo "  Next: Upload $OUT_DIR to HF Hub, then run Stage-2 on santosh."
