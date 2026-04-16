#!/bin/bash
# Fix A6000 setup issues for depth-guided experiments
# Run this on the A6000 machine before starting experiments.
#
# Fixes:
#   1. leftImg8bit_sequence symlink (CUPS Stage-2 needs it)
#   2. Verify val depth maps exist
#   3. Kill stale training processes

set -euo pipefail

CS_ROOT="${1:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"

echo "=== Fix 1: Create leftImg8bit_sequence symlink ==="
if [ -d "$CS_ROOT/leftImg8bit_sequence" ] || [ -L "$CS_ROOT/leftImg8bit_sequence" ]; then
    echo "  Already exists: $(ls -la $CS_ROOT/leftImg8bit_sequence | head -1)"
else
    ln -s "$CS_ROOT/leftImg8bit" "$CS_ROOT/leftImg8bit_sequence"
    echo "  Created symlink: leftImg8bit_sequence -> leftImg8bit"
fi

echo ""
echo "=== Fix 2: Check val depth maps ==="
VAL_DEPTH_DIR="$CS_ROOT/depth_depthpro/val"
if [ -d "$VAL_DEPTH_DIR" ]; then
    VAL_COUNT=$(find "$VAL_DEPTH_DIR" -name "*.npy" | wc -l)
    echo "  Val depth maps: $VAL_COUNT"
    if [ "$VAL_COUNT" -lt 500 ]; then
        echo "  WARNING: Expected 500, found $VAL_COUNT"
        echo "  Sync from local: rsync -avz ~/Desktop/datasets/cityscapes/depth_depthpro/val/ cvpr_ug_5@master:$VAL_DEPTH_DIR/"
    else
        echo "  OK"
    fi
else
    echo "  MISSING: $VAL_DEPTH_DIR"
    echo "  Sync from local: rsync -avz ~/Desktop/datasets/cityscapes/depth_depthpro/ cvpr_ug_5@master:$CS_ROOT/depth_depthpro/"
fi

TRAIN_DEPTH_DIR="$CS_ROOT/depth_depthpro/train"
if [ -d "$TRAIN_DEPTH_DIR" ]; then
    TRAIN_COUNT=$(find "$TRAIN_DEPTH_DIR" -name "*.npy" | wc -l)
    echo "  Train depth maps: $TRAIN_COUNT"
else
    echo "  MISSING: $TRAIN_DEPTH_DIR"
fi

echo ""
echo "=== Fix 3: Check for stale processes ==="
STALE=$(ps aux | grep -E "train_cause_depth|run_approach_b" | grep -v grep || true)
if [ -n "$STALE" ]; then
    echo "  Found stale processes:"
    echo "$STALE"
    echo "  Kill with: pkill -f train_cause_depth_finetune; pkill -f run_approach_b"
else
    echo "  No stale processes found"
fi

echo ""
echo "=== Fix 4: Verify CAUSE checkpoint ==="
CAUSE_SEG="refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth"
if [ -f "$CAUSE_SEG" ]; then
    echo "  Segment_TR checkpoint: OK ($(du -h $CAUSE_SEG | cut -f1))"
else
    echo "  MISSING: $CAUSE_SEG"
fi

CAUSE_BACKBONE="refs/cause/checkpoint/dinov2_vit_base_14.pth"
if [ -f "$CAUSE_BACKBONE" ]; then
    echo "  DINOv2 backbone: OK ($(du -h $CAUSE_BACKBONE | cut -f1))"
else
    echo "  MISSING: $CAUSE_BACKBONE"
fi

CAUSE_CODEBOOK="refs/cause/CAUSE/cityscapes/modularity/dinov2_vit_base_14/2048/modular.npy"
if [ -f "$CAUSE_CODEBOOK" ]; then
    echo "  Codebook: OK ($(du -h $CAUSE_CODEBOOK | cut -f1))"
else
    echo "  MISSING: $CAUSE_CODEBOOK"
fi

echo ""
echo "=== Fix 5: Verify CroppedDataset ==="
CROP_DIR="$CS_ROOT/cropped/cityscapes_five_crop_0.5/img/train"
if [ -d "$CROP_DIR" ]; then
    CROP_COUNT=$(ls "$CROP_DIR" | wc -l)
    echo "  Train crops: $CROP_COUNT (expected 14875)"
else
    echo "  MISSING: $CROP_DIR"
    echo "  Generate with: python scripts/generate_cause_crops.py --data_dir $(dirname $CS_ROOT)"
fi

echo ""
echo "=== Summary ==="
echo "After fixing any MISSING items above, run Approach B:"
echo ""
echo "  PROJECT_DIR=\$(pwd) DATA_DIR=$(dirname $CS_ROOT) nohup bash scripts/run_approach_b_a6000.sh > logs/approach_b/full_sweep.log 2>&1 &"
echo "  tail -f logs/approach_b/B_lambda0.0.log"
