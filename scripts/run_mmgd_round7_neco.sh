#!/bin/bash
# MMGD-Cut Round 7: NeCo post-training of DINOv3
# MUST run on remote GPU (santosh@172.17.254.146, conda env cups).
# Step 1: Train NeCo on COCO train2017 (~40-60 GPU hours on 1080Ti×2)
# Step 2: Re-extract features with NeCo backbone
# Step 3: Evaluate NeCo-DINOv3 + SSD-1B + NAMR (best expected combination)
#
# On remote, run:
#   nohup bash scripts/run_mmgd_round7_neco.sh > /tmp/mmgd_round7.log 2>&1 &
#   tail -f /tmp/mmgd_round7.log
#
# Note: COCO root on remote is /media/santosh/data/coco (adjust if different)

set -e

PYTHON="python"  # remote: use conda cups env python
COCO="/media/santosh/data/coco"  # adjust to remote COCO path
CKPT_DIR="checkpoints/neco_dinov3"
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/mbps_pytorch"

echo "=== MMGD-Cut Round 7: NeCo Post-Training ==="
echo "Start: $(date)"
echo "COCO: $COCO"
echo "CKPT: $CKPT_DIR"

# ── Step 1: Train NeCo on COCO train2017 ──
echo ""
echo "=== Step 1: NeCo Training (5 epochs on COCO train) ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT_DIR/train_neco_dinov3.py \
    --coco_root $COCO \
    --output_dir $CKPT_DIR \
    --device $DEVICE \
    --img_size 448 \
    --patch_size 14 \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --temperature 0.07 \
    --r_pos 2 \
    --r_neg 6 \
    --n_pairs 64 \
    --proj_dim 256 \
    --unfreeze_blocks 4 \
    --num_workers 4 \
    --log_interval 100
echo "Step 1 done: $(date)"

# ── Step 2: Re-extract DINOv3 features with NeCo backbone ──
echo ""
echo "=== Step 2: Extract NeCo-enhanced features for val2017 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT_DIR/extract_dinov3_features_neco.py \
    --coco_root $COCO \
    --checkpoint $CKPT_DIR/neco_best.pth \
    --output_subdir dinov3_neco_features \
    --img_size 512 \
    --device $DEVICE
echo "Step 2 done: $(date)"

# ── Step 3a: Evaluate NeCo-DINOv3 only ──
# Note: mmgd_cut.py needs "dinov3_neco" added to FEATURE_SOURCES before this step.
# Alternatively, replace dinov3_features/val2017 symlink with dinov3_neco_features/val2017.
# For now, manually symlink:
#   ln -sfn $COCO/dinov3_neco_features/val2017 $COCO/dinov3_features_neco/val2017
# Then run with --sources dinov3_neco (after adding to FEATURE_SOURCES in mmgd_cut.py).

echo ""
echo "=== Step 3: Manual instructions for evaluation ==="
echo "After Step 2, add 'dinov3_neco' to FEATURE_SOURCES in mmgd_cut.py:"
echo "  'dinov3_neco': ('dinov3_neco_features/val2017', 1024, 768),"
echo "Then run:"
echo "  R7-1: python mmgd_cut.py --sources dinov3_neco --diffusion none ..."
echo "  R7-2: python mmgd_cut.py --sources dinov3_neco ssd1b --step 10 --diffusion none ..."
echo "  R7-3: python mmgd_cut.py --sources dinov3_neco ssd1b --step 10 --diffusion none --namr ..."

echo ""
echo "=== Round 7 training complete ==="
echo "End: $(date)"
