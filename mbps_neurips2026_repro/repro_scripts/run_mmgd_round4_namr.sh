#!/bin/bash
# MMGD-Cut Round 4: NAMR Post-Processing
# Tests NAMR (Nonlinear Adaptive Mask Refinement) on top of the best baseline.
# Expected gain: +1-3 mIoU by closing the post-processing gap to Falcon.
# Usage: nohup bash scripts/run_mmgd_round4_namr.sh > /tmp/mmgd_round4.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/mmgd_cut.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"

echo "=== MMGD-Cut Round 4: NAMR Post-Processing ==="
echo "Start: $(date)"

# --- R4-1: DINOv3+SSD-1B + NAMR (best baseline + NAMR, 7 temperatures) ---
echo ""
echo "=== R4-1: DINOv3+SSD-1B + NAMR (7 temperatures) ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --namr --namr_window 2
echo "Done R4-1: $(date)"

# --- R4-2: DINOv3+SSD-1B + NAMR single temperature T=0.10 ---
echo ""
echo "=== R4-2: DINOv3+SSD-1B + NAMR single T=0.10 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --namr --namr_single_temp 0.10
echo "Done R4-2: $(date)"

# --- R4-3: DINOv3 only + NAMR (NAMR value on single-modal) ---
echo ""
echo "=== R4-3: DINOv3 only + NAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --namr --namr_window 2
echo "Done R4-3: $(date)"

# --- R4-4: DINOv3+SSD-1B + NAMR wider window (window=3) ---
echo ""
echo "=== R4-4: DINOv3+SSD-1B + NAMR window=3 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --namr --namr_window 3
echo "Done R4-4: $(date)"

echo ""
echo "=== Round 4 complete ==="
echo "End: $(date)"
