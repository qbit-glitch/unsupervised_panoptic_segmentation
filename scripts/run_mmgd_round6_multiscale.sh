#!/bin/bash
# MMGD-Cut Round 6: Multi-Scale NCut (32x32 coarse + 64x64 fine)
# Tests whether fine 64×64 boundaries improve coarse 32×32 semantic labels.
# Fine DINOv3 64×64 features must already be extracted at:
#   $COCO/dinov3_features_64x64/val2017/
# Expected gain: +0-2 mIoU (uncertain — 64×64 was worse alone, may help as boundary)
# Usage: nohup bash scripts/run_mmgd_round6_multiscale.sh > /tmp/mmgd_round6.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/mmgd_cut.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"

echo "=== MMGD-Cut Round 6: Multi-Scale NCut ==="
echo "Start: $(date)"

# --- R6-1: DINOv3+SSD-1B + multi-scale (coarse 32×32 K=54 + fine 64×64 K=108) ---
echo ""
echo "=== R6-1: DINOv3+SSD-1B + multiscale ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --multiscale --k_fine_multiplier 2
echo "Done R6-1: $(date)"

# --- R6-2: DINOv3+SSD-1B + multi-scale + NAMR (stacked R4+R6) ---
echo ""
echo "=== R6-2: DINOv3+SSD-1B + multiscale + NAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --multiscale --k_fine_multiplier 2 \
    --namr --namr_window 2
echo "Done R6-2: $(date)"

# --- R6-3: DINOv3+SSD-1B + multi-scale + adaptive K + NAMR (full stack R4+R5+R6) ---
echo ""
echo "=== R6-3: DINOv3+SSD-1B + multiscale + adaptive K + NAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --multiscale --k_fine_multiplier 2 \
    --adaptive_k --k_min 10 --k_max 150 \
    --namr --namr_window 2
echo "Done R6-3: $(date)"

# --- R6-4: DINOv3+SSD-1B + multi-scale with 3× fine K (K_fine=162) ---
echo ""
echo "=== R6-4: DINOv3+SSD-1B + multiscale k_fine_multiplier=3 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --multiscale --k_fine_multiplier 3
echo "Done R6-4: $(date)"

echo ""
echo "=== Round 6 complete ==="
echo "End: $(date)"
