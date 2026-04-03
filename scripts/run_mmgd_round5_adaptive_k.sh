#!/bin/bash
# MMGD-Cut Round 5: Adaptive K via Laplacian Eigengap
# Tests per-image K selection instead of fixed K=54.
# Expected gain: +0.5-2 mIoU from better-matched segmentation granularity.
# Usage: nohup bash scripts/run_mmgd_round5_adaptive_k.sh > /tmp/mmgd_round5.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/mmgd_cut.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"

echo "=== MMGD-Cut Round 5: Adaptive K via Eigengap ==="
echo "Start: $(date)"

# --- R5-1: DINOv3+SSD-1B + adaptive K, wide range [10, 150] ---
echo ""
echo "=== R5-1: DINOv3+SSD-1B + adaptive K [10, 150] ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --adaptive_k --k_min 10 --k_max 150
echo "Done R5-1: $(date)"

# --- R5-2: DINOv3+SSD-1B + adaptive K, tight range [20, 80] ---
echo ""
echo "=== R5-2: DINOv3+SSD-1B + adaptive K [20, 80] ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --adaptive_k --k_min 20 --k_max 80
echo "Done R5-2: $(date)"

# --- R5-3: DINOv3+SSD-1B + adaptive K + NAMR (stacked R4+R5) ---
echo ""
echo "=== R5-3: DINOv3+SSD-1B + adaptive K [10,150] + NAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --adaptive_k --k_min 10 --k_max 150 \
    --namr --namr_window 2
echo "Done R5-3: $(date)"

# --- R5-4: DINOv3 only + adaptive K [10, 150] (single-modal baseline) ---
echo ""
echo "=== R5-4: DINOv3 only + adaptive K [10, 150] ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --adaptive_k --k_min 10 --k_max 150
echo "Done R5-4: $(date)"

echo ""
echo "=== Round 5 complete ==="
echo "End: $(date)"
