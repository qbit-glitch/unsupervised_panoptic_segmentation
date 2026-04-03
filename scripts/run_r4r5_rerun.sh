#!/bin/bash
# Rerun: fix config_key collisions (R4) and corrupted R5-1 cache.
# R4-1 (7T w=2), R4-2 (single T=0.10), R4-4 (w=3): now use unique keys _namrW{w}T{n}.
# R5-1 (adaptK10-150): cache deleted, rerun fresh.
# R5-3 (adaptK10-150 + NAMR): will reuse corrected R5-1 cache.
# Usage: nohup bash scripts/run_r4r5_rerun.sh > /tmp/mmgd_r4r5_rerun.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/mmgd_cut.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"

echo "=== R4/R5 Rerun: Config-key fixes + R5-1 cache fix ==="
echo "Start: $(date)"

# ── R4-1 (7T, window=2): key = _namrW2T7 ──
echo ""
echo "=== R4-1: DINOv3+SSD-1B + NAMR (7T, window=2) → _namrW2T7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --namr --namr_window 2
echo "Done R4-1: $(date)"

# ── R4-2 (single T=0.10, window=2): key = _namrW2T1 ──
echo ""
echo "=== R4-2: DINOv3+SSD-1B + NAMR (single T=0.10) → _namrW2T1 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --namr --namr_single_temp 0.10
echo "Done R4-2: $(date)"

# ── R4-4 (7T, window=3): key = _namrW3T7 ──
echo ""
echo "=== R4-4: DINOv3+SSD-1B + NAMR (7T, window=3) → _namrW3T7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --namr --namr_window 3
echo "Done R4-4: $(date)"

# ── R5-1 (adaptK10-150, DINOv3+SSD-1B): fresh Phase 1 ──
echo ""
echo "=== R5-1: DINOv3+SSD-1B + adaptive K [10,150] (fresh) ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --adaptive_k --k_min 10 --k_max 150
echo "Done R5-1: $(date)"

# ── R5-3 (adaptK10-150 + NAMR 7T w=2): reuses R5-1 cache ──
echo ""
echo "=== R5-3: DINOv3+SSD-1B + adaptive K [10,150] + NAMR → _adaptK10-150_namrW2T7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only \
    --adaptive_k --k_min 10 --k_max 150 \
    --namr --namr_window 2
echo "Done R5-3: $(date)"

echo ""
echo "=== Rerun complete ==="
echo "End: $(date)"
