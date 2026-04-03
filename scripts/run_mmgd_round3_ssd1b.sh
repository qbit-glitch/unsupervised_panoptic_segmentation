#!/bin/bash
# MMGD-Cut Round 3: SSD-1B fusion experiments
# Critical test: DINOv3+SSD-1B should show genuine multi-modal gain (r=0.772)
# Usage: nohup bash scripts/run_mmgd_round3_ssd1b.sh > /tmp/mmgd_round3.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/mmgd_cut.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"

echo "=== MMGD-Cut Round 3: SSD-1B Experiments ==="
echo "Start: $(date)"

# --- R3-1: SSD-1B only, no diffusion (isolate SSD-1B backbone quality) ---
echo ""
echo "=== R3-1: SSD-1B only, no diffusion ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only
echo "Done R3-1: $(date)"

# --- R3-2: DINOv3 + SSD-1B fusion, no diffusion (the critical test) ---
echo ""
echo "=== R3-2: DINOv3 + SSD-1B fusion, no diffusion ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion none \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only
echo "Done R3-2: $(date)"

# --- R3-3: DINOv3 + SSD-1B + feature diffusion 3 steps ---
echo ""
echo "=== R3-3: DINOv3 + SSD-1B + feature diffusion 3 steps ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion ppr --diff_mode feature --diff_steps 3 \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only
echo "Done R3-3: $(date)"

# --- R3-4: DINOv3 + SSD-1B + affinity diffusion 3 steps ---
echo ""
echo "=== R3-4: DINOv3 + SSD-1B + affinity diffusion 3 steps ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --coco_root $COCO --device $DEVICE \
    --sources dinov3 ssd1b --step 10 --diffusion ppr --diff_mode affinity --diff_steps 3 \
    --K 54 --alpha 5.5 --reg_lambda 0.7 --dino_only
echo "Done R3-4: $(date)"

echo ""
echo "=== All Round 3 configs complete ==="
echo "End: $(date)"
