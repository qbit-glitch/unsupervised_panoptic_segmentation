#!/bin/bash
# Falcon 2D grid sweep: alpha x reg_lambda (no PAMR)
# Goal: find joint optimum given non-additive interaction
# Known: alpha=4.5, reg=0.5 -> 40.59% (skip this config)
# Usage: nohup bash scripts/run_falcon_2d_grid_sweep.sh > /tmp/falcon_2d_grid.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/falcon_pseudo_semantics.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"
DINO="--dino_only"

echo "=== Falcon 2D Grid Sweep: alpha x reg_lambda ==="
echo "Start: $(date)"

# --- alpha=4.5, reg_lambda=0.3 ---
echo ""
echo "=== G1: alpha=4.5, reg_lambda=0.3 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.3 $DINO
echo "Done G1: $(date)"

# --- alpha=4.5, reg_lambda=0.7 ---
echo ""
echo "=== G2: alpha=4.5, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done G2: $(date)"

# --- alpha=5.0, reg_lambda=0.3 ---
echo ""
echo "=== G3: alpha=5.0, reg_lambda=0.3 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 5.0 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.3 $DINO
echo "Done G3: $(date)"

# --- alpha=5.0, reg_lambda=0.5 ---
echo ""
echo "=== G4: alpha=5.0, reg_lambda=0.5 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 5.0 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.5 $DINO
echo "Done G4: $(date)"

# --- alpha=5.0, reg_lambda=0.7 ---
echo ""
echo "=== G5: alpha=5.0, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 5.0 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done G5: $(date)"

# --- alpha=5.5, reg_lambda=0.3 ---
echo ""
echo "=== G6: alpha=5.5, reg_lambda=0.3 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 5.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.3 $DINO
echo "Done G6: $(date)"

# --- alpha=5.5, reg_lambda=0.5 ---
echo ""
echo "=== G7: alpha=5.5, reg_lambda=0.5 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 5.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.5 $DINO
echo "Done G7: $(date)"

# --- alpha=5.5, reg_lambda=0.7 ---
echo ""
echo "=== G8: alpha=5.5, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 5.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done G8: $(date)"

echo ""
echo "=== All 2D grid configs complete ==="
echo "End: $(date)"
