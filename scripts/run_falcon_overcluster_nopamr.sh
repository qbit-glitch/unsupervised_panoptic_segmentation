#!/bin/bash
# Falcon overclustering sweep: K=32/54/80 with best config (α=5.5, reg_λ=0.7, no PAMR)
# Also test K=54/80 with α=4.5/reg_λ=0.7 (runner-up) for robustness
# Usage: nohup bash scripts/run_falcon_overcluster_nopamr.sh > /tmp/falcon_overcluster.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/falcon_pseudo_semantics.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"
DINO="--dino_only"

echo "=== Falcon Overclustering Sweep (no PAMR) ==="
echo "Start: $(date)"

# --- OC1: K=32, alpha=5.5, reg_lambda=0.7 ---
echo ""
echo "=== OC1: K=32, alpha=5.5, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 32 --alpha 5.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done OC1: $(date)"

# --- OC2: K=54, alpha=5.5, reg_lambda=0.7 ---
echo ""
echo "=== OC2: K=54, alpha=5.5, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 54 --alpha 5.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done OC2: $(date)"

# --- OC3: K=80, alpha=5.5, reg_lambda=0.7 ---
echo ""
echo "=== OC3: K=80, alpha=5.5, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 80 --alpha 5.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done OC3: $(date)"

# --- OC4: K=54, alpha=4.5, reg_lambda=0.7 (runner-up α) ---
echo ""
echo "=== OC4: K=54, alpha=4.5, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 54 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done OC4: $(date)"

# --- OC5: K=80, alpha=4.5, reg_lambda=0.7 (runner-up α) ---
echo ""
echo "=== OC5: K=80, alpha=4.5, reg_lambda=0.7 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 80 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.7 $DINO
echo "Done OC5: $(date)"

echo ""
echo "=== All overclustering configs complete ==="
echo "End: $(date)"
