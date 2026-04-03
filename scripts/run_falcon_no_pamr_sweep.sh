#!/bin/bash
# Falcon no-PAMR refinement sweep: F1 was best at 38.65% without PAMR
# Now test α, β, reg_lambda variations without PAMR
# Usage: nohup bash scripts/run_falcon_no_pamr_sweep.sh > /tmp/falcon_no_pamr_sweep.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/falcon_pseudo_semantics.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"
DINO="--dino_only"

echo "=== Falcon No-PAMR Refinement Sweep ==="
echo "Start: $(date)"

# --- F1-ref1: higher alpha (α=6.0, no PAMR) ---
echo ""
echo "=== F1-ref1: alpha=6.0, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 6.0 --beta 0.5 --n_iter 15 --init kmeans $DINO
echo "Done F1-ref1: $(date)"

# --- F1-ref2: alpha=5.5, no PAMR (paper sweet spot) ---
echo ""
echo "=== F1-ref2: alpha=5.5, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 5.5 --beta 0.5 --n_iter 15 --init kmeans $DINO
echo "Done F1-ref2: $(date)"

# --- F1-ref3: alpha=4.5, reg_lambda=0.5, no PAMR ---
echo ""
echo "=== F1-ref3: alpha=4.5, reg_lambda=0.5, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.5 $DINO
echo "Done F1-ref3: $(date)"

# --- F1-ref4: alpha=4.5, reg_lambda=0.1, no PAMR ---
echo ""
echo "=== F1-ref4: alpha=4.5, reg_lambda=0.1, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.1 $DINO
echo "Done F1-ref4: $(date)"

# --- F1-ref5: alpha=6.0, reg_lambda=0.5, no PAMR ---
echo ""
echo "=== F1-ref5: alpha=6.0, reg_lambda=0.5, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 6.0 --beta 0.5 --n_iter 15 --init kmeans --reg_lambda 0.5 $DINO
echo "Done F1-ref5: $(date)"

# --- F1-ref6: alpha=4.5, beta=0.1 (sharper reweight), no PAMR ---
echo ""
echo "=== F1-ref6: alpha=4.5, beta=0.1, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.1 --n_iter 15 --init kmeans $DINO
echo "Done F1-ref6: $(date)"

# --- F1-ref7: alpha=4.5, n_iter=25, no PAMR ---
echo ""
echo "=== F1-ref7: alpha=4.5, n_iter=25, no PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase all --coco_root $COCO --device $DEVICE \
    --K 27 --alpha 4.5 --beta 0.5 --n_iter 25 --init kmeans $DINO
echo "Done F1-ref7: $(date)"

echo ""
echo "=== All Falcon no-PAMR configs complete ==="
echo "End: $(date)"
