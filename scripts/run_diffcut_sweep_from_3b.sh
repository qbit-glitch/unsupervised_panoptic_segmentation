#!/bin/bash
# DiffCut sweep continuation: configs 3b-3j (3a already complete)
# Usage: nohup bash scripts/run_diffcut_sweep_from_3b.sh > /tmp/diffcut_sweep_3b.log 2>&1 &

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
SCRIPT="mbps_pytorch/diffcut_pseudo_semantics.py"
COCO="/Users/qbit-glitch/Desktop/datasets/coco"
DEVICE="mps"
DINO="--dino_only"

echo "=== DiffCut Sweep Continuation (3b-3j) ==="
echo "Start: $(date)"

# --- Config 3b: + PAMR ---
echo ""
echo "=== Config 3b: tau=0.5, alpha=10, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 10.0 --step 50 --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 10.0 --step 50 --pamr --K_global 27 --feature_source dinov3 $DINO
echo "Done 3b: $(date)"

# --- Config 3c: lower tau (τ=0.3, PAMR) ---
echo ""
echo "=== Config 3c: tau=0.3, alpha=10, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --tau 0.3 --alpha 10.0 --step 50 --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --tau 0.3 --alpha 10.0 --step 50 --pamr --K_global 27 --feature_source dinov3 $DINO
echo "Done 3c: $(date)"

# --- Config 3d: higher tau (τ=0.7, PAMR) ---
echo ""
echo "=== Config 3d: tau=0.7, alpha=10, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --tau 0.7 --alpha 10.0 --step 50 --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --tau 0.7 --alpha 10.0 --step 50 --pamr --K_global 27 --feature_source dinov3 $DINO
echo "Done 3d: $(date)"

# --- Config 3e: lower alpha (α=5, PAMR) ---
echo ""
echo "=== Config 3e: tau=0.5, alpha=5, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 5.0 --step 50 --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 5.0 --step 50 --pamr --K_global 27 --feature_source dinov3 $DINO
echo "Done 3e: $(date)"

# --- Config 3f: higher alpha (α=15, PAMR) ---
echo ""
echo "=== Config 3f: tau=0.5, alpha=15, PAMR ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase segment --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 15.0 --step 50 --pamr $DINO
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 15.0 --step 50 --pamr --K_global 27 --feature_source dinov3 $DINO
echo "Done 3f: $(date)"

# --- Config 3g: step=50 (same as 3b, skip) ---
echo ""
echo "=== Config 3g: step=50 (same as 3b, skipping) ==="

# --- Config 3h: step=200 (deferred) ---
echo ""
echo "=== Config 3h: step=200 (DEFERRED) ==="

# --- Config 3i: overclustering K=80 (reuses 3b segments) ---
echo ""
echo "=== Config 3i: tau=0.5, alpha=10, PAMR, K_global=80 ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 10.0 --step 50 --pamr --K_global 80 --feature_source dinov3 $DINO
echo "Done 3i: $(date)"

# --- Config 3j: SD features for clustering ---
echo ""
echo "=== Config 3j: tau=0.5, alpha=10, PAMR, SD features ==="
echo "Start: $(date)"
$PYTHON -u $SCRIPT --phase cluster --coco_root $COCO --device $DEVICE \
    --tau 0.5 --alpha 10.0 --step 50 --pamr --K_global 27 --feature_source sd $DINO
echo "Done 3j: $(date)"

echo ""
echo "=== All DiffCut configs complete ==="
echo "End: $(date)"
