#!/bin/bash
# Run pseudo-label diagnostics on A6000 machine.
# Usage: bash scripts/run_diag_a6000.sh
set -euo pipefail

PROJ_ROOT="$HOME/umesh/unsupervised_panoptic_segmentation"
CS_ROOT="$HOME/umesh/datasets/cityscapes"
PSEUDO_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020"

echo "=== Pulling latest code ==="
cd "$PROJ_ROOT"
git fetch origin main
git checkout origin/main -- scripts/diagnose_pseudo_labels.py

echo ""
echo "=== Running diagnostics ==="
python scripts/diagnose_pseudo_labels.py \
    --pseudo_dir "$PSEUDO_DIR" \
    --cityscapes_root "$CS_ROOT" \
    --output "$PROJ_ROOT/logs/a6000_diag.json"

echo ""
echo "=== Quick comparison with santosh ==="
echo "Santosh: 2975 instances, 15 things, 19.2 avg instances/img, 0 empty"
echo "Santosh things: [3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75]"
echo "Santosh file sizes (first 3):"
echo "  aachen_000000_000019_leftImg8bit_instance.png: 6120 bytes"
echo "  aachen_000001_000019_leftImg8bit_instance.png: 6296 bytes"
echo "  aachen_000002_000019_leftImg8bit_instance.png: 6470 bytes"
echo ""
echo "Check logs/a6000_diag.json for full results."
echo "If file sizes DIFFER from santosh, labels were generated independently (LIKELY root cause)."
echo "If thing classes DIFFER, that's a guaranteed training divergence."
