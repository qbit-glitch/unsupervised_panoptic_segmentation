#!/bin/bash
# Run pseudo-label diagnostics on BOTH machines and download results for comparison.
# Usage: bash scripts/run_diag_both_machines.sh
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG_SCRIPT="scripts/diagnose_pseudo_labels.py"

SANTOSH_HOST="santosh@100.93.203.100"
SANTOSH_PSEUDO="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020"
SANTOSH_CS="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes"
SANTOSH_PROJ="/media/santosh/Kuldeep/panoptic_segmentation/unsupervised_panoptic_segmentation"

A6000_HOST="cvpr_ug_5@master"
A6000_PSEUDO="$HOME/umesh/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020"
A6000_CS="$HOME/umesh/datasets/cityscapes"

echo "=== Step 1: Push diagnostic script to santosh ==="
scp "$PROJ_ROOT/$DIAG_SCRIPT" "$SANTOSH_HOST:$SANTOSH_PROJ/$DIAG_SCRIPT"

echo ""
echo "=== Step 2: Run on santosh ==="
ssh "$SANTOSH_HOST" "cd $SANTOSH_PROJ && python $DIAG_SCRIPT \
    --pseudo_dir '$SANTOSH_PSEUDO' \
    --cityscapes_root '$SANTOSH_CS' \
    --output /tmp/santosh_diag.json"

echo ""
echo "=== Step 3: Download santosh results ==="
scp "$SANTOSH_HOST:/tmp/santosh_diag.json" "$PROJ_ROOT/logs/santosh_diag.json"

echo ""
echo "=== Step 4: Run on A6000 (if this IS the A6000) ==="
if [ -d "$A6000_PSEUDO" ]; then
    cd "$PROJ_ROOT"
    python "$DIAG_SCRIPT" \
        --pseudo_dir "$A6000_PSEUDO" \
        --cityscapes_root "$A6000_CS" \
        --output "logs/a6000_diag.json"
else
    echo "A6000 pseudo-dir not found at $A6000_PSEUDO — run manually on A6000"
fi

echo ""
echo "=== Step 5: Compare results ==="
echo "Diagnostics saved to:"
echo "  logs/santosh_diag.json"
echo "  logs/a6000_diag.json"
echo ""
echo "Key things to check:"
echo "  1. file_counts — should match"
echo "  2. val_contamination — must be False on both"
echo "  3. thing_stuff_split — things_classes must match exactly"
echo "  4. instance_stats — avg_instances, empty_pct should be similar"
echo "  5. size_fingerprint — same files should have same sizes IF labels are identical"
