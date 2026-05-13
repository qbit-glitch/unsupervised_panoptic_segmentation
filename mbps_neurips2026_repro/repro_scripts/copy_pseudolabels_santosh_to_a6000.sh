#!/bin/bash
# Copy santosh's proven pseudo-labels (PQ=28.4) to A6000.
#
# ROOT CAUSE: A6000 pseudo-labels were generated independently, producing a
# completely different thing/stuff class split:
#   Santosh: 15 things [3,11,14,15,16,29,32,37,38,45,46,62,65,73,75]
#   A6000:   12 things [4,8,13,23,28,29,32,52,53,56,59,71]
#   Overlap: ONLY 2 classes (29, 32)
# This means the A6000 model learns a fundamentally different task → PQ plateau at 22%.
#
# This script copies santosh's exact pseudo-labels to A6000 via this local machine as relay.
#
# Usage: bash scripts/copy_pseudolabels_santosh_to_a6000.sh
set -euo pipefail

SANTOSH_HOST="santosh@100.93.203.100"
SANTOSH_PSEUDO="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020"

A6000_PSEUDO="$HOME/umesh/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020"

LOCAL_STAGING="/tmp/cups_pseudo_labels_depthpro_tau020"

echo "=== Step 1: Check santosh source ==="
FILE_COUNT=$(ssh -o ConnectTimeout=10 "$SANTOSH_HOST" "ls '$SANTOSH_PSEUDO' | wc -l")
echo "Santosh has $FILE_COUNT files (expected 8925 = 2975 × 3)"
if [ "$FILE_COUNT" -ne 8925 ]; then
    echo "ERROR: Expected 8925 files, got $FILE_COUNT"
    exit 1
fi

echo ""
echo "=== Step 2: Download from santosh to local staging ==="
echo "Downloading ~8925 files to $LOCAL_STAGING ..."
mkdir -p "$LOCAL_STAGING"
rsync -avz --progress "$SANTOSH_HOST:$SANTOSH_PSEUDO/" "$LOCAL_STAGING/"

echo ""
echo "=== Step 3: Verify local copy ==="
LOCAL_COUNT=$(ls "$LOCAL_STAGING" | wc -l)
echo "Local staging has $LOCAL_COUNT files"
if [ "$LOCAL_COUNT" -ne 8925 ]; then
    echo "ERROR: Expected 8925 files, got $LOCAL_COUNT"
    exit 1
fi

echo ""
echo "=== Step 4: Backup A6000 old labels ==="
echo "Renaming $A6000_PSEUDO to ${A6000_PSEUDO}_old_wrong_split"
if [ -d "$A6000_PSEUDO" ]; then
    mv "$A6000_PSEUDO" "${A6000_PSEUDO}_old_wrong_split"
    echo "Backed up old labels"
else
    echo "No existing labels to backup"
fi

echo ""
echo "=== Step 5: Copy to A6000 ==="
echo "NOTE: Run this step ON the A6000 machine, or modify for SSH/rsync."
echo "If running on A6000 directly:"
echo "  cp -r $LOCAL_STAGING $A6000_PSEUDO"
echo ""
echo "If the A6000 is not this machine, you need to:"
echo "  1. Push local staging to A6000 via rsync/scp"
echo "  2. Or run the santosh → A6000 rsync directly if they can see each other"
cp -r "$LOCAL_STAGING" "$A6000_PSEUDO" 2>/dev/null || echo "Copy skipped (A6000 path not local)"

echo ""
echo "=== Step 6: Verify ==="
if [ -d "$A6000_PSEUDO" ]; then
    A6000_COUNT=$(ls "$A6000_PSEUDO" | wc -l)
    echo "A6000 now has $A6000_COUNT files"
    # Check no val cities
    VAL_COUNT=$(ls "$A6000_PSEUDO" | grep -E '^(frankfurt|lindau|munster)_' | wc -l)
    echo "Val city files: $VAL_COUNT (should be 0)"
else
    echo "A6000 pseudo-label dir not found at $A6000_PSEUDO"
    echo "Upload $LOCAL_STAGING to the A6000 manually."
fi

echo ""
echo "=== Done ==="
echo "After copying, restart training on A6000:"
echo "  cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups"
echo "  nohup python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml > ~/cups_bs1_fixed_labels.log 2>&1 &"
echo ""
echo "Expected thing classes after fix: [3,11,14,15,16,29,32,37,38,45,46,62,65,73,75] (15 things)"
