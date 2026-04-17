#!/bin/bash
# Download santosh's pseudo-labels from HuggingFace Hub and set up for CUPS training.
#
# Run on A6000:
#   cd ~/umesh/unsupervised_panoptic_segmentation
#   git pull origin main
#   bash scripts/download_pseudolabels_a6000.sh
set -euo pipefail

CS_ROOT="$HOME/umesh/datasets/cityscapes"
PSEUDO_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020"
TAR_FILE="/tmp/cups_pseudo_labels_depthpro_tau020.tar.gz"
URL="https://huggingface.co/datasets/qbit-glitch/cityscapes-cups-pseudo-labels/resolve/main/cups_pseudo_labels_depthpro_tau020.tar.gz"

echo "=== Step 1: Download from HuggingFace Hub (79MB) ==="
if [ -f "$TAR_FILE" ]; then
    echo "Already downloaded at $TAR_FILE, skipping."
else
    wget -O "$TAR_FILE" "$URL"
fi
ls -lh "$TAR_FILE"

echo ""
echo "=== Step 2: Backup old pseudo-labels ==="
if [ -d "$PSEUDO_DIR" ]; then
    mv "$PSEUDO_DIR" "${PSEUDO_DIR}_old_wrong_split"
    echo "Backed up old labels to ${PSEUDO_DIR}_old_wrong_split"
fi

echo ""
echo "=== Step 3: Extract ==="
cd "$CS_ROOT"
tar xzf "$TAR_FILE"
echo "Extracted to $PSEUDO_DIR"

echo ""
echo "=== Step 4: Verify ==="
INST_COUNT=$(ls "$PSEUDO_DIR"/*_instance.png 2>/dev/null | wc -l)
SEM_COUNT=$(ls "$PSEUDO_DIR"/*_semantic.png 2>/dev/null | wc -l)
PT_COUNT=$(ls "$PSEUDO_DIR"/*.pt 2>/dev/null | wc -l)
VAL_COUNT=$(ls "$PSEUDO_DIR" | grep -cE '^(frankfurt|lindau|munster)_' || true)
echo "Instance: $INST_COUNT, Semantic: $SEM_COUNT, .pt: $PT_COUNT"
echo "Val contamination: $VAL_COUNT (should be 0)"
echo "Expected: 2975 each, 0 val contamination"

FIRST_FILE="$PSEUDO_DIR/aachen_000000_000019_leftImg8bit_instance.png"
if [ -f "$FIRST_FILE" ]; then
    SIZE=$(stat -c%s "$FIRST_FILE" 2>/dev/null || stat -f%z "$FIRST_FILE")
    echo "First file size: $SIZE bytes (santosh reference: 6120)"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Now kill old training, clean checkpoints, and restart:"
echo "  pkill -f 'train.py.*dinov3' || true"
echo "  rm -rf ~/umesh/experiments/cups_dinov3_vitb_depthpro_tau020_anydesk*"
echo "  cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups"
echo "  export WANDB_MODE=disabled"
echo "  nohup python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml > ~/cups_fixed_labels.log 2>&1 &"
echo "  tail -f ~/cups_fixed_labels.log"
echo ""
echo "Verify: Thing classes should show 15 classes: (16, 14, 73, 3, 29, 46, 38, 75, 37, 11, 65, 32, 15, 62, 45)"
