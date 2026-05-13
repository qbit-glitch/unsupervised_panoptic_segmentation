#!/bin/bash
# Fix: CUPS loads all files from flat pseudo-label dir as training data.
# Val cities (frankfurt, lindau, munster) must be in a separate subdir.
# Move val files to cups_pseudo_labels_depthpro_tau020_val/
set -euo pipefail

CS_ROOT="$HOME/umesh/datasets/cityscapes"
PSEUDO_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020"
VAL_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020_val"

mkdir -p "$VAL_DIR"

VAL_CITIES="frankfurt lindau munster"
moved=0

for city in $VAL_CITIES; do
    count=$(ls "$PSEUDO_DIR"/${city}_* 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        mv "$PSEUDO_DIR"/${city}_* "$VAL_DIR/"
        echo "Moved $count files for $city"
        moved=$((moved + count))
    fi
done

echo ""
echo "Total moved: $moved val files to $VAL_DIR"
echo "Remaining train files: $(ls "$PSEUDO_DIR"/*_instance.png 2>/dev/null | wc -l) instances"
echo "Val files: $(ls "$VAL_DIR"/*_instance.png 2>/dev/null | wc -l) instances"
