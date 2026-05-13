#!/usr/bin/env bash
# Inventory Cityscapes layout under $1 (or $CITYSCAPES_ROOT). Reports which of
# the six packages CUPS Stage-1 expects are present, with on-disk sizes.
#
# Usage:
#   bash scripts/e1_check_cityscapes.sh /home/cvpr_ug_5/umesh/datasets/cityscapes

set -u

ROOT="${1:-${CITYSCAPES_ROOT:?path required}}"

# CUPS Stage-1 (CityscapesStereoVideo) reads:
#   leftImg8bit/{train,val}              - packageID=3,  ~11 GB
#   rightImg8bit/{train,val}             - packageID=4,  ~11 GB
#   leftImg8bit_sequence/{train,val}     - packageID=14, ~324 GB
#   rightImg8bit_sequence/{train,val}    - packageID=15, ~324 GB
#   camera/{train,val}                   - packageID=8,  ~2 MB
#   gtFine/{train,val}                   - packageID=1,  ~241 MB (eval only)

declare -A NEEDS=(
    [leftImg8bit]=3
    [rightImg8bit]=4
    [leftImg8bit_sequence]=14
    [rightImg8bit_sequence]=15
    [camera]=8
    [gtFine]=1
)

printf "Cityscapes inventory at: %s\n\n" "$ROOT"
printf "%-26s %-6s %-12s %-10s %-10s %s\n" "PACKAGE" "pkgID" "STATUS" "TRAIN_GB" "VAL_GB" "TRAIN_FILES"
printf -- "------------------------- ------ ------------ ---------- ---------- -------------\n"

ANY_MISSING=0
for sub in leftImg8bit rightImg8bit leftImg8bit_sequence rightImg8bit_sequence camera gtFine; do
    pid="${NEEDS[$sub]}"
    if [[ ! -d "$ROOT/$sub" ]]; then
        printf "%-26s %-6s %-12s\n" "$sub" "$pid" "MISSING"
        ANY_MISSING=1
        continue
    fi
    train_gb="-"; val_gb="-"; train_files="-"
    [[ -d "$ROOT/$sub/train" ]] && train_gb=$(du -sBG "$ROOT/$sub/train" 2>/dev/null | awk '{gsub(/G/,"",$1); print $1}')
    [[ -d "$ROOT/$sub/val"   ]] && val_gb=$(du -sBG   "$ROOT/$sub/val"   2>/dev/null | awk '{gsub(/G/,"",$1); print $1}')
    [[ -d "$ROOT/$sub/train" ]] && train_files=$(find "$ROOT/$sub/train" -maxdepth 3 -type f \( -name "*.png" -o -name "*.json" -o -name "*.txt" \) 2>/dev/null | wc -l | tr -d ' ')
    printf "%-26s %-6s %-12s %-10s %-10s %s\n" "$sub" "$pid" "ok" "$train_gb" "$val_gb" "$train_files"
done

echo
if (( ANY_MISSING )); then
    echo "STATUS: incomplete — packages marked MISSING above must be downloaded."
    echo "        Use scripts/e1_download_cityscapes.sh to fetch."
    exit 1
else
    echo "STATUS: complete — all six packages present."
fi
