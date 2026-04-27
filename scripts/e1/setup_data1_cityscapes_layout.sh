#!/bin/bash
# scripts/e1/setup_data1_cityscapes_layout.sh
#
# Build a unified CITYSCAPES_ROOT at /Data1/cityscapes/ that combines:
#   - existing left/seq/gtFine on /home/cvpr_ug_5/umesh/datasets/cityscapes (via symlinks)
#   - newly downloaded right/right_seq/camera (real dirs on /Data1)
#
# Idempotent: safe to re-run. Reports per-package status and exits non-zero
# if a critical inconsistency is found.
#
# Also performs a sanity check on leftImg8bit_sequence/ — CUPS SMURF needs
# multiple frames per labeled image (typically 30). Warns if only 1 frame
# per labeled sample is present.
#
# Usage:
#   bash scripts/e1/setup_data1_cityscapes_layout.sh
#   # Optional overrides:
#   EXISTING_CITYSCAPES=/some/other/path bash scripts/e1/setup_data1_cityscapes_layout.sh
#   DATA1_CITYSCAPES=/Data1/cityscapes_v2 bash scripts/e1/setup_data1_cityscapes_layout.sh

set -u

EXISTING_CITYSCAPES="${EXISTING_CITYSCAPES:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"
DATA1_CITYSCAPES="${DATA1_CITYSCAPES:-/Data1/cityscapes}"

GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; CYN=$'\033[36m'; RST=$'\033[0m'
ok()   { echo "  ${GRN}OK${RST}  $*"; }
miss() { echo "  ${RED}--${RST}  $*"; }
warn() { echo "  ${YLW}!!${RST}  $*"; }

echo "${CYN}=== E1 /Data1 Cityscapes layout ===${RST}"
echo "  EXISTING_CITYSCAPES = $EXISTING_CITYSCAPES"
echo "  DATA1_CITYSCAPES    = $DATA1_CITYSCAPES"
echo

if [ ! -d "$EXISTING_CITYSCAPES" ]; then
    miss "EXISTING_CITYSCAPES does not exist."
    exit 2
fi
mkdir -p "$DATA1_CITYSCAPES"

# 1. Symlink the packages we already have on /home (left, left_seq, gtFine).
echo "${CYN}-- Symlinking existing packages from /home --${RST}"
for sub in leftImg8bit leftImg8bit_sequence gtFine; do
    src="$EXISTING_CITYSCAPES/$sub"
    dst="$DATA1_CITYSCAPES/$sub"

    if [ ! -d "$src" ]; then
        miss "$src not present on /home — cannot symlink."
        continue
    fi

    if [ -L "$dst" ]; then
        cur_target=$(readlink -f "$dst")
        if [ "$cur_target" = "$(readlink -f "$src")" ]; then
            ok "$sub already symlinked to $cur_target"
        else
            warn "$dst points to $cur_target — replacing with $src"
            rm "$dst"
            ln -s "$src" "$dst"
            ok "$sub re-linked"
        fi
    elif [ -d "$dst" ]; then
        warn "$dst is a real directory (not a symlink). Leaving it alone."
    else
        ln -s "$src" "$dst"
        ok "$sub linked  ($dst -> $src)"
    fi
done
echo

# 2. Make placeholder dirs for the 3 packages we need to download.
echo "${CYN}-- Preparing destinations for missing packages --${RST}"
for sub in rightImg8bit rightImg8bit_sequence camera; do
    dst="$DATA1_CITYSCAPES/$sub"
    if [ -d "$dst" ] && [ "$(find "$dst" -mindepth 2 -type f 2>/dev/null | head -1)" ]; then
        ok "$sub already populated under /Data1"
    else
        mkdir -p "$dst"
        warn "$sub: empty directory ready under $dst — needs e1_download_cityscapes.sh"
    fi
done
echo

# 3. leftImg8bit_sequence completeness check.
echo "${CYN}-- leftImg8bit_sequence frame-count sanity --${RST}"
SEQ_DIR="$DATA1_CITYSCAPES/leftImg8bit_sequence/train"
if [ -d "$SEQ_DIR" ]; then
    sample_city=$(find "$SEQ_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
    if [ -z "$sample_city" ]; then
        miss "no city subdirs under $SEQ_DIR"
    else
        # group files by labeled-frame stem (the "_<seq>_<frame>" stripped to "_<seq>")
        # and count members per group, then take the median.
        n_total=$(find "$sample_city" -maxdepth 1 -type f -name '*_leftImg8bit.png' 2>/dev/null | wc -l)
        n_unique=$(find "$sample_city" -maxdepth 1 -type f -name '*_leftImg8bit.png' 2>/dev/null \
                    | sed -E 's/_[0-9]{6}_leftImg8bit\.png$//' | sort -u | wc -l)
        if [ "$n_unique" -eq 0 ]; then
            miss "$sample_city has no *_leftImg8bit.png files"
        else
            ratio=$(awk -v t="$n_total" -v u="$n_unique" 'BEGIN { if (u==0) print 0; else printf "%.1f", t/u }')
            echo "  sample city: $(basename "$sample_city")"
            echo "  total frames: $n_total"
            echo "  unique image stems: $n_unique"
            echo "  frames / labeled-image: $ratio"
            if awk -v r="$ratio" 'BEGIN { exit !(r > 25) }'; then
                ok "looks like the full 30-frame sequence is present"
            elif awk -v r="$ratio" 'BEGIN { exit !(r > 1.5) }'; then
                warn "partial sequence — ratio $ratio. CUPS SMURF needs ≥2 adjacent frames; may still work but verify upstream."
            else
                miss "looks like only the labeled frame is present (ratio $ratio). CUPS SMURF will fail. Re-download package 14 — see scripts/e1_download_cityscapes.sh warning."
            fi
        fi
    fi
else
    miss "$SEQ_DIR not present"
fi
echo

# 4. Final inventory using existing helper (if available).
if [ -x "$(dirname "$0")/../e1_check_cityscapes.sh" ]; then
    echo "${CYN}-- Inventory via scripts/e1_check_cityscapes.sh --${RST}"
    bash "$(dirname "$0")/../e1_check_cityscapes.sh" "$DATA1_CITYSCAPES" || true
fi

echo
echo "${CYN}=== Layout setup complete ===${RST}"
echo "  CITYSCAPES_ROOT for downstream scripts:  $DATA1_CITYSCAPES"
