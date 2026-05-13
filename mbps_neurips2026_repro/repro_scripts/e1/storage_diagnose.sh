#!/bin/bash
# scripts/e1/storage_diagnose.sh
#
# Quick diagnostic for "running low on storage" on the A6000.
# Reports:
#   1. df for /, /home, /Data1
#   2. per-package disk usage under $CITYSCAPES_ROOT_DST
#   3. orphan zips in _downloads/ that can be safely deleted
#   4. anything left on /home/.../cityscapes/ (should be empty after migration)
#
# Read-only by default. Pass `--clean` to delete any obviously-corrupt or
# leftover artifacts identified.

set -uo pipefail

CITYSCAPES_HOME="${CITYSCAPES_HOME:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"
CITYSCAPES_ROOT_DST="${CITYSCAPES_ROOT_DST:-/Data1/cityscapes}"
CLEAN=0
[ "${1:-}" = "--clean" ] && CLEAN=1

GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; CYN=$'\033[36m'; RST=$'\033[0m'
section() { echo; echo "${CYN}=== $* ===${RST}"; }
ok()   { echo "  ${GRN}OK${RST}  $*"; }
warn() { echo "  ${YLW}!!${RST}  $*"; }

section "1. Filesystem usage (df -h)"
df -h / /home /Data1 2>/dev/null | sed 's/^/  /' \
    || df -h | grep -E '(^Filesystem|/Data1|/home|/$)' | sed 's/^/  /'

section "2. /Data1/cityscapes per-package usage"
if [ -d "$CITYSCAPES_ROOT_DST" ]; then
    for sub in leftImg8bit leftImg8bit_sequence rightImg8bit rightImg8bit_sequence camera gtFine; do
        d="$CITYSCAPES_ROOT_DST/$sub"
        if [ -d "$d" ]; then
            sz=$(du -sh "$d" 2>/dev/null | cut -f1)
            n=$(find "$d" -mindepth 2 -type f 2>/dev/null | wc -l | tr -d ' ')
            ok "$sub: $sz, $n files"
        else
            warn "$sub: ABSENT"
        fi
    done
else
    warn "$CITYSCAPES_ROOT_DST does not exist."
fi

section "3. Download cache (_downloads/) — orphan zips"
DL_DIR_DATA1="$CITYSCAPES_ROOT_DST/_downloads"
DL_DIR_HOME="$CITYSCAPES_HOME/_downloads"
TOTAL_RECLAIMABLE=0
for d in "$DL_DIR_DATA1" "$DL_DIR_HOME"; do
    [ -d "$d" ] || continue
    echo "  inspecting $d:"
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        sz_h=$(du -h "$f" 2>/dev/null | cut -f1)
        # Decide: is this orphan? A zip is orphan if its corresponding
        # subdir already has files (extract done) OR if it's tiny (HTML).
        base=$(basename "$f" .zip)
        case "$base" in
            gtFine_trainvaltest)             corr_sub="gtFine" ;;
            leftImg8bit_trainvaltest)        corr_sub="leftImg8bit" ;;
            rightImg8bit_trainvaltest)       corr_sub="rightImg8bit" ;;
            camera_trainvaltest)             corr_sub="camera" ;;
            leftImg8bit_sequence_trainvaltest)  corr_sub="leftImg8bit_sequence" ;;
            rightImg8bit_sequence_trainvaltest) corr_sub="rightImg8bit_sequence" ;;
            *) corr_sub="" ;;
        esac
        verdict="keep (in-flight or partial)"
        if (( sz < 1000000 )); then
            verdict="DELETE — tiny ($sz bytes), likely auth-redirect HTML"
        elif [ -n "$corr_sub" ] && [ -d "$CITYSCAPES_ROOT_DST/$corr_sub/train" ]; then
            verdict="DELETE — extract complete at $CITYSCAPES_ROOT_DST/$corr_sub"
        fi
        echo "    $f  ($sz_h)  ->  $verdict"
        if [[ "$verdict" == DELETE* ]]; then
            TOTAL_RECLAIMABLE=$((TOTAL_RECLAIMABLE + sz))
            if [ "$CLEAN" = "1" ]; then
                rm -f "$f"
                ok "removed $f"
            fi
        fi
    done < <(find "$d" -maxdepth 1 -type f -name '*.zip' 2>/dev/null)
    # Also report cookies.txt — tiny, noise; report only if >1KB
    if [ -f "$d/cookies.txt" ]; then
        cs=$(stat -c%s "$d/cookies.txt" 2>/dev/null || stat -f%z "$d/cookies.txt" 2>/dev/null || echo 0)
        echo "    $d/cookies.txt ($cs bytes)  ->  keep (active session cookie)"
    fi
done

awk -v r="$TOTAL_RECLAIMABLE" 'BEGIN { printf "\n  Total reclaimable from orphan zips: %.1f GB\n", r / 1024 / 1024 / 1024 }'
[ "$CLEAN" = "0" ] && (( TOTAL_RECLAIMABLE > 0 )) && \
    warn "re-run with --clean to delete the DELETE-marked files."

section "4. Leftover content on /home Cityscapes (should be empty after migration)"
if [ -d "$CITYSCAPES_HOME" ]; then
    for sub in leftImg8bit leftImg8bit_sequence rightImg8bit rightImg8bit_sequence camera gtFine; do
        d="$CITYSCAPES_HOME/$sub"
        if [ -d "$d" ] && [ -n "$(find "$d" -maxdepth 3 -type f 2>/dev/null | head -1)" ]; then
            sz=$(du -sh "$d" 2>/dev/null | cut -f1)
            warn "$sub: STILL PRESENT on /home ($sz)"
        fi
    done
else
    ok "$CITYSCAPES_HOME does not exist (good — cleanup complete)."
fi

section "5. Largest single directories under /Data1 (top 10)"
du -h --max-depth=2 /Data1 2>/dev/null | sort -rh | head -10 | sed 's/^/  /' \
    || warn "could not enumerate /Data1 top-level usage"

echo
echo "${CYN}=== DIAGNOSE COMPLETE ===${RST}"
