#!/usr/bin/env bash
# Download specific Cityscapes packages from cityscapes-dataset.com using
# credentialed login, then unzip into $CITYSCAPES_ROOT.
#
# Cityscapes is CC BY-NC-SA 4.0 with mandatory credentialed access — we
# CANNOT mirror it on Hugging Face. You must use your own account.
#
# Credentials (do NOT pass on the command line; export instead):
#   export CITYSCAPES_USER='your_login'
#   export CITYSCAPES_PASS='your_password'
#
# Usage (one or more package IDs):
#   bash scripts/e1_download_cityscapes.sh 15
#   bash scripts/e1_download_cityscapes.sh 3 4 14 15 8 1
#
# Package IDs and sizes (compressed):
#   1   gtFine_trainvaltest                   ~  241 MB
#   3   leftImg8bit_trainvaltest              ~ 11 GB
#   4   rightImg8bit_trainvaltest             ~ 11 GB
#   8   camera_trainvaltest                   ~  2 MB
#  14   leftImg8bit_sequence_trainvaltest     ~324 GB  <-- huge
#  15   rightImg8bit_sequence_trainvaltest    ~324 GB  <-- huge
#
# Resume-friendly: re-running with a partial archive resumes the download
# (wget -c). Already-extracted folders are detected and skipped.

set -euo pipefail

: "${CITYSCAPES_ROOT:?Set CITYSCAPES_ROOT, e.g. /home/cvpr_ug_5/umesh/datasets/cityscapes}"
: "${CITYSCAPES_USER:?Set CITYSCAPES_USER (export, do not paste on command line)}"
: "${CITYSCAPES_PASS:?Set CITYSCAPES_PASS (export, do not paste on command line)}"

if (( $# == 0 )); then
    echo "usage: $0 <packageID> [packageID ...]" >&2
    echo "  e.g. $0 15           # rightImg8bit_sequence only" >&2
    echo "       $0 3 4 14 15 8 1   # full CUPS Stage-1 set" >&2
    exit 64
fi

declare -A PKG_NAMES=(
    [1]="gtFine_trainvaltest"
    [3]="leftImg8bit_trainvaltest"
    [4]="rightImg8bit_trainvaltest"
    [8]="camera_trainvaltest"
    [14]="leftImg8bit_sequence_trainvaltest"
    [15]="rightImg8bit_sequence_trainvaltest"
)
declare -A PKG_SUBDIR=(
    [1]="gtFine"
    [3]="leftImg8bit"
    [4]="rightImg8bit"
    [8]="camera"
    [14]="leftImg8bit_sequence"
    [15]="rightImg8bit_sequence"
)

DL_DIR="${CITYSCAPES_DOWNLOAD_DIR:-$CITYSCAPES_ROOT/_downloads}"
COOKIE="$DL_DIR/cookies.txt"
mkdir -p "$DL_DIR" "$CITYSCAPES_ROOT"

trap 'rm -f "$COOKIE"' EXIT

echo "[cs] logging in as $CITYSCAPES_USER ..."
wget --quiet \
     --keep-session-cookies --save-cookies="$COOKIE" \
     --post-data="username=${CITYSCAPES_USER}&password=${CITYSCAPES_PASS}&submit=Login" \
     -O /dev/null \
     "https://www.cityscapes-dataset.com/login/"

if ! grep -q PHPSESSID "$COOKIE"; then
    echo "[cs] ERROR: login failed — check CITYSCAPES_USER / CITYSCAPES_PASS." >&2
    exit 2
fi
echo "[cs] login ok."

for pid in "$@"; do
    name="${PKG_NAMES[$pid]:-}"
    sub="${PKG_SUBDIR[$pid]:-}"
    if [[ -z "$name" ]]; then
        echo "[cs] skipping unknown packageID=$pid"
        continue
    fi
    if [[ -d "$CITYSCAPES_ROOT/$sub/train" || -d "$CITYSCAPES_ROOT/$sub/val" ]]; then
        echo "[cs] [skip] $name (pkg=$pid) already extracted at $CITYSCAPES_ROOT/$sub"
        continue
    fi
    zip="$DL_DIR/${name}.zip"
    echo "[cs] downloading $name (packageID=$pid) ..."
    wget -c --content-disposition \
         --load-cookies "$COOKIE" \
         -O "$zip" \
         "https://www.cityscapes-dataset.com/file-handling/?packageID=${pid}"

    echo "[cs] extracting $name -> $CITYSCAPES_ROOT ..."
    ( cd "$CITYSCAPES_ROOT" && unzip -q -o "$zip" )
    echo "[cs] [done] $name extracted; archive kept at $zip (rm to free disk)."
done

echo "[cs] all requested packages processed. Re-run e1_check_cityscapes.sh to verify."
