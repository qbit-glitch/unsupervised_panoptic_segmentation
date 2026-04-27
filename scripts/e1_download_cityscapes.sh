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
#
# To delete the .zip immediately after a successful extract (saves ~half
# the disk for the huge sequence packages):
#   DELETE_ZIPS=1 bash scripts/e1_download_cityscapes.sh 4 8 14 15
#
# WARNING — leftImg8bit_sequence "stubs": if your existing
# leftImg8bit_sequence/train/<city>/ only contains files with frame index
# 000019 (the labeled frame), the full sequence package was never
# extracted there. The downloader detects an existing directory and would
# skip pkg 14, so DELETE the stub first:
#   rm -rf $CITYSCAPES_ROOT/leftImg8bit_sequence
# then run with packageID=14.

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

# Re-login per package. The Cityscapes PHPSESSID expires after ~30-60
# minutes of inactivity AND on long downloads — so a multi-package run
# that takes hours will fail on the second+ package with a 302 redirect
# to /login/ that silently saves the login HTML in place of the .zip.
# Calling do_login() before every package is cheap (sub-second) and
# fully eliminates that failure mode.
do_login() {
    rm -f "$COOKIE"
    wget --quiet \
         --keep-session-cookies --save-cookies="$COOKIE" \
         --post-data="username=${CITYSCAPES_USER}&password=${CITYSCAPES_PASS}&submit=Login" \
         -O /dev/null \
         "https://www.cityscapes-dataset.com/login/"
    grep -q PHPSESSID "$COOKIE"
}

echo "[cs] initial login as $CITYSCAPES_USER ..."
if ! do_login; then
    echo "[cs] ERROR: login failed — check CITYSCAPES_USER / CITYSCAPES_PASS." >&2
    exit 2
fi
echo "[cs] login ok."

# Smallest legitimate Cityscapes package (camera_trainvaltest) is ~35 KB
# of HTML when the server redirects us to /login/. The smallest real zip
# (camera_trainvaltest) is ~2 MB. So any pre-existing zip < 1 MB is
# certainly a corrupt auth-redirect; nuke it before resume.
SUSPICIOUS_MIN_BYTES=1000000

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

    # Clean up any pre-existing tiny zip that is almost certainly the
    # login HTML page from a prior cookie-expiry failure. wget -c
    # would otherwise try to "resume" appending to it — useless.
    if [[ -f "$zip" ]]; then
        sz=$(stat -c%s "$zip" 2>/dev/null || stat -f%z "$zip" 2>/dev/null || echo 0)
        if (( sz < SUSPICIOUS_MIN_BYTES )); then
            echo "[cs] [cleanup] $zip is only $sz bytes — likely an auth-redirect HTML; deleting before retry."
            rm -f "$zip"
        fi
    fi

    # Re-login before every package so the cookie is fresh.
    if ! do_login; then
        echo "[cs] ERROR: re-login failed before $name (pkg=$pid)." >&2
        exit 2
    fi

    echo "[cs] downloading $name (packageID=$pid) ..."
    wget -c --content-disposition \
         --load-cookies "$COOKIE" \
         -O "$zip" \
         "https://www.cityscapes-dataset.com/file-handling/?packageID=${pid}"

    # Defensive: validate the .zip is a real archive before extracting.
    # If unzip can't open it, the server returned login HTML again.
    if ! unzip -tq "$zip" >/dev/null 2>&1; then
        sz=$(stat -c%s "$zip" 2>/dev/null || stat -f%z "$zip" 2>/dev/null || echo 0)
        echo "[cs] ERROR: $zip ($sz bytes) failed zip-integrity check." >&2
        echo "[cs]   first 200 bytes of the saved file:" >&2
        head -c 200 "$zip" >&2 || true
        echo >&2
        echo "[cs]   deleting and aborting; re-run will re-login and retry this package." >&2
        rm -f "$zip"
        exit 9
    fi

    echo "[cs] extracting $name -> $CITYSCAPES_ROOT ..."
    ( cd "$CITYSCAPES_ROOT" && unzip -q -o "$zip" )
    if [[ "${DELETE_ZIPS:-0}" == "1" ]]; then
        echo "[cs] [done] $name extracted; deleting $zip to free disk."
        rm -f "$zip"
    else
        echo "[cs] [done] $name extracted; archive kept at $zip (set DELETE_ZIPS=1 to auto-rm)."
    fi
done

echo "[cs] all requested packages processed. Re-run e1_check_cityscapes.sh to verify."
