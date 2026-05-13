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

# Tail-truncation slack used when removing HTML appended to a partial
# download. Generous (1 MiB) so we never accidentally truncate INTO
# the valid zip body. wget -c will refill those bytes from the real
# server on the next attempt.
HTML_TRUNCATE_SLACK=1048576

# Returns 0 (true) if the last 16 KB of the file contains HTML markers
# typically present in the Cityscapes login redirect page.
_has_html_tail() {
    local f="$1"
    [ -f "$f" ] || return 1
    tail -c 16384 "$f" 2>/dev/null \
        | grep -q -E -i '<html|<!doctype|<head|<body|</html|cityscapes-dataset\.com|<meta|<title|PHPSESSID' \
        2>/dev/null
}

# Truncate the last HTML_TRUNCATE_SLACK bytes off a file so wget -c can
# resume cleanly from the real server next attempt. No-op if the file
# is smaller than the slack (we just delete it).
_truncate_html_tail() {
    local f="$1"
    local sz
    sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    if (( sz <= HTML_TRUNCATE_SLACK )); then
        echo "[cs] [trunc] $f size=${sz} <= slack=${HTML_TRUNCATE_SLACK}; deleting outright."
        rm -f "$f"
    else
        local newsz=$((sz - HTML_TRUNCATE_SLACK))
        truncate -s "$newsz" "$f"
        echo "[cs] [trunc] $f: dropped ${HTML_TRUNCATE_SLACK} bytes of HTML tail; new size ${newsz} bytes."
    fi
}

# Robust per-package download. Handles:
#   - mid-download cookie expiration (server returns 302 -> login HTML appended)
#   - transient network errors (wget exits non-zero, file is partial)
#   - HTML tail at end of partial file (truncate, then resume)
# Re-logs-in between attempts. Uses wget -c to resume instead of
# re-downloading from scratch.
download_pkg_with_retry() {
    local pid="$1"
    local zip="$2"
    local label="$3"
    local url="https://www.cityscapes-dataset.com/file-handling/?packageID=${pid}"
    local max_attempts=20
    local attempt=1

    while (( attempt <= max_attempts )); do
        # Always re-login before each attempt — the cookie is the
        # primary failure mode on long downloads.
        if ! do_login; then
            echo "[cs] [$label] login failed (attempt $attempt). Sleeping 60s ..."
            sleep 60
            attempt=$((attempt+1)); continue
        fi

        # Pre-flight: if the existing partial file ends in HTML (from a
        # previous interrupted attempt), truncate before resuming.
        if [ -f "$zip" ] && _has_html_tail "$zip"; then
            _truncate_html_tail "$zip"
        fi

        # Pre-flight: drop suspiciously tiny files entirely.
        if [ -f "$zip" ]; then
            local sz0
            sz0=$(stat -c%s "$zip" 2>/dev/null || stat -f%z "$zip" 2>/dev/null || echo 0)
            if (( sz0 < SUSPICIOUS_MIN_BYTES )); then
                echo "[cs] [$label] partial file is only $sz0 bytes — deleting and starting fresh."
                rm -f "$zip"
            fi
        fi

        echo "[cs] [$label] download attempt $attempt/$max_attempts ..."
        # --tries=1: fail fast on transient errors so our loop can re-login.
        wget -c --tries=1 --timeout=300 --content-disposition \
             --load-cookies "$COOKIE" \
             -O "$zip" \
             "$url"
        local rc=$?

        if (( rc == 0 )); then
            # wget thinks it's done — but a 302+200 HTML redirect can
            # also exit 0. Validate the zip integrity AND check for
            # HTML tail before trusting.
            if _has_html_tail "$zip"; then
                echo "[cs] [$label] wget exit 0 but HTML tail detected — cookie expired mid-download."
                _truncate_html_tail "$zip"
                attempt=$((attempt+1))
                echo "[cs] [$label] sleeping 30s before next attempt ..."
                sleep 30
                continue
            fi
            if unzip -tq "$zip" >/dev/null 2>&1; then
                echo "[cs] [$label] download complete and zip-validated."
                return 0
            fi
            # Integrity failed but no HTML tail — probably wget reported
            # success on a partial. Just retry; wget -c will resume.
            echo "[cs] [$label] wget exit 0 but zip integrity failed; resuming with wget -c."
        else
            echo "[cs] [$label] wget exited rc=$rc on attempt $attempt."
            # Even on rc != 0, the file might have HTML appended from
            # a redirect we caught. Clean up before next resume.
            if [ -f "$zip" ] && _has_html_tail "$zip"; then
                _truncate_html_tail "$zip"
            fi
        fi

        attempt=$((attempt+1))
        local backoff=$(( attempt * 15 < 300 ? attempt * 15 : 300 ))
        echo "[cs] [$label] sleeping ${backoff}s before next attempt ..."
        sleep "$backoff"
    done

    echo "[cs] [$label] ERROR: gave up after $max_attempts attempts." >&2
    return 9
}

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

    if ! download_pkg_with_retry "$pid" "$zip" "$name"; then
        echo "[cs] ERROR: failed to download $name after retries." >&2
        echo "[cs]   partial file (if any) preserved at $zip — wget -c will resume on rerun." >&2
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
