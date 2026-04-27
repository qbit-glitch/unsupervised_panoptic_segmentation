#!/bin/bash
# scripts/e1/download_cityscapes_all.sh
#
# One-shot, idempotent downloader that produces a COMPLETE Cityscapes
# Stereo+Video tree at $CITYSCAPES_ROOT (default /Data1/cityscapes/) with
# all six packages CUPS Stage-1 needs to extract pseudo-labels:
#
#     pkg  subdir                       ~size unpacked   purpose
#     ---  ---------------------------  ---------------  --------------------
#     1    gtFine                       ~  0.25 GB       eval / class taxonomy
#     3    leftImg8bit                  ~ 11 GB          Stage-2/3 input images
#     4    rightImg8bit                 ~ 11 GB          stereo for SF2SE3
#     8    camera                       ~  2 MB          intrinsics for SF2SE3
#     14   leftImg8bit_sequence         ~119 GB          adjacent frames for RAFT-SMURF flow
#     15   rightImg8bit_sequence        ~119 GB          stereo + temporal for SMURF disparity
#                                       --------
#                                       ~260 GB total on disk
#
# By default this script symlinks the three packages already on /home
# (left, leftImg8bit_sequence, gtFine) into /Data1/cityscapes/ rather than
# re-downloading them — saves ~120 GB / ~6 h of bandwidth. Override with
# FRESH=1 to ignore /home entirely.
#
# ============================================================================
# REQUIRED ENV VARS (export before invocation; never paste on command line)
# ============================================================================
#   CITYSCAPES_USER   — your cityscapes-dataset.com login
#   CITYSCAPES_PASS   — your cityscapes-dataset.com password
#
# ============================================================================
# OPTIONAL ENV VARS
# ============================================================================
#   CITYSCAPES_ROOT       — destination root. Default: /Data1/cityscapes
#   EXISTING_CITYSCAPES   — source for /home symlinks. Default:
#                           /home/cvpr_ug_5/umesh/datasets/cityscapes
#   FRESH                 — 1 ⇒ skip the symlink stage, force fresh download
#                           of every package. Default: 0.
#   DELETE_ZIPS           — 1 ⇒ delete each .zip after successful extract.
#                           Default: 1.
#   LIGHT                 — 1 ⇒ skip pkg 15 (rightImg8bit_sequence, 119 GB).
#                           CUPS Stage-1 will then run with reduced flow
#                           quality on the right camera; emergency fallback
#                           only if /Data1 fills up. Default: 0.
#   PACKAGES              — explicit space-separated pkg-ID list (overrides
#                           the default "1 3 4 8 14 15" computed below).
#
# ============================================================================
# USAGE
# ============================================================================
#
# Standard run on the A6000 (uses /home symlinks for left/seq/gtFine,
# downloads right/right_seq/camera onto /Data1):
#
#   export CITYSCAPES_USER='your_login'
#   export CITYSCAPES_PASS='your_password'
#   bash scripts/e1/download_cityscapes_all.sh
#
# Long-running ⇒ background it:
#
#   nohup bash scripts/e1/download_cityscapes_all.sh \
#       > /Data1/e1_cityscapes_download.log 2>&1 &
#   echo $! > /Data1/e1_cityscapes_download.pid
#
# Force fresh re-download of everything onto /Data1:
#
#   FRESH=1 bash scripts/e1/download_cityscapes_all.sh
#
# Emergency: skip the giant right-sequence package:
#
#   LIGHT=1 bash scripts/e1/download_cityscapes_all.sh
#
set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve paths & defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/Data1/cityscapes}"
EXISTING_CITYSCAPES="${EXISTING_CITYSCAPES:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"
FRESH="${FRESH:-0}"
DELETE_ZIPS="${DELETE_ZIPS:-1}"
LIGHT="${LIGHT:-0}"

GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; CYN=$'\033[36m'; RST=$'\033[0m'
log()  { echo "${CYN}[cs-all $(date +%H:%M:%S)]${RST} $*"; }
ok()   { echo "  ${GRN}OK${RST}  $*"; }
warn() { echo "  ${YLW}!!${RST}  $*"; }
err()  { echo "  ${RED}--${RST}  $*" >&2; }

# Build default package list
if [ -z "${PACKAGES:-}" ]; then
    if [ "$LIGHT" = "1" ]; then
        PACKAGES="1 3 4 8 14"   # everything but rightImg8bit_sequence
    else
        PACKAGES="1 3 4 8 14 15"
    fi
fi

log "================================================================"
log "E1 Cityscapes complete-download wrapper"
log "  CITYSCAPES_ROOT     = $CITYSCAPES_ROOT"
log "  EXISTING_CITYSCAPES = $EXISTING_CITYSCAPES"
log "  PACKAGES            = $PACKAGES"
log "  FRESH               = $FRESH"
log "  DELETE_ZIPS         = $DELETE_ZIPS"
log "  LIGHT               = $LIGHT"
log "================================================================"

# ---------------------------------------------------------------------------
# 1. Pre-flight: credentials + tools + disk space
# ---------------------------------------------------------------------------
log "Pre-flight checks..."

if [ -z "${CITYSCAPES_USER:-}" ] || [ -z "${CITYSCAPES_PASS:-}" ]; then
    err "CITYSCAPES_USER and CITYSCAPES_PASS must be exported before running."
    err "  export CITYSCAPES_USER='your_login'"
    err "  export CITYSCAPES_PASS='your_password'"
    err "  (register at https://www.cityscapes-dataset.com/register/)"
    exit 64
fi

for cmd in wget unzip find awk df readlink; do
    command -v "$cmd" >/dev/null 2>&1 || { err "$cmd not found in PATH"; exit 1; }
done

mkdir -p "$CITYSCAPES_ROOT"

# Disk-space sanity. /Data1 needs ~280 GB free for full set + zips, ~160 GB
# free if DELETE_ZIPS=1 (zips are removed as we go).
DEST_FREE_GB=$(df -BG "$CITYSCAPES_ROOT" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
log "Free disk in $CITYSCAPES_ROOT: ${DEST_FREE_GB:-unknown} GB"
if [ -n "${DEST_FREE_GB:-}" ]; then
    NEED=160
    [ "$DELETE_ZIPS" = "1" ] || NEED=280
    [ "$LIGHT" = "1" ] && NEED=$((NEED - 119))
    [ "$FRESH" = "1" ] || NEED=$((NEED - 120))   # symlinks save ~120 GB
    if (( DEST_FREE_GB < NEED )); then
        warn "free space (${DEST_FREE_GB} GB) is below the recommended ${NEED} GB."
        warn "set LIGHT=1 to skip rightImg8bit_sequence, or DELETE_ZIPS=1 (already on)."
    fi
fi

# ---------------------------------------------------------------------------
# 2. Symlink existing /home packages (unless FRESH=1)
# ---------------------------------------------------------------------------
if [ "$FRESH" = "1" ]; then
    log "FRESH=1 — skipping symlink stage; will re-download every package."
else
    log "Stage 1/3 — symlinking existing /home packages into $CITYSCAPES_ROOT ..."
    if [ ! -d "$EXISTING_CITYSCAPES" ]; then
        warn "EXISTING_CITYSCAPES=$EXISTING_CITYSCAPES not present — skipping symlink stage."
    else
        for sub in leftImg8bit leftImg8bit_sequence gtFine; do
            src="$EXISTING_CITYSCAPES/$sub"
            dst="$CITYSCAPES_ROOT/$sub"
            if [ ! -d "$src" ]; then
                warn "$src absent — will fall through to download."
                continue
            fi
            if [ -L "$dst" ]; then
                if [ "$(readlink -f "$dst")" = "$(readlink -f "$src")" ]; then
                    ok "$sub already symlinked"
                    continue
                fi
                warn "$dst points elsewhere — replacing"
                rm "$dst"
            elif [ -d "$dst" ]; then
                warn "$dst is a real directory — leaving it. Set FRESH=1 to override."
                continue
            fi
            ln -s "$src" "$dst"
            ok "$sub linked  ($dst -> $src)"
        done
    fi
fi
echo

# ---------------------------------------------------------------------------
# 3. leftImg8bit_sequence stub-detection (CUPS SMURF needs ≥2 frames per image)
# ---------------------------------------------------------------------------
SEQ_DIR="$CITYSCAPES_ROOT/leftImg8bit_sequence/train"
if [ -d "$SEQ_DIR" ]; then
    sample_city=$(find "$SEQ_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
    if [ -n "$sample_city" ]; then
        n_total=$(find "$sample_city" -maxdepth 1 -type f -name '*_leftImg8bit.png' 2>/dev/null | wc -l)
        n_unique=$(find "$sample_city" -maxdepth 1 -type f -name '*_leftImg8bit.png' 2>/dev/null \
                   | sed -E 's/_[0-9]{6}_leftImg8bit\.png$//' | sort -u | wc -l)
        if [ "$n_unique" -gt 0 ]; then
            ratio=$(awk -v t="$n_total" -v u="$n_unique" 'BEGIN { printf "%.1f", t/u }')
            log "leftImg8bit_sequence frame-ratio sanity (sample $(basename "$sample_city")): ${ratio} frames per labeled image"
            if awk -v r="$ratio" 'BEGIN { exit !(r > 1.5) }'; then
                ok "looks like the full sequence package is present (ratio $ratio > 1.5)"
            else
                warn "ratio $ratio looks like single-frame stubs — CUPS SMURF will fail."
                warn "delete $CITYSCAPES_ROOT/leftImg8bit_sequence (or its source on /home),"
                warn "then re-run with PACKAGES='14' to fetch the real package."
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 4. Delegate the actual download to scripts/e1_download_cityscapes.sh
#    (idempotent: skips packages whose subdir already exists, including via
#     our symlinks above).
# ---------------------------------------------------------------------------
log "Stage 2/3 — invoking scripts/e1_download_cityscapes.sh for packages: $PACKAGES"

DOWNLOADER="$REPO_ROOT/scripts/e1_download_cityscapes.sh"
if [ ! -x "$DOWNLOADER" ]; then
    err "missing or non-executable: $DOWNLOADER"
    err "did you 'git pull origin implement-dora-adapters' on this host?"
    exit 1
fi

# shellcheck disable=SC2086    # we want word-splitting on $PACKAGES
CITYSCAPES_ROOT="$CITYSCAPES_ROOT" \
DELETE_ZIPS="$DELETE_ZIPS" \
bash "$DOWNLOADER" $PACKAGES
DL_RC=$?

if [ $DL_RC -ne 0 ]; then
    err "downloader returned non-zero exit code $DL_RC — abort."
    exit $DL_RC
fi

# ---------------------------------------------------------------------------
# 5. Validate via scripts/e1_check_cityscapes.sh
# ---------------------------------------------------------------------------
log "Stage 3/3 — verifying layout via scripts/e1_check_cityscapes.sh ..."

CHECKER="$REPO_ROOT/scripts/e1_check_cityscapes.sh"
if [ -x "$CHECKER" ]; then
    bash "$CHECKER" "$CITYSCAPES_ROOT"
    CHECK_RC=$?
else
    warn "$CHECKER not found — skipping inventory check."
    CHECK_RC=0
fi

# ---------------------------------------------------------------------------
# 6. Cleanup hints + summary
# ---------------------------------------------------------------------------
echo
log "================================================================"
if [ $CHECK_RC -eq 0 ]; then
    log "${GRN}E1 Cityscapes layout COMPLETE.${RST}"
else
    log "${YLW}E1 Cityscapes layout INCOMPLETE — see warnings above.${RST}"
fi
log "  CITYSCAPES_ROOT for downstream scripts:  $CITYSCAPES_ROOT"
log "  Next: run scripts/e1_master_stage_assets.sh (depthg.ckpt + raft_smurf.pt)"
log "  Then: run scripts/e1_stage1_pseudolabel_gen_a6000.sh (full pseudo-label gen)"
log "================================================================"

exit $CHECK_RC
