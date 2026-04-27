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
# By DEFAULT this script downloads ALL six packages fresh into
# $CITYSCAPES_ROOT — no symlinks, no skipped packages. Rationale: /Data1
# is presumed to be a fast local volume, while /home on this HPC node
# is typically NFS-mounted; symlinking the 119 GB leftImg8bit_sequence
# from /home would force every CUPS pseudo-label read to hit NFS and
# slow the multi-hour Stage-1 run substantially. Local /Data1 reads
# avoid that bottleneck. With 1.8 TB free there is no reason to skip
# anything.
#
# Set USE_HOME_SYMLINKS=1 to opt back into the symlink optimisation if
# you specifically want to save the ~120 GB of left/seq/gtFine
# bandwidth and you know /Data1 and /home share the same physical
# volume (or NFS performance is acceptable for your run).
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
#   USE_HOME_SYMLINKS     — 1 ⇒ symlink left/leftImg8bit_sequence/gtFine
#                           from $EXISTING_CITYSCAPES instead of
#                           downloading them. Default: 0 (full fresh
#                           download on /Data1).
#   EXISTING_CITYSCAPES   — source for /home symlinks (only used if
#                           USE_HOME_SYMLINKS=1). Default:
#                           /home/cvpr_ug_5/umesh/datasets/cityscapes
#   DELETE_ZIPS           — 1 ⇒ delete each .zip after successful
#                           extract. Default: 1.
#   PACKAGES              — explicit space-separated pkg-ID list
#                           (overrides the default "1 3 4 8 14 15").
#
# ============================================================================
# USAGE
# ============================================================================
#
# Standard run on the A6000 — full fresh download, all 6 packages on /Data1:
#
#   export CITYSCAPES_USER='your_login'
#   export CITYSCAPES_PASS='your_password'
#   nohup bash scripts/e1/download_cityscapes_all.sh \
#       > /Data1/e1_cityscapes_download.log 2>&1 &
#   echo $! > /Data1/e1_cityscapes_download.pid
#   tail -f /Data1/e1_cityscapes_download.log
#
# Save bandwidth by symlinking left/leftImg8bit_sequence/gtFine from /home:
#
#   USE_HOME_SYMLINKS=1 bash scripts/e1/download_cityscapes_all.sh
#
set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve paths & defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/Data1/cityscapes}"
EXISTING_CITYSCAPES="${EXISTING_CITYSCAPES:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"
USE_HOME_SYMLINKS="${USE_HOME_SYMLINKS:-0}"
DELETE_ZIPS="${DELETE_ZIPS:-1}"

GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; CYN=$'\033[36m'; RST=$'\033[0m'
log()  { echo "${CYN}[cs-all $(date +%H:%M:%S)]${RST} $*"; }
ok()   { echo "  ${GRN}OK${RST}  $*"; }
warn() { echo "  ${YLW}!!${RST}  $*"; }
err()  { echo "  ${RED}--${RST}  $*" >&2; }

# Always download all six packages by default (full Cityscapes Stereo+Video
# set required by CUPS Stage-1).
PACKAGES="${PACKAGES:-1 3 4 8 14 15}"

log "================================================================"
log "E1 Cityscapes complete-download wrapper"
log "  CITYSCAPES_ROOT     = $CITYSCAPES_ROOT"
log "  PACKAGES            = $PACKAGES"
log "  USE_HOME_SYMLINKS   = $USE_HOME_SYMLINKS"
[ "$USE_HOME_SYMLINKS" = "1" ] && log "  EXISTING_CITYSCAPES = $EXISTING_CITYSCAPES"
log "  DELETE_ZIPS         = $DELETE_ZIPS"
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

# Disk-space sanity. Full unpacked set is ~260 GB. With DELETE_ZIPS=1 the
# peak headroom we need on top of the unpacked tree is the size of the
# largest single zip (~119 GB for leftImg8bit_sequence). Recommend ≥350 GB
# free as a comfortable working margin.
DEST_FREE_GB=$(df -BG "$CITYSCAPES_ROOT" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
log "Free disk in $CITYSCAPES_ROOT: ${DEST_FREE_GB:-unknown} GB"
if [ -n "${DEST_FREE_GB:-}" ]; then
    NEED=350
    [ "$DELETE_ZIPS" = "1" ] || NEED=520
    [ "$USE_HOME_SYMLINKS" = "1" ] && NEED=$((NEED - 130))   # symlinks save ~130 GB
    if (( DEST_FREE_GB < NEED )); then
        warn "free space (${DEST_FREE_GB} GB) is below the recommended ${NEED} GB."
        warn "set USE_HOME_SYMLINKS=1 to reuse left/seq/gtFine from /home,"
        warn "or set PACKAGES='1 3 4 8 14' to skip rightImg8bit_sequence (~119 GB)."
    fi
fi

# ---------------------------------------------------------------------------
# 2. Optional: symlink existing /home packages (off by default)
# ---------------------------------------------------------------------------
if [ "$USE_HOME_SYMLINKS" = "1" ]; then
    log "Stage 1/3 — USE_HOME_SYMLINKS=1, symlinking existing /home packages ..."
    if [ ! -d "$EXISTING_CITYSCAPES" ]; then
        warn "EXISTING_CITYSCAPES=$EXISTING_CITYSCAPES not present — falling through to full download."
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
                warn "$dst is a real directory — leaving it. Remove first to re-link."
                continue
            fi
            ln -s "$src" "$dst"
            ok "$sub linked  ($dst -> $src)"
        done
    fi
else
    log "Stage 1/3 — full fresh download (USE_HOME_SYMLINKS=0). All 6 packages will land on $CITYSCAPES_ROOT."
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
