#!/bin/bash
# scripts/e1/resume_cityscapes_on_data1.sh
#
# Recovery script: migrate any Cityscapes packages that ended up on /home
# (because CITYSCAPES_ROOT was previously set to a /home path) to
# /Data1/cityscapes, then resume the download on /Data1 with a fresh login
# (so a stale cookie cannot corrupt the next zip).
#
# Symptoms it fixes:
#   - "downloader returned non-zero exit code 9 — abort" caused by the
#     Cityscapes server returning a login-redirect HTML page instead of
#     the .zip (cookie expired during the previous long download).
#   - Data sitting on /home/cvpr_ug_5/umesh/datasets/cityscapes/ when the
#     intent was /Data1/cityscapes/.
#
# Required env vars:
#   CITYSCAPES_USER  — your cityscapes-dataset.com login
#   CITYSCAPES_PASS  — your cityscapes-dataset.com password
#
# Optional env vars:
#   CITYSCAPES_HOME      — source path with stray data (default:
#                          /home/cvpr_ug_5/umesh/datasets/cityscapes)
#   CITYSCAPES_ROOT_DST  — final destination (default: /Data1/cityscapes)
#   DRY_RUN              — 1 ⇒ print actions only, do not move/download
#   SKIP_MIGRATE         — 1 ⇒ skip the migration stage; just resume the
#                          download on /Data1
#
# Usage:
#   export CITYSCAPES_USER='your_login' CITYSCAPES_PASS='your_password'
#   nohup bash scripts/e1/resume_cityscapes_on_data1.sh \
#       > /Data1/e1_cityscapes_resume.log 2>&1 &
#   echo $! > /Data1/e1_cityscapes_resume.pid
#   tail -f /Data1/e1_cityscapes_resume.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CITYSCAPES_HOME="${CITYSCAPES_HOME:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"
CITYSCAPES_ROOT_DST="${CITYSCAPES_ROOT_DST:-/Data1/cityscapes}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MIGRATE="${SKIP_MIGRATE:-0}"

GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; CYN=$'\033[36m'; RST=$'\033[0m'
log()  { echo "${CYN}[resume $(date +%H:%M:%S)]${RST} $*"; }
ok()   { echo "  ${GRN}OK${RST}  $*"; }
warn() { echo "  ${YLW}!!${RST}  $*"; }
err()  { echo "  ${RED}--${RST}  $*" >&2; }

log "================================================================"
log "E1 Cityscapes recovery & resume on /Data1"
log "  CITYSCAPES_HOME      = $CITYSCAPES_HOME"
log "  CITYSCAPES_ROOT_DST  = $CITYSCAPES_ROOT_DST"
log "  DRY_RUN              = $DRY_RUN"
log "  SKIP_MIGRATE         = $SKIP_MIGRATE"
log "================================================================"

if [ -z "${CITYSCAPES_USER:-}" ] || [ -z "${CITYSCAPES_PASS:-}" ]; then
    err "CITYSCAPES_USER and CITYSCAPES_PASS must be exported."
    exit 64
fi

mkdir -p "$CITYSCAPES_ROOT_DST"

# ---------------------------------------------------------------------------
# 1. Inventory: what is on /home, what is already on /Data1?
# ---------------------------------------------------------------------------
log "Inventory before migration ..."
for sub in leftImg8bit leftImg8bit_sequence rightImg8bit rightImg8bit_sequence camera gtFine; do
    src="$CITYSCAPES_HOME/$sub"
    dst="$CITYSCAPES_ROOT_DST/$sub"

    src_state="absent"
    dst_state="absent"

    if [ -d "$src" ] && [ -n "$(find "$src" -maxdepth 3 -type f 2>/dev/null | head -1)" ]; then
        src_size=$(du -sh "$src" 2>/dev/null | cut -f1)
        src_state="present ($src_size)"
    fi
    if [ -d "$dst" ] && [ -n "$(find "$dst" -maxdepth 3 -type f 2>/dev/null | head -1)" ]; then
        dst_size=$(du -sh "$dst" 2>/dev/null | cut -f1)
        dst_state="present ($dst_size)"
    elif [ -L "$dst" ]; then
        dst_state="symlink -> $(readlink -f "$dst")"
    fi

    printf "  %-24s  /home: %-22s  /Data1: %s\n" "$sub" "$src_state" "$dst_state"
done
echo

# ---------------------------------------------------------------------------
# 2. Migrate from /home -> /Data1 (rsync with --remove-source-files).
#    Each package subdir is moved as a unit; if a partial copy exists on
#    /Data1 we merge into it and remove sources only after successful copy.
# ---------------------------------------------------------------------------
if [ "$SKIP_MIGRATE" = "1" ]; then
    log "SKIP_MIGRATE=1 — leaving /home untouched."
else
    log "Stage 1/3 — migrating packages /home -> /Data1 (rsync, no compression — same host)"
    if [ ! -d "$CITYSCAPES_HOME" ]; then
        warn "$CITYSCAPES_HOME does not exist — nothing to migrate."
    else
        for sub in leftImg8bit leftImg8bit_sequence rightImg8bit rightImg8bit_sequence camera gtFine; do
            src="$CITYSCAPES_HOME/$sub"
            dst="$CITYSCAPES_ROOT_DST/$sub"

            if [ ! -d "$src" ] || [ -z "$(find "$src" -maxdepth 3 -type f 2>/dev/null | head -1)" ]; then
                continue   # nothing to migrate for this package
            fi

            # If /Data1 destination is a stale symlink (e.g. from a prior
            # USE_HOME_SYMLINKS=1 run) pointing back into /home, drop it
            # before migrating real bytes.
            if [ -L "$dst" ]; then
                cur_target=$(readlink -f "$dst" 2>/dev/null || true)
                warn "$dst is a symlink -> $cur_target ; removing before migration"
                [ "$DRY_RUN" = "1" ] || rm "$dst"
            fi

            log "  -> migrating $sub  ($(du -sh "$src" | cut -f1))"
            if [ "$DRY_RUN" = "1" ]; then
                ok "(dry-run) would: rsync -aH --info=progress2 --remove-source-files \"$src/\" \"$dst/\""
                continue
            fi

            mkdir -p "$dst"
            # -a: archive (preserves perms/times)
            # -H: preserve hardlinks (Cityscapes gtFine has some)
            # --info=progress2: single rolling progress line
            # --remove-source-files: delete each source file once it's safely copied
            if rsync -aH --info=progress2 --remove-source-files "$src/" "$dst/"; then
                # Clean up the now-empty source directories.
                find "$src" -depth -type d -empty -delete 2>/dev/null || true
                if [ ! -d "$src" ] || [ -z "$(find "$src" -mindepth 1 2>/dev/null | head -1)" ]; then
                    rm -rf "$src" 2>/dev/null || true
                    ok "$sub migrated and source removed"
                else
                    warn "$sub: rsync ok but source dir not fully empty — inspect manually:"
                    warn "  ls -la $src"
                fi
            else
                err "$sub: rsync failed — leaving source intact, will need manual recovery."
                exit 2
            fi
        done

        # Also nuke the /home _downloads dir which contains the corrupted camera zip.
        if [ -d "$CITYSCAPES_HOME/_downloads" ]; then
            log "Removing $CITYSCAPES_HOME/_downloads (contains corrupted camera zip)"
            [ "$DRY_RUN" = "1" ] || rm -rf "$CITYSCAPES_HOME/_downloads"
        fi
    fi
fi
echo

# ---------------------------------------------------------------------------
# 3. Inventory after migration
# ---------------------------------------------------------------------------
log "Inventory after migration ..."
for sub in leftImg8bit leftImg8bit_sequence rightImg8bit rightImg8bit_sequence camera gtFine; do
    src="$CITYSCAPES_HOME/$sub"
    dst="$CITYSCAPES_ROOT_DST/$sub"
    src_state="absent"; dst_state="absent"
    [ -d "$src" ] && [ -n "$(find "$src" -maxdepth 3 -type f 2>/dev/null | head -1)" ] && src_state="STILL PRESENT"
    [ -d "$dst" ] && [ -n "$(find "$dst" -maxdepth 3 -type f 2>/dev/null | head -1)" ] && dst_state="present ($(du -sh "$dst" 2>/dev/null | cut -f1))"
    printf "  %-24s  /home: %-22s  /Data1: %s\n" "$sub" "$src_state" "$dst_state"
done
echo

# ---------------------------------------------------------------------------
# 4. Resume download on /Data1 with a fresh login
# ---------------------------------------------------------------------------
log "Stage 2/3 — resuming download on /Data1 (fresh login, all 6 pkgs idempotent)"

if [ "$DRY_RUN" = "1" ]; then
    ok "(dry-run) would now exec: CITYSCAPES_ROOT=$CITYSCAPES_ROOT_DST bash $SCRIPT_DIR/download_cityscapes_all.sh"
    exit 0
fi

# Force-override any pre-existing CITYSCAPES_ROOT in this process
export CITYSCAPES_ROOT="$CITYSCAPES_ROOT_DST"
unset CITYSCAPES_DOWNLOAD_DIR    # let the existing downloader rebuild this under /Data1

# The existing downloader skips packages whose subdir already exists on
# CITYSCAPES_ROOT, so right/left/seq/gtFine that we just migrated will be
# skipped automatically and only camera + missing pkgs will be fetched.
bash "$SCRIPT_DIR/download_cityscapes_all.sh"
DL_RC=$?

if [ $DL_RC -ne 0 ]; then
    err "download wrapper returned exit code $DL_RC."
    exit $DL_RC
fi

log "================================================================"
log "${GRN}E1 Cityscapes recovery + resume COMPLETE.${RST}"
log "  CITYSCAPES_ROOT for downstream: $CITYSCAPES_ROOT_DST"
log "================================================================"
