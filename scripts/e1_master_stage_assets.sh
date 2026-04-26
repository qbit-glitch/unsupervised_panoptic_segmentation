#!/usr/bin/env bash
# E1 master-node stager — runs on the internet-connected, GPU-less login node.
#
# Air-gap workflow:
#   - master node     : has internet, NO GPU. Runs THIS script.
#   - GPU compute node: has GPU, NO internet. Runs e1_stage1_pseudolabel_gen_a6000.sh
#                       with SKIP_GIT=1 SKIP_DOWNLOAD=1 once this script finishes.
#
# Assumes /home is a shared filesystem (NFS) between master and GPU node — the
# standard HPC layout. If it isn't, see § 3 (rsync fallback) of
# scripts/E1_HF_UPLOAD_README.md.
#
# Idempotent: safe to re-run. Skips files that already exist with the right
# size; verifies SHA256 of any pre-existing or freshly downloaded file.
#
# Usage (paste on master):
#   REPO_ROOT=/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation \
#   bash scripts/e1_master_stage_assets.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/qbit-glitch/unsupervised_panoptic_segmentation.git}"
REPO_BRANCH="${REPO_BRANCH:-implement-dora-adapters}"
REPO_ROOT="${REPO_ROOT:-$HOME/unsupervised_panoptic_segmentation}"
HF_REPO_BASE="${HF_REPO_BASE:-https://huggingface.co/qbit-glitch/mbps-cups-e1/resolve/main}"

DEPTHG_SHA="09556e83b791a8177a369c218800d54bc8bfa77a61759840d9758f6211ecab39"
SMURF_SHA="692f1deece783ae69d058608b18ddeb97096c2505bd4529a1a08b8f5cab20d47"

log() { echo "[stage $(date +%H:%M:%S)] $*"; }

log "================================================================"
log "E1 master-node stager"
log "REPO_ROOT=$REPO_ROOT"
log "HF_REPO_BASE=$HF_REPO_BASE"
log "================================================================"

command -v wget    >/dev/null || { log "ERROR: wget not found";    exit 1; }
command -v git     >/dev/null || { log "ERROR: git not found";     exit 1; }
command -v sha256sum >/dev/null || command -v shasum >/dev/null || { log "ERROR: need sha256sum or shasum"; exit 1; }
SHACMD="sha256sum"; command -v "$SHACMD" >/dev/null || SHACMD="shasum -a 256"

# ---------------------------------------------------------------------------
# 1. Pull repo to the path the GPU node will read
# ---------------------------------------------------------------------------
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    log "Cloning $REPO_URL (branch=$REPO_BRANCH) into $REPO_ROOT ..."
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_ROOT"
else
    log "Repo at $REPO_ROOT — fetching $REPO_BRANCH"
    git -C "$REPO_ROOT" fetch origin "$REPO_BRANCH"
    git -C "$REPO_ROOT" checkout "$REPO_BRANCH"
    git -C "$REPO_ROOT" pull --ff-only origin "$REPO_BRANCH"
fi
COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
log "Repo at commit $COMMIT"

# ---------------------------------------------------------------------------
# 2. Stage HF checkpoints at the launcher's expected paths
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/weights" "$REPO_ROOT/refs/cups/cups/optical_flow/checkpoints"

DEPTH_DEST="$REPO_ROOT/weights/depthg.ckpt"
SMURF_DEST="$REPO_ROOT/refs/cups/cups/optical_flow/checkpoints/raft_smurf.pt"

stage_one() {
    local url="$1" dest="$2" expect_sha="$3" label="$4"
    if [[ -s "$dest" ]]; then
        log "[exists] $label  ($(du -h "$dest" | cut -f1))  — verifying checksum"
    else
        log "[fetch]  $label  <-  $url"
        wget -q --show-progress -O "$dest.partial" "$url"
        mv "$dest.partial" "$dest"
        log "[done]   $label  ($(du -h "$dest" | cut -f1))"
    fi
    actual=$($SHACMD "$dest" | awk '{print $1}')
    if [[ "$actual" != "$expect_sha" ]]; then
        log "ERROR: $label SHA256 mismatch."
        log "  expected: $expect_sha"
        log "  actual:   $actual"
        log "  -> file is corrupt; delete it and re-run."
        exit 2
    fi
    log "[ok]     $label SHA256 verified"
}

stage_one "$HF_REPO_BASE/depthg.ckpt"   "$DEPTH_DEST"  "$DEPTHG_SHA"  "depthg.ckpt"
stage_one "$HF_REPO_BASE/raft_smurf.pt" "$SMURF_DEST"  "$SMURF_SHA"   "raft_smurf.pt"

# ---------------------------------------------------------------------------
# 3. Summary + next-step on the GPU node
# ---------------------------------------------------------------------------
log "================================================================"
log "STAGER COMPLETE."
log "Repo:        $REPO_ROOT  (commit $COMMIT)"
log "depthg:      $DEPTH_DEST"
log "raft_smurf:  $SMURF_DEST"
log ""
log "Now SSH/Anydesk into the GPU node (offline) and paste:"
log ""
cat <<NEXT
  cd $REPO_ROOT
  module load gcc-9.3.0
  source ~/umesh/ups_env/bin/activate
  export PATH=\$HOME/cuda-12.1/bin:\$PATH
  export LD_LIBRARY_PATH=\$HOME/cuda-12.1/lib64:\$LD_LIBRARY_PATH
  export CUDA_HOME=\$HOME/cuda-12.1

  export REPO_ROOT="$REPO_ROOT"
  export CITYSCAPES_ROOT=/home/cvpr_ug_5/umesh/datasets/cityscapes
  export WORK_DIR=/home/cvpr_ug_5/umesh
  export OUTPUT_DIR=/home/cvpr_ug_5/umesh/cups_pseudo_labels_e1
  export GPU_ID=0
  export NUM_SUBSPLITS=2
  export NUM_WORKERS=6
  export SKIP_DEPS=1
  export SKIP_GIT=1
  export SKIP_DOWNLOAD=1
  export PYTHON_BIN="\$(which python)"

  nohup bash scripts/e1_stage1_pseudolabel_gen_a6000.sh \\
      > /home/cvpr_ug_5/umesh/e1_stage1.out 2>&1 &
  echo "PID=\$!"
NEXT
log "================================================================"
