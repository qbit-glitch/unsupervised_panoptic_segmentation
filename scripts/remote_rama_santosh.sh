#!/usr/bin/env bash
# Generate RAMA MultiCut coarse masks on the Santosh remote machine.
#
# RAMA MultiCut does not run on Apple MPS (no GPU bindings on macOS) — this
# script is the canonical entry point for shipping the job to Santosh, where
# CUDA is available.
#
# Usage:
#   bash scripts/remote_rama_santosh.sh <split> [extra args]
#
# Examples:
#   bash scripts/remote_rama_santosh.sh train
#   bash scripts/remote_rama_santosh.sh val --max_images 100
#
# Notes:
#   * The local SGM adapter (mbps_pytorch/train_sgm_adapter.py) uses depth-aware
#     SLIC superpixels and DOES NOT require RAMA. Only call this script when a
#     RAMA-based coarse-mask alternative is wanted as an additional supervision
#     source.
#   * Outputs are mirrored back to the user's Cityscapes root by rsync.
#
# Remote host:
#   user@host    : santosh@172.17.254.146
#   conda env    : ups
#   storage      : Kuldeep hard drive  (default: /media/santosh/Kuldeep)
#                  Override with REMOTE_KULDEEP_ROOT=/your/path
#   reference    : test-instance-labels/Superpixels/scripts/run_rama_official.py

set -euo pipefail

REMOTE_USER="santosh"
REMOTE_HOST="172.17.254.146"
# Kuldeep hard drive on the Santosh box (capital K — udisks2 mount path).
# RAMA outputs and caches live here so the home partition stays small.
# Override at call time if Kuldeep moves: REMOTE_KULDEEP_ROOT=/other/path bash ...
REMOTE_KULDEEP_ROOT="${REMOTE_KULDEEP_ROOT:-/media/santosh/Kuldeep}"
REMOTE_DIR="${REMOTE_KULDEEP_ROOT}/mbps_panoptic_segmentation"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SPLIT="${1:-train}"
shift || true

if [[ "$SPLIT" != "train" && "$SPLIT" != "val" ]]; then
  echo "split must be train or val (got: $SPLIT)" >&2
  exit 2
fi

REMOTE_OUT="${REMOTE_DIR}/test-instance-labels/Superpixels/runs/rama_${SPLIT}"

echo "[remote-rama] kuldeep root  : ${REMOTE_KULDEEP_ROOT}"
echo "[remote-rama] remote dir    : ${REMOTE_DIR}"
echo "[remote-rama] remote output : ${REMOTE_OUT}"

echo "[remote-rama] ensuring ${REMOTE_DIR} exists on Kuldeep drive..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}/test-instance-labels/Superpixels/scripts' '${REMOTE_OUT}'"

echo "[remote-rama] syncing scripts to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
rsync -avz --exclude '__pycache__/' --exclude '*.pyc' \
  "${LOCAL_DIR}/test-instance-labels/Superpixels/scripts/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/test-instance-labels/Superpixels/scripts/"

echo "[remote-rama] launching RAMA on ${SPLIT} split..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "bash -lc '
  source ~/.bashrc
  conda activate ups
  cd ${REMOTE_DIR}
  python test-instance-labels/Superpixels/scripts/run_rama_official.py \
    --split ${SPLIT} \
    --output_dir ${REMOTE_OUT} \
    $@
'"

echo "[remote-rama] pulling outputs back to ${LOCAL_DIR}..."
mkdir -p "${LOCAL_DIR}/test-instance-labels/Superpixels/runs/rama_${SPLIT}"
rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUT}/" \
  "${LOCAL_DIR}/test-instance-labels/Superpixels/runs/rama_${SPLIT}/"

echo "[remote-rama] done. local outputs: ${LOCAL_DIR}/test-instance-labels/Superpixels/runs/rama_${SPLIT}"
