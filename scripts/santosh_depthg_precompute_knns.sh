#!/usr/bin/env bash
# Pre-flight for the DepthG smoke run on santosh: build the DINO-CLS NN cache.
# Produces:
#   $DEPTHG_CITYSCAPES_ROOT/nns/nns_vit_base_cityscapes_train_None_224.npz
#
# ~15 min on a single 1080 Ti for 2,975 Cityscapes train images at res=224.

set -eo pipefail
set +u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cups
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEPTHG_CITYSCAPES_ROOT="${DEPTHG_CITYSCAPES_ROOT:-/home/santosh/datasets/cityscapes}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}/refs/cups"
export PYTHONPATH="${REPO_ROOT}/refs/cups/external/depthg:${PYTHONPATH:-}"

echo "[$(date +'%F %T')] precompute_knns on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -u external/depthg/src/precompute_knns.py \
    --cityscapes_root "${DEPTHG_CITYSCAPES_ROOT}" \
    --split train \
    --res 224 \
    --K 7
echo "[$(date +'%F %T')] precompute_knns done"
