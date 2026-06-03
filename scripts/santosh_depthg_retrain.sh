#!/usr/bin/env bash
# Full DepthG retrain on santosh — CUPS-published defaults + DepthPro monocular depth.
# Single 1080 Ti, bs=16, res=224, 7000 steps. ~6 h wall on a stock 1080 Ti.
#
# Run:
#   ssh santosh@172.17.254.146
#   cd ~/mbps_panoptic_segmentation
#   nohup bash scripts/santosh_depthg_retrain.sh > /home/santosh/datasets/cityscapes/depthg_retrain/logs/retrain_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f /home/santosh/datasets/cityscapes/depthg_retrain/logs/retrain_*.log
#
# Output checkpoint:
#   $DEPTHG_OUTPUT_ROOT/checkpoints/cityscapes_depthpro_monocular_date_*/last.ckpt
#   ↑ drop this back into refs/cups/cups/pseudo_labels/gen_pseudo_labels.py --MODEL.CHECKPOINT
#     to regenerate the CUPS semantic pseudo-labels with the fully monocular DepthG.
#
# Acceptance: final test/cluster/mIoU >= 22.0 (CUPS published ckpt 22.3 +/- 0.5).

set -eo pipefail

# --- env ---------------------------------------------------------------------
set +u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cups
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# 2-GPU DDP: bs=16 per GPU = effective bs=32 (matches upstream CUPS bs=32).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# --- paths -------------------------------------------------------------------
export DEPTHG_CITYSCAPES_ROOT="${DEPTHG_CITYSCAPES_ROOT:-/home/santosh/datasets/cityscapes}"
export DEPTHG_DEPTHPRO_ROOT="${DEPTHG_DEPTHPRO_ROOT:-/home/santosh/datasets/cityscapes/depth_depthpro}"
export DEPTHG_OUTPUT_ROOT="${DEPTHG_OUTPUT_ROOT:-/home/santosh/datasets/cityscapes/depthg_retrain}"
mkdir -p "${DEPTHG_OUTPUT_ROOT}/logs" "${DEPTHG_OUTPUT_ROOT}/checkpoints"

# --- launch ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}/refs/cups"
export PYTHONPATH="${REPO_ROOT}/refs/cups/external/depthg:${PYTHONPATH:-}"

echo "[$(date +'%F %T')] starting full DepthG retrain on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  cityscapes  : ${DEPTHG_CITYSCAPES_ROOT}"
echo "  depthpro    : ${DEPTHG_DEPTHPRO_ROOT}"
echo "  output      : ${DEPTHG_OUTPUT_ROOT}"

python -u external/depthg/src/train_segmentation.py \
    experiment_name=depthpro_monocular_ddp \
    gpus=2 \
    batch_size=16 \
    max_steps=7000 \
    val_freq=500 \
    checkpoint_freq=500 \
    scalar_log_freq=10 \
    depth_feat_weight=0.036864 \
    depth_feat_shift=0.012288 \
    wandb_logging=false

echo "[$(date +'%F %T')] retrain complete"
