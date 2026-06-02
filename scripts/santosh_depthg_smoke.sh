#!/usr/bin/env bash
# santosh DepthG smoke run — 200 steps, GPU 0, canonical depth_feat_(weight,shift).
# Purpose: confirm the data.py depthpro branch, the cityscapes_root env override, and the
# 1080 Ti memory budget all work end-to-end before committing to the 9-run HP sweep.
#
# Run on santosh, NOT locally:
#   ssh santosh@172.17.254.146
#   cd <repo>/refs/cups/external/depthg/src
#   nohup bash <repo>/scripts/santosh_depthg_smoke.sh > /home/santosh/logs/depthg_smoke.log 2>&1 &
#   tail -f /home/santosh/logs/depthg_smoke.log
#
# Expected:
#   - first 20 steps print loss values (correspondence_weight=1.0, depth_feat loss > 0).
#   - peak GPU memory ~9-10 GB on a 1080 Ti at bs=16.
#   - val_check at step 200 produces a test/cluster/mIoU number.
#   - smoke checkpoint saved under $DEPTHG_OUTPUT_ROOT/checkpoints/.

set -euo pipefail

# --- env ----------------------------------------------------------------------
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cups
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- paths (override via env if needed) --------------------------------------
export DEPTHG_CITYSCAPES_ROOT="${DEPTHG_CITYSCAPES_ROOT:-/home/santosh/datasets/cityscapes}"
export DEPTHG_DEPTHPRO_ROOT="${DEPTHG_DEPTHPRO_ROOT:-/home/santosh/datasets/cityscapes/depth_depthpro}"
export DEPTHG_OUTPUT_ROOT="${DEPTHG_OUTPUT_ROOT:-/home/santosh/datasets/cityscapes/depthg_retrain}"
mkdir -p "${DEPTHG_OUTPUT_ROOT}/logs" "${DEPTHG_OUTPUT_ROOT}/checkpoints"

# --- sanity (cheap, fail fast) ------------------------------------------------
python - <<'PY'
import os, numpy as np
from pathlib import Path
cs   = Path(os.environ["DEPTHG_CITYSCAPES_ROOT"])
dp   = Path(os.environ["DEPTHG_DEPTHPRO_ROOT"])
sample_img = next((cs / "leftImg8bit" / "train").rglob("*_leftImg8bit.png"), None)
assert sample_img is not None, f"no leftImg8bit pngs under {cs}/leftImg8bit/train"
city = sample_img.parent.name
stem = sample_img.stem.replace("_leftImg8bit", "")
dp_path = dp / "train" / city / f"{stem}.npy"
assert dp_path.exists(), f"missing DepthPro cache for {sample_img}: expected {dp_path}"
arr = np.load(dp_path)
assert arr.ndim == 2 and arr.dtype == np.float32, f"unexpected depthpro shape/dtype: {arr.shape}/{arr.dtype}"
print(f"[sanity ok] sample img={sample_img.name}  depth={dp_path.name}  shape={arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]")
PY

# --- launch -------------------------------------------------------------------
cd "$(dirname "$0")/../refs/cups/external/depthg/src"
echo "[$(date +'%F %T')] starting smoke (200 steps) on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -u train_segmentation.py \
    experiment_name=smoke_depthpro \
    max_steps=200 \
    val_freq=200 \
    checkpoint_freq=200 \
    wandb_logging=false
echo "[$(date +'%F %T')] smoke run complete"
