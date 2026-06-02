#!/usr/bin/env bash
# santosh DepthG 2-knob HP sweep — depth_feat_weight × depth_feat_shift.
# Nine runs, 1500 steps each (~1 h on 1080 Ti at bs=16), round-robin GPU 0/1, max two concurrent.
# Acceptance criterion per run: final test/cluster/mIoU >= 22.0 (upstream depthg.ckpt = 22.3).
#
# Run on santosh AFTER smoke has succeeded:
#   ssh santosh@172.17.254.146
#   nohup bash <repo>/scripts/santosh_depthg_sweep.sh > /home/santosh/datasets/cityscapes/depthg_retrain/sweep_master.log 2>&1 &
#   tail -f /home/santosh/datasets/cityscapes/depthg_retrain/sweep_master.log
#
# Why this grid? DepthPro returns normalized [0,1] (we normalize per-image in data.py), same range as ZoeDepth,
# so the upstream-canonical depth_feat_weight=0.036864 / depth_feat_shift=0.012288 should be close but is not
# guaranteed optimal. 3x3 around the defaults at half/x1/x2 covers the plausible range without overspending compute.

set -eo pipefail

# --- env ---------------------------------------------------------------------
# Relax -u around conda activate (santosh's activate hooks reference unbound ADDR2LINE).
set +u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cups
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --- paths -------------------------------------------------------------------
export DEPTHG_CITYSCAPES_ROOT="${DEPTHG_CITYSCAPES_ROOT:-/home/santosh/datasets/cityscapes}"
export DEPTHG_DEPTHPRO_ROOT="${DEPTHG_DEPTHPRO_ROOT:-/home/santosh/datasets/cityscapes/depth_depthpro}"
export DEPTHG_OUTPUT_ROOT="${DEPTHG_OUTPUT_ROOT:-/home/santosh/datasets/cityscapes/depthg_retrain}"
SWEEP_LOG_DIR="${DEPTHG_OUTPUT_ROOT}/logs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_LOG_DIR}"

# --- sweep grid --------------------------------------------------------------
# upstream defaults: w=0.036864, s=0.012288
WEIGHTS=(0.018432 0.036864 0.073728)   # 0.5x  1x  2x
SHIFTS=(0.006144 0.012288 0.024576)    # 0.5x  1x  2x

# --- run scheduler -----------------------------------------------------------
# Up to 2 concurrent processes (1 per GPU). Block when 2 are running.
cd "$(dirname "$0")/../refs/cups/external/depthg/src"
PIDS=()
run_idx=0
for w in "${WEIGHTS[@]}"; do
  for s in "${SHIFTS[@]}"; do
    run_idx=$((run_idx + 1))
    gpu=$(( (run_idx - 1) % 2 ))
    tag="w${w}_s${s}"
    log="${SWEEP_LOG_DIR}/run${run_idx}_${tag}_gpu${gpu}.log"
    echo "[$(date +'%F %T')] launching run ${run_idx}/9: w=${w} s=${s} on GPU ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      nohup python -u train_segmentation.py \
        experiment_name="sweep_${tag}" \
        max_steps=1500 \
        val_freq=300 \
        checkpoint_freq=500 \
        depth_feat_weight="${w}" \
        depth_feat_shift="${s}" \
        wandb_logging=false \
        > "${log}" 2>&1 &
    PIDS+=($!)

    # block until at most 1 in-flight (so the next iter can pick up the freed GPU)
    if [[ ${#PIDS[@]} -ge 2 ]]; then
      wait "${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
    fi
  done
done
# drain remaining
for pid in "${PIDS[@]}"; do wait "$pid"; done

echo "[$(date +'%F %T')] sweep complete. Pick best by tailing test/cluster/mIoU in each run log:"
echo "  for f in ${SWEEP_LOG_DIR}/run*.log; do echo \"\$f\"; grep -E 'test/cluster/mIoU' \"\$f\" | tail -3; done"
