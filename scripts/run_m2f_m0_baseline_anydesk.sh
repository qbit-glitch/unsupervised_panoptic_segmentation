#!/usr/bin/env bash
# Stage-2 M0 baseline: Mask2Former + ViT-Adapter on frozen DINOv3 ViT-B/16.
# Target: Anydesk (cvpr_ug_5@gpunode2), RTX A6000 48GB.
#
# Usage (on anydesk):
#   cd ~/umesh/unsupervised_panoptic_segmentation
#   bash scripts/run_m2f_m0_baseline_anydesk.sh
set -euo pipefail

PROJ_ROOT="$HOME/umesh/unsupervised_panoptic_segmentation"
CFG="configs/stage2_m2f/M0_baseline_dinov3_vitb_k80_anydesk.yaml"
LOG_DIR="$HOME/umesh/experiments/stage2_m2f/M0_baseline_anydesk"
LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
cd "$PROJ_ROOT"

# Required on anydesk for CUDA 12.1 toolchain
module load gcc-9.3.0 2>/dev/null || true
source ~/umesh/ups_env/bin/activate 2>/dev/null || true

export WANDB_MODE=disabled
export PYTHONPATH="$PROJ_ROOT/refs/cups:${PYTHONPATH:-}"

echo "=== Stage-2 M2F+ViTAdapter M0 baseline (anydesk A6000) ==="
echo "Config:   $CFG"
echo "Log:      $LOGFILE"
echo "Expected: 20k optimizer steps, bs=8 x accum=2 = eff 16, bf16-mixed (A6000 48GB)"
echo "OOM fallback: edit config to BATCH_SIZE=4 ACCUMULATE_GRAD_BATCHES=4 (or 2 and 8)"

nohup python -u refs/cups/train.py \
  --experiment_config_file "$CFG" \
  --disable_wandb \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "PID=$PID"
echo "LOG=$LOGFILE"
echo ""
echo "Monitor with: tail -f $LOGFILE"
echo "Kill with:    kill $PID"
