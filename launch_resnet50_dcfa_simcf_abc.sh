#!/bin/bash
# Launch CUPS-original DINO ResNet-50 + Cascade Mask R-CNN
# on DCFA + DepthPro + SIMCF-ABC pseudo-labels (Stage-2).
#
# Target: santosh@172.17.254.146, 2x GTX 1080 Ti (11 GB each)
# Effective batch: 2 GPUs x bs=2 x accum=4 = 16
# Expected runtime: ~75-90 min for 8000 steps + 3 self-training rounds.
#
# Usage (on the remote, after `cd /home/santosh/cups`):
#   bash launch_resnet50_dcfa_simcf_abc.sh
#   tail -f logs/resnet50_dcfa_simcf_abc_seed43_stage2.log

set -e

eval "$(/home/santosh/anaconda3/bin/conda shell.bash hook)"
conda activate cups

# Fix library path (required for PIL/Pillow on this env)
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"

cd /home/santosh/cups

mkdir -p logs

LOG_FILE="logs/resnet50_dcfa_simcf_abc_seed43_stage2.log"

# Run unbuffered (-u) so `tail -f` shows live progress bars
nohup python -u train.py \
  --experiment_config_file configs/train_cityscapes_resnet50_dcfa_simcf_abc_santosh.yaml \
  --disable_wandb \
  SYSTEM.SEED 43 \
  SYSTEM.RUN_NAME "cups_resnet50_dcfa_simcf_abc_seed43_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Launched PID=${PID}"
echo "Log: ${LOG_FILE}"
echo
echo "Tail it with:"
echo "  ssh santosh@172.17.254.146 \"tail -f /home/santosh/cups/${LOG_FILE}\""
