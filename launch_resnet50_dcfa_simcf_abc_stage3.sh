#!/bin/bash
# Stage-3 self-training: CUPS-original DINO ResNet-50 + Cascade Mask R-CNN
# seeded from Stage-2 best_pq_step=003125.ckpt on DCFA+DepthPro+SIMCF-ABC labels.
#
# Target: santosh@172.17.254.146, 2x GTX 1080 Ti (11 GB each)
# Effective batch: 2 GPUs x bs=1 x accum=8 = 16
# Schedule: 3 rounds x 4000 steps = 12000 optimizer steps
# Expected runtime: ~5-7 hours.
# Saves top 6 best_pq checkpoints (train_self.py patched: save_top_k=6).
#
# Usage (on the remote, after `cd /home/santosh/cups`):
#   bash launch_resnet50_dcfa_simcf_abc_stage3.sh
#   tail -f logs/resnet50_dcfa_simcf_abc_seed43_stage3.log

set -e

eval "$(/home/santosh/anaconda3/bin/conda shell.bash hook)"
conda activate cups

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"

cd /home/santosh/cups

mkdir -p logs

LOG_FILE="logs/resnet50_dcfa_simcf_abc_seed43_stage3.log"

nohup python -u train_self.py \
  --experiment_config_file configs/train_self_cityscapes_resnet50_dcfa_simcf_abc_santosh.yaml \
  --disable_wandb \
  SYSTEM.SEED 43 \
  SYSTEM.RUN_NAME "cups_resnet50_dcfa_simcf_abc_seed43_stage3" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Launched PID=${PID}"
echo "Log: ${LOG_FILE}"
echo
echo "Tail it with:"
echo "  ssh santosh@172.17.254.146 \"tail -f /home/santosh/cups/${LOG_FILE}\""
