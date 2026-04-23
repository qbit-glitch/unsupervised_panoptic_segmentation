#!/bin/bash
# Launch DCFA+SIMCF-ABC Stage 2 with effective batch 16 (2 GPUs)
# Uses existing config: train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml

eval "$(/home/santosh/anaconda3/bin/conda shell.bash hook)"
conda activate cups

# Fix library path
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"

cd /home/santosh/cups

# Seed 43 - DCFA+SIMCF-ABC Stage 2
python train.py \
  --experiment_config_file configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
  --disable_wandb \
  SYSTEM.SEED 43 \
  SYSTEM.RUN_NAME "dcfa_simcf_abc_seed43_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  > logs/dcfa_simcf_abc_seed43_stage2.log 2>&1
