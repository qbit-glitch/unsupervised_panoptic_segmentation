#!/bin/bash
# Launch DCFA+DepthPro+SIMCF-ABC Stage 2 with effective batch 16 (2 GPUs)
# Uses config: train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml
# Seed: 44 (changed from original)

eval "$(/home/santosh/anaconda3/bin/conda shell.bash hook)"
conda activate cups

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"

cd /home/santosh/cups

python train.py \
  --experiment_config_file configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
  --disable_wandb \
  SYSTEM.SEED 44 \
  SYSTEM.RUN_NAME "dcfa_simcf_abc_seed44_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  > logs/dcfa_simcf_abc_seed44_stage2.log 2>&1
