#!/bin/bash
# Launch experiments on remote server with proper conda activation

eval "$(/home/santosh/anaconda3/bin/conda shell.bash hook)"
conda activate cups

# Fix library path (required for PIL/Pillow)
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"

cd /home/santosh/cups

# Kill any existing train.py processes
pkill -f "train.py" 2>/dev/null || true
sleep 2

# Seed 43 - GPU 0 (single GPU mode)
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml \
  --disable_wandb \
  SYSTEM.SEED 43 \
  SYSTEM.NUM_GPUS 1 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed43_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  > logs/seed43_stage2.log 2>&1 &
echo "Seed 43 launched on GPU 0, PID: $!"

# Seed 44 - GPU 1 (single GPU mode)
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
  --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml \
  --disable_wandb \
  SYSTEM.SEED 44 \
  SYSTEM.NUM_GPUS 1 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed44_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  > logs/seed44_stage2.log 2>&1 &
echo "Seed 44 launched on GPU 1, PID: $!"

sleep 5
echo "--- Running processes ---"
ps aux | grep -E 'train.py' | grep -v grep | awk '{print $2, $11, $14}'
echo "--- GPU status ---"
nvidia-smi
