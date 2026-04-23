#!/bin/bash
# launch_seed_experiments.sh
# Launch Stage 2 seed robustness experiments on remote server
# Run this from your local machine

set -e

REMOTE_HOST="santosh@172.17.254.146"
REMOTE_DIR="/home/santosh/cups"
CONDA_ENV="cups"
BASE_CONFIG="configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml"

echo "Launching Stage 2 experiments..."
echo "Seed 43"
ssh ${REMOTE_HOST} << EOF
cd ${REMOTE_DIR}
conda activate ${CONDA_ENV}
nohup python train.py \
  --experiment_config_file ${BASE_CONFIG} \
  SYSTEM.SEED 43 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed43_stage2" \
  SYSTEM.LOG_PATH "${REMOTE_DIR}/experiments" \
  > logs/seed43_stage2.log 2>&1 &
echo "Seed 43 launched, PID: \$!"
EOF

echo "Seed 44"
ssh ${REMOTE_HOST} << EOF
cd ${REMOTE_DIR}
conda activate ${CONDA_ENV}
nohup python train.py \
  --experiment_config_file ${BASE_CONFIG} \
  SYSTEM.SEED 44 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed44_stage2" \
  SYSTEM.LOG_PATH "${REMOTE_DIR}/experiments" \
  > logs/seed44_stage2.log 2>&1 &
echo "Seed 44 launched, PID: \$!"
EOF

echo "Both seed experiments launched. Monitor with:"
echo "  ssh ${REMOTE_HOST} 'tail -f ${REMOTE_DIR}/logs/seed43_stage2.log'"
echo "  ssh ${REMOTE_HOST} 'tail -f ${REMOTE_DIR}/logs/seed44_stage2.log'"
echo "  ssh ${REMOTE_HOST} 'nvidia-smi -l 5'"
