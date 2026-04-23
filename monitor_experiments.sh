#!/bin/bash
# monitor_experiments.sh
# Quick monitoring of running experiments

REMOTE_HOST="santosh@172.17.254.146"

echo "=== GPU Status ==="
ssh ${REMOTE_HOST} "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv"

echo ""
echo "=== Running Python Processes ==="
ssh ${REMOTE_HOST} "ps aux | grep 'train.py\|train_self.py' | grep -v grep | awk '{print \$2, \$11, \$12, \$13}'"

echo ""
echo "=== Recent Experiment Outputs ==="
ssh ${REMOTE_HOST} "ls -lt /home/santosh/cups/experiments/ | head -10"

echo ""
echo "=== Log Tails ==="
for seed in 43 44; do
    echo "--- Seed ${seed} Stage 2 ---"
    ssh ${REMOTE_HOST} "tail -n 5 /home/santosh/cups/logs/seed${seed}_stage2.log 2>/dev/null || echo 'Log not found (experiment may not be running yet)'"
    echo ""
done
