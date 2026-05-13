#!/bin/bash
# Approach B v2 sweep on MPS (M4 Pro 48GB)
# Contrastive loss DISABLED (lambda_contrastive=0.0)
# Sweeps: lambda_depth x lr
#
# Usage:
#   bash scripts/run_approach_b_mps.sh          # full sweep (8 runs)
#   bash scripts/run_approach_b_mps.sh quick     # quick test (1 run, 3 epochs)

set -euo pipefail

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
DATA_DIR="/Users/qbit-glitch/Desktop/datasets"
SCRIPT="mbps_pytorch/train_cause_depth_finetune.py"
LOG_DIR="logs/approach_b_v2"
RESULT_DIR="results/approach_b_v2"

mkdir -p "$LOG_DIR"

if [ "${1:-}" = "quick" ]; then
    echo "=== Quick test: 1 run, 3 epochs ==="
    nohup $PYTHON -u $SCRIPT \
        --data_dir "$DATA_DIR" \
        --output_dir "$RESULT_DIR/lambda0.05_lr1e-5" \
        --lambda_depth 0.05 \
        --lambda_contrastive 0.0 \
        --lr 1e-5 \
        --epochs 3 \
        --batch_size 4 \
        --save_every 1 \
        > "$LOG_DIR/quick_test.log" 2>&1 &
    echo "PID: $!"
    echo "Log: tail -f $LOG_DIR/quick_test.log"
    exit 0
fi

echo "=== Approach B v2 sweep: 8 runs ==="
echo "  lambda_depth: 0.01 0.05 0.1 0.2"
echo "  lr: 1e-5 5e-6"
echo "  lambda_contrastive: 0.0 (disabled)"
echo ""

for LAMBDA in 0.01 0.05 0.1 0.2; do
    for LR in 1e-5 5e-6; do
        RUN_NAME="lambda${LAMBDA}_lr${LR}"
        OUT_DIR="$RESULT_DIR/$RUN_NAME"
        LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

        echo "--- Running: $RUN_NAME ---"
        $PYTHON -u $SCRIPT \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUT_DIR" \
            --lambda_depth "$LAMBDA" \
            --lambda_contrastive 0.0 \
            --lr "$LR" \
            --epochs 20 \
            --batch_size 4 \
            --save_every 5 \
            2>&1 | tee "$LOG_FILE"

        echo "--- Finished: $RUN_NAME ---"
        echo ""
    done
done

echo "=== All runs complete ==="
echo "Results in: $RESULT_DIR"
echo "Next: evaluate each with generate_depth_overclustered_semantics.py"
