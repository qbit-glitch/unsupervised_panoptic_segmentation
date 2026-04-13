#!/bin/bash
# Train all 5 contrastive embedding runs sequentially.
# Usage: nohup bash scripts/train_contrastive_all_runs.sh > logs/train_contrastive_all.log 2>&1 &

PYTHON=/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
CS_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes

echo "=== Contrastive Embedding Training: 5 Runs ==="
echo "Started: $(date)"
echo ""

for RUN_ID in 1 2 3 4 5; do
    echo "========================================"
    echo "RUN $RUN_ID — Started: $(date)"
    echo "========================================"

    $PYTHON -u mbps_pytorch/train_contrastive_embed.py \
        --cityscapes_root $CS_ROOT \
        --run_id $RUN_ID \
        --epochs 20 \
        --batch_size 4 \
        --num_workers 4 \
        --max_anchors 256 \
        --n_negatives 32

    echo ""
    echo "RUN $RUN_ID — Finished: $(date)"
    echo ""
done

echo "=== All 5 runs complete: $(date) ==="

# After training: evaluate all checkpoints
echo ""
echo "=== Starting Evaluation ==="
for RUN_ID in 1 2 3 4 5; do
    CKPT="checkpoints/contrastive_embed/run${RUN_ID}/best.pth"
    if [ -f "$CKPT" ]; then
        echo "Evaluating run $RUN_ID..."
        $PYTHON -u mbps_pytorch/eval_contrastive_learned.py \
            --cityscapes_root $CS_ROOT \
            --checkpoint "$CKPT" \
            --sweep
        echo ""
    else
        echo "SKIP: $CKPT not found"
    fi
done

echo "=== All evaluations complete: $(date) ==="
