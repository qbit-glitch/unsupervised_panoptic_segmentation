#!/bin/zsh
# Approach C architecture sweep — 5 variants, all with lp=20.0
set -euo pipefail

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
CS_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
LP=20.0

# V1: sinusoidal 16D, hidden=128, 2 layers
# V2: sinusoidal 16D, hidden=256, 2 layers
# V3: sinusoidal 16D, hidden=384, 2 layers
# V4: sinusoidal 16D, hidden=256, 3 layers
# V5: raw 1D, hidden=256, 2 layers

NAMES=(V1 V2 V3 V4 V5)
DDS=(16 16 16 16 1)
HDS=(128 256 384 256 256)
NLS=(2 2 2 3 2)

for i in 1 2 3 4 5; do
    V=${NAMES[$i]}
    DD=${DDS[$i]}
    HD=${HDS[$i]}
    NL=${NLS[$i]}
    OUT="results/depth_adapter/${V}_dd${DD}_h${HD}_l${NL}"
    PRED_DIR="${CS_ROOT}/pseudo_semantic_adapter_${V}_k300"

    echo ""
    echo "========================================"
    echo "  ${V}: depth_dim=${DD} hidden=${HD} layers=${NL}"
    echo "========================================"

    # Train
    $PYTHON -u mbps_pytorch/train_depth_adapter.py \
        --cityscapes_root "$CS_ROOT" \
        --output_dir "$OUT" \
        --epochs 20 --batch_size 8 --lr 1e-3 \
        --lambda_preserve "$LP" \
        --depth_dim "$DD" --hidden_dim "$HD" --num_layers "$NL" \
        2>&1 | grep -E "lambda|DepthAdapter|Zero-init|Epoch.*/19|Training complete|best"

    # Generate pseudo-labels
    $PYTHON -u mbps_pytorch/generate_depth_overclustered_semantics.py \
        --cityscapes_root "$CS_ROOT" \
        --split val \
        --adapter_checkpoint "${OUT}/best.pt" \
        --variant sinusoidal --alpha 0.1 --k 300 \
        --output_subdir "pseudo_semantic_adapter_${V}_k300" \
        --skip_crf \
        2>&1 | grep -E "Loaded|clusters$|Done"

    # Evaluate
    echo "--- ${V} Results ---"
    $PYTHON -u mbps_pytorch/evaluate_semantic_pseudolabels.py \
        --pred_dir "${PRED_DIR}/val" \
        --gt_dir "${CS_ROOT}/gtFine/val" \
        2>&1 | grep -E "mIoU|ALL.*PQ|STUFF|THINGS|motorcycle|train "

    echo ""
done

echo "========================================"
echo "  SWEEP COMPLETE"
echo "========================================"
