#!/bin/bash
# Evaluate all DCFA v2 ablation checkpoints at k=80 mIoU.
# Two steps per experiment:
#   1. generate_depth_overclustered_semantics.py — fit k-means + generate pseudo-labels
#   2. evaluate_cascade_pseudolabels.py — compute mIoU against GT
set -euo pipefail

CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PROJ_ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
BASE_OUT="$PROJ_ROOT/results/depth_adapter/DCFA_v2"

cd "$PROJ_ROOT"

EXPERIMENTS=(
    "A1_cross_attn"
    "A2_normals"
    "A3_gradients"
    "B1_film"
    "B2_deep"
    "B3_window_attn"
    "C1_contrastive"
    "C2_cross_image"
    "X_combined"
)

echo "================================================================"
echo "  DCFA v2 Evaluation Sweep (k=80 mIoU)"
echo "  Started: $(date)"
echo "================================================================"

for exp in "${EXPERIMENTS[@]}"; do
    ckpt="$BASE_OUT/${exp}/best.pt"
    output_subdir="pseudo_semantic_dcfa_v2_${exp}_k80"
    eval_json="$BASE_OUT/${exp}/eval_k80.json"

    if [ ! -f "$ckpt" ]; then
        echo "  SKIP $exp — no checkpoint found"
        continue
    fi

    # Skip if already evaluated
    if [ -f "$eval_json" ]; then
        echo "  SKIP $exp — already evaluated ($eval_json)"
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$exp] Step 1: Generate pseudo-labels (k=80)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    t0=$(date +%s)
    python3 -u mbps_pytorch/generate_depth_overclustered_semantics.py \
        --cityscapes_root "$CS_ROOT" \
        --k 80 --variant sinusoidal --alpha 0.1 --skip_crf \
        --adapter_checkpoint "$ckpt" \
        --output_subdir "$output_subdir"
    t1=$(date +%s)
    echo "  [$exp] Pseudo-labels done in $((t1 - t0))s"

    echo ""
    echo "  [$exp] Step 2: Evaluate mIoU"
    python3 -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
        --cityscapes_root "$CS_ROOT" \
        --split val \
        --semantic_subdir "$output_subdir" \
        --num_clusters 80 \
        --skip_instance \
        --output "$eval_json"
    t2=$(date +%s)
    echo "  [$exp] Eval done in $((t2 - t1))s (total: $((t2 - t0))s)"
done

echo ""
echo "================================================================"
echo "  Results Summary"
echo "================================================================"
echo ""
printf "%-20s  %8s  %8s\n" "Experiment" "mIoU" "val_loss"
printf "%-20s  %8s  %8s\n" "--------------------" "--------" "--------"

for exp in "${EXPERIMENTS[@]}"; do
    eval_json="$BASE_OUT/${exp}/eval_k80.json"
    if [ -f "$eval_json" ]; then
        miou=$(python3 -c "
import json
d = json.load(open('$eval_json'))
sem = d.get('semantic', {})
print(f\"{sem.get('miou', 0):.2f}\")
")
    else
        miou="N/A"
    fi
    val_loss=$(python3 -c "
import torch
c = torch.load('$BASE_OUT/${exp}/best.pt', map_location='cpu', weights_only=True)
if isinstance(c, dict) and 'val_loss' in c:
    print(f'{c[\"val_loss\"]:.4f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
    printf "%-20s  %8s  %8s\n" "$exp" "$miou" "$val_loss"
done

echo ""
echo "  Baseline V3: mIoU=55.29%"
echo ""
echo "================================================================"
echo "  Evaluation complete: $(date)"
echo "================================================================"
