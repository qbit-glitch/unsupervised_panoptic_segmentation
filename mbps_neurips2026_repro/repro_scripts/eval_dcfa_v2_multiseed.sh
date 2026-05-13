#!/bin/bash
# Multi-seed k-means evaluation for DCFA v2.
# Runs generate + evaluate with 5 different seeds per adapter.
# Feature extraction is the bottleneck (~2 min/run), so total ~2 min * 5 seeds * 6 methods = ~60 min.
# Compatible with macOS bash 3 (no associative arrays).
set -euo pipefail

CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PROJ_ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
BASE_OUT="$PROJ_ROOT/results/depth_adapter/DCFA_v2"

cd "$PROJ_ROOT"

SEEDS=(42 123 456 789 1024)

# Focus on top methods + baseline + worst to see the spread
EXPERIMENTS=(
    "V3_baseline:results/depth_adapter/V3_dd16_h384_l2/best.pt"
    "B2_deep:results/depth_adapter/DCFA_v2/B2_deep/best.pt"
    "A1_cross_attn:results/depth_adapter/DCFA_v2/A1_cross_attn/best.pt"
    "C2_cross_image:results/depth_adapter/DCFA_v2/C2_cross_image/best.pt"
    "X_combined:results/depth_adapter/DCFA_v2/X_combined/best.pt"
    "B1_film:results/depth_adapter/DCFA_v2/B1_film/best.pt"
)

echo "================================================================"
echo "  DCFA v2 Multi-Seed Evaluation (k=80, ${#SEEDS[@]} seeds)"
echo "  Started: $(date)"
echo "================================================================"

for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp ckpt <<< "$entry"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $exp"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    MIOUS=""
    for seed in "${SEEDS[@]}"; do
        out_sub="pseudo_semantic_dcfa_v2_${exp}_k80_s${seed}"
        eval_json="$BASE_OUT/${exp}/eval_k80_s${seed}.json"
        mkdir -p "$(dirname "$eval_json")"

        if [ -f "$eval_json" ]; then
            miou=$(python3 -c "import json; d=json.load(open('$eval_json')); print(d['semantic']['miou'])")
            echo "  seed=$seed: mIoU=${miou}% (cached)"
        else
            t0=$(date +%s)
            python3 -u mbps_pytorch/generate_depth_overclustered_semantics.py \
                --cityscapes_root "$CS_ROOT" \
                --k 80 --variant sinusoidal --alpha 0.1 --skip_crf \
                --adapter_checkpoint "$ckpt" \
                --output_subdir "$out_sub" \
                --kmeans_seed "$seed" 2>&1 | tail -3

            python3 -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
                --cityscapes_root "$CS_ROOT" \
                --split val \
                --semantic_subdir "$out_sub" \
                --num_clusters 80 \
                --skip_instance \
                --output "$eval_json" 2>&1 | grep -E "mIoU|Saved"

            t1=$(date +%s)
            miou=$(python3 -c "import json; d=json.load(open('$eval_json')); print(d['semantic']['miou'])")
            echo "  seed=$seed: mIoU=${miou}% ($((t1 - t0))s)"
        fi
        MIOUS="$MIOUS $miou"
    done

    # Compute mean and std
    python3 -c "
import numpy as np
vals = [float(x) for x in '$MIOUS'.split()]
print(f'  >> $exp: {np.mean(vals):.2f} +/- {np.std(vals):.2f}% (seeds: {vals})')
"
done

echo ""
echo "================================================================"
echo "  Summary Table"
echo "================================================================"
echo ""

for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp ckpt <<< "$entry"
    MIOUS=""
    for seed in "${SEEDS[@]}"; do
        eval_json="$BASE_OUT/${exp}/eval_k80_s${seed}.json"
        if [ -f "$eval_json" ]; then
            m=$(python3 -c "import json; d=json.load(open('$eval_json')); print(d['semantic']['miou'])")
            MIOUS="$MIOUS $m"
        fi
    done
    python3 -c "
import numpy as np
vals = [float(x) for x in '$MIOUS'.split()]
if vals:
    seeds_str = ', '.join(f'{v:.1f}' for v in vals)
    print(f'$exp'.ljust(20) + f'  {np.mean(vals):>6.2f} +/- {np.std(vals):.2f}%  seeds: [{seeds_str}]')
"
done

echo ""
echo "  Done: $(date)"
echo "================================================================"
