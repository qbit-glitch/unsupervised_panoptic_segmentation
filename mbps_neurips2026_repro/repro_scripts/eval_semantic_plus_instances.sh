#!/bin/zsh
# Evaluate V3 adapter k=80 semantics + DepthPro depth-guided instance splitting → panoptic PQ
# Uses depth_guided_instances() from sweep_depthpro.py (NOT pre-computed sparse instances)
set -euo pipefail

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
CS_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
LOG="logs/v3_adapter_depthpro_panoptic_eval.log"
mkdir -p logs

echo "=== V3 Adapter + DepthPro Depth-Guided Panoptic Evaluation ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

# DepthPro winning params: tau=0.01, min_area=1000, sigma=0.0, dilation=3
$PYTHON -u mbps_pytorch/evaluate_panoptic_combined.py \
    --sem_dir "${CS_ROOT}/pseudo_semantic_adapter_V3_k80/val" \
    --cityscapes_root "${CS_ROOT}" \
    --depth_subdir "depth_depthpro" \
    --tau 0.01 --min_area 1000 --sigma 0.0 --dilation 3 \
    --output "results/depth_adapter/V3_k80_depthpro_panoptic_eval.json" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
