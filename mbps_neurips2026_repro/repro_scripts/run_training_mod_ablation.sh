#!/usr/bin/env bash
# Run CUPS training modification ablation: TM-0 through TM-3.
# 100 steps each on CPU with DINOv3 ViT-B/16.
#
# Prerequisites:
#   - CUPS codebase at refs/cups/
#   - k=80 pseudo-labels at cups_pseudo_labels_k80/
#   - DepthPro depth at depth_depthpro/ (for TM-2, TM-3)
set -euo pipefail

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CUPS_DIR="${PROJECT_ROOT}/refs/cups"
YAML="${CUPS_DIR}/configs/train_cityscapes_dinov3_vitb_k80_tm_ablation.yaml"
RESULTS="${PROJECT_ROOT}/results/tm_ablation"
mkdir -p "$RESULTS"

echo "============================================"
echo "  Training Modification Ablation (TM-0..TM-3)"
echo "  Config: $YAML"
echo "  Results: $RESULTS"
echo "============================================"

# TM-0: Baseline CUPS Stage-2
echo ""
echo ">>> TM-0: Baseline CUPS Stage-2 (100 steps)"
cd "$CUPS_DIR"
$PYTHON -u train.py \
    --experiment_config_file "$YAML" \
    --disable_wandb \
    SYSTEM.RUN_NAME "tm0_baseline" \
    2>&1 | tee "${RESULTS}/tm0_baseline.log"

# TM-1: + Stuff preservation KD loss
echo ""
echo ">>> TM-1: + Stuff KD loss (weight=0.2, T=2.0)"
$PYTHON -u train.py \
    --experiment_config_file "$YAML" \
    --disable_wandb \
    SYSTEM.RUN_NAME "tm1_stuff_kd" \
    MODEL.SEM_SEG_HEAD.STUFF_KD_WEIGHT 0.2 \
    MODEL.SEM_SEG_HEAD.KD_TEMPERATURE 2.0 \
    2>&1 | tee "${RESULTS}/tm1_stuff_kd.log"

# TM-2: + Depth FiLM conditioning
echo ""
echo ">>> TM-2: + Depth FiLM conditioning"
$PYTHON -u train.py \
    --experiment_config_file "$YAML" \
    --disable_wandb \
    SYSTEM.RUN_NAME "tm2_depth_film" \
    DATA.DEPTH_SUBDIR "depth_depthpro" \
    MODEL.SEM_SEG_HEAD.USE_DEPTH_FILM True \
    MODEL.SEM_SEG_HEAD.DEPTH_CHANNELS 15 \
    2>&1 | tee "${RESULTS}/tm2_depth_film.log"

# TM-3: + All combined (KD + FiLM)
echo ""
echo ">>> TM-3: All combined (Stuff KD + Depth FiLM)"
$PYTHON -u train.py \
    --experiment_config_file "$YAML" \
    --disable_wandb \
    SYSTEM.RUN_NAME "tm3_all_combined" \
    DATA.DEPTH_SUBDIR "depth_depthpro" \
    MODEL.SEM_SEG_HEAD.STUFF_KD_WEIGHT 0.2 \
    MODEL.SEM_SEG_HEAD.KD_TEMPERATURE 2.0 \
    MODEL.SEM_SEG_HEAD.USE_DEPTH_FILM True \
    MODEL.SEM_SEG_HEAD.DEPTH_CHANNELS 15 \
    2>&1 | tee "${RESULTS}/tm3_all_combined.log"

echo ""
echo "============================================"
echo "  Results saved to: ${RESULTS}/"
echo "============================================"
echo ""
echo "Compare loss curves:"
echo "  grep 'loss_sem_seg' ${RESULTS}/tm*.log"
echo "  grep 'loss_stuff_kd' ${RESULTS}/tm*.log"
