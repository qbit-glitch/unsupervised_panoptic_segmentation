#!/bin/bash
# Download V3 adapter pseudo-labels from HF Hub and launch CUPS Stage-2 training.
# Target: santosh@100.93.203.100, 2x GTX 1080 Ti, conda env: cups
#
# Run: bash scripts/run_v3_adapter_stage2_santosh.sh
set -euo pipefail

CS_ROOT="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes"
PROJ_ROOT="/media/santosh/Kuldeep/panoptic_segmentation/mbps_panoptic_segmentation"
OUT_DIR="$CS_ROOT/cups_pseudo_labels_adapter_V3_tau020"
URL="https://huggingface.co/datasets/qbit-glitch/cityscapes-cups-pseudo-labels/resolve/main/cups_pseudo_labels_adapter_V3_tau020.tar.gz"

echo "=== V3 Adapter + DepthPro tau=0.20 CUPS Stage-2 ==="

# ─── Step 1: Download pseudo-labels ───
if [ -d "$OUT_DIR" ] && [ "$(find "$OUT_DIR" -name "*_semantic.png" | wc -l)" -ge 2975 ]; then
    echo "Pseudo-labels already exist at $OUT_DIR ($(find "$OUT_DIR" -name "*_semantic.png" | wc -l) files). Skipping download."
else
    echo "--- Step 1: Download from HuggingFace Hub (64MB) ---"
    cd "$CS_ROOT"
    wget -q --show-progress "$URL" -O cups_pseudo_labels_adapter_V3_tau020.tar.gz
    echo "Extracting..."
    tar -xzf cups_pseudo_labels_adapter_V3_tau020.tar.gz
    rm cups_pseudo_labels_adapter_V3_tau020.tar.gz

    SEM=$(find "$OUT_DIR" -name "*_semantic.png" | wc -l)
    INST=$(find "$OUT_DIR" -name "*_instance.png" | wc -l)
    PT=$(find "$OUT_DIR" -name "*.pt" | wc -l)
    echo "  Extracted: $SEM semantic, $INST instance, $PT .pt files"

    if [ "$SEM" -lt 2975 ]; then
        echo "ERROR: Expected >= 2975 files. Download may be corrupted."
        exit 1
    fi
    echo "  PASS!"
fi

# ─── Step 2: Launch Stage-2 training ───
echo ""
echo "--- Step 2: Launch CUPS Stage-2 training ---"
echo "  Config: train_cityscapes_dinov3_vitb_adapter_V3_tau020.yaml"
echo "  Pseudo-labels: $OUT_DIR"
echo "  Effective batch: 2 GPUs x bs=1 x 8 accum = 16"
echo "  Steps: 8000 optimizer steps"

cd "$PROJ_ROOT/refs/cups"
export WANDB_MODE=disabled
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}

nohup python -u train.py \
    --experiment_config_file configs/train_cityscapes_dinov3_vitb_adapter_V3_tau020.yaml \
    > /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_adapter_V3_tau020_stage2.log 2>&1 &

echo "  PID: $!"
echo "  Log: /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_adapter_V3_tau020_stage2.log"
echo ""
echo "Monitor with: tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/cups_adapter_V3_tau020_stage2.log"
