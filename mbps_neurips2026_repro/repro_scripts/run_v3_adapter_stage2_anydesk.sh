#!/bin/bash
# Download V3 adapter pseudo-labels from HF Hub and launch CUPS Stage-2 training.
# Target: Anydesk RTX A6000 48GB, conda env: ups
#
# Run: bash scripts/run_v3_adapter_stage2_anydesk.sh
set -euo pipefail

CS_ROOT="$HOME/umesh/datasets/cityscapes"
PROJ_ROOT="$HOME/umesh/unsupervised_panoptic_segmentation"
OUT_DIR="$CS_ROOT/cups_pseudo_labels_adapter_V3_tau020"
URL="https://huggingface.co/datasets/qbit-glitch/cityscapes-cups-pseudo-labels/resolve/main/cups_pseudo_labels_adapter_V3_tau020.tar.gz"

echo "=== V3 Adapter + DepthPro tau=0.20 CUPS Stage-2 (A6000) ==="

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
    # Clean macOS resource forks
    find "$OUT_DIR" -name '._*' -delete 2>/dev/null || true

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
echo "  Config: train_cityscapes_dinov3_vitb_adapter_V3_tau020_anydesk.yaml"
echo "  Pseudo-labels: $OUT_DIR"
echo "  Effective batch: 1 GPU x bs=1 x 16 accum = 16"
echo "  Steps: 8000 optimizer steps"

cd "$PROJ_ROOT/refs/cups"
export WANDB_MODE=disabled

nohup python -u train.py \
    --experiment_config_file configs/train_cityscapes_dinov3_vitb_adapter_V3_tau020_anydesk.yaml \
    --disable_wandb \
    > "$HOME/umesh/experiments/cups_adapter_V3_tau020_stage2.log" 2>&1 &

echo "  PID: $!"
echo "  Log: ~/umesh/experiments/cups_adapter_V3_tau020_stage2.log"
echo ""
echo "Monitor with: tail -f ~/umesh/experiments/cups_adapter_V3_tau020_stage2.log"
