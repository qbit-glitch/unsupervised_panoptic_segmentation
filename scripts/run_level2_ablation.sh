#!/usr/bin/env bash
# Level-2 Dead-Class Recovery Ablation Runner for CUPS on AnyDesk
#
# Usage:
#   bash scripts/run_level2_ablation.sh baseline    # Baseline run
#   bash scripts/run_level2_ablation.sh all         # Run all ablations sequentially
#   bash scripts/run_level2_ablation.sh L2A         # Run specific ablation
#   bash scripts/run_level2_ablation.sh parallel    # Run 2 ablations in parallel (48GB VRAM)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs/cups_ablations"
LOG_DIR="$PROJECT_DIR/logs/level2_ablation"

mkdir -p "$LOG_DIR"

# Ablation mapping
declare -A ABLATIONS=(
    [baseline]="$PROJECT_DIR/configs/cups_cityscapes.yaml"
    [L2A]="$CONFIG_DIR/L2A_BACC_TS.yaml"
    [L2B]="$CONFIG_DIR/L2B_SSRC_TS.yaml"
    [L2C]="$CONFIG_DIR/L2C_EQLv2.yaml"
    [L2D]="$CONFIG_DIR/L2D_RFCL.yaml"
    [L2E]="$CONFIG_DIR/L2E_Combined.yaml"
    [L3A]="$CONFIG_DIR/L3A_AMR_ST.yaml"
)

run_ablation() {
    local name="$1"
    local config="${ABLATIONS[$name]}"
    local logfile="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "============================================================"
    echo "  Running Level-2 Ablation: $name"
    echo "  Config: $config"
    echo "  Log: $logfile"
    echo "============================================================"

    cd "$PROJECT_DIR"
    python refs/cups/train.py \
        --experiment_config_file "$config" \
        --disable_wandb \
        2>&1 | tee "$logfile"

    echo ""
    echo "Ablation $name complete. Log: $logfile"
    echo ""
}

run_parallel() {
    echo "Running L2A + L2B in parallel (GPU split)..."
    cd "$PROJECT_DIR"

    # Run L2A on first half of GPU memory
    CUDA_VISIBLE_DEVICES=0 python refs/cups/train.py \
        --experiment_config_file "${ABLATIONS[L2A]}" \
        --disable_wandb \
        2>&1 | tee "$LOG_DIR/L2A_parallel_$(date +%Y%m%d_%H%M%S).log" &
    PID1=$!

    # Run L2B on same GPU but with memory fraction limit
    CUDA_VISIBLE_DEVICES=0 python -c "
import torch; torch.cuda.set_per_process_memory_fraction(0.45, 0)
" && python refs/cups/train.py \
        --experiment_config_file "${ABLATIONS[L2B]}" \
        --disable_wandb \
        2>&1 | tee "$LOG_DIR/L2B_parallel_$(date +%Y%m%d_%H%M%S).log" &
    PID2=$!

    wait $PID1
    wait $PID2
    echo "Parallel run complete."
}

case "${1:-help}" in
    baseline)
        run_ablation baseline
        ;;
    L2A)
        run_ablation L2A
        ;;
    L2B)
        run_ablation L2B
        ;;
    L2C)
        run_ablation L2C
        ;;
    L2D)
        run_ablation L2D
        ;;
    L2E)
        run_ablation L2E
        ;;
    L3A)
        run_ablation L3A
        ;;
    all)
        for name in baseline L2A L2B L2C L2D L2E L3A; do
            run_ablation "$name"
        done
        ;;
    parallel)
        run_parallel
        ;;
    *)
        echo "Usage: $0 {baseline|L2A|L2B|L2C|L2D|L2E|all|parallel}"
        echo ""
        echo "  baseline  — Baseline CUPS run on AnyDesk (bs=4)"
        echo "  L2A       — BACC-TS (Boundary + Class-balanced + Temp scaling)"
        echo "  L2B       — SSRCTS (Rare-class ROI resampling)"
        echo "  L2C       — EQL v2 (Equalization Loss v2)"
        echo "  L2D       — RFCL (Rare-First Curriculum Learning)"
        echo "  L2E       — Combined stack (L2A + L2B + L2C + L2D)"
        echo "  L3A       — AMR-ST (Asymmetric Multi-Round Self-Training)"
        echo "  all       — Run all sequentially"
        echo "  parallel  — Run L2A + L2B in parallel (48GB VRAM)"
        ;;
esac
