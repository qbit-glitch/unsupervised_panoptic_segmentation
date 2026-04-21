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
    [baseline]="$PROJECT_DIR/configs/cups_ablations/anydesk_baseline.yaml"
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
    local pidfile="$LOG_DIR/${name}.pid"

    if [ ! -f "$config" ]; then
        echo "ERROR: Config file not found: $config" >&2
        return 1
    fi

    # Clean up stale PID file if present
    if [ -f "$pidfile" ]; then
        local old_pid
        old_pid=$(cat "$pidfile" 2>/dev/null) || true
        if [ -n "$old_pid" ] && ! kill -0 "$old_pid" 2>/dev/null; then
            rm -f "$pidfile"
        fi
    fi

    echo "============================================================"
    echo "  Running Level-2 Ablation: $name"
    echo "  Config: $config"
    echo "  Log: $logfile"
    echo "  PID file: $pidfile"
    echo "============================================================"

    cd "$PROJECT_DIR"
    nohup python refs/cups/train.py \
        --experiment_config_file "$config" \
        --disable_wandb \
        > "$logfile" 2>&1 &

    local pid=$!
    echo "$pid" > "$pidfile"
    echo ""
    echo "Ablation $name started in background (PID: $pid)"
    echo "Monitor with: tail -f $logfile"
    echo "Kill with: kill \$(cat $pidfile)"
    echo ""
}

run_parallel() {
    echo "Running L2A + L2B in parallel (single GPU, serialized by CUDA)"
    run_ablation L2A
    run_ablation L2B
    echo ""
    echo "Parallel runs started. Both jobs are queued on the same GPU."
    echo "Monitor with: tail -f $LOG_DIR/L2A_*.log"
    echo "              tail -f $LOG_DIR/L2B_*.log"
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
            local pidfile="$LOG_DIR/${name}.pid"
            local pid
            pid=$(cat "$pidfile" 2>/dev/null) || true
            if [ -n "$pid" ]; then
                echo "Waiting for $name (PID: $pid) to complete before starting next..."
                wait "$pid" || true
            fi
            # Clean up PID file after completion
            rm -f "$pidfile"
        done
        ;;
    status)
        echo "=== Running ablations ==="
        local any_pid=false
        for pidfile in "$LOG_DIR"/*.pid; do
            [ -f "$pidfile" ] || continue
            any_pid=true
            local name
            local pid
            name=$(basename "$pidfile" .pid)
            pid=$(cat "$pidfile" 2>/dev/null) || true
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "  $name: RUNNING (PID: $pid)"
            else
                echo "  $name: FINISHED (PID: ${pid:-unknown})"
            fi
        done
        if [ "$any_pid" = false ]; then
            echo "  No ablation PID files found."
        fi
        ;;
    parallel)
        run_parallel
        ;;
    *)
        echo "Usage: $0 {baseline|L2A|L2B|L2C|L2D|L2E|L3A|all|parallel|status}"
        echo ""
        echo "  baseline  — Baseline CUPS run on AnyDesk (bs=8)"
        echo "  L2A       — BACC-TS (Boundary + Class-balanced + Temp scaling)"
        echo "  L2B       — SSRCTS (Rare-class ROI resampling)"
        echo "  L2C       — EQL v2 (Equalization Loss v2)"
        echo "  L2D       — RFCL (Rare-First Curriculum Learning)"
        echo "  L2E       — Combined stack (L2A + L2B + L2C + L2D)"
        echo "  L3A       — AMR-ST (Asymmetric Multi-Round Self-Training)"
        echo "  all       — Run all sequentially"
        echo "  parallel  — Run L2A + L2B in parallel (queued on same GPU)"
        echo "  status    — Show running ablations"
        ;;
esac
