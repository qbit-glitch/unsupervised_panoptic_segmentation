#!/usr/bin/env bash
# E2 DepthPro Conv-DoRA — single GPU training on santosh
# Usage:
#   bash scripts/run_e2_depthpro_dora.sh verify   # pre-flight checks
#   bash scripts/run_e2_depthpro_dora.sh stage2    # launch Stage-2 training
#   bash scripts/run_e2_depthpro_dora.sh stage3    # launch Stage-3 self-training
#   bash scripts/run_e2_depthpro_dora.sh status    # check training progress

set -euo pipefail

CONFIG="configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_1gpu.yaml"
CUPS_DIR="/home/santosh/cups"
LOG_DIR="/home/santosh/cups/logs"
PSEUDO_DIR="/home/santosh/datasets/cityscapes/cups_pseudo_labels_depthpro"
WEIGHTS_DIR="/home/santosh/cups/weights"
GPU_ID=0

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/santosh/anaconda3/envs/cups/bin/python"

mkdir -p "$LOG_DIR"

verify() {
    echo "=== E2 DepthPro Conv-DoRA Pre-flight ==="
    echo ""

    # 1. Check pseudo-labels
    n_sem=$(ls "$PSEUDO_DIR"/*_semantic.png 2>/dev/null | wc -l)
    n_inst=$(ls "$PSEUDO_DIR"/*_instance.png 2>/dev/null | wc -l)
    echo "[1/5] Pseudo-labels: $n_sem semantic, $n_inst instance"
    if [ "$n_sem" -eq 2975 ] && [ "$n_inst" -eq 2975 ]; then
        echo "  OK: 5950 files present"
    else
        echo "  FAIL: Expected 2975 semantic + 2975 instance"
        exit 1
    fi

    # 2. Check non-empty instance ratio
    n_nonempty=0
    total=0
    for f in $(ls "$PSEUDO_DIR"/*_instance.png | shuf -n 100); do
        total=$((total + 1))
        max_val=$(python3 -c "from PIL import Image; import numpy as np; print(np.array(Image.open('$f')).max())")
        if [ "$max_val" -gt 0 ]; then
            n_nonempty=$((n_nonempty + 1))
        fi
    done
    pct=$((n_nonempty * 100 / total))
    echo "[2/5] Non-empty instances: $n_nonempty/$total ($pct%)"
    if [ "$pct" -gt 50 ]; then
        echo "  OK"
    else
        echo "  WARN: Low non-empty ratio"
    fi

    # 3. Check DINOv3 weights
    if [ -f "$WEIGHTS_DIR/dinov3_vitb16_official.pth" ]; then
        echo "[3/5] DINOv3 weights: FOUND (dinov3_vitb16_official.pth)"
    else
        echo "[3/5] DINOv3 weights: MISSING at $WEIGHTS_DIR/dinov3_vitb16_official.pth"
        ls "$WEIGHTS_DIR"/ 2>/dev/null
    fi

    # 4. Check GPU
    echo "[4/5] GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$procs" -eq 0 ]; then
        echo "  OK: No GPU processes running"
    else
        echo "  WARN: $procs processes on GPU"
        nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
    fi

    # 5. Check config
    if [ -f "$CUPS_DIR/$CONFIG" ]; then
        echo "[5/5] Config: FOUND"
        echo "  ROOT_PSEUDO: $(grep ROOT_PSEUDO "$CUPS_DIR/$CONFIG" | awk '{print $2}' | tr -d '\"')"
        echo "  NUM_GPUS: $(grep NUM_GPUS "$CUPS_DIR/$CONFIG" | awk '{print $2}')"
        echo "  ACCUMULATE_GRAD_BATCHES: $(grep ACCUMULATE_GRAD "$CUPS_DIR/$CONFIG" | awk '{print $2}')"
        echo "  LORA ENABLED: $(grep -A1 'LORA:' "$CUPS_DIR/$CONFIG" | grep ENABLED | awk '{print $2}')"
    else
        echo "[5/5] Config: MISSING at $CUPS_DIR/$CONFIG"
        exit 1
    fi

    echo ""
    echo "=== All checks passed ==="
}

stage2() {
    echo "=== Launching E2 DepthPro Conv-DoRA Stage-2 ==="
    LOGFILE="$LOG_DIR/e2_depthpro_dora_stage2_$(date +%Y%m%d_%H%M%S).log"

    cd "$CUPS_DIR"

    cd "$CUPS_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $PYTHON -u train.py \
        --experiment_config_file "$CONFIG" \
        --disable_wandb \
        > "$LOGFILE" 2>&1 &

    PID=$!
    echo "PID: $PID"
    echo "Log: $LOGFILE"
    echo ""
    echo "Monitor with: tail -f $LOGFILE"
    echo "Kill with:    kill $PID"
}

stage3() {
    echo "=== Launching E2 DepthPro Conv-DoRA Stage-3 ==="

    # Find best Stage-2 checkpoint
    EXP_DIR="/home/santosh/cups/experiments/e2_depthpro_conv_dora_r4_1gpu"
    if [ ! -d "$EXP_DIR" ]; then
        echo "ERROR: Stage-2 experiment dir not found: $EXP_DIR"
        exit 1
    fi

    CKPT=$(ls -t "$EXP_DIR"/checkpoints/*.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoints found in $EXP_DIR/checkpoints/"
        exit 1
    fi

    echo "Using checkpoint: $CKPT"
    LOGFILE="$LOG_DIR/e2_depthpro_dora_stage3_$(date +%Y%m%d_%H%M%S).log"

    cd "$CUPS_DIR"

    cd "$CUPS_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $PYTHON -u train_self.py \
        --experiment_config_file "$CONFIG" \
        --ckpt_path "$CKPT" \
        --disable_wandb \
        > "$LOGFILE" 2>&1 &

    PID=$!
    echo "PID: $PID"
    echo "Log: $LOGFILE"
    echo ""
    echo "Monitor with: tail -f $LOGFILE"
    echo "Kill with:    kill $PID"
}

status() {
    echo "=== E2 DepthPro Conv-DoRA Status ==="

    # Check if training is running
    pids=$(pgrep -f "train_net_pseudo.*depthpro_e2_dora" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Stage-2 RUNNING (PIDs: $pids)"
    else
        pids=$(pgrep -f "train_net_self.*depthpro_e2_dora" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "Stage-3 RUNNING (PIDs: $pids)"
        else
            echo "No training process found"
        fi
    fi

    # Show latest log tail
    LATEST_LOG=$(ls -t "$LOG_DIR"/e2_depthpro_dora_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo ""
        echo "Latest log: $LATEST_LOG"
        echo "--- Last 30 lines ---"
        tail -30 "$LATEST_LOG"
    fi
}

case "${1:-help}" in
    verify) verify ;;
    stage2) stage2 ;;
    stage3) stage3 ;;
    status) status ;;
    *)
        echo "Usage: $0 {verify|stage2|stage3|status}"
        exit 1
        ;;
esac
