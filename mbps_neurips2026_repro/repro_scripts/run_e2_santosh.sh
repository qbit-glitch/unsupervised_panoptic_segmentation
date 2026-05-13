#!/bin/bash
# E2 Conv-DoRA Experiment: DINOv3 ViT-B/16 + k=80 pseudo-labels + Conv-DoRA adapters
# Target: santosh@172.17.254.146, 2x GTX 1080 Ti
#
# Usage:
#   Verify:   bash scripts/run_e2_santosh.sh verify
#   Stage-2:  bash scripts/run_e2_santosh.sh stage2
#   Stage-3:  bash scripts/run_e2_santosh.sh stage3 /path/to/stage2/best_checkpoint.ckpt

set -euo pipefail

CUPS_ROOT="/home/santosh/cups"
DATASET_ROOT="/home/santosh/datasets/cityscapes"
PSEUDO_LABELS="${DATASET_ROOT}/cups_pseudo_labels_k80"
WEIGHTS="${CUPS_ROOT}/weights/dinov3_vitb16_official.pth"
LOG_DIR="${CUPS_ROOT}/logs"
EXPERIMENT_DIR="${CUPS_ROOT}/experiments"
CONFIG="${CUPS_ROOT}/configs/train_cityscapes_dinov3_vitb_k80_e2_conv_dora_santosh.yaml"

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/santosh/anaconda3/envs/cups/bin/python"

mkdir -p "${LOG_DIR}"

verify() {
    echo "=== E2 Conv-DoRA Pre-flight Checks ==="
    local ok=0
    local fail=0

    # Check pseudo-labels
    if [ -d "${PSEUDO_LABELS}" ]; then
        local sem_count inst_count non_empty
        sem_count=$(find "${PSEUDO_LABELS}" -name "*_semantic.png" 2>/dev/null | wc -l)
        inst_count=$(find "${PSEUDO_LABELS}" -name "*_instance.png" 2>/dev/null | wc -l)
        non_empty=$(${PYTHON} -c "
import numpy as np; from PIL import Image; import os
d='${PSEUDO_LABELS}'
n=sum(1 for f in os.listdir(d) if 'instance' in f and f.endswith('.png') and np.array(Image.open(os.path.join(d,f))).max()>0)
print(n)
" 2>/dev/null || echo "?")
        echo "[OK] Pseudo-labels: ${sem_count} semantic, ${inst_count} instance, ${non_empty} non-empty"
        if [ "${sem_count}" -lt 2900 ]; then
            echo "[FAIL] Expected ~2975 semantic labels, got ${sem_count}"
            ((fail++))
        else
            ((ok++))
        fi
    else
        echo "[FAIL] Pseudo-labels NOT found: ${PSEUDO_LABELS}"
        ((fail++))
    fi

    # Check DINOv3 weights
    if [ -f "${WEIGHTS}" ]; then
        echo "[OK] DINOv3 weights: ${WEIGHTS}"
        ((ok++))
    else
        echo "[FAIL] DINOv3 weights NOT found: ${WEIGHTS}"
        ((fail++))
    fi

    # Check config
    if [ -f "${CONFIG}" ]; then
        echo "[OK] E2 config: ${CONFIG}"
        # Verify LORA.ENABLED is True
        if grep -q "ENABLED: True" "${CONFIG}"; then
            echo "[OK] Conv-DoRA ENABLED: True"
            ((ok++))
        else
            echo "[FAIL] Conv-DoRA not enabled in config"
            ((fail++))
        fi
    else
        echo "[FAIL] E2 config NOT found: ${CONFIG}"
        ((fail++))
    fi

    # Check Cityscapes images + GT
    if [ -d "${DATASET_ROOT}/leftImg8bit/train" ]; then
        echo "[OK] Cityscapes train images"
        ((ok++))
    else
        echo "[FAIL] Cityscapes images NOT found"
        ((fail++))
    fi

    if [ -d "${DATASET_ROOT}/gtFine/val" ]; then
        echo "[OK] Cityscapes GT val"
        ((ok++))
    else
        echo "[FAIL] Cityscapes GT val NOT found"
        ((fail++))
    fi

    # Check GPUs
    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        echo "[OK] GPUs: ${gpu_count} available"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
        ((ok++))
    else
        echo "[FAIL] nvidia-smi not found"
        ((fail++))
    fi

    echo ""
    echo "=== ${ok} passed, ${fail} failed ==="
    if [ "${fail}" -gt 0 ]; then
        echo "Fix failures before running."
        return 1
    fi
    echo "Ready to run E2 Conv-DoRA."
    return 0
}

run_stage2() {
    echo "=== E2 Stage-2: DINOv3 ViT-B/16 + Conv-DoRA (r=4) + k=80 PLs ==="
    local logfile="${LOG_DIR}/e2_conv_dora_stage2_$(date +%Y%m%d_%H%M%S).log"

    verify || exit 1

    echo ""
    echo "Config:  ${CONFIG}"
    echo "Log:     ${logfile}"
    echo "GPUs:    2x GTX 1080 Ti"
    echo "Batch:   2 GPUs x bs=1 x accum=8 = effective 16"
    echo "Conv-DoRA: rank=4, alpha=4.0, variant=conv_dora"
    echo ""
    echo "Starting E2 Stage-2 training (8000 steps)..."
    echo "Monitor: tail -f ${logfile}"
    echo ""

    cd "${CUPS_ROOT}"
    nohup ${PYTHON} -u train.py \
        --experiment_config_file "${CONFIG}" \
        --disable_wandb \
        > "${logfile}" 2>&1 &

    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/e2_stage2.pid"
    echo ""
    echo "Stage-2 launched. When done, find best checkpoint in:"
    echo "  ${EXPERIMENT_DIR}/experiments/e2_dinov3_vitb_k80_conv_dora_r4/"
    echo ""
    echo "Then run Stage-3:"
    echo "  bash run_e2_santosh.sh stage3 /path/to/best_checkpoint.ckpt"
}

run_stage3() {
    local ckpt_path="${1:-}"
    if [ -z "${ckpt_path}" ]; then
        echo "ERROR: Stage-3 requires a Stage-2 checkpoint path."
        echo "Usage: bash run_e2_santosh.sh stage3 /path/to/checkpoint.ckpt"
        exit 1
    fi

    if [ ! -f "${ckpt_path}" ]; then
        echo "ERROR: Checkpoint not found: ${ckpt_path}"
        exit 1
    fi

    echo "=== E2 Stage-3: Conv-DoRA Self-Training with EMA Teacher ==="
    echo "Checkpoint: ${ckpt_path}"
    echo "NOTE: Stage-3 config not yet created. Create it based on E2 Stage-2 config."
    exit 1
}

case "${1:-}" in
    verify)
        verify
        ;;
    stage2)
        run_stage2
        ;;
    stage3)
        run_stage3 "${2:-}"
        ;;
    *)
        echo "Usage: bash run_e2_santosh.sh {verify|stage2|stage3} [checkpoint_path]"
        echo ""
        echo "  verify  — Check pseudo-labels, weights, GPU before running"
        echo "  stage2  — Launch Stage-2 training (8000 steps, ~1.5h on 2x 1080 Ti)"
        echo "  stage3  — Launch Stage-3 self-training (needs Stage-2 checkpoint)"
        exit 1
        ;;
esac
