#!/bin/bash
# Parameter sweep for COCONUT pseudo-label generation
# Ablates: k (clusters), grad_threshold (tau), min_area
# DINOv2 features and DAv2 depth are pre-computed and shared across all runs.
#
# Usage: bash scripts/sweep_coconut_params.sh 2>&1 | tee sweep_coconut.log

set -euo pipefail

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
COCO_ROOT="/Users/qbit-glitch/Desktop/datasets/coco"
COCONUT_ROOT="/Users/qbit-glitch/Desktop/datasets/coconut"
SCRIPT="mbps_pytorch/generate_coconut_pseudolabels.py"
EVAL_SCRIPT="mbps_pytorch/evaluate_coconut_pseudolabels.py"
SPLIT="val"
DEVICE="mps"

# ── Sweep grid ──
# k:              50, 100, 150, 200, 300
# grad_threshold: 0.02, 0.05, 0.10, 0.20
# min_area:       500, 1000, 2000, 4000

# Strategy: fix two params, sweep the third
# Base config: k=150, grad_threshold=0.05, min_area=1000

RESULTS_FILE="${COCONUT_ROOT}/sweep_results.csv"
echo "run_name,k,grad_threshold,min_area,n_things,miou_all,miou_things,miou_stuff,pq_all,pq_things,pq_stuff,sq_all,rq_all,total_tp,total_fp,total_fn,total_instances,avg_inst_per_img" > "$RESULTS_FILE"

run_config() {
    local RUN_NAME=$1
    local K=$2
    local GRAD=$3
    local MIN_AREA=$4
    local N_THINGS=$5
    local LOG="${COCONUT_ROOT}/sweep_logs/${RUN_NAME}.log"

    mkdir -p "${COCONUT_ROOT}/sweep_logs"

    echo ""
    echo "================================================================"
    echo "  RUN: ${RUN_NAME} (k=${K}, tau=${GRAD}, min_area=${MIN_AREA}, n_things=${N_THINGS})"
    echo "================================================================"

    # Compute output dirs
    local SEM_DIR="${COCONUT_ROOT}/pseudo_semantic_k${K}"
    local INST_DIR="${COCONUT_ROOT}/sweep_instance_${RUN_NAME}"
    local PAN_DIR="${COCONUT_ROOT}/sweep_panoptic_${RUN_NAME}"
    local ST_PATH="${SEM_DIR}/stuff_things_${RUN_NAME}.json"

    # Step 1: Semantics (only if not already generated for this k)
    if [ ! -d "${SEM_DIR}/${SPLIT}" ] || [ $(ls "${SEM_DIR}/${SPLIT}/"*.png 2>/dev/null | wc -l) -lt 5000 ]; then
        echo "  [1/5] Generating semantics (k=${K})..."
        $PYTHON -u "$SCRIPT" \
            --coco_root "$COCO_ROOT" --coconut_root "$COCONUT_ROOT" \
            --split "$SPLIT" --step semantics --k "$K" \
            --depth_model depth_anything_v2 --device "$DEVICE" \
            >> "$LOG" 2>&1
    else
        echo "  [1/5] Semantics k=${K} already exists, skipping"
    fi

    # Step 2: Stuff/Things classification (per-run since n_things varies)
    echo "  [2/5] Classifying stuff/things (n_things=${N_THINGS})..."
    $PYTHON -u -c "
import json, sys
sys.path.insert(0, '.')
from mbps_pytorch.generate_coconut_pseudolabels import classify_stuff_things
classify_stuff_things(
    semantic_dir='${SEM_DIR}/${SPLIT}',
    depth_dir='${COCONUT_ROOT}/depth_dav2/${SPLIT}',
    output_path='${ST_PATH}',
    num_classes=${K},
    n_things=${N_THINGS},
    grad_threshold=${GRAD},
    depth_blur_sigma=1.0,
)
" >> "$LOG" 2>&1

    # Step 3: Instances (per-run since grad/min_area vary)
    echo "  [3/5] Generating instances..."
    rm -rf "${INST_DIR}"
    $PYTHON -u -c "
import json, sys
sys.path.insert(0, '.')
from mbps_pytorch.generate_coconut_pseudolabels import generate_instances
generate_instances(
    semantic_dir='${SEM_DIR}/${SPLIT}',
    depth_dir='${COCONUT_ROOT}/depth_dav2/${SPLIT}',
    output_dir='${INST_DIR}/${SPLIT}',
    stuff_things_path='${ST_PATH}',
    grad_threshold=${GRAD},
    min_area=${MIN_AREA},
    dilation_iters=3,
    depth_blur=1.0,
)
" >> "$LOG" 2>&1

    # Step 4: Panoptic merge
    echo "  [4/5] Generating panoptic..."
    rm -rf "${PAN_DIR}"
    $PYTHON -u -c "
import json, sys
sys.path.insert(0, '.')
from mbps_pytorch.generate_coconut_pseudolabels import generate_panoptic
generate_panoptic(
    semantic_dir='${SEM_DIR}/${SPLIT}',
    instance_dir='${INST_DIR}/${SPLIT}',
    output_dir='${PAN_DIR}/${SPLIT}',
    stuff_things_path='${ST_PATH}',
)
" >> "$LOG" 2>&1

    # Step 5: Evaluate
    echo "  [5/5] Evaluating..."
    $PYTHON -u "$EVAL_SCRIPT" \
        --coconut_root "$COCONUT_ROOT" \
        --split "$SPLIT" \
        --k "$K" \
        --panoptic_subdir "sweep_panoptic_${RUN_NAME}" \
        >> "$LOG" 2>&1

    # Extract results from eval JSON
    local EVAL_JSON="${COCONUT_ROOT}/eval_results_${SPLIT}.json"
    if [ -f "$EVAL_JSON" ]; then
        local MIOU_ALL=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['semantic']['miou']*100:.2f}\")")
        local MIOU_TH=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['semantic']['miou_things']*100:.2f}\")")
        local MIOU_ST=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['semantic']['miou_stuff']*100:.2f}\")")
        local PQ_ALL=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['panoptic']['pq']*100:.2f}\")")
        local PQ_TH=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['panoptic']['pq_things']*100:.2f}\")")
        local PQ_ST=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['panoptic']['pq_stuff']*100:.2f}\")")
        local SQ_ALL=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['panoptic']['sq']*100:.2f}\")")
        local RQ_ALL=$($PYTHON -c "import json; d=json.load(open('${EVAL_JSON}')); print(f\"{d['panoptic']['rq']*100:.2f}\")")

        # Get TP/FP/FN from per_category
        local TP_FP_FN=$($PYTHON -c "
import json
d=json.load(open('${EVAL_JSON}'))
tp = sum(v['tp'] for v in d['panoptic']['per_category'].values())
fp = sum(v['fp'] for v in d['panoptic']['per_category'].values())
fn = sum(v['fn'] for v in d['panoptic']['per_category'].values())
print(f'{tp},{fp},{fn}')
")
        local TP=$(echo "$TP_FP_FN" | cut -d, -f1)
        local FP=$(echo "$TP_FP_FN" | cut -d, -f2)
        local FN=$(echo "$TP_FP_FN" | cut -d, -f3)

        # Get instance count from stats
        local INST_STATS="${INST_DIR}/${SPLIT}/stats.json"
        local TOTAL_INST="0"
        local AVG_INST="0"
        if [ -f "$INST_STATS" ]; then
            TOTAL_INST=$($PYTHON -c "import json; d=json.load(open('${INST_STATS}')); print(d.get('total_instances', 0))")
            AVG_INST=$($PYTHON -c "import json; d=json.load(open('${INST_STATS}')); print(d.get('avg_instances_per_image', 0))")
        fi

        echo "${RUN_NAME},${K},${GRAD},${MIN_AREA},${N_THINGS},${MIOU_ALL},${MIOU_TH},${MIOU_ST},${PQ_ALL},${PQ_TH},${PQ_ST},${SQ_ALL},${RQ_ALL},${TP},${FP},${FN},${TOTAL_INST},${AVG_INST}" >> "$RESULTS_FILE"

        # Copy eval JSON for this run
        cp "$EVAL_JSON" "${COCONUT_ROOT}/sweep_logs/${RUN_NAME}_eval.json"

        echo "  RESULT: mIoU=${MIOU_ALL}% PQ=${PQ_ALL}% (PQ_th=${PQ_TH}% PQ_st=${PQ_ST}%) RQ=${RQ_ALL}% TP=${TP} FP=${FP} FN=${FN}"
    else
        echo "  ERROR: No eval results found"
        echo "${RUN_NAME},${K},${GRAD},${MIN_AREA},${N_THINGS},ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR" >> "$RESULTS_FILE"
    fi

    echo "  Done: ${RUN_NAME}"
}

echo "Starting COCONUT parameter sweep..."
echo "Base config: k=150, grad_threshold=0.05, min_area=1000"
echo ""

# ── Ablation 1: Sweep k (fix tau=0.05, min_area=1000) ──
echo "=== SWEEP 1: k (clusters) ==="
for K in 50 100 150 200 300; do
    N_THINGS=$(( K * 60 / 100 ))
    run_config "k${K}_tau005_area1000" "$K" 0.05 1000 "$N_THINGS"
done

# ── Ablation 2: Sweep grad_threshold/tau (fix k=150, min_area=1000) ──
echo ""
echo "=== SWEEP 2: grad_threshold (tau) ==="
for TAU in 0.02 0.05 0.10 0.20; do
    TAU_LABEL=$(echo "$TAU" | tr -d '.')
    run_config "k150_tau${TAU_LABEL}_area1000" 150 "$TAU" 1000 90
done

# ── Ablation 3: Sweep min_area (fix k=150, tau=0.05) ──
echo ""
echo "=== SWEEP 3: min_area ==="
for AREA in 500 1000 2000 4000 8000; do
    run_config "k150_tau005_area${AREA}" 150 0.05 "$AREA" 90
done

# ── Ablation 4: Sweep n_things (fix k=150, tau=0.05, min_area=1000) ──
echo ""
echo "=== SWEEP 4: n_things ==="
for NT in 45 60 75 90 105 120; do
    run_config "k150_tau005_area1000_nt${NT}" 150 0.05 1000 "$NT"
done

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE"
echo "================================================================"
echo "Results: ${RESULTS_FILE}"
echo ""
echo "Summary:"
cat "$RESULTS_FILE" | column -t -s,
