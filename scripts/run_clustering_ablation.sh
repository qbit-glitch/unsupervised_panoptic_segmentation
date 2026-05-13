#!/bin/bash
# Clustering ablation: generate + evaluate + summary for all methods.
#
# Usage:
#   bash scripts/run_clustering_ablation.sh all        # full pipeline
#   bash scripts/run_clustering_ablation.sh generate    # generate only
#   bash scripts/run_clustering_ablation.sh evaluate    # evaluate only
#   bash scripts/run_clustering_ablation.sh summary     # print summary table
#   bash scripts/run_clustering_ablation.sh single vmf  # one method only

set -euo pipefail

CS_ROOT="${CITYSCAPES_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PYTHON="${PYTHON:-/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python}"
K=80
SEED=42
LOG_DIR="logs/clustering_ablation"
RESULTS_DIR="results/clustering_ablation"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

METHODS=(
    "euclidean_kmeans"
    "spherical_kmeans"
    "vmf"
    "kmeans_sinkhorn"
    "vmf_sinkhorn"
    "eagle_spectral"
    "cause_codebook"
)

generate_one() {
    local METHOD="$1"
    local OUT_SUBDIR="pseudo_semantic_raw_dinov3_k${K}_${METHOD}"
    echo "[GEN] Method=$METHOD -> $OUT_SUBDIR"

    local EXTRA_ARGS=""
    case "$METHOD" in
        vmf|vmf_sinkhorn)
            EXTRA_ARGS="--vmf_max_iter 50 --kappa_init 100"
            ;;
        kmeans_sinkhorn|vmf_sinkhorn)
            EXTRA_ARGS="$EXTRA_ARGS --sinkhorn_iters 5 --sinkhorn_temp 0.1"
            ;;
        eagle_spectral)
            EXTRA_ARGS="--n_eig 20 --alpha 0.7 --max_images_spectral 200"
            ;;
        cause_codebook)
            EXTRA_ARGS="--lr 1e-3 --num_epochs 10 --diversity_weight 0.1"
            ;;
    esac

    $PYTHON -u mbps_pytorch/generate_clustering_ablation.py \
        --cityscapes_root "$CS_ROOT" \
        --method "$METHOD" \
        --k "$K" --seed "$SEED" \
        --splits train val \
        $EXTRA_ARGS \
        > "$LOG_DIR/gen_${METHOD}.log" 2>&1

    echo "  Done. Log: $LOG_DIR/gen_${METHOD}.log"
}

generate_all() {
    for METHOD in "${METHODS[@]}"; do
        generate_one "$METHOD"
    done
}

evaluate_one() {
    local METHOD="$1"
    local OUT_SUBDIR="pseudo_semantic_raw_dinov3_k${K}_${METHOD}"
    local EVAL_OUT="$RESULTS_DIR/eval_${METHOD}.json"

    if [ ! -d "$CS_ROOT/$OUT_SUBDIR/val" ]; then
        echo "[SKIP] $METHOD — no val labels at $CS_ROOT/$OUT_SUBDIR/val"
        return
    fi

    echo "[EVAL] $METHOD"
    $PYTHON -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
        --cityscapes_root "$CS_ROOT" \
        --split val \
        --semantic_subdir "$OUT_SUBDIR" \
        --instance_subdir pseudo_instance_spidepth \
        --num_clusters "$K" \
        --cluster_mapping majority \
        --eval_size 512 1024 \
        --output "$EVAL_OUT" \
        > "$LOG_DIR/eval_${METHOD}.log" 2>&1

    echo "  Done -> $EVAL_OUT"
}

evaluate_all() {
    for METHOD in "${METHODS[@]}"; do
        evaluate_one "$METHOD"
    done
}

summary() {
    echo ""
    echo "┌─────────────────────┬───────┬───────┬────────┬─────────┬─────────┬───────┐"
    echo "│ Method              │ mIoU  │ PQ    │ PQ_st  │ PQ_th   │ Entropy │ Empty │"
    echo "├─────────────────────┼───────┼───────┼────────┼─────────┼─────────┼───────┤"
    for METHOD in "${METHODS[@]}"; do
        local JSON="$RESULTS_DIR/eval_${METHOD}.json"
        local STATS="$CS_ROOT/pseudo_semantic_raw_dinov3_k${K}_${METHOD}/cluster_stats.json"

        if [ ! -f "$JSON" ]; then
            printf "│ %-19s │ %5s │ %5s │ %6s │ %7s │ %7s │ %5s │\n" \
                "$METHOD" "—" "—" "—" "—" "—" "—"
            continue
        fi

        local MIOU PQ PQ_ST PQ_TH ENTROPY EMPTY
        MIOU=$($PYTHON -c "import json; d=json.load(open('$JSON')); print(f\"{d.get('semantic',{}).get('miou',0):.2f}\")" 2>/dev/null || echo "—")
        PQ=$($PYTHON -c "import json; d=json.load(open('$JSON')); print(f\"{d.get('panoptic',{}).get('PQ',0):.2f}\")" 2>/dev/null || echo "—")
        PQ_ST=$($PYTHON -c "import json; d=json.load(open('$JSON')); print(f\"{d.get('panoptic',{}).get('PQ_stuff',0):.2f}\")" 2>/dev/null || echo "—")
        PQ_TH=$($PYTHON -c "import json; d=json.load(open('$JSON')); print(f\"{d.get('panoptic',{}).get('PQ_things',0):.2f}\")" 2>/dev/null || echo "—")

        if [ -f "$STATS" ]; then
            ENTROPY=$($PYTHON -c "import json; d=json.load(open('$STATS')); print(f\"{d['entropy']:.3f}\")" 2>/dev/null || echo "—")
            EMPTY=$($PYTHON -c "import json; d=json.load(open('$STATS')); print(d['empty_clusters'])" 2>/dev/null || echo "—")
        else
            ENTROPY="—"
            EMPTY="—"
        fi

        printf "│ %-19s │ %5s │ %5s │ %6s │ %7s │ %7s │ %5s │\n" \
            "$METHOD" "$MIOU" "$PQ" "$PQ_ST" "$PQ_TH" "$ENTROPY" "$EMPTY"
    done
    echo "└─────────────────────┴───────┴───────┴────────┴─────────┴─────────┴───────┘"
    echo ""
}

case "${1:-all}" in
    generate)  generate_all ;;
    evaluate)  evaluate_all ;;
    summary)   summary ;;
    single)    generate_one "${2:?Usage: $0 single <method>}"; evaluate_one "$2" ;;
    all)       generate_all; evaluate_all; summary ;;
    *)         echo "Usage: $0 {generate|evaluate|summary|single <method>|all}"; exit 1 ;;
esac
