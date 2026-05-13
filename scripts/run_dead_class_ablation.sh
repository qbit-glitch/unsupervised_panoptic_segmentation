#!/bin/bash
# Dead-class recovery ablation: evaluate + summarize.
#
# Usage:
#   bash scripts/run_dead_class_ablation.sh evaluate
#   bash scripts/run_dead_class_ablation.sh summary
#   bash scripts/run_dead_class_ablation.sh eval_one <name> <subdir>

set -euo pipefail

CS_ROOT="${CITYSCAPES_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PYTHON="${PYTHON:-/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python}"
K=100
RESULTS_DIR="results/clustering_ablation"
LOG_DIR="logs"

mkdir -p "$RESULTS_DIR"

NAMES="baseline shiftavg pcl_sinkhorn recursive_dsc"

get_subdir() {
    case "$1" in
        baseline)       echo "pseudo_semantic_raw_dinov3_k100_spherical_kmeans_vitl16" ;;
        shiftavg)       echo "pseudo_semantic_raw_dinov3_k100_spherical_kmeans_vitl16_shiftavg64x128" ;;
        pcl_sinkhorn)   echo "pseudo_semantic_raw_dinov3_k100_pcl_sinkhorn_vitl16" ;;
        recursive_dsc)  echo "pseudo_semantic_raw_dinov3_k100_recursive_dsc_vitl16" ;;
        *)              echo "$1" ;;
    esac
}

evaluate_one() {
    local NAME="$1"
    local SUBDIR
    SUBDIR=$(get_subdir "$NAME")
    local EVAL_OUT="$RESULTS_DIR/eval_${NAME}_vitl16_k100.json"

    if [ ! -d "$CS_ROOT/$SUBDIR/val" ]; then
        echo "[SKIP] $NAME — no val labels at $CS_ROOT/$SUBDIR/val"
        return
    fi

    echo "[EVAL] $NAME ($SUBDIR)"
    $PYTHON -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
        --cityscapes_root "$CS_ROOT" \
        --split val \
        --semantic_subdir "$SUBDIR" \
        --instance_subdir pseudo_instance_spidepth \
        --num_clusters "$K" \
        --cluster_mapping majority \
        --eval_size 512 1024 \
        --output "$EVAL_OUT" \
        > "$LOG_DIR/eval_${NAME}.log" 2>&1

    echo "  Done -> $EVAL_OUT"
}

evaluate_all() {
    for NAME in $NAMES; do
        evaluate_one "$NAME"
    done
}

summary() {
    echo ""
    echo "┌────────────────────┬───────┬───────┬────────┬─────────┬───────┐"
    echo "│ Method             │ mIoU  │ PQ    │ PQ_st  │ PQ_th   │ Dead  │"
    echo "├────────────────────┼───────┼───────┼────────┼─────────┼───────┤"

    for NAME in $NAMES; do
        local JSON="$RESULTS_DIR/eval_${NAME}_vitl16_k100.json"

        if [ ! -f "$JSON" ]; then
            printf "│ %-18s │ %5s │ %5s │ %6s │ %7s │ %5s │\n" \
                "$NAME" "—" "—" "—" "—" "—"
            continue
        fi

        $PYTHON -c "
import json
d=json.load(open('$JSON'))
sem = d.get('semantic',{})
pan = d.get('panoptic',{})
iou = sem.get('per_class_iou',{})
miou = sem.get('miou',0)
pq = pan.get('PQ',0)
pqs = pan.get('PQ_stuff',0)
pqt = pan.get('PQ_things',0)
names = ['wall','fence','traffic light','rider','truck','train','motorcycle']
dead = sum(1 for n in names if iou.get(n,0.0) == 0.0)
print(f'│ $NAME' + ' '*(19-len('$NAME')) + f'│ {miou:5.1f} │ {pq:5.1f} │ {pqs:6.1f} │ {pqt:7.1f} │ {dead}/7   │')
" 2>/dev/null || printf "│ %-18s │ %5s │ %5s │ %6s │ %7s │ %5s │\n" "$NAME" "ERR" "ERR" "ERR" "ERR" "ERR"
    done
    echo "└────────────────────┴───────┴───────┴────────┴─────────┴───────┘"
    echo ""

    echo "Per-class IoU for originally-dead classes:"
    echo "┌────────────────────┬───────┬───────┬──────┬───────┬───────┬───────┬──────────┐"
    echo "│ Method             │ wall  │ fence │ t.lt │ rider │ truck │ train │ m.cycle  │"
    echo "├────────────────────┼───────┼───────┼──────┼───────┼───────┼───────┼──────────┤"

    for NAME in $NAMES; do
        local JSON="$RESULTS_DIR/eval_${NAME}_vitl16_k100.json"

        if [ ! -f "$JSON" ]; then
            printf "│ %-18s │ %5s │ %5s │ %4s │ %5s │ %5s │ %5s │ %8s │\n" \
                "$NAME" "—" "—" "—" "—" "—" "—" "—"
            continue
        fi

        $PYTHON -c "
import json
d=json.load(open('$JSON'))
iou = d.get('semantic',{}).get('per_class_iou',{})
names = ['wall','fence','traffic light','rider','truck','train','motorcycle']
vals = [f'{iou.get(n,0.0):.1f}' for n in names]
print(f'│ $NAME' + ' '*(19-len('$NAME')) + '│ ' + ' │ '.join(f'{v:>5s}' for v in vals) + ' │')
" 2>/dev/null || printf "│ %-18s │ %5s │ %5s │ %4s │ %5s │ %5s │ %5s │ %8s │\n" "$NAME" "—" "—" "—" "—" "—" "—" "—"
    done
    echo "└────────────────────┴───────┴───────┴──────┴───────┴───────┴───────┴──────────┘"
}

case "${1:-summary}" in
    evaluate)   evaluate_all ;;
    eval_one)   evaluate_one "${2:?Usage: $0 eval_one <name>}" ;;
    summary)    summary ;;
    *)          echo "Usage: $0 {evaluate|eval_one <name>|summary}"; exit 1 ;;
esac
