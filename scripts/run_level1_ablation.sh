#!/usr/bin/env bash
# Level 1 Dead-Class Recovery Ablation Runner
#
# Usage:
#   bash scripts/run_level1_ablation.sh generate    # Generate all variants
#   bash scripts/run_level1_ablation.sh evaluate    # Evaluate all variants
#   bash scripts/run_level1_ablation.sh all         # Generate + evaluate

set -euo pipefail

CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
BASELINE_DIR="$CS_ROOT/cups_pseudo_labels_dcfa_simcf_abc"
OUTPUT_ROOT="$CS_ROOT/level1_ablation"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs/level1_ablation"

mkdir -p "$LOG_DIR"

generate() {
    echo "=== Generating Level 1 Ablation Variants ==="
    echo "Baseline: $BASELINE_DIR"
    echo "Output:   $OUTPUT_ROOT"
    echo ""

    python3 "$SCRIPT_DIR/generate_level1_ablation.py" \
        --cityscapes_root "$CS_ROOT" \
        --baseline_dir "$BASELINE_DIR" \
        --output_root "$OUTPUT_ROOT" \
        --methods A1,A2,A3,A4,A1A2,ALL \
        2>&1 | tee "$LOG_DIR/generate.log"
}

evaluate() {
    echo "=== Evaluating Level 1 Ablation Variants ==="
    echo ""

    RESULTS_FILE="$LOG_DIR/results_summary.csv"
    echo "variant,PQ,PQ_stuff,PQ_things,mIoU,ignore_pct" > "$RESULTS_FILE"

    VARIANTS="baseline A1 A2 A3 A4 A1A2 ALL"
    for name in $VARIANTS; do
        if [ "$name" = "baseline" ]; then
            dir="$BASELINE_DIR"
        else
            dir="$OUTPUT_ROOT/$name"
        fi

        if [ ! -d "$dir" ]; then
            echo "SKIP $name: directory not found ($dir)"
            continue
        fi

        echo "--- Evaluating $name ---"
        python3 "$SCRIPT_DIR/evaluate_pseudolabel_quality.py" \
            --pseudo_dir "$dir" \
            --cityscapes_root "$CS_ROOT" \
            --baseline_dir "$BASELINE_DIR" \
            2>&1 | tee "$LOG_DIR/eval_${name}.log"

        # Extract summary line
        summary=$(grep "^SUMMARY;" "$LOG_DIR/eval_${name}.log" | tail -1)
        if [ -n "$summary" ]; then
            pq=$(echo "$summary" | sed 's/.*PQ=\([0-9.]*\).*/\1/')
            pq_st=$(echo "$summary" | sed 's/.*PQ_st=\([0-9.]*\).*/\1/')
            pq_th=$(echo "$summary" | sed 's/.*PQ_th=\([0-9.]*\).*/\1/')
            miou=$(echo "$summary" | sed 's/.*mIoU=\([0-9.]*\).*/\1/')
            ignore=$(echo "$summary" | sed 's/.*ignore=\([0-9.]*\).*/\1/')
            echo "$name,$pq,$pq_st,$pq_th,$miou,$ignore" >> "$RESULTS_FILE"
        fi
        echo ""
    done

    echo "=== Results saved to $RESULTS_FILE ==="
    cat "$RESULTS_FILE"
}

summary() {
    RESULTS_FILE="$LOG_DIR/results_summary.csv"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "No results found. Run 'evaluate' first."
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "  Level 1 Dead-Class Recovery Ablation Results"
    echo "============================================================"
    echo ""
    printf "%-10s | %6s | %8s | %9s | %6s | %7s\n" \
        "Variant" "PQ" "PQ_stuff" "PQ_things" "mIoU" "Ignore"
    echo "------------------------------------------------------------"

    baseline_pq=$(grep "^baseline," "$RESULTS_FILE" | cut -d',' -f2)

    while IFS=',' read -r name pq pq_st pq_th miou ignore; do
        [ "$name" = "variant" ] && continue
        if [ -n "$baseline_pq" ] && [ "$name" != "baseline" ]; then
            delta=$(echo "$pq - $baseline_pq" | bc 2>/dev/null || echo "?")
            printf "%-10s | %5s%% | %7s%% | %8s%% | %5s%% | %6s%% [%+.2f]\n" \
                "$name" "$pq" "$pq_st" "$pq_th" "$miou" "$ignore" "$delta"
        else
            printf "%-10s | %5s%% | %7s%% | %8s%% | %5s%% | %6s%%\n" \
                "$name" "$pq" "$pq_st" "$pq_th" "$miou" "$ignore"
        fi
    done < "$RESULTS_FILE"

    echo ""
    echo "Baseline PQ: $baseline_pq%"
    echo "============================================================"
}

case "${1:-help}" in
    generate)  generate ;;
    evaluate)  evaluate ;;
    all)       generate; evaluate; summary ;;
    summary)   summary ;;
    *)
        echo "Usage: $0 {generate|evaluate|all|summary}"
        echo ""
        echo "  generate  — create all pseudo-label variants"
        echo "  evaluate  — evaluate all variants against GT"
        echo "  all       — generate + evaluate + summary"
        echo "  summary   — print results table"
        ;;
esac
