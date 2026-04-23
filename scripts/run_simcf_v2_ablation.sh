#!/usr/bin/env bash
# SIMCF-v2 Ablation: Tier 1 (individual steps) + Evaluate
#
# Runs all 5 individual SIMCF-v2 steps on A3 (SIMCF-ABC) output,
# then evaluates each variant against Cityscapes GT.
#
# Usage:
#   nohup bash scripts/run_simcf_v2_ablation.sh > logs/simcf_v2_ablation.log 2>&1 &

set -euo pipefail

CS_ROOT="${CS_ROOT:-$HOME/Desktop/datasets/cityscapes}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPTS_DIR/../logs/pseudolabel_ablation"
PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"

A3_DIR="$CS_ROOT/cups_pseudo_labels_simcf_abc"
CENTROIDS="$CS_ROOT/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  SIMCF-v2 Ablation — Tier 1 (Individual Steps)"
echo "  $(date)"
echo "  Input: $A3_DIR"
echo "============================================================"
echo ""

# Verify input
n_input=$(ls "$A3_DIR"/*_semantic.png 2>/dev/null | wc -l | tr -d ' ')
echo "Input check: $n_input semantic PNGs (need 2975)"
[ "$n_input" -ge 2975 ] || { echo "FAIL: A3 output incomplete"; exit 1; }
echo ""

# ---------- Tier 1: Generate ----------

run_step() {
    local name="$1"
    local steps="$2"
    local extra_args="${3:-}"
    local out_dir="$CS_ROOT/cups_pseudo_labels_simcf_v2_${name}"

    echo "=== [$name] Steps=$steps ==="
    echo "  Start: $(date)"

    $PYTHON -u "$SCRIPTS_DIR/refine_simcf_v2.py" \
        --input_dir "$A3_DIR" \
        --output_dir "$out_dir" \
        --centroids_path "$CENTROIDS" \
        --cityscapes_root "$CS_ROOT" \
        --steps "$steps" \
        $extra_args \
        2>&1 | tee "$LOG_DIR/v2_${name}.log"

    local n_out=$(ls "$out_dir"/*_semantic.png 2>/dev/null | wc -l | tr -d ' ')
    echo "  Output: $n_out semantic PNGs"
    echo "  End: $(date)"
    echo ""
}

echo "=========================================="
echo "  Phase 1: Generate all Tier 1 variants"
echo "=========================================="
echo ""

run_step "D" "D"
run_step "E" "E"
run_step "F" "F"
run_step "G" "G"
run_step "H" "H"

echo "=========================================="
echo "  Phase 2: Evaluate all variants"
echo "=========================================="
echo ""

RESULTS_FILE="$LOG_DIR/results_v2.csv"
echo "variant,PQ,PQ_stuff,PQ_things,mIoU,ignore_pct" > "$RESULTS_FILE"

eval_variant() {
    local name="$1"
    local dir="$2"

    if [ ! -d "$dir" ]; then
        echo "SKIP $name: not found"
        return
    fi

    echo "--- Evaluating $name ---"
    echo "  Start: $(date)"
    $PYTHON -u "$SCRIPTS_DIR/evaluate_pseudolabel_quality.py" \
        --pseudo_dir "$dir" \
        --cityscapes_root "$CS_ROOT" \
        2>&1 | tee "$LOG_DIR/eval_v2_${name}.log"

    summary=$(grep "^SUMMARY;" "$LOG_DIR/eval_v2_${name}.log" | tail -1)
    if [ -n "$summary" ]; then
        pq=$(echo "$summary" | sed 's/.*PQ=\([0-9.]*\).*/\1/')
        pq_st=$(echo "$summary" | sed 's/.*PQ_st=\([0-9.]*\).*/\1/')
        pq_th=$(echo "$summary" | sed 's/.*PQ_th=\([0-9.]*\).*/\1/')
        miou=$(echo "$summary" | sed 's/.*mIoU=\([0-9.]*\).*/\1/')
        ignore=$(echo "$summary" | sed 's/.*ignore=\([0-9.]*\).*/\1/')
        echo "$name,$pq,$pq_st,$pq_th,$miou,$ignore" >> "$RESULTS_FILE"
        echo "  PQ=$pq PQ_st=$pq_st PQ_th=$pq_th mIoU=$miou ignore=$ignore"
    fi
    echo "  End: $(date)"
    echo ""
}

# Evaluate baselines first
eval_variant "a0_baseline" "$CS_ROOT/cups_pseudo_labels_depthpro_tau020"
eval_variant "a3_simcf_abc" "$A3_DIR"

# Evaluate Tier 1
for step in D E F G H; do
    eval_variant "v2_${step}" "$CS_ROOT/cups_pseudo_labels_simcf_v2_${step}"
done

echo ""
echo "============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"
echo ""
printf "%-18s | %6s | %8s | %9s | %6s | %7s\n" \
    "Variant" "PQ" "PQ_stuff" "PQ_things" "mIoU" "Ignore"
echo "-----------------------------------------------------------"

baseline_pq=$(grep "a3_simcf_abc" "$RESULTS_FILE" | cut -d',' -f2)

while IFS=',' read -r name pq pq_st pq_th miou ignore; do
    [ "$name" = "variant" ] && continue
    if [ -n "$baseline_pq" ] && [ "$name" != "a0_baseline" ] && [ "$name" != "a3_simcf_abc" ]; then
        delta=$(echo "$pq - $baseline_pq" | bc 2>/dev/null || echo "?")
        printf "%-18s | %5s%% | %7s%% | %8s%% | %5s%% | %6s%% [%+.2f]\n" \
            "$name" "$pq" "$pq_st" "$pq_th" "$miou" "$ignore" "$delta"
    else
        printf "%-18s | %5s%% | %7s%% | %8s%% | %5s%% | %6s%%\n" \
            "$name" "$pq" "$pq_st" "$pq_th" "$miou" "$ignore"
    fi
done < "$RESULTS_FILE"

echo ""
echo "Gate: Delta > 0 over A3 (PQ=$baseline_pq)"
echo "Results: $RESULTS_FILE"
echo "Completed: $(date)"
echo "============================================================"
