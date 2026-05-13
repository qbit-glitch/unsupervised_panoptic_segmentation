#!/usr/bin/env bash
# SIMCF Step-by-Step Ablation (rebuttal harness)
#
# Question: What is the contribution of each SIMCF step (A, B, C) on top
# of DCFA-conditioned k=80 pseudo-labels?
#
# Variants (all four use the same DCFA-conditioned input + DepthPro instances):
#   1. no_simcf         = cups_pseudo_labels_adapter_V3_tau020 (existing baseline)
#   2. simcf_a          = + Step A only       (NEW)
#   3. simcf_ab         = + Steps A+B         (NEW)
#   4. simcf_abc        = + Steps A+B+C       (existing cups_pseudo_labels_dcfa_simcf_abc)
#
# Time budget: ~60 min on M4 Pro CPU.
#
# Usage:
#   bash scripts/run_simcf_step_ablation.sh

set -euo pipefail

CS_ROOT="${CS_ROOT:-$HOME/Desktop/datasets/cityscapes}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/simcf_step_ablation"
PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"

# DCFA-conditioned pseudo-label source (no SIMCF) — paired with DepthPro instances.
DCFA_DIR="$CS_ROOT/cups_pseudo_labels_adapter_V3_tau020"
# DCFA-specific centroids (same cluster-to-class mapping used at SIMCF train time).
CENTROIDS="$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz"
# Existing fully-filtered output to verify reproducibility against the paper number.
ABC_DIR="$CS_ROOT/cups_pseudo_labels_dcfa_simcf_abc"

# New per-step variants.
A_DIR="$CS_ROOT/cups_pseudo_labels_dcfa_simcf_step_a"
AB_DIR="$CS_ROOT/cups_pseudo_labels_dcfa_simcf_step_ab"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  SIMCF Step Ablation: A vs A+B vs A+B+C"
echo "  Started: $(date)"
echo "  Source DCFA dir: $DCFA_DIR"
echo "  Centroids:       $CENTROIDS"
echo "============================================================"
echo ""

# --- Pre-flight ---
n_in=$(ls "$DCFA_DIR"/*_semantic.png 2>/dev/null | wc -l | tr -d ' ')
echo "[pre-flight] DCFA source has $n_in semantic PNGs (need 2975)"
[ "$n_in" -ge 2975 ] || { echo "FAIL: DCFA source incomplete"; exit 1; }

n_abc=$(ls "$ABC_DIR"/*_semantic.png 2>/dev/null | wc -l | tr -d ' ')
echo "[pre-flight] Full SIMCF-ABC dir has $n_abc semantic PNGs"
[ "$n_abc" -ge 2975 ] || { echo "FAIL: SIMCF-ABC dir incomplete"; exit 1; }
[ -f "$CENTROIDS" ] || { echo "FAIL: centroids not found at $CENTROIDS"; exit 1; }
echo ""

# --- Generation: SIMCF-A only ---
if [ ! -f "$A_DIR/aachen_000000_000019_leftImg8bit_semantic.png" ]; then
  echo "=== [GEN] DCFA + SIMCF-A only ==="
  echo "  Start: $(date)"
  $PYTHON -u "$SCRIPTS_DIR/refine_simcf.py" \
      --input_dir "$DCFA_DIR" \
      --output_dir "$A_DIR" \
      --centroids_path "$CENTROIDS" \
      --cityscapes_root "$CS_ROOT" \
      --steps A \
      2>&1 | tee "$LOG_DIR/gen_simcf_a.log"
  echo "  End: $(date)"
  echo ""
else
  echo "=== [SKIP-GEN] SIMCF-A output already exists at $A_DIR ==="
  echo ""
fi

# --- Generation: SIMCF-A+B ---
if [ ! -f "$AB_DIR/aachen_000000_000019_leftImg8bit_semantic.png" ]; then
  echo "=== [GEN] DCFA + SIMCF-A+B ==="
  echo "  Start: $(date)"
  $PYTHON -u "$SCRIPTS_DIR/refine_simcf.py" \
      --input_dir "$DCFA_DIR" \
      --output_dir "$AB_DIR" \
      --centroids_path "$CENTROIDS" \
      --cityscapes_root "$CS_ROOT" \
      --steps A,B \
      2>&1 | tee "$LOG_DIR/gen_simcf_ab.log"
  echo "  End: $(date)"
  echo ""
else
  echo "=== [SKIP-GEN] SIMCF-A+B output already exists at $AB_DIR ==="
  echo ""
fi

# --- Evaluation ---
RESULTS_FILE="$LOG_DIR/results.csv"
echo "variant,PQ,PQ_stuff,PQ_things,mIoU,ignore_pct" > "$RESULTS_FILE"

eval_variant() {
  local name="$1"
  local dir="$2"
  if [ ! -d "$dir" ]; then
    echo "  SKIP $name: not found"
    return
  fi
  echo "=== [EVAL] $name ==="
  echo "  Dir: $dir"
  echo "  Start: $(date)"
  $PYTHON -u "$SCRIPTS_DIR/evaluate_pseudolabel_quality.py" \
      --pseudo_dir "$dir" \
      --cityscapes_root "$CS_ROOT" \
      --centroids_path "$CENTROIDS" \
      --output "$LOG_DIR/eval_${name}.json" \
      2>&1 | tee "$LOG_DIR/eval_${name}.log"

  summary=$(grep "^SUMMARY;" "$LOG_DIR/eval_${name}.log" | tail -1)
  if [ -n "$summary" ]; then
    pq=$(echo "$summary" | sed 's/.*PQ=\([0-9.]*\).*/\1/')
    pq_st=$(echo "$summary" | sed 's/.*PQ_st=\([0-9.]*\).*/\1/')
    pq_th=$(echo "$summary" | sed 's/.*PQ_th=\([0-9.]*\).*/\1/')
    miou=$(echo "$summary" | sed 's/.*mIoU=\([0-9.]*\).*/\1/')
    ignore=$(echo "$summary" | sed 's/.*ignore=\([0-9.]*\).*/\1/')
    echo "$name,$pq,$pq_st,$pq_th,$miou,$ignore" >> "$RESULTS_FILE"
    echo "  -> PQ=$pq PQ_st=$pq_st PQ_th=$pq_th mIoU=$miou ignore=$ignore"
  else
    echo "  WARN: no SUMMARY line for $name"
  fi
  echo "  End: $(date)"
  echo ""
}

eval_variant "no_simcf"   "$DCFA_DIR"
eval_variant "simcf_a"    "$A_DIR"
eval_variant "simcf_ab"   "$AB_DIR"
eval_variant "simcf_abc"  "$ABC_DIR"

# --- Instance-count proxy ---
echo "=== [STATS] Instance counts (over-fragmentation proxy) ==="
$PYTHON -u "$SCRIPTS_DIR/compute_instance_stats.py" \
    --variant no_simcf  --dir "$DCFA_DIR"  --output "$LOG_DIR/inst_no_simcf.json" \
    2>&1 | tee "$LOG_DIR/inst_no_simcf.log"
$PYTHON -u "$SCRIPTS_DIR/compute_instance_stats.py" \
    --variant simcf_a   --dir "$A_DIR"     --output "$LOG_DIR/inst_simcf_a.json" \
    2>&1 | tee "$LOG_DIR/inst_simcf_a.log"
$PYTHON -u "$SCRIPTS_DIR/compute_instance_stats.py" \
    --variant simcf_ab  --dir "$AB_DIR"    --output "$LOG_DIR/inst_simcf_ab.json" \
    2>&1 | tee "$LOG_DIR/inst_simcf_ab.log"
$PYTHON -u "$SCRIPTS_DIR/compute_instance_stats.py" \
    --variant simcf_abc --dir "$ABC_DIR"   --output "$LOG_DIR/inst_simcf_abc.json" \
    2>&1 | tee "$LOG_DIR/inst_simcf_abc.log"

echo ""
echo "============================================================"
echo "  Results CSV: $RESULTS_FILE"
echo "  Logs dir:    $LOG_DIR"
echo "  Completed:   $(date)"
echo "============================================================"
cat "$RESULTS_FILE"
