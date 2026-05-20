#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="${CACHE_DIR:-outputs/code_upsampler/cache_dcfa_v3_90d_64x128}"
GT_DIR="${GT_DIR:-/Users/qbit-glitch/Desktop/datasets/cityscapes/gtFine/val}"
RUN_ROOT="${RUN_ROOT:-outputs/code_upsampler/runs}"
PSEUDO_ROOT="${PSEUDO_ROOT:-outputs/code_upsampler}"
LOG_DIR="${LOG_DIR:-outputs/code_upsampler/logs/next_ablation_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="${SUMMARY_CSV:-outputs/code_upsampler/next_ablation_summary.csv}"

DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
LIMIT_IMAGES="${LIMIT_IMAGES:-0}"
VAL_LIMIT_IMAGES="${VAL_LIMIT_IMAGES:-0}"
K="${K:-80}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.025}"
INFER_BATCH="${INFER_BATCH:-8}"
KMEANS_BATCH="${KMEANS_BATCH:-4096}"
N_INIT="${N_INIT:-5}"
MAX_ITER="${MAX_ITER:-100}"
LIMIT_TRAIN="${LIMIT_TRAIN:-0}"
LIMIT_ASSIGN="${LIMIT_ASSIGN:-0}"

mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_CSV")"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "run_name,variant,mIoU,mIoU_stuff,mIoU_things,pixel_accuracy,eval_json" > "$SUMMARY_CSV"
fi

append_summary() {
  local run_name="$1"
  local variant="$2"
  local eval_json="$3"
  python3 - "$run_name" "$variant" "$eval_json" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
run_name, variant, eval_json, summary_csv = sys.argv[1:5]
with open(eval_json) as f:
    data = json.load(f)
row = {
    "run_name": run_name,
    "variant": variant,
    "mIoU": data["mIoU"],
    "mIoU_stuff": data["mIoU_stuff"],
    "mIoU_things": data["mIoU_things"],
    "pixel_accuracy": data["pixel_accuracy"],
    "eval_json": eval_json,
}
with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
PY
}

run_one() {
  local run_name="$1"
  local variant="$2"
  shift 2
  local pseudo_dir="${PSEUDO_ROOT}/pseudo_semantic_${run_name}"
  local eval_json="${PSEUDO_ROOT}/${run_name}_cityscapes27.json"

  echo "=== Training ${run_name} (${variant}) ==="
  python3 mbps_pytorch/train_code_upsampler.py \
    --variant "$variant" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$RUN_ROOT" \
    --run_name "$run_name" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --limit_images "$LIMIT_IMAGES" \
    --val_limit_images "$VAL_LIMIT_IMAGES" \
    "$@" 2>&1 | tee "$LOG_DIR/${run_name}_train.log"

  echo "=== Generating K=${K} val clusters for ${run_name} ==="
  python3 mbps_pytorch/generate_code_upsampler_kmeans.py \
    --cache_dir "$CACHE_DIR" \
    --checkpoint "${RUN_ROOT}/${run_name}/best.pt" \
    --output_dir "$pseudo_dir" \
    --k "$K" \
    --sample_frac "$SAMPLE_FRAC" \
    --inference_batch_size "$INFER_BATCH" \
    --kmeans_batch_size "$KMEANS_BATCH" \
    --n_init "$N_INIT" \
    --max_iter "$MAX_ITER" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --limit_train "$LIMIT_TRAIN" \
    --limit_assign "$LIMIT_ASSIGN" 2>&1 | tee "$LOG_DIR/${run_name}_kmeans.log"

  echo "=== Evaluating ${run_name} on strict Cityscapes-27 ==="
  python3 mbps_pytorch/evaluate_cityscapes27_clusters.py \
    --pred_dir "${pseudo_dir}/val" \
    --gt_dir "$GT_DIR" \
    --num_clusters "$K" \
    --output "$eval_json" 2>&1 | tee "$LOG_DIR/${run_name}_eval.log"

  append_summary "$run_name" "$variant" "$eval_json"
  echo "=== Done ${run_name}; summary appended to ${SUMMARY_CSV} ==="
}

run_one "jafar_attn_90d_dcfa_v3_h64w128_k80_seed${SEED}" "attentive" \
  --use_coords \
  --coord_freqs 4 \
  --guidance_hidden 48 \
  --hidden_ch 64 \
  --num_blocks 2 \
  --attn_dim 64 \
  --attn_window 5 \
  --residual_scale 0.15

run_one "loftup_coord_mask_90d_dcfa_v3_h64w128_k80_seed${SEED}" "dynamic" \
  --use_coords \
  --coord_freqs 4 \
  --lambda_mask 0.05 \
  --mask_edge_alpha 12.0 \
  --mask_teacher_temp 10.0 \
  --mask_neg_margin 0.25

run_one "anyup_crop_teacher_90d_dcfa_v3_h64w128_k80_seed${SEED}" "dynamic" \
  --crop_teacher \
  --crop_h 32 \
  --crop_w 64

run_one "neco_neighbor_90d_dcfa_v3_h64w128_k80_seed${SEED}" "dynamic" \
  --lambda_neco 0.02 \
  --neco_samples 256 \
  --neco_tau 0.1

echo "All ablations completed."
echo "Logs: ${LOG_DIR}"
echo "Summary: ${SUMMARY_CSV}"
