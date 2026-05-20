#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
CITY="${CITYSCAPES_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
SEM="pseudo_semantic_raw_k80"
CENTROIDS="$CITY/pseudo_semantic_raw_k80/kmeans_centroids.npz"
DEPTH="depth_depthpro"
FEAT="dinov3_features_vitl16"
TRAIN_IMAGES="${TRAIN_IMAGES:-298}"
VAL_IMAGES="${VAL_IMAGES:-50}"
EVAL_IMAGES="${EVAL_IMAGES:-$VAL_IMAGES}"
EPOCHS="${EPOCHS:-10}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="logs/superpixel_depth_dino_ablations_${STAMP}.log"

mkdir -p logs checkpoints results
OVERALL_STATUS=0

log_msg() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$MASTER_LOG"
}

run_one() {
  local name="$1"
  local merge="$2"
  local min_area="$3"
  local class_min_area="$4"
  shift 4

  local ckpt_dir="checkpoints/${name}"
  local inst_dir="results/${name}_instances"
  local eval_json="results/${name}_eval${EVAL_IMAGES}.json"
  local train_log="logs/${name}.train.log"
  local gen_log="logs/${name}.generate.log"
  local eval_log="logs/${name}.eval.log"

  log_msg "START ${name}"
  log_msg "checkpoint=${ckpt_dir}"
  log_msg "subset=train:${TRAIN_IMAGES} val:${VAL_IMAGES} eval:${EVAL_IMAGES} epochs:${EPOCHS}"
  "$PY" mbps_pytorch/train_superpixel_affinity_adapter.py \
    --cityscapes_root "$CITY" \
    --semantic_subdir "$SEM" \
    --centroids_path "$CENTROIDS" \
    --depth_subdir "$DEPTH" \
    --feature_subdir "$FEAT" \
    --output_dir "$ckpt_dir" \
    --pseudo_tau 0.20 \
    --pseudo_min_area 1000 \
    --max_train_images "$TRAIN_IMAGES" \
    --max_val_images "$VAL_IMAGES" \
    --epochs "$EPOCHS" \
    --batch_size 4096 \
    --default_merge_threshold "$merge" \
    --output_min_area "$min_area" \
    "$@" > "$train_log" 2>&1
  local train_status=$?
  log_msg "TRAIN_DONE ${name} status=${train_status}"
  if [[ "$train_status" -ne 0 ]]; then
    return "$train_status"
  fi

  "$PY" mbps_pytorch/generate_superpixel_affinity_instances.py \
    --cityscapes_root "$CITY" \
    --checkpoint "$ckpt_dir/best.pth" \
    --output_dir "$inst_dir" \
    --split val \
    --max_images "$EVAL_IMAGES" \
    --merge_threshold "$merge" \
    --min_area "$min_area" \
    --class_min_area "$class_min_area" > "$gen_log" 2>&1
  local gen_status=$?
  log_msg "GENERATE_DONE ${name} status=${gen_status}"
  if [[ "$gen_status" -ne 0 ]]; then
    return "$gen_status"
  fi

  "$PY" mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CITY" \
    --split val \
    --max_images "$EVAL_IMAGES" \
    --semantic_subdir "$SEM" \
    --instance_subdir "$ROOT/$inst_dir" \
    --num_clusters 80 \
    --cluster_mapping_path "$CENTROIDS" \
    --thing_mode maskcut \
    --output "$eval_json" > "$eval_log" 2>&1
  local eval_status=$?
  log_msg "EVAL_DONE ${name} status=${eval_status} output=${eval_json}"
  return "$eval_status"
}

run_one \
  "superpixel_affinity_a1_classcut10pct_k80_depthpro_dinov3" \
  0.55 \
  600 \
  "11:300,12:300,18:300" \
  --n_segments 900 \
  --max_edges_per_image 2200 \
  --positive_affinity_min 0.58 \
  --negative_affinity_max 0.32 \
  --negative_boundary_min 0.055 \
  --intra_instance_negative_affinity_max 0.42 \
  --intra_instance_negative_boundary_min 0.055 \
  --intra_instance_negative_weight 1.50 \
  --class_aware_negative_classes 11,12,18 \
  --class_aware_positive_affinity_min 0.62 \
  --class_aware_negative_weight 2.00 \
  --class_aware_positive_weight 1.00 \
  --class_aware_intra_instance_negative_affinity_max 0.48 \
  --class_aware_intra_instance_negative_boundary_min 0.045
run_status=$?
if [[ "$run_status" -ne 0 ]]; then
  OVERALL_STATUS="$run_status"
fi

run_one \
  "superpixel_affinity_a2_overseg1200_10pct_k80_depthpro_dinov3" \
  0.65 \
  450 \
  "11:250,12:250,18:250" \
  --n_segments 1200 \
  --min_superpixel_area 8 \
  --max_edges_per_image 3200 \
  --positive_affinity_min 0.55 \
  --negative_affinity_max 0.35 \
  --negative_boundary_min 0.045 \
  --intra_instance_negative_affinity_max 0.45 \
  --intra_instance_negative_boundary_min 0.045 \
  --intra_instance_negative_weight 1.25 \
  --class_aware_negative_classes 11,12,18 \
  --class_aware_negative_weight 1.50 \
  --class_aware_intra_instance_negative_affinity_max 0.50 \
  --class_aware_intra_instance_negative_boundary_min 0.040
run_status=$?
if [[ "$run_status" -ne 0 ]]; then
  OVERALL_STATUS="$run_status"
fi

run_one \
  "superpixel_affinity_a3_depthstrict10pct_k80_depthpro_dinov3" \
  0.60 \
  600 \
  "11:300,12:300,18:300" \
  --n_segments 900 \
  --max_edges_per_image 2400 \
  --sigma_depth 0.025 \
  --sigma_color 0.10 \
  --dino_temperature 0.15 \
  --positive_affinity_min 0.60 \
  --negative_affinity_max 0.30 \
  --negative_boundary_min 0.050 \
  --intra_instance_negative_affinity_max 0.38 \
  --intra_instance_negative_boundary_min 0.050 \
  --intra_instance_negative_weight 1.35 \
  --class_aware_negative_classes 11,12,18 \
  --class_aware_positive_affinity_min 0.64 \
  --class_aware_negative_weight 1.75 \
  --class_aware_intra_instance_negative_affinity_max 0.45 \
  --class_aware_intra_instance_negative_boundary_min 0.045
run_status=$?
if [[ "$run_status" -ne 0 ]]; then
  OVERALL_STATUS="$run_status"
fi

log_msg "ALL_DONE status=${OVERALL_STATUS}"
exit "$OVERALL_STATUS"
