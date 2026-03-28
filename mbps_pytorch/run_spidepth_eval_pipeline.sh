#!/bin/bash
# Full SPIdepth instance pseudo-label evaluation pipeline
# Uses CAUSE-TR semantic pseudo-labels + SPIdepth depth-guided instances
#
# Stages:
#   1. Generate depth-guided instances (val + train)
#   2a. Evaluate instance pseudo-labels WITHOUT semantic (instance AR/AP only)
#   2b. Evaluate WITH CAUSE-TR semantic labels (full panoptic PQ)
#   3. Compute unsupervised stuff-things classifier
#   4. Regenerate instances + evaluate with unsupervised stuff-things split
#
# Usage: bash mbps_pytorch/run_spidepth_eval_pipeline.sh

set -e

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
CS_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
LOG_DIR="logs"
DEPTH_DIR="$CS_ROOT/depth_spidepth"
SEM_DIR="$CS_ROOT/pseudo_semantic_cause_trainid"
INSTANCE_OUT="$CS_ROOT/pseudo_instance_spidepth"
INSTANCE_OUT_UNSUP="$CS_ROOT/pseudo_instance_spidepth_unsup_st"
STUFF_THINGS_OUT="$CS_ROOT/pseudo_semantic_cause_trainid/stuff_things_spidepth.json"

# Relative subdirs (for evaluate_cascade_pseudolabels.py — it appends /{split}/)
SEM_SUBDIR="pseudo_semantic_cause_trainid"
INST_SUBDIR="pseudo_instance_spidepth"
INST_SUBDIR_UNSUP="pseudo_instance_spidepth_unsup_st"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "SPIdepth Instance Pseudo-Label Pipeline"
echo "=========================================="
echo "Depth dir:    $DEPTH_DIR"
echo "Semantic dir: $SEM_DIR"
echo "Instance out: $INSTANCE_OUT"
echo ""

# ============================================
# Stage 1a: Generate depth-guided instances (val)
# Uses default Cityscapes GT thing IDs (trainID 11-18)
# ============================================
echo "--- Stage 1a: Generate depth-guided instances (val) ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/generate_depth_guided_instances.py \
    --semantic_dir "$SEM_DIR/val" \
    --depth_dir "$DEPTH_DIR/val" \
    --output_dir "$INSTANCE_OUT/val" \
    --grad_threshold 0.05 \
    --min_area 100 \
    --dilation_iters 3 \
    --depth_blur 1.0 \
    2>&1 | tee "$LOG_DIR/spidepth_instance_gen_val.log"

echo ""
echo "--- Stage 1b: Generate depth-guided instances (train) ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/generate_depth_guided_instances.py \
    --semantic_dir "$SEM_DIR/train" \
    --depth_dir "$DEPTH_DIR/train" \
    --output_dir "$INSTANCE_OUT/train" \
    --grad_threshold 0.05 \
    --min_area 100 \
    --dilation_iters 3 \
    --depth_blur 1.0 \
    2>&1 | tee "$LOG_DIR/spidepth_instance_gen_train.log"

# ============================================
# Stage 2a: Evaluate instance pseudo-labels WITHOUT semantic
# Only computes instance AR/AP metrics (no semantic, no panoptic)
# ============================================
echo ""
echo "--- Stage 2a: Evaluate instances WITHOUT semantic labels ---"
echo "(Instance AR@100, AP@50, AP@75 only)"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --semantic_subdir "$SEM_SUBDIR" \
    --instance_subdir "$INST_SUBDIR" \
    --skip_semantic \
    --skip_panoptic \
    --output "$INSTANCE_OUT/eval_instance_only.json" \
    2>&1 | tee "$LOG_DIR/spidepth_eval_instance_only.log"

# ============================================
# Stage 2b: Evaluate WITH CAUSE-TR semantic labels (full panoptic)
# Semantic mIoU + Panoptic PQ using depth-guided thing instances
# ============================================
echo ""
echo "--- Stage 2b: Evaluate WITH CAUSE-TR semantic labels (panoptic) ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --semantic_subdir "$SEM_SUBDIR" \
    --instance_subdir "$INST_SUBDIR" \
    --thing_mode maskcut \
    --output "$INSTANCE_OUT/eval_panoptic_with_semantic.json" \
    2>&1 | tee "$LOG_DIR/spidepth_eval_panoptic.log"

# ============================================
# Stage 3: Compute unsupervised stuff-things classifier
# Uses depth-edge statistics to classify 19 classes as stuff or things
# ============================================
echo ""
echo "--- Stage 3: Compute stuff-things classification ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/classify_stuff_things.py \
    --semantic_dir "$SEM_DIR/val" \
    --depth_dir "$DEPTH_DIR/val" \
    --output_path "$STUFF_THINGS_OUT" \
    --num_classes 19 \
    --n_things 11 \
    2>&1 | tee "$LOG_DIR/spidepth_stuff_things.log"

# ============================================
# Stage 4a: Regenerate instances with unsupervised stuff-things split
# ============================================
echo ""
echo "--- Stage 4a: Regenerate instances with unsupervised stuff-things ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/generate_depth_guided_instances.py \
    --semantic_dir "$SEM_DIR/val" \
    --depth_dir "$DEPTH_DIR/val" \
    --output_dir "$INSTANCE_OUT_UNSUP/val" \
    --stuff_things "$STUFF_THINGS_OUT" \
    --grad_threshold 0.05 \
    --min_area 100 \
    --dilation_iters 3 \
    --depth_blur 1.0 \
    2>&1 | tee "$LOG_DIR/spidepth_instance_gen_unsup_st.log"

# ============================================
# Stage 4b: Evaluate with unsupervised stuff-things split
# ============================================
echo ""
echo "--- Stage 4b: Evaluate panoptic with unsupervised stuff-things ---"
PYTHONUNBUFFERED=1 $PYTHON mbps_pytorch/evaluate_cascade_pseudolabels.py \
    --cityscapes_root "$CS_ROOT" \
    --split val \
    --semantic_subdir "$SEM_SUBDIR" \
    --instance_subdir "$INST_SUBDIR_UNSUP" \
    --thing_mode maskcut \
    --stuff_things "$STUFF_THINGS_OUT" \
    --output "$INSTANCE_OUT_UNSUP/eval_panoptic_unsup_st.json" \
    2>&1 | tee "$LOG_DIR/spidepth_eval_panoptic_unsup_st.log"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results:"
echo "  Instance only:               $INSTANCE_OUT/eval_instance_only.json"
echo "  Panoptic (GT stuff-things):  $INSTANCE_OUT/eval_panoptic_with_semantic.json"
echo "  Stuff-things classifier:     $STUFF_THINGS_OUT"
echo "  Panoptic (unsup stuff-things): $INSTANCE_OUT_UNSUP/eval_panoptic_unsup_st.json"
