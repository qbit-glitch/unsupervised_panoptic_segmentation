#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Mobile Panoptic Ablation Chain (Instance Heads Only)
# Runs instance head ablations sequentially on a single GPU.
# Augmented baseline already completed (PQ=23.73).
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_mobile_ablations.sh &
# ═══════════════════════════════════════════════════════════════════════════════

# Do NOT use set -e — each run should continue even if a previous one fails

export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

PYTHON=/home/santosh/anaconda3/envs/cups/bin/python
ROOT=/media/santosh/Kuldeep/panoptic_segmentation
CS_ROOT=$ROOT/datasets/cityscapes
SCRIPT=$ROOT/mbps_pytorch/train_mobile_panoptic.py
LOG_DIR=$ROOT/logs
CKPT_DIR=$ROOT/checkpoints

mkdir -p $LOG_DIR

# Common args
COMMON="--cityscapes_root $CS_ROOT \
  --semantic_subdir pseudo_semantic_mapped_k80 \
  --backbone repvit_m0_9.dist_450e_in1k \
  --num_classes 19 --fpn_dim 128 \
  --num_epochs 50 --batch_size 4 --lr 1e-3 \
  --backbone_lr_ratio 0.1 --freeze_backbone_epochs 5 \
  --crop_h 384 --crop_w 768 --eval_interval 2 \
  --label_smoothing 0.1 --num_workers 4"

echo "════════════════════════════════════════════════════════════════"
echo "  MOBILE PANOPTIC ABLATION CHAIN (Instance Heads)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Started: $(date)"
echo "  Skipping Run 1 (augmented baseline already done, PQ=23.73)"
echo "════════════════════════════════════════════════════════════════"

# ─── Run 1/5: I-A Embedding Head ──────────────────────────────────────────
echo ""
echo "═══ [1/5] I-A: EMBEDDING HEAD ═══"
echo "Started: $(date)"
$PYTHON $SCRIPT $COMMON \
  --augmentation full \
  --instance_head embedding \
  --instance_subdir instance_targets \
  --lambda_instance 1.0 \
  --output_dir $CKPT_DIR/mobile_instance_embedding \
  2>&1 | tee $LOG_DIR/mobile_instance_embedding.log
echo "[1/5] I-A Embedding EXIT CODE: $? — $(date)"

# ─── Run 2/5: I-B Center/Offset Head ─────────────────────────────────────
echo ""
echo "═══ [2/5] I-B: CENTER/OFFSET HEAD ═══"
echo "Started: $(date)"
$PYTHON $SCRIPT $COMMON \
  --augmentation full \
  --instance_head center_offset \
  --instance_subdir instance_targets \
  --lambda_instance 1.0 \
  --output_dir $CKPT_DIR/mobile_instance_center_offset \
  2>&1 | tee $LOG_DIR/mobile_instance_center_offset.log
echo "[2/5] I-B Center/Offset EXIT CODE: $? — $(date)"

# ─── Run 3/5: I-C Boundary Head ──────────────────────────────────────────
echo ""
echo "═══ [3/5] I-C: BOUNDARY HEAD ═══"
echo "Started: $(date)"
$PYTHON $SCRIPT $COMMON \
  --augmentation full \
  --instance_head boundary \
  --instance_subdir instance_targets \
  --lambda_instance 1.0 \
  --output_dir $CKPT_DIR/mobile_instance_boundary \
  2>&1 | tee $LOG_DIR/mobile_instance_boundary.log
echo "[3/5] I-C Boundary EXIT CODE: $? — $(date)"

# ─── Run 4/5: I-BC Center + Boundary ─────────────────────────────────────
echo ""
echo "═══ [4/5] I-BC: CENTER/OFFSET + BOUNDARY ═══"
echo "Started: $(date)"
$PYTHON $SCRIPT $COMMON \
  --augmentation full \
  --instance_head center_boundary \
  --instance_subdir instance_targets \
  --lambda_instance 1.0 \
  --output_dir $CKPT_DIR/mobile_instance_center_boundary \
  2>&1 | tee $LOG_DIR/mobile_instance_center_boundary.log
echo "[4/5] I-BC Center+Boundary EXIT CODE: $? — $(date)"

# ─── Run 5/5: I-ABC All Heads ────────────────────────────────────────────
echo ""
echo "═══ [5/5] I-ABC: ALL HEADS (embedding + center/offset + boundary) ═══"
echo "Started: $(date)"
$PYTHON $SCRIPT $COMMON \
  --augmentation full \
  --instance_head all \
  --instance_subdir instance_targets \
  --lambda_instance 1.0 \
  --output_dir $CKPT_DIR/mobile_instance_all \
  2>&1 | tee $LOG_DIR/mobile_instance_all.log
echo "[5/5] I-ABC All Heads EXIT CODE: $? — $(date)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL 5 INSTANCE HEAD ABLATIONS COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Results summary — check best.pth in each checkpoint dir:"
echo "    $CKPT_DIR/mobile_augmented_baseline/     (already done, PQ=23.73)"
echo "    $CKPT_DIR/mobile_instance_embedding/"
echo "    $CKPT_DIR/mobile_instance_center_offset/"
echo "    $CKPT_DIR/mobile_instance_boundary/"
echo "    $CKPT_DIR/mobile_instance_center_boundary/"
echo "    $CKPT_DIR/mobile_instance_all/"
echo "════════════════════════════════════════════════════════════════"
