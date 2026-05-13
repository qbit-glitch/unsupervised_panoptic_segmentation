#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Panoptic-DeepLab + RepViT Ablation Chain
#
# Three ablation groups:
#   Group 1: Architecture Ablation (4 runs) — compare architectures with full CUPS
#   Group 2: Training Recipe Ablation (6 runs) — isolate CUPS components on best arch
#   Group 3: FPN Ablation (3 runs) — compare FPN variants on best arch
#
# Total: 13 runs, ~65 GPU-hours on GTX 1080 Ti
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_panoptic_deeplab_ablations.sh [group] &
#
#   group = "arch" | "recipe" | "fpn" | "all" (default: all)
# ═══════════════════════════════════════════════════════════════════════════════

export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

PYTHON=/home/santosh/anaconda3/envs/cups/bin/python
ROOT=/media/santosh/Kuldeep/panoptic_segmentation
CS_ROOT=$ROOT/datasets/cityscapes
SCRIPT=$ROOT/mbps_pytorch/train_panoptic_deeplab.py
LOG_DIR=$ROOT/logs
CKPT_DIR=$ROOT/checkpoints

mkdir -p $LOG_DIR

# Common args for ALL runs
COMMON="--cityscapes_root $CS_ROOT \
  --semantic_subdir pseudo_semantic_mapped_k80 \
  --instance_subdir instance_targets \
  --backbone repvit_m0_9.dist_450e_in1k \
  --num_classes 19 --fpn_dim 128 \
  --num_epochs 50 --batch_size 4 \
  --freeze_backbone_epochs 5 --backbone_lr_ratio 0.1 \
  --crop_h 384 --crop_w 768 --eval_interval 2 \
  --label_smoothing 0.1 --num_workers 4"

# CUPS Stage-2 + Stage-3 args
# LR=1e-3 matches baseline (1e-4 was too low for 5.42M model)
# lambda_instance=1.0 restored (cascade scaling handles balance)
CUPS_FULL="--cups_stage2 --cups_stage3 \
  --lr 1e-3 --weight_decay 1e-4 \
  --lambda_instance 1.0 --grad_clip 1.0 \
  --st_rounds 3 --st_epochs_per_round 5 \
  --st_tta --ema_momentum 0.999"

GROUP=${1:-all}

echo "════════════════════════════════════════════════════════════════"
echo "  PANOPTIC-DEEPLAB + RepViT ABLATION SUITE"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Group: $GROUP"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1: ARCHITECTURE ABLATION (4 runs)
# Compare architectures with identical CUPS training recipe
# ═══════════════════════════════════════════════════════════════════════════════

run_arch_ablations() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  GROUP 1: ARCHITECTURE ABLATION (4 runs)                   ║"
  echo "╚══════════════════════════════════════════════════════════════╝"

  # ─── A-1: Panoptic-DeepLab + BiFPN ───
  echo ""
  echo "═══ [A-1] Panoptic-DeepLab + BiFPN + Full CUPS ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $CUPS_FULL \
    --arch panoptic_deeplab --fpn_type bifpn \
    --output_dir $CKPT_DIR/pdl_arch_panoptic_deeplab \
    2>&1 | tee $LOG_DIR/pdl_arch_panoptic_deeplab.log
  echo "[A-1] Panoptic-DeepLab EXIT CODE: $? — $(date)"

  # ─── A-2: kMaX-DeepLab + BiFPN ───
  echo ""
  echo "═══ [A-2] kMaX-DeepLab + BiFPN + Full CUPS ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $CUPS_FULL \
    --arch kmax_deeplab --fpn_type bifpn \
    --output_dir $CKPT_DIR/pdl_arch_kmax_deeplab \
    2>&1 | tee $LOG_DIR/pdl_arch_kmax_deeplab.log
  echo "[A-2] kMaX-DeepLab EXIT CODE: $? — $(date)"

  # ─── A-3: MaskConver + BiFPN ───
  echo ""
  echo "═══ [A-3] MaskConver + BiFPN + Full CUPS ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $CUPS_FULL \
    --arch maskconver --fpn_type bifpn \
    --output_dir $CKPT_DIR/pdl_arch_maskconver \
    2>&1 | tee $LOG_DIR/pdl_arch_maskconver.log
  echo "[A-3] MaskConver EXIT CODE: $? — $(date)"

  # ─── A-4: Mask2Former-Lite + BiFPN ───
  echo ""
  echo "═══ [A-4] Mask2Former-Lite + BiFPN + Full CUPS ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $CUPS_FULL \
    --arch mask2former_lite --fpn_type bifpn \
    --output_dir $CKPT_DIR/pdl_arch_mask2former_lite \
    2>&1 | tee $LOG_DIR/pdl_arch_mask2former_lite.log
  echo "[A-4] Mask2Former-Lite EXIT CODE: $? — $(date)"

  echo ""
  echo "GROUP 1 COMPLETE: $(date)"
}


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2: TRAINING RECIPE ABLATION (6 runs)
# On best architecture (Panoptic-DeepLab), isolate each CUPS component
# ═══════════════════════════════════════════════════════════════════════════════

run_recipe_ablations() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  GROUP 2: TRAINING RECIPE ABLATION (6 runs)                ║"
  echo "╚══════════════════════════════════════════════════════════════╝"

  ARCH_COMMON="--arch panoptic_deeplab --fpn_type bifpn"

  # ─── T-0: Baseline (CE only, no CUPS) ───
  echo ""
  echo "═══ [T-0] BASELINE (CE only, no CUPS tricks) ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $ARCH_COMMON \
    --lr 1e-3 --weight_decay 1e-4 \
    --lambda_instance 1.0 \
    --output_dir $CKPT_DIR/pdl_recipe_baseline \
    2>&1 | tee $LOG_DIR/pdl_recipe_baseline.log
  echo "[T-0] Baseline EXIT CODE: $? — $(date)"

  # ─── T-1: + DropLoss only ───
  echo ""
  echo "═══ [T-1] + DropLoss ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $ARCH_COMMON \
    --cups_stage2 \
    --no_copy_paste --no_resolution_jitter --no_ignore_unknown \
    --lr 1e-3 --weight_decay 1e-4 \
    --lambda_instance 1.0 \
    --output_dir $CKPT_DIR/pdl_recipe_droploss \
    2>&1 | tee $LOG_DIR/pdl_recipe_droploss.log
  echo "[T-1] +DropLoss EXIT CODE: $? — $(date)"

  # ─── T-2: + Copy-paste only ───
  echo ""
  echo "═══ [T-2] + Copy-Paste ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $ARCH_COMMON \
    --cups_stage2 \
    --no_droploss --no_resolution_jitter --no_ignore_unknown \
    --lr 1e-3 --weight_decay 1e-4 \
    --lambda_instance 1.0 \
    --output_dir $CKPT_DIR/pdl_recipe_copypaste \
    2>&1 | tee $LOG_DIR/pdl_recipe_copypaste.log
  echo "[T-2] +Copy-Paste EXIT CODE: $? — $(date)"

  # ─── T-3: + Resolution jitter only ───
  echo ""
  echo "═══ [T-3] + Resolution Jitter ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $ARCH_COMMON \
    --cups_stage2 \
    --no_droploss --no_copy_paste --no_ignore_unknown \
    --lr 1e-3 --weight_decay 1e-4 \
    --lambda_instance 1.0 \
    --output_dir $CKPT_DIR/pdl_recipe_resjitter \
    2>&1 | tee $LOG_DIR/pdl_recipe_resjitter.log
  echo "[T-3] +ResJitter EXIT CODE: $? — $(date)"

  # ─── T-4: All stage-2 tricks ───
  echo ""
  echo "═══ [T-4] All Stage-2 Tricks ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $ARCH_COMMON \
    --cups_stage2 \
    --lr 1e-3 --weight_decay 1e-4 \
    --lambda_instance 1.0 \
    --output_dir $CKPT_DIR/pdl_recipe_stage2_full \
    2>&1 | tee $LOG_DIR/pdl_recipe_stage2_full.log
  echo "[T-4] Stage-2 Full EXIT CODE: $? — $(date)"

  # ─── T-5: + Stage-3 self-training ───
  echo ""
  echo "═══ [T-5] + Stage-3 Self-Training ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $ARCH_COMMON $CUPS_FULL \
    --output_dir $CKPT_DIR/pdl_recipe_full_cups \
    2>&1 | tee $LOG_DIR/pdl_recipe_full_cups.log
  echo "[T-5] Full CUPS EXIT CODE: $? — $(date)"

  echo ""
  echo "GROUP 2 COMPLETE: $(date)"
}


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3: FPN ABLATION (3 runs)
# On best architecture (Panoptic-DeepLab), compare FPN types
# ═══════════════════════════════════════════════════════════════════════════════

run_fpn_ablations() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  GROUP 3: FPN ABLATION (3 runs)                            ║"
  echo "╚══════════════════════════════════════════════════════════════╝"

  # ─── F-1: SimpleFPN ───
  echo ""
  echo "═══ [F-1] SimpleFPN ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $CUPS_FULL \
    --arch panoptic_deeplab --fpn_type simple \
    --output_dir $CKPT_DIR/pdl_fpn_simple \
    2>&1 | tee $LOG_DIR/pdl_fpn_simple.log
  echo "[F-1] SimpleFPN EXIT CODE: $? — $(date)"

  # ─── F-2: BiFPN (same as A-1, skip if already run) ───
  echo ""
  echo "═══ [F-2] BiFPN (reference: same as A-1) ═══"
  if [ -f "$CKPT_DIR/pdl_arch_panoptic_deeplab/best.pth" ]; then
    echo "SKIPPED: A-1 already has BiFPN results"
  else
    echo "Started: $(date)"
    $PYTHON $SCRIPT $COMMON $CUPS_FULL \
      --arch panoptic_deeplab --fpn_type bifpn \
      --output_dir $CKPT_DIR/pdl_fpn_bifpn \
      2>&1 | tee $LOG_DIR/pdl_fpn_bifpn.log
    echo "[F-2] BiFPN EXIT CODE: $? — $(date)"
  fi

  # ─── F-3: PANet ───
  echo ""
  echo "═══ [F-3] PANet ═══"
  echo "Started: $(date)"
  $PYTHON $SCRIPT $COMMON $CUPS_FULL \
    --arch panoptic_deeplab --fpn_type panet \
    --output_dir $CKPT_DIR/pdl_fpn_panet \
    2>&1 | tee $LOG_DIR/pdl_fpn_panet.log
  echo "[F-3] PANet EXIT CODE: $? — $(date)"

  echo ""
  echo "GROUP 3 COMPLETE: $(date)"
}


# ═══════════════════════════════════════════════════════════════════════════════
# Run selected group(s)
# ═══════════════════════════════════════════════════════════════════════════════

case "$GROUP" in
  arch)
    run_arch_ablations
    ;;
  recipe)
    run_recipe_ablations
    ;;
  fpn)
    run_fpn_ablations
    ;;
  all)
    run_arch_ablations
    run_recipe_ablations
    run_fpn_ablations
    ;;
  *)
    echo "Unknown group: $GROUP"
    echo "Usage: $0 [arch|recipe|fpn|all]"
    exit 1
    ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ABLATION SUITE COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Results summary — check best.pth in each checkpoint dir:"
echo ""
echo "  Architecture ablation:"
echo "    $CKPT_DIR/pdl_arch_panoptic_deeplab/"
echo "    $CKPT_DIR/pdl_arch_kmax_deeplab/"
echo "    $CKPT_DIR/pdl_arch_maskconver/"
echo "    $CKPT_DIR/pdl_arch_mask2former_lite/"
echo ""
echo "  Recipe ablation:"
echo "    $CKPT_DIR/pdl_recipe_baseline/"
echo "    $CKPT_DIR/pdl_recipe_droploss/"
echo "    $CKPT_DIR/pdl_recipe_copypaste/"
echo "    $CKPT_DIR/pdl_recipe_resjitter/"
echo "    $CKPT_DIR/pdl_recipe_stage2_full/"
echo "    $CKPT_DIR/pdl_recipe_full_cups/"
echo ""
echo "  FPN ablation:"
echo "    $CKPT_DIR/pdl_fpn_simple/"
echo "    $CKPT_DIR/pdl_fpn_bifpn/ (or pdl_arch_panoptic_deeplab/)"
echo "    $CKPT_DIR/pdl_fpn_panet/"
echo "════════════════════════════════════════════════════════════════"
