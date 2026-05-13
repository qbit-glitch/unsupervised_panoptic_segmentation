#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# CUPS Semantic-Only Ablation (4 focused runs)
#
# All runs use semantic-only pipeline (--instance_head none) with pre-computed
# pseudo instances for PQ scoring. No center/offset learning.
#
# R1: BiFPN + Full CUPS (stage-2 + stage-3)   — best expected
# R2: BiFPN + No CUPS (augmented baseline)    — measures CUPS contribution
# R3: BiFPN + Stage-2 only (no self-training) — measures stage-3 value
# R4: SimpleFPN + Full CUPS                   — measures BiFPN value
#
# Baseline reference: PQ=23.73 (SimpleFPN + full augmentation, no CUPS)
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_cups_semantic_ablations.sh [run] &
#
#   run = "r1" | "r2" | "r3" | "r4" | "all" (default: all)
# ═══════════════════════════════════════════════════════════════════════════════

export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

PYTHON=/home/santosh/anaconda3/envs/cups/bin/python
ROOT=/media/santosh/Kuldeep/panoptic_segmentation
CS_ROOT=$ROOT/datasets/cityscapes
SCRIPT=$ROOT/mbps_pytorch/train_mobile_panoptic.py
LOG_DIR=$ROOT/logs
CKPT_DIR=$ROOT/checkpoints

mkdir -p $LOG_DIR

# Common args for ALL runs
COMMON="--cityscapes_root $CS_ROOT \
  --semantic_subdir pseudo_semantic_mapped_k80 \
  --instance_subdir instance_targets \
  --backbone repvit_m0_9.dist_450e_in1k \
  --num_classes 19 --fpn_dim 128 \
  --instance_head none \
  --num_epochs 50 --batch_size 4 \
  --lr 1e-3 --weight_decay 1e-4 \
  --freeze_backbone_epochs 5 --backbone_lr_ratio 0.1 \
  --crop_h 384 --crop_w 768 --eval_interval 2 \
  --label_smoothing 0.1 --num_workers 4 \
  --grad_clip 1.0"

RUN=${1:-all}

echo "════════════════════════════════════════════════════════════════"
echo "  CUPS SEMANTIC-ONLY ABLATION SUITE"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Run: $RUN"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"


# ─── R1: BiFPN + Full CUPS (stage-2 + stage-3) ──────────────────────────────
run_r1() {
  echo ""
  echo "═══ [R1] BiFPN + Full CUPS (stage-2 + stage-3) ═══"
  echo "Expected: PQ 26-28 | Started: $(date)"
  $PYTHON $SCRIPT $COMMON \
    --fpn_type bifpn --augmentation cups \
    --cups_stage2 --cups_stage3 \
    --st_rounds 3 --st_epochs_per_round 5 --st_tta \
    --ema_momentum 0.999 \
    --output_dir $CKPT_DIR/cups_sem_bifpn_full \
    2>&1 | tee $LOG_DIR/cups_sem_bifpn_full.log
  echo "[R1] BiFPN+FullCUPS EXIT CODE: $? — $(date)"
}

# ─── R2: BiFPN + No CUPS (augmented baseline) ───────────────────────────────
run_r2() {
  echo ""
  echo "═══ [R2] BiFPN + No CUPS (augmented baseline) ═══"
  echo "Expected: PQ 24-25 | Started: $(date)"
  $PYTHON $SCRIPT $COMMON \
    --fpn_type bifpn --augmentation full \
    --output_dir $CKPT_DIR/cups_sem_bifpn_baseline \
    2>&1 | tee $LOG_DIR/cups_sem_bifpn_baseline.log
  echo "[R2] BiFPN+Baseline EXIT CODE: $? — $(date)"
}

# ─── R3: BiFPN + Stage-2 only (no self-training) ────────────────────────────
run_r3() {
  echo ""
  echo "═══ [R3] BiFPN + Stage-2 only (no stage-3) ═══"
  echo "Expected: PQ 25-27 | Started: $(date)"
  $PYTHON $SCRIPT $COMMON \
    --fpn_type bifpn --augmentation cups \
    --cups_stage2 \
    --output_dir $CKPT_DIR/cups_sem_bifpn_stage2 \
    2>&1 | tee $LOG_DIR/cups_sem_bifpn_stage2.log
  echo "[R3] BiFPN+Stage2 EXIT CODE: $? — $(date)"
}

# ─── R4: SimpleFPN + Full CUPS ───────────────────────────────────────────────
run_r4() {
  echo ""
  echo "═══ [R4] SimpleFPN + Full CUPS ═══"
  echo "Expected: PQ 25-27 | Started: $(date)"
  $PYTHON $SCRIPT $COMMON \
    --fpn_type simple --augmentation cups \
    --cups_stage2 --cups_stage3 \
    --st_rounds 3 --st_epochs_per_round 5 --st_tta \
    --ema_momentum 0.999 \
    --output_dir $CKPT_DIR/cups_sem_simple_full \
    2>&1 | tee $LOG_DIR/cups_sem_simple_full.log
  echo "[R4] SimpleFPN+FullCUPS EXIT CODE: $? — $(date)"
}


# ═══════════════════════════════════════════════════════════════════════════════
# Run selected run(s)
# ═══════════════════════════════════════════════════════════════════════════════

case "$RUN" in
  r1)
    run_r1
    ;;
  r2)
    run_r2
    ;;
  r3)
    run_r3
    ;;
  r4)
    run_r4
    ;;
  all)
    run_r1
    run_r2
    run_r3
    run_r4
    ;;
  *)
    echo "Unknown run: $RUN"
    echo "Usage: $0 [r1|r2|r3|r4|all]"
    exit 1
    ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ABLATION COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
echo "    R1 (BiFPN+FullCUPS):  $CKPT_DIR/cups_sem_bifpn_full/"
echo "    R2 (BiFPN+Baseline):  $CKPT_DIR/cups_sem_bifpn_baseline/"
echo "    R3 (BiFPN+Stage2):    $CKPT_DIR/cups_sem_bifpn_stage2/"
echo "    R4 (Simple+FullCUPS): $CKPT_DIR/cups_sem_simple_full/"
echo ""
echo "  Baseline reference: PQ=23.73 (SimpleFPN + full aug, no CUPS)"
echo "════════════════════════════════════════════════════════════════"
