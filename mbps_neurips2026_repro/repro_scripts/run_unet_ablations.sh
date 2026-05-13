#!/bin/bash
# UNet Improvement Ablation Study
# Baseline: PQ=27.73 (ep8, checkpoints/hires_unet_depth_guided/)
# 7 ablations: focal, low_lr, feat_aug, rich_skip, extra_block, depth_bw, combined

PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
CS="/Users/qbit-glitch/Desktop/datasets/cityscapes"
export PYTHONPATH="$ROOT"

# Common base args (matching UNet baseline config)
BASE_ARGS="--cityscapes_root $CS \
  --semantic_subdir pseudo_semantic_mapped_k80 \
  --num_classes 19 \
  --model_type unet \
  --bridge_dim 192 \
  --num_bottleneck_blocks 2 \
  --skip_dim 32 \
  --coupling_strength 0.1 \
  --num_epochs 20 \
  --batch_size 4 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --eval_interval 2 \
  --lambda_distill 1.0 \
  --lambda_distill_min 0.85 \
  --lambda_align 0.25 \
  --lambda_proto 0.025 \
  --lambda_ent 0.025 \
  --label_smoothing 0.1"

RUN=$1

case "$RUN" in
  B|focal)
    echo "=== Run B: Focal Loss (gamma=1.0) ==="
    $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --focal_gamma 1.0 \
      --output_dir checkpoints/unet_ablation_focal
    ;;
  C|low_lr)
    echo "=== Run C: Lower LR (2e-5) ==="
    $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --lr 2e-5 \
      --output_dir checkpoints/unet_ablation_low_lr
    ;;
  D|feat_aug)
    echo "=== Run D: Feature Augmentation ==="
    $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --feature_augment \
      --feature_noise_sigma 0.02 \
      --feature_dropout_rate 0.05 \
      --output_dir checkpoints/unet_ablation_feat_aug
    ;;
  E|rich_skip)
    echo "=== Run E: Rich Depth Skip (2nd conv + Laplacian) ==="
    $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --rich_skip \
      --output_dir checkpoints/unet_ablation_rich_skip
    ;;
  F|extra_block)
    echo "=== Run F: Extra Block at 128x256 ==="
    $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --num_final_blocks 1 \
      --output_dir checkpoints/unet_ablation_extra_block
    ;;
  G|depth_bw)
    echo "=== Run G: Depth-Boundary Weighting ==="
    $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --lambda_depth_boundary 0.5 \
      --output_dir checkpoints/unet_ablation_depth_bw
    ;;
  H|combined)
    echo "=== Run H: All Combined ==="
    $PYTHON mbps_pytorch/train_refine_net.py $BASE_ARGS \
      --lr 2e-5 \
      --focal_gamma 1.0 \
      --feature_augment \
      --feature_noise_sigma 0.02 \
      --feature_dropout_rate 0.05 \
      --rich_skip \
      --num_final_blocks 1 \
      --lambda_depth_boundary 0.5 \
      --output_dir checkpoints/unet_ablation_combined
    ;;
  all)
    echo "=== Running all ablations sequentially ==="
    for run in B C D E F G H; do
      bash "$0" $run
    done
    ;;
  *)
    echo "Usage: $0 {B|C|D|E|F|G|H|focal|low_lr|feat_aug|rich_skip|extra_block|depth_bw|combined|all}"
    exit 1
    ;;
esac
