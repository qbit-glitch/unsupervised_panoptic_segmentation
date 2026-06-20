#!/usr/bin/env bash
# Seeded, fully GT-free DepthGuidedUNet (DINOv3 ViT-L/16 + DepthPro, k=27) on fics-lab.
set -euo pipefail
CS_ROOT="${CS_ROOT:?set CS_ROOT to the cityscapes data root on fics-lab}"
SEM=pseudo_semantic_raw_dinov3_k27_spherical_kmeans_vitl16
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # train_refine_net uses `from mbps_pytorch...`

for sub in dinov3_features_vitl16/train depth_depthpro/train "$SEM/train"; do
  test -d "$CS_ROOT/$sub" || { echo "MISSING: $CS_ROOT/$sub"; exit 1; }
done

PY="${PY:-/mnt/HDD_16TB/umesh/envs/gadepthg/bin/python}"
"$PY" mbps_pytorch/train_refine_net.py \
  --cityscapes_root "$CS_ROOT" \
  --model_type unet --num_classes 27 \
  --feature_subdir dinov3_features_vitl16 --feature_dim 1024 \
  --semantic_subdir "$SEM" \
  --depth_subdir depth_depthpro \
  --block_type attention --num_decoder_stages 2 --num_bottleneck_blocks 2 \
  --bridge_dim 192 --skip_dim 32 --window_size 8 --num_heads 4 \
  --num_epochs 12 --batch_size 4 --lr 1e-4 --seed 42 \
  --eval_interval 999 \
  --output_dir checkpoints/gtfree_k27 \
  --device cuda --gpu 0 2>&1 | tee logs/gtfree_k27_train.log
