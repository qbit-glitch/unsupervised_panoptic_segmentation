#!/usr/bin/env bash
# CUPS-protocol eval on fics-lab (same DINOv3 features as training → consistent).
#   E1 extract val DINOv3 features  E2 val k27 labels (dataset-load only)
#   E3 GT-free CC thing/stuff split  E4 eval each checkpoint (1-to-1 + many-to-1)
set -euo pipefail
CS_ROOT="${CS_ROOT:?set CS_ROOT (e.g. /mnt/HDD_16TB/datasets/Cityscapes)}"
PY="${PY:-/mnt/HDD_16TB/umesh/envs/gadepthg/bin/python}"
SEM=pseudo_semantic_raw_dinov3_k27_spherical_kmeans_vitl16
CKPT_DIR="${CKPT_DIR:-checkpoints/gtfree_k27}"
export PYTHONPATH="$PWD:$PWD/refs/cups:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# E1: val DINOv3 ViT-L/16 features (needs val images; idempotent)
if [ ! -f "$CS_ROOT/dinov3_features_vitl16/val/metadata.json" ]; then
  echo "[E1] extracting val DINOv3 features..."
  "$PY" mbps_pytorch/extract_dinov3_features.py \
    --data_dir "$CS_ROOT/leftImg8bit/val" \
    --output_dir "$CS_ROOT/dinov3_features_vitl16/val" \
    --model_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --image_height 512 --image_width 1024 --batch_size 16 --device cuda
else
  echo "[E1] val features present, skip"
fi

# E2: val k27 labels (seed 42 → same centroids as train; only needed so the
# eval dataset loads — pseudo-labels are UNUSED in the PQ metric).
if [ ! -d "$CS_ROOT/$SEM/val" ]; then
  echo "[E2] generating val k27 labels..."
  "$PY" mbps_pytorch/generate_clustering_ablation.py \
    --cityscapes_root "$CS_ROOT" --feat_subdir dinov3_features_vitl16 \
    --method spherical_kmeans --k 27 --splits val --seed 42 --output_suffix vitl16
else
  echo "[E2] val labels present, skip"
fi

# E3: GT-free CC thing/stuff split (no instance maps)
if [ ! -f "$CS_ROOT/stuff_things_k27.json" ]; then
  echo "[E3] computing CC thing/stuff split..."
  "$PY" mbps_pytorch/classify_stuff_things_freq.py \
    --semantic_dir "$CS_ROOT/$SEM/train" \
    --num_clusters 27 --threshold 0.08 \
    --output "$CS_ROOT/stuff_things_k27.json"
else
  echo "[E3] split present, skip"
fi

# E4: CUPS-protocol eval (each ckpt prints one-to-one + many-to-one)
shopt -s nullglob
for ckpt in "$CKPT_DIR"/best.pth "$CKPT_DIR"/checkpoint_epoch_*.pth; do
  echo "===================== EVAL $(basename "$ckpt") ====================="
  "$PY" mbps_pytorch/eval_cups_protocol.py \
    --checkpoint "$ckpt" \
    --cityscapes_root "$CS_ROOT" \
    --stuff_things "$CS_ROOT/stuff_things_k27.json" \
    --device cuda || echo "EVAL FAILED for $ckpt"
done
echo "EVAL DONE"
