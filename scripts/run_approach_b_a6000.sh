#!/bin/bash
# Approach B: CAUSE + Depth Contrastive Fine-Tuning on A6000
#
# Prerequisites on A6000:
#   1. Project code synced to ~/umesh/mbps_panoptic_segmentation/
#   2. Cityscapes dataset at $DATA_DIR/cityscapes/ (with leftImg8bit/, gtFine/)
#   3. DepthPro depth maps at $DATA_DIR/cityscapes/depth_depthpro/train/ (2975 .npy files)
#   4. CAUSE checkpoint at refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth
#   5. Python env with torch, sklearn, tqdm, PIL
#
# Usage:
#   bash scripts/run_approach_b_a6000.sh          # Run full lambda sweep
#   bash scripts/run_approach_b_a6000.sh 0.05     # Run single lambda

set -euo pipefail

# ─── Config (Anydesk A6000: cvpr_ug_5@master) ───
PROJECT_DIR="${PROJECT_DIR:-/home/cvpr_ug_5/umesh/mbps_panoptic_segmentation}"
DATA_DIR="${DATA_DIR:-/home/cvpr_ug_5/umesh/datasets}"
PYTHON="${PYTHON:-/home/cvpr_ug_5/umesh/ups_env/bin/python}"
DEVICE="cuda:0"
BATCH_SIZE=4  # A6000 with ~24GB free (other training running)
EPOCHS=20
LOG_DIR="$PROJECT_DIR/logs/approach_b"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" results

# ─── Verify prerequisites ───
echo "=== Verifying prerequisites ==="
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

if [ ! -f "refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth" ]; then
    echo "ERROR: CAUSE checkpoint not found. Sync refs/cause/ first."
    exit 1
fi

DEPTH_COUNT=$(find "$DATA_DIR/cityscapes/depth_depthpro/train/" -name "*.npy" 2>/dev/null | wc -l)
echo "DepthPro train depth maps: $DEPTH_COUNT"
if [ "$DEPTH_COUNT" -lt 2975 ]; then
    echo "WARNING: Expected 2975 depth maps, found $DEPTH_COUNT"
fi

echo "Prerequisites OK."
echo ""

# ─── Lambda sweep ───
if [ $# -ge 1 ]; then
    LAMBDAS="$1"
else
    LAMBDAS="0.0 0.01 0.05 0.1"
fi

for LAMBDA in $LAMBDAS; do
    OUT_DIR="results/cause_depth_ft_lambda${LAMBDA}"
    LOG_FILE="$LOG_DIR/B_lambda${LAMBDA}.log"

    if [ -d "$OUT_DIR/epoch_000" ]; then
        echo "[SKIP] lambda=$LAMBDA — checkpoint exists at $OUT_DIR/epoch_000"
        continue
    fi

    echo "=== Training: lambda_depth=$LAMBDA ==="
    echo "  Output: $OUT_DIR"
    echo "  Log: $LOG_FILE"

    nohup $PYTHON -u mbps_pytorch/train_cause_depth_finetune.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUT_DIR" \
        --lambda_depth "$LAMBDA" \
        --epochs "$EPOCHS" \
        --lr 1e-5 \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --save_every 5 \
        --seed 42 \
        > "$LOG_FILE" 2>&1

    echo "  Done. Checking best checkpoint..."
    if [ -d "$OUT_DIR/epoch_000" ]; then
        echo "  Best checkpoint saved to $OUT_DIR/epoch_000/"
    else
        echo "  WARNING: No best checkpoint found"
    fi
    echo ""
done

echo "=== Lambda sweep complete ==="
echo ""

# ─── Generate pseudo-labels from best checkpoints ───
echo "=== Generating pseudo-labels from fine-tuned checkpoints ==="

for LAMBDA in $LAMBDAS; do
    CKPT_DIR="results/cause_depth_ft_lambda${LAMBDA}/epoch_000"
    OUT_SUBDIR="pseudo_semantic_depth_ft_lambda${LAMBDA}_k300"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[SKIP] No checkpoint for lambda=$LAMBDA"
        continue
    fi

    if [ -d "$DATA_DIR/cityscapes/$OUT_SUBDIR/val" ]; then
        echo "[SKIP] Pseudo-labels already exist for lambda=$LAMBDA"
        continue
    fi

    echo "  Generating for lambda=$LAMBDA ..."
    $PYTHON -u mbps_pytorch/generate_depth_overclustered_semantics.py \
        --cityscapes_root "$DATA_DIR/cityscapes" \
        --split val --k 300 \
        --checkpoint_dir "$CKPT_DIR" \
        --variant none --device "$DEVICE" \
        --output_subdir "$OUT_SUBDIR" \
        > "$LOG_DIR/gen_lambda${LAMBDA}.log" 2>&1
    echo "  Done -> $DATA_DIR/cityscapes/$OUT_SUBDIR/val/"
done

echo ""

# ─── Evaluate all ───
echo "=== Evaluating all Approach B variants ==="

for LAMBDA in $LAMBDAS; do
    OUT_SUBDIR="pseudo_semantic_depth_ft_lambda${LAMBDA}_k300"
    EVAL_OUT="eval_depth_ablation_B_lambda${LAMBDA}.json"

    if [ ! -d "$DATA_DIR/cityscapes/$OUT_SUBDIR/val" ]; then
        echo "[SKIP] No pseudo-labels for lambda=$LAMBDA"
        continue
    fi

    if [ -f "$EVAL_OUT" ]; then
        echo "[SKIP] Already evaluated lambda=$LAMBDA"
        continue
    fi

    echo "  Evaluating lambda=$LAMBDA ..."
    $PYTHON -u mbps_pytorch/evaluate_cascade_pseudolabels.py \
        --cityscapes_root "$DATA_DIR/cityscapes" \
        --split val \
        --semantic_subdir "$OUT_SUBDIR" \
        --num_clusters 300 --skip_instance \
        --output "$EVAL_OUT" \
        > "$LOG_DIR/eval_lambda${LAMBDA}.log" 2>&1
    echo "  Done -> $EVAL_OUT"
done

echo ""
echo "=== All done! ==="
echo ""
echo "Results summary:"
for LAMBDA in $LAMBDAS; do
    EVAL_OUT="eval_depth_ablation_B_lambda${LAMBDA}.json"
    if [ -f "$EVAL_OUT" ]; then
        MIOU=$($PYTHON -c "import json; d=json.load(open('$EVAL_OUT')); print(f\"{d['semantic']['miou']:.2f}\")")
        echo "  B (lambda=$LAMBDA): mIoU=$MIOU%"
    fi
done
echo ""
echo "Compare with Approach A best: sinusoidal alpha=0.1 = 56.76%"
