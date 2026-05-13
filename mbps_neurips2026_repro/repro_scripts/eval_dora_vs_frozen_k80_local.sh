#!/usr/bin/env bash
# LOCAL (Mac MPS) apples-to-apples eval:
#   (a) frozen DINOv2 + CAUSE-TR          (no adapters)
#   (b) DoRA-adapted DINOv2 + CAUSE-TR    (best.pt)
# Uses eval_cause_k80.py (CUPS-standard many-to-one Hungarian + argmax fallback).
#
# Run:
#   cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
#   nohup bash scripts/eval_dora_vs_frozen_k80_local.sh \
#       > logs/dora_vs_frozen_k80_local.log 2>&1 &
#   echo "PID: $!"
#   tail -f logs/dora_vs_frozen_k80_local.log
set -euo pipefail

REPO="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
CITYSCAPES_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
ADAPTER_CKPT="$REPO/checkpoints/dino_adapter_distill_r4/best.pt"
PYTHON="/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"
DEVICE="${DEVICE:-mps}"
K="${K:-80}"

cd "$REPO"
mkdir -p logs

for f in "$ADAPTER_CKPT" \
         "$PYTHON" \
         "$CITYSCAPES_ROOT/leftImg8bit/val" \
         "$CITYSCAPES_ROOT/gtFine/val" \
         "refs/cause/checkpoint/dinov2_vit_base_14.pth" \
         "refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth" \
         "refs/cause/CAUSE/cityscapes/modularity/dinov2_vit_base_14/2048/modular.npy"; do
    if [ ! -e "$f" ]; then
        echo "MISSING: $f" >&2
        exit 1
    fi
done

echo "Device: $DEVICE   K: $K   Python: $PYTHON"
echo

echo "=========================================================="
echo "PART 1/2  Frozen DINOv2 + CAUSE-TR baseline   (no adapters)"
echo "=========================================================="
"$PYTHON" -u mbps_pytorch/eval_cause_k80.py \
    --backbone dinov2 \
    --cityscapes_root "$CITYSCAPES_ROOT" \
    --k_values "$K" \
    --device "$DEVICE" \
    2>&1 | tee logs/eval_k80_frozen_baseline_local.log

echo
echo "=========================================================="
echo "PART 2/2  DoRA-adapted DINOv2 + CAUSE-TR      (best.pt)"
echo "=========================================================="
"$PYTHON" -u mbps_pytorch/eval_cause_k80.py \
    --backbone dinov2 \
    --cityscapes_root "$CITYSCAPES_ROOT" \
    --k_values "$K" \
    --device "$DEVICE" \
    --adapter_checkpoint "$ADAPTER_CKPT" \
    2>&1 | tee logs/eval_k80_dora_adapter_local.log

echo
echo "=========================================================="
echo "DONE.  Summary:"
echo "  logs/eval_k80_frozen_baseline_local.log"
echo "  logs/eval_k80_dora_adapter_local.log"
echo
echo "Key lines (grep mIoU + SUMMARY):"
grep -E "^(mIoU =|K=[0-9]+  mIoU)" logs/eval_k80_frozen_baseline_local.log | tail -3
echo "---"
grep -E "^(mIoU =|K=[0-9]+  mIoU)" logs/eval_k80_dora_adapter_local.log | tail -3
echo "=========================================================="
