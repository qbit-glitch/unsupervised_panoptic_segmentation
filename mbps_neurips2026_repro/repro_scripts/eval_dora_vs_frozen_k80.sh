#!/usr/bin/env bash
# Fair apples-to-apples comparison of:
#   (a) frozen DINOv2 + CAUSE-TR          (no adapters)
#   (b) DoRA-adapted DINOv2 + CAUSE-TR    (best.pt from 50-epoch training)
# Both use eval_cause_k80.py which uses CUPS-standard many-to-one Hungarian
# matching (Hungarian + argmax fallback). This is the same pipeline that
# produced the published 52.69% frozen baseline in memory.
#
# Run on santosh:
#   cd /home/santosh/mbps_panoptic_segmentation
#   nohup bash scripts/eval_dora_vs_frozen_k80.sh > logs/dora_vs_frozen_k80.log 2>&1 &
#   echo "PID: $!"
#   tail -f logs/dora_vs_frozen_k80.log
set -euo pipefail

REPO="${REPO:-/home/santosh/mbps_panoptic_segmentation}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes}"
ADAPTER_CKPT="${ADAPTER_CKPT:-/media/santosh/Kuldeep/panoptic_segmentation/experiments/semantic_adapter_dinov2_cause_dora_r4_ddp/best.pt}"
DEVICE="${DEVICE:-cuda:0}"
K="${K:-80}"

cd "$REPO"
mkdir -p logs results/dora_eval

for f in "$ADAPTER_CKPT" \
         "$CITYSCAPES_ROOT/leftImg8bit/val" \
         "refs/cause/checkpoint/dinov2_vit_base_14.pth" \
         "refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth" \
         "refs/cause/CAUSE/cityscapes/modularity/dinov2_vit_base_14/2048/modular.npy"; do
    if [ ! -e "$f" ]; then
        echo "MISSING: $f" >&2
        exit 1
    fi
done

echo "=========================================================="
echo "PART 1/2  Frozen DINOv2 + CAUSE-TR baseline   (no adapters)"
echo "=========================================================="
python3 mbps_pytorch/eval_cause_k80.py \
    --backbone dinov2 \
    --cityscapes_root "$CITYSCAPES_ROOT" \
    --k_values "$K" \
    --device "$DEVICE" \
    2>&1 | tee logs/eval_k80_frozen_baseline.log

echo
echo "=========================================================="
echo "PART 2/2  DoRA-adapted DINOv2 + CAUSE-TR      (best.pt)"
echo "=========================================================="
python3 mbps_pytorch/eval_cause_k80.py \
    --backbone dinov2 \
    --cityscapes_root "$CITYSCAPES_ROOT" \
    --k_values "$K" \
    --device "$DEVICE" \
    --adapter_checkpoint "$ADAPTER_CKPT" \
    2>&1 | tee logs/eval_k80_dora_adapter.log

echo
echo "=========================================================="
echo "DONE.  Logs:"
echo "  logs/eval_k80_frozen_baseline.log   (frozen)"
echo "  logs/eval_k80_dora_adapter.log      (DoRA)"
echo "Grep the SUMMARY lines at the bottom of each log for the"
echo "mIoU and PQ numbers."
echo "=========================================================="
