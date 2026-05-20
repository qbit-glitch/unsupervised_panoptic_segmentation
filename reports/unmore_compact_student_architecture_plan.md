# unMORE Compact Student Architecture Plan

Date: 2026-05-20

## Goal

Distill the official unMORE ImageNet/VoteCut object reasoning supervision into a compact class-agnostic instance segmentation student, then evaluate that student only on the official benchmark protocol we selected:

- COCO20K
- KITTI

COCO20K and KITTI are evaluation sets, not the primary distillation training data. The training source is the ImageNet-1K plus VoteCut cache, matching the official unMORE training setup.

## Training Cache

Prepared cache:

```text
/Volumes/code_files/mbps_datasets/unmore_teacher_cache/imagenet_votecut_w050
```

Final cache stats:

| Item | Value |
|---|---:|
| Image records | 1,280,558 |
| VoteCut masks kept | 1,805,648 |
| Score threshold | weight >= 0.5 |
| Shards | 2,502 |
| Cache size | 4.8 GB |
| Missing images | 0 |

This cache stores, per image, teacher-style boxes, scores, category ids, bbox areas, and compressed COCO RLE masks. It is directly compatible with `UnmoreTeacherCacheDataset`.

## Student Architecture

Selected architecture:

```text
Pretrained DINOv3-S ViT-S/16 backbone
    -> intermediate transformer blocks: 3, 6, 9, 11
    -> lightweight same-stride feature fusion
    -> compact FPN pyramid P3/P4/P5/P6
    -> class-agnostic Mask R-CNN heads
```

Model implementation:

```text
mbps_pytorch/unmore_distill/model.py
```

Measured parameter sizes:

| FPN dim | Parameters | FP32 size |
|---:|---:|---:|
| 96 | 30,259,929 | 115.4 MB |
| 128 | 32,160,697 | 122.7 MB |
| 160 | 34,081,945 | 130.0 MB |
| 192 | 36,023,673 | 137.4 MB |

Chosen initial student: `fpn_dim=128`.

Reason:

- inside the target 80-150 MB range
- preserves enough FPN capacity for masks
- keeps training/inference cheaper than the 4 GB unMORE center-boundary checkpoint
- can be quantized later if the fp32 model is strong enough

## Training Schedule

### Official-Teacher Cache Track

The next quality step is not to keep training only on VoteCut. We are now
building an ImageNet cache from the official unMORE Cascade Mask R-CNN teacher:

```text
Teacher checkpoint:
test-instance-labels/unMORE/checkpoints/unMORE_model.pth

Cache target:
/Volumes/code_files/mbps_datasets/unmore_teacher_cache/imagenet_official_unmore_teacher_s005_10k_v2

Live log:
logs/cache_unmore_official_teacher_imagenet10k_v2.log
```

This cache uses ImageNet images from the `imagenet_votecut_w050` source index,
but replaces VoteCut masks with official teacher predictions:

- score threshold: `0.05`
- max detections per image: `100`
- teacher inference resolution: `MIN_SIZE_TEST=800`, `MAX_SIZE_TEST=1333`
- device: CPU
- output format: grouped sharded cache compatible with `UnmoreTeacherCacheDataset`

Because the local environment has no CUDA/MPS, the official teacher pass is
slow. The job is intentionally chained so that after the 10k teacher cache
finishes, it starts student training from the official-teacher cache:

```text
checkpoints/unmore_dinov3s_official_teacher_imagenet10k_v2_phase1
```

This official-teacher stage is the first meaningful attempt to improve AP beyond
the VoteCut-only sanity model.

### Phase 1: Frozen-Backbone Warm Start

Purpose:

- verify stable learning from ImageNet/VoteCut at scale
- train FPN, RPN, box head, and mask head while keeping pretrained DINOv3-S fixed
- establish CPU throughput and loss behavior before heavier training

Current run:

```bash
.venv/bin/python scripts/train_unmore_dinov3_student.py \
  --cache-dir /Volumes/code_files/mbps_datasets/unmore_teacher_cache/imagenet_votecut_w050 \
  --output-dir checkpoints/unmore_dinov3s_imagenet_w050_phase1_frozen \
  --score-min 0.5 \
  --max-instances 10 \
  --batch-size 1 \
  --epochs 1 \
  --max-steps 2000 \
  --device cpu \
  --pretrained-backbone \
  --freeze-backbone \
  --fpn-dim 128 \
  --min-size 256 \
  --max-size 384 \
  --val-fraction 0.05 \
  --val-batches 10 \
  --checkpoint-every-steps 500 \
  --val-every-steps 500 \
  --log-every 25 \
  --skip-empty \
  --lr 0.0001 \
  --weight-decay 0.05
```

Live log:

```bash
tail -f logs/train_unmore_dinov3s_imagenet_phase1.log
```

### Phase 2: Higher-Resolution Student Training

If Phase 1 is stable, continue from the best Phase 1 checkpoint with:

- `min-size=384`
- `max-size=640`
- `max-instances=20`
- frozen DINOv3-S initially
- then optionally unfreeze only the last transformer blocks if compute allows

Expected effect:

- better medium/large object masks
- better boxes for COCO20K/KITTI evaluation
- still compact, because model size does not change

### Phase 3: Benchmark Evaluation

Export student predictions on:

- COCO20K official cache/images
- KITTI official cache/images

Primary metric:

- official unMORE class-agnostic AP protocol

Secondary checks:

- AP50/AP75
- mask versus box AP gap
- throughput
- comparison with official unMORE teacher checkpoints

### Phase 4: Surpass-Teacher Attempt

Only after the compact student reaches a competitive baseline:

- noisy-student self-training on high-confidence student predictions
- multi-scale consistency
- teacher-student agreement filtering
- optional final quantization or distillation to an even smaller head

Do not quantize before the fp32 student is strong; otherwise quantization only compresses a weak model.

## Current Environment Constraint

Local PyTorch reports:

```text
CUDA: unavailable
MPS: unavailable
```

So the current run is a CPU Phase 1 warm-start, not the final full-capacity training run. If Phase 1 loss and early AP are promising, the next practical move is to run the same code on a GPU machine for Phase 2/3.
