# DINOv3 + CUPS Cascade Mask R-CNN — Full Evaluation Results

## Setup
- **Backbone**: Official DINOv3 ViT-B/16 (facebookresearch/dinov3, pretrained on LVD-1689M)
- **Detection Head**: Cascade Mask R-CNN (CUPS architecture)
- **Pseudo-labels**: k=80 overclustered semantic labels + depth-guided instances
- **Training**: 2x GTX 1080 Ti, effective batch 16 (bs=1 x 2 GPUs x 8 accum), fp32
- **Evaluation**: Full Cityscapes val (500 images), 27-class CAUSE + Hungarian matching
- **Metric**: Same as CUPS CVPR 2025 paper (Table 1)

## Stage-2: Pseudo-label Bootstrapping

| Metric | Value |
|--------|-------|
| PQ | 0.27865 |
| SQ | 0.57830 |
| RQ | 0.36290 |
| PQ_things | 0.23170 |
| SQ_things | 0.62786 |
| RQ_things | 0.30650 |
| PQ_stuff | 0.30627 |
| SQ_stuff | 0.54916 |
| RQ_stuff | 0.39607 |
| Acc | 0.86340 |
| mIoU | 0.43579 |

Checkpoint: `checkpoints/dinov3_stage3/dinov3_official_stage2_last.ckpt` (global_step=1000)

## Stage-3: Self-Training — Full Eval at Each Checkpoint

| Step | PQ | SQ | RQ | PQ_things | SQ_things | RQ_things | PQ_stuff | SQ_stuff | RQ_stuff | Acc | mIoU |
|------|---------|---------|---------|-----------|-----------|-----------|----------|----------|----------|---------|---------|
| 600 | 0.29070 | 0.58804 | 0.37412 | 0.25751 | 0.64783 | 0.32787 | 0.31022 | 0.55288 | 0.40133 | 0.85741 | 0.43899 |
| 800 | 0.29002 | 0.59044 | 0.37110 | 0.26610 | 0.65343 | 0.34090 | 0.30410 | 0.55340 | 0.38885 | 0.86551 | 0.43898 |
| 1800 | 0.30255 | 0.62388 | 0.38756 | 0.28495 | 0.73113 | 0.35919 | 0.31291 | 0.56079 | 0.40425 | 0.85707 | 0.44396 |
| 2000 | 0.29939 | 0.61421 | 0.38121 | 0.27928 | 0.69137 | 0.35168 | 0.31121 | 0.56882 | 0.39858 | 0.85705 | 0.44178 |
| 3400 | 0.30806 | 0.60933 | 0.38785 | 0.31327 | 0.67926 | 0.38688 | 0.30500 | 0.56820 | 0.38842 | 0.86092 | 0.44650 |
| 5200 | 0.31800 | 0.60646 | 0.39611 | 0.31126 | 0.67649 | 0.38526 | 0.32197 | 0.56527 | 0.40249 | 0.89438 | 0.45743 |
| 8000 | 0.32761 | 0.62573 | 0.40751 | 0.34132 | 0.71329 | 0.41551 | 0.31954 | 0.57423 | 0.40281 | 0.86389 | 0.45137 |

## Best Result: Step 8000

| Metric | Value |
|--------|-------|
| PQ | 0.32761 |
| SQ | 0.62573 |
| RQ | 0.40751 |
| PQ_things | 0.34132 |
| SQ_things | 0.71329 |
| RQ_things | 0.41551 |
| PQ_stuff | 0.31954 |
| SQ_stuff | 0.57423 |
| RQ_stuff | 0.40281 |
| Acc | 0.86389 |
| mIoU | 0.45137 |

## Comparison to CUPS CVPR 2025 (Table 1)

| Metric | DINOv3 (Ours, step 8000) | CUPS RN50 (paper) | Delta |
|--------|--------------------------|-------------------|-------|
| PQ | 0.32761 | 0.278 | +0.04961 |
| PQ_things | 0.34132 | 0.177 | +0.16432 |
| PQ_stuff | 0.31954 | 0.351 | -0.03146 |

## Key Observations
- PQ_things improvement is massive: +16.4% absolute over CUPS (0.341 vs 0.177)
- PQ_stuff slightly lower than CUPS (-3.1%), likely due to patch_size=16 losing fine boundary detail vs ResNet's hierarchical features
- Overall PQ beats CUPS by +5.0% absolute (0.328 vs 0.278)
- Self-training (Stage-3) adds +4.9 PQ over Stage-2 only (0.328 vs 0.279)
- PQ continues to improve through training; step 8000 is the current best but training is still running
- mIoU peaks at step 5200 (0.457) then slightly decreases at step 8000 (0.451)
