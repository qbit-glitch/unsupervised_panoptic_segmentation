# Path-C AuxThingAdapter Eval Results

**Protocol:** CUPS k80 global Hungarian assignment, then target-space Path-C fusion.
**Adapter:** checkpoints/aux_thing_adapter_run1/best.pt
**Stage-3 checkpoint:** checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt
**Fusion class scope:** things
**Thresholds:** [0.99, 0.995, 0.999]
**Best threshold by PQ_things:** 0.999

## Summary

| method | PQ | SQ | RQ | PQ_stuff | PQ_things | mIoU | Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-3 baseline | 40.55 | 61.14 | 49.29 | 40.34 | 41.06 | 43.20 | 89.07 |
| Path-C fused tau=0.999 | 40.55 | 61.14 | 49.29 | 40.34 | 41.06 | 45.79 | 93.61 |
| Delta | +0.00 |  |  | +0.00 | +0.00 | +2.59 | +4.54 |

## Threshold Sweep

| tau | PQ | delta PQ | PQ_stuff | delta stuff | PQ_things | delta things | mIoU | active px | suppressed px |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.990 | 40.53 | -0.02 | 40.34 | +0.00 | 40.98 | -0.08 | 45.74 | 759730 | 1514410 |
| 0.995 | 40.55 | -0.01 | 40.34 | +0.00 | 41.03 | -0.03 | 45.78 | 740672 | 1273311 |
| 0.999 | 40.55 | +0.00 | 40.34 | +0.00 | 41.06 | +0.00 | 45.79 | 684331 | 802001 |

## Eval Counters

- Pass-1 images: 20
- Pass-2 images: 20
- Missing P4 cache fallback images: 0
- Active adapter pixels by threshold: {0.99: 759730, 0.995: 740672, 0.999: 684331}
- Suppressed thing pixels without CUPS instances by threshold: {0.99: 1514410, 0.995: 1273311, 0.999: 802001}

## Per-Class PQ

| cls | class | baseline PQ | fused PQ | delta |
|---:|---|---:|---:|---:|
| 0 | road | 96.64 | 96.64 | +0.00 |
| 1 | sidewalk | 61.62 | 61.62 | +0.00 |
| 2 | parking | 0.00 | 0.00 | +0.00 |
| 3 | rail track | 0.00 | 0.00 | +0.00 |
| 4 | building | 86.33 | 86.33 | +0.00 |
| 5 | wall | 19.54 | 19.54 | +0.00 |
| 6 | fence | 28.97 | 28.97 | +0.00 |
| 7 | guard rail | 0.00 | 0.00 | +0.00 |
| 8 | bridge | 0.00 | 0.00 | +0.00 |
| 9 | tunnel | 0.00 | 0.00 | +0.00 |
| 10 | pole | 4.64 | 4.64 | +0.00 |
| 11 | polegroup | 0.00 | 0.00 | +0.00 |
| 12 | traffic light | 7.96 | 7.96 | +0.00 |
| 13 | traffic sign | 37.18 | 37.18 | +0.00 |
| 14 | vegetation | 88.21 | 88.21 | +0.00 |
| 15 | terrain | 45.70 | 45.70 | +0.00 |
| 16 | sky | 87.93 | 87.93 | +0.00 |
| 17 | person | 7.24 | 7.24 | +0.00 |
| 18 | rider | 40.66 | 40.66 | +0.00 |
| 19 | car | 32.60 | 32.60 | +0.00 |
| 20 | truck | 78.38 | 78.38 | +0.00 |
| 21 | bus | 0.00 | 0.00 | +0.00 |
| 22 | caravan | 0.00 | 0.00 | +0.00 |
| 23 | trailer | 0.00 | 0.00 | +0.00 |
| 24 | train | 0.00 | 0.00 | +0.00 |
| 25 | motorcycle | 0.00 | 0.00 | +0.00 |
| 26 | bicycle | 87.50 | 87.50 | +0.00 |

## CUPS Assignment

```text
[5, 3, 7, 8, 9, 11, 4, 0, 0, 6, 1, 4, 0, 0, 4, 14, 0, 0, 0, 1, 4, 0, 4, 13, 4, 4, 0, 0, 0, 0, 0, 10, 14, 14, 6, 12, 0, 4, 0, 15, 10, 1, 4, 16, 0, 14, 4, 0, 0, 1, 4, 2, 0, 0, 14, 0, 0, 4, 0, 0, 0, 0, 4, 0, 21, 22, 20, 23, 24, 19, 25, 17, 17, 17, 17, 19, 18, 17, 17, 26]
```
