# Path-C AuxThingAdapter Eval Results

**Protocol:** CUPS k80 global Hungarian assignment, then target-space Path-C fusion.
**Adapter:** checkpoints/aux_thing_adapter_run1/best.pt
**Stage-3 checkpoint:** checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt
**Fusion class scope:** things
**Thresholds:** [0.999]
**Best threshold by PQ_things:** 0.999

## Summary

| method | PQ | SQ | RQ | PQ_stuff | PQ_things | mIoU | Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-3 baseline | 35.83 | 62.79 | 43.78 | 35.56 | 36.26 | 44.56 | 87.30 |
| Path-C fused tau=0.999 | 35.46 | 62.10 | 43.38 | 35.56 | 35.30 | 45.66 | 92.04 |
| Delta | -0.37 |  |  | +0.00 | -0.96 | +1.10 | +4.75 |

## Threshold Sweep

| tau | PQ | delta PQ | PQ_stuff | delta stuff | PQ_things | delta things | mIoU | active px | suppressed px |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.999 | 35.46 | -0.37 | 35.56 | +0.00 | 35.30 | -0.96 | 45.66 | 12412432 | 16094228 |

## Eval Counters

- Pass-1 images: 500
- Pass-2 images: 500
- Missing P4 cache fallback images: 0
- Active adapter pixels by threshold: {0.999: 12412432}
- Suppressed thing pixels without CUPS instances by threshold: {0.999: 16094228}

## Per-Class PQ

| cls | class | baseline PQ | fused PQ | delta |
|---:|---|---:|---:|---:|
| 0 | road | 92.99 | 92.99 | +0.00 |
| 1 | sidewalk | 62.44 | 62.44 | +0.00 |
| 2 | parking | 0.00 | 0.00 | +0.00 |
| 3 | rail track | 8.40 | 8.40 | +0.00 |
| 4 | building | 83.54 | 83.54 | +0.00 |
| 5 | wall | 32.32 | 32.32 | +0.00 |
| 6 | fence | 20.26 | 20.26 | +0.00 |
| 7 | guard rail | 0.00 | 0.00 | +0.00 |
| 8 | bridge | 17.21 | 17.21 | +0.00 |
| 9 | tunnel | 0.00 | 0.00 | +0.00 |
| 10 | pole | 2.05 | 2.05 | +0.00 |
| 11 | polegroup | 0.00 | 0.00 | +0.00 |
| 12 | traffic light | 6.20 | 6.20 | +0.00 |
| 13 | traffic sign | 37.19 | 37.19 | +0.00 |
| 14 | vegetation | 84.70 | 84.70 | +0.00 |
| 15 | terrain | 35.68 | 35.68 | +0.00 |
| 16 | sky | 86.05 | 86.05 | +0.00 |
| 17 | person | 13.37 | 13.36 | -0.01 |
| 18 | rider | 22.94 | 23.06 | +0.12 |
| 19 | car | 70.71 | 67.98 | -2.73 |
| 20 | truck | 62.64 | 61.96 | -0.68 |
| 21 | bus | 76.67 | 70.82 | -5.85 |
| 22 | caravan | 0.00 | 0.00 | +0.00 |
| 23 | trailer | 0.00 | 0.00 | +0.00 |
| 24 | train | 77.17 | 77.07 | -0.10 |
| 25 | motorcycle | 0.10 | 0.17 | +0.07 |
| 26 | bicycle | 38.99 | 38.59 | -0.41 |

## CUPS Assignment

```text
[5, 9, 4, 4, 13, 2, 4, 0, 0, 6, 14, 4, 0, 0, 4, 14, 14, 0, 0, 1, 4, 0, 4, 13, 4, 4, 8, 3, 16, 4, 0, 10, 14, 14, 7, 12, 0, 4, 0, 15, 11, 1, 4, 16, 0, 14, 4, 0, 0, 1, 4, 0, 0, 0, 14, 0, 0, 4, 0, 0, 0, 1, 4, 1, 24, 21, 20, 17, 22, 19, 17, 23, 17, 17, 17, 25, 18, 17, 17, 26]
```
