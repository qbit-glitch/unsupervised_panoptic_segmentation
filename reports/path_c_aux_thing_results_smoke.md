# Path-C AuxThingAdapter Eval Results

**Protocol:** CUPS k80 global Hungarian assignment, then target-space Path-C fusion.
**Adapter:** checkpoints/aux_thing_adapter_run1/best.pt
**Stage-3 checkpoint:** checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt
**Fusion class scope:** things
**Thresholds:** [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
**Best threshold by PQ_things:** 0.990

## Summary

| method | PQ | SQ | RQ | PQ_stuff | PQ_things | mIoU | Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-3 baseline | 46.57 | 50.86 | 56.41 | 47.12 | 44.72 | 46.03 | 82.47 |
| Path-C fused tau=0.990 | 46.42 | 50.71 | 56.41 | 47.12 | 44.08 | 51.37 | 90.79 |
| Delta | -0.15 |  |  | +0.00 | -0.65 | +5.34 | +8.32 |

## Threshold Sweep

| tau | PQ | delta PQ | PQ_stuff | delta stuff | PQ_things | delta things | mIoU | active px | suppressed px |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 37.92 | -8.65 | 47.12 | +0.00 | 14.91 | -29.82 | 47.04 | 67862 | 409521 |
| 0.700 | 37.92 | -8.65 | 47.12 | +0.00 | 14.91 | -29.82 | 46.67 | 66519 | 336894 |
| 0.800 | 37.92 | -8.65 | 47.12 | +0.00 | 14.91 | -29.82 | 47.00 | 65454 | 297667 |
| 0.900 | 37.92 | -8.65 | 47.12 | +0.00 | 14.91 | -29.82 | 47.72 | 64188 | 249624 |
| 0.950 | 37.92 | -8.65 | 47.12 | +0.00 | 14.91 | -29.82 | 49.02 | 62866 | 213566 |
| 0.980 | 46.12 | -0.45 | 47.12 | +0.00 | 42.79 | -1.93 | 51.03 | 61130 | 173249 |
| 0.990 | 46.42 | -0.15 | 47.12 | +0.00 | 44.08 | -0.65 | 51.37 | 60631 | 147826 |

## Eval Counters

- Pass-1 images: 2
- Pass-2 images: 2
- Missing P4 cache fallback images: 0
- Active adapter pixels by threshold: {0.5: 67862, 0.7: 66519, 0.8: 65454, 0.9: 64188, 0.95: 62866, 0.98: 61130, 0.99: 60631}
- Suppressed thing pixels without CUPS instances by threshold: {0.5: 409521, 0.7: 336894, 0.8: 297667, 0.9: 249624, 0.95: 213566, 0.98: 173249, 0.99: 147826}

## Per-Class PQ

| cls | class | baseline PQ | fused PQ | delta |
|---:|---|---:|---:|---:|
| 0 | road | 95.15 | 95.15 | +0.00 |
| 1 | sidewalk | 69.87 | 69.87 | +0.00 |
| 2 | parking | 0.00 | 0.00 | +0.00 |
| 3 | rail track | 0.00 | 0.00 | +0.00 |
| 4 | building | 76.50 | 76.50 | +0.00 |
| 5 | wall | 0.00 | 0.00 | +0.00 |
| 6 | fence | 0.00 | 0.00 | +0.00 |
| 7 | guard rail | 0.00 | 0.00 | +0.00 |
| 8 | bridge | 0.00 | 0.00 | +0.00 |
| 9 | tunnel | 0.00 | 0.00 | +0.00 |
| 10 | pole | 0.00 | 0.00 | +0.00 |
| 11 | polegroup | 0.00 | 0.00 | +0.00 |
| 12 | traffic light | 0.00 | 0.00 | +0.00 |
| 13 | traffic sign | 52.01 | 52.01 | +0.00 |
| 14 | vegetation | 86.09 | 86.09 | +0.00 |
| 15 | terrain | 0.00 | 0.00 | +0.00 |
| 16 | sky | 91.60 | 91.60 | +0.00 |
| 17 | person | 0.00 | 0.00 | +0.00 |
| 18 | rider | 74.54 | 72.60 | -1.94 |
| 19 | car | 59.62 | 59.62 | +0.00 |
| 20 | truck | 0.00 | 0.00 | +0.00 |
| 21 | bus | 0.00 | 0.00 | +0.00 |
| 22 | caravan | 0.00 | 0.00 | +0.00 |
| 23 | trailer | 0.00 | 0.00 | +0.00 |
| 24 | train | 0.00 | 0.00 | +0.00 |
| 25 | motorcycle | 0.00 | 0.00 | +0.00 |
| 26 | bicycle | 0.00 | 0.00 | +0.00 |

## CUPS Assignment

```text
[2, 3, 5, 7, 8, 9, 4, 11, 0, 0, 15, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 10, 14, 6, 0, 12, 0, 0, 0, 0, 14, 1, 0, 16, 0, 14, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 19, 25, 26, 17, 17, 17, 19, 18, 17, 17, 17]
```
