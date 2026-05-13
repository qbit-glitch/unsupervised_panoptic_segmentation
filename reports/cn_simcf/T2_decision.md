# T2 Bootstrap Remap Decision

Date: 2026-04-27

## Verdict

Reject T2 bootstrap remapping for this local ablation branch.

## Metrics

| Run | PQ | PQ_stuff | PQ_things | mIoU | Decision |
|---|---:|---:|---:|---:|---|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | Reference |
| T2 iter1 | 25.20 | 33.92 | 13.21 | 56.26 | Neutral, no gain |
| T2 iter2 | 0.62 | 0.95 | 0.16 | 2.79 | Abort |

## Notes

- Iter1 exactly reproduced the T1 metric surface, so it did not recover additional rare-class signal.
- Iter2 catastrophically collapsed the class mapping. Most classes went to zero PQ and only a small subset of classes remained active.
- The T3 protected-mask ablation should therefore branch from the original CN-SIMCF/T1 mapping rather than from the iter2 bootstrap mapping.
