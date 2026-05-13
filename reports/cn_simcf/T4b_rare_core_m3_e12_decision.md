# T4b Rare-Core M3 Erode-12 Decision

Date: 2026-04-27

## Verdict

Do not promote T4b e12 as the Stage-2 root.

T4b fixed the overprotection problem mechanically, but it also removed the rare-class positives that T3/T4 recovered. It is close to T1 on aggregate PQ, which is useful as a sanity check, but it is not better than T1 for the dead-class recovery objective.

## Configuration

- Mask source: `--agreement_required 3`.
- Protected-mask policy: `--protected_mask_policy rare_core`.
- Core erosion: `--rare_core_erode_px 12`.
- Boundary guard: `--rare_core_boundary_px 8`.
- Output: `/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_rarecore_m3_e12`.
- Evaluation: `reports/cn_simcf/T4b_rare_core_m3_e12_hungarian.json`.

## Metrics

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore | Decision |
|---|---:|---:|---:|---:|---:|---|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 | Reference |
| T3 broad protected v2 | 25.13 | 33.83 | 13.17 | 56.21 | 1.04 | Rare signal found, too blunt |
| T4 rare-core protected | 25.15 | 33.85 | 13.19 | 56.22 | 1.18 | Rare signal preserved, still broad |
| T4b M3 rare-core e12 | 25.19 | 33.91 | 13.21 | 56.25 | 1.38 | Aggregate recovered, rare signal lost |

## Generation Stats

| Run | Protected coverage | Rare-core coverage | Step B merges | Step B blocked | Step C masked |
|---|---:|---:|---:|---:|---:|
| T3 broad protected v2 | 31.57% | n/a | 0 | n/a | 65,042,279 px / 1.04% |
| T4 rare-core protected | 31.57% | 22.86% | 39 | 1,691 | 73,475,432 px / 1.18% |
| T4b M3 rare-core e12 | 0.86% | 0.24% | 91 | 566 | 86,068,603 px / 1.38% |

## Rare-Class Signal

| Class | T1 TP | T3 v2 TP | T4 TP | T4b TP | T4b PQ | Note |
|---|---:|---:|---:|---:|---:|---|
| pole | 0 | 1 | 1 | 0 | 0.00 | Rare signal lost |
| traffic light | 0 | 0 | 0 | 0 | 0.00 | Still dead |
| motorcycle | 0 | 0 | 0 | 0 | 0.00 | Still dead |
| person | 355 | 358 | 358 | 355 | 1.47 | Back to T1 |
| rider | 53 | 55 | 55 | 53 | 2.33 | Back to T1 |
| bicycle | 145 | 147 | 147 | 145 | 2.50 | Back to T1 |

## Interpretation

T4b proves the strict-mask direction can recover aggregate behavior: Step B rises from 39 to 91 merges, and PQ recovers from 25.15 to 25.19. But the rare-core became too tiny to protect the weak rare evidence. The result is almost a T1 clone, not a better Stage-2 seed.

## Next Configuration

Do not run the e16 variant next. Since e12 already loses the rare positives, e16 is very likely to make the rare core even less useful.

Run T4c instead:

1. Keep `--agreement_required 3`.
2. Reduce core erosion to `--rare_core_erode_px 8`.
3. Keep `--rare_core_boundary_px 8`.
4. Accept only if pole/person/rider/bicycle recoveries return while PQ stays at or above T4 and close to T1.
