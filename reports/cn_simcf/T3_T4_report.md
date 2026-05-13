# T3/T4 Rare-Class Protection Report

Date: 2026-04-27

## Objective

Recover weak rare-class pseudo-label signal without destroying the CN-SIMCF behavior that keeps the label map trainable for later Stage-2/Stage-3 learning.

The main target is not immediate pseudo-label PQ maximization. The target is a better Stage-2 root: preserve rare positives while keeping common-class and instance structure stable enough for the downstream learner to recover PQ.

## Baseline

T1 CN-SIMCF is the reference point.

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore |
|---|---:|---:|---:|---:|---:|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 |

T1 keeps the strongest aggregate pseudo-label PQ among these branches, but it still leaves several rare/dead classes unrecovered.

## T3: Broad Protected Mask

T3 applied the rare protected mask directly to CN-SIMCF. The corrected T3 v2 run loaded the mask successfully and protected all marked pixels through the legacy policy.

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore |
|---|---:|---:|---:|---:|---:|
| T3 broad protected v2 | 25.13 | 33.83 | 13.17 | 56.21 | 1.04 |

Generation behavior:

| Step | T3 Result |
|---|---:|
| Protected mask coverage | 31.57% |
| Step A changed | 0 pixels |
| Step B merges | 0 |
| Step C masked | 65,042,279 px / 1.04% |

T3 recovered a faint rare-class signal:

| Class | T1 TP | T3 TP | T3 PQ | Note |
|---|---:|---:|---:|---|
| pole | 0 | 1 | 0.02 | First recovered pole match |
| traffic light | 0 | 0 | 0.00 | Still dead |
| motorcycle | 0 | 0 | 0.00 | Still dead |
| person | 355 | 358 | 1.48 | Small gain |
| rider | 53 | 55 | 2.41 | Small gain |
| bicycle | 145 | 147 | 2.53 | Small gain |

Interpretation: T3 proves the protected-mask idea is not empty. It preserves weak rare positives. But the mask is too blunt: Step B is completely disabled, which removes useful instance merging.

## T4: Rare-Core Policy

T4 changed the policy rather than the mask source.

- Step A: no protection.
- Step B: normal same-cluster + similarity merge behavior, except merge candidates are blocked if their merge corridor crosses a rare core.
- Step C: only rare-core pixels are protected from outlier masking.

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore |
|---|---:|---:|---:|---:|---:|
| T4 rare-core protected | 25.15 | 33.85 | 13.19 | 56.22 | 1.18 |

Generation behavior:

| Step | T4 Result |
|---|---:|
| Protected mask coverage | 31.57% |
| Rare-core coverage | 22.86% |
| Step A changed | 0 pixels |
| Step B merges | 39 |
| Step B blocked | 1,691 candidates |
| Step C masked | 73,475,432 px / 1.18% |

Rare-class signal:

| Class | T1 TP | T3 TP | T4 TP | T4 PQ | Note |
|---|---:|---:|---:|---:|---|
| pole | 0 | 1 | 1 | 0.02 | Preserved |
| traffic light | 0 | 0 | 0 | 0.00 | Still dead |
| motorcycle | 0 | 0 | 0 | 0.00 | Still dead |
| person | 355 | 358 | 358 | 1.48 | Preserved |
| rider | 53 | 55 | 55 | 2.41 | Preserved |
| bicycle | 145 | 147 | 147 | 2.53 | Preserved |

Interpretation: T4 is directionally better than T3. It restores some Step B merge behavior while preserving the rare positives. However, a 22.86% rare-core mask is still too broad; it blocks too many merge candidates and remains below T1 by 0.05 PQ.

## Conclusion

T3 and T4 should not be read as final pseudo-label wins. They should be read as evidence that rare-signal preservation is possible:

- T3 proves broad protection can recover faint rare positives.
- T4 proves the protection can be made less destructive by separating rare cores from the broad mask.
- The failure mode is now specific: rare-core coverage is still too high.

The next run should tighten the mask source and core extraction.

## T4b Plan

Run T4b with:

- `--agreement_required 3` when building masks.
- `--protected_mask_policy rare_core`.
- `--rare_core_erode_px 12`.
- `--rare_core_boundary_px 8`.

Acceptance criteria:

1. Preserve the recovered rare positives from T3/T4.
2. Increase Step B merges above T4's 39.
3. Reduce rare-core coverage well below 22.86%.
4. Match or exceed T4 PQ; ideally close the gap to T1 while remaining a better rare-signal Stage-2 root.

## T4b Result: M3 Rare-Core, Erode 12

T4b was run with stricter source agreement and stronger rare-core erosion:

- Mask build: `--agreement_required 3`.
- CN-SIMCF policy: `--protected_mask_policy rare_core`.
- Core extraction: `--rare_core_erode_px 12`, `--rare_core_boundary_px 8`.
- Output root: `/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_rarecore_m3_e12`.
- Evaluation: `reports/cn_simcf/T4b_rare_core_m3_e12_hungarian.json`.

Aggregate metrics:

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore |
|---|---:|---:|---:|---:|---:|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 |
| T3 broad protected v2 | 25.13 | 33.83 | 13.17 | 56.21 | 1.04 |
| T4 rare-core protected | 25.15 | 33.85 | 13.19 | 56.22 | 1.18 |
| T4b M3 rare-core e12 | 25.19 | 33.91 | 13.21 | 56.25 | 1.38 |

Generation behavior:

| Step | T4b Result |
|---|---:|
| Strict protected mask coverage | 0.86% |
| Rare-core coverage | 0.24% |
| Step A changed | 0 pixels |
| Step B merges | 91 |
| Step B blocked | 566 candidates |
| Step C masked | 86,068,603 px / 1.38% |

Rare-class signal:

| Class | T1 TP | T3 TP | T4 TP | T4b TP | T4b PQ | Note |
|---|---:|---:|---:|---:|---:|---|
| pole | 0 | 1 | 1 | 0 | 0.00 | Rare recovery lost |
| traffic light | 0 | 0 | 0 | 0 | 0.00 | Still dead |
| motorcycle | 0 | 0 | 0 | 0 | 0.00 | Still dead |
| person | 355 | 358 | 358 | 355 | 1.47 | Back to T1 |
| rider | 53 | 55 | 55 | 53 | 2.33 | Back to T1 |
| bicycle | 145 | 147 | 147 | 145 | 2.50 | Back to T1 |

Interpretation: T4b is structurally cleaner than T4. It reduces rare-core coverage from 22.86% to 0.24%, restores Step B from 39 to 91 merges, and nearly matches T1 aggregate PQ. But it fails the main acceptance criterion: the rare positives recovered by T3/T4 disappear.

## Updated Decision

T4b e12 should not replace T4 as the rare-recovery root. It is a good control showing that stricter masks recover aggregate PQ, but it is too strict for dead-class recovery.

Do not spend the next run on `--rare_core_erode_px 16`: e12 already removed the rare positives, and e16 should shrink the protected core further. The better next branch is to keep the high-precision M3 source mask but relax the core extraction:

1. T4c: `--agreement_required 3`, `--rare_core_erode_px 8`, `--rare_core_boundary_px 8`.
2. If pole/person/rider/bicycle gains return but Step B is still healthy, try `--rare_core_erode_px 6`.
3. Promote only if it keeps T4's rare TP gains while staying within roughly 0.03 PQ of T1.

## T4c Result: M3 Rare-Core, Erode 8

T4c kept the stricter M3 source mask from T4b, but relaxed rare-core erosion:

- Mask source: existing `agreement_required=3` masks.
- CN-SIMCF policy: `--protected_mask_policy rare_core`.
- Core extraction: `--rare_core_erode_px 8`, `--rare_core_boundary_px 8`.
- Output root: `/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_rarecore_m3_e8`.
- Evaluation: `reports/cn_simcf/T4c_rare_core_m3_e8_hungarian.json`.

Aggregate metrics:

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore |
|---|---:|---:|---:|---:|---:|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 |
| T4b M3 rare-core e12 | 25.19 | 33.91 | 13.21 | 56.25 | 1.38 |
| T4c M3 rare-core e8 | 25.19 | 33.91 | 13.21 | 56.25 | 1.38 |

Generation behavior:

| Step | T4c Result |
|---|---:|
| Strict protected mask coverage | 0.86% |
| Rare-core coverage | 0.42% |
| Step A changed | 0 pixels |
| Step B merges | 89 |
| Step B blocked | 699 candidates |
| Step C masked | 85,829,984 px / 1.38% |

Rare-class signal:

| Class | T1 TP | T3 TP | T4 TP | T4b TP | T4c TP | T4c PQ | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| pole | 0 | 1 | 1 | 0 | 0 | 0.00 | Rare recovery still lost |
| traffic light | 0 | 0 | 0 | 0 | 0 | 0.00 | Still dead |
| motorcycle | 0 | 0 | 0 | 0 | 0 | 0.00 | Still dead |
| person | 355 | 358 | 358 | 355 | 355 | 1.47 | Back to T1 |
| rider | 53 | 55 | 55 | 53 | 53 | 2.33 | Back to T1 |
| bicycle | 145 | 147 | 147 | 145 | 146 | 2.52 | Tiny partial recovery only |

Interpretation: T4c was the right test, but it did not recover the rare positives. Relaxing erosion from 12 to 8 increased rare-core coverage from 0.24% to 0.42%, but the M3 source mask still appears too source-selective for dead-class recovery. The rare positives from T3/T4 likely came from M2-only evidence that M3 removes before the rare-core policy can help.

## Updated Decision After T4c

T4c should not replace T4 as the rare-recovery root. It is a clean aggregate control, nearly tied with T1, but it fails the rare-class objective.

Do not continue a plain M3 erosion sweep. The next useful branch should recover the T3/T4 rare positives while reducing T4's broad 22.86% rare-core coverage:

1. T4d-M2-tight: return to M2 masks and use a tighter rare core, for example `--rare_core_erode_px 12` with `--rare_core_boundary_px 4` or `6`.
2. T4d-hybrid: use M3 as high-precision seeds, then expand/protect only the connected M2 rare components touched by those seeds.
3. Promote only if pole/person/rider/bicycle match T3/T4 rare TP while Step B remains closer to T4b/T4c than T4.

## Appendix: Per-Class PQ and TP

This table tracks the complete per-class comparison for the main branches. T4b and T4c recover the aggregate profile, but the rows that matter for rare recovery move back to T1.

| Class | Type | T1 PQ | T1 TP | T3 PQ | T3 TP | T4 PQ | T4 TP | T4b PQ | T4b TP | T4c PQ | T4c TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| road | stuff | 78.04 | 2902 | 77.97 | 2902 | 78.00 | 2902 | 78.04 | 2902 | 78.04 | 2902 |
| sidewalk | stuff | 39.75 | 1633 | 39.92 | 1640 | 39.88 | 1638 | 39.75 | 1633 | 39.75 | 1633 |
| building | stuff | 69.56 | 2645 | 69.49 | 2646 | 69.49 | 2645 | 69.56 | 2645 | 69.55 | 2645 |
| wall | stuff | 14.91 | 212 | 14.89 | 212 | 14.89 | 212 | 14.91 | 212 | 14.91 | 212 |
| fence | stuff | 14.53 | 310 | 14.57 | 311 | 14.57 | 311 | 14.53 | 310 | 14.53 | 310 |
| pole | stuff | 0.00 | 0 | 0.02 | 1 | 0.02 | 1 | 0.00 | 0 | 0.00 | 0 |
| traffic light | stuff | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 |
| traffic sign | stuff | 12.65 | 533 | 12.89 | 543 | 12.82 | 540 | 12.65 | 533 | 12.65 | 533 |
| vegetation | stuff | 65.65 | 2571 | 65.56 | 2571 | 65.57 | 2571 | 65.65 | 2571 | 65.65 | 2571 |
| terrain | stuff | 14.73 | 314 | 14.77 | 315 | 14.77 | 315 | 14.73 | 314 | 14.73 | 314 |
| sky | stuff | 63.24 | 2273 | 62.01 | 2250 | 62.34 | 2256 | 63.18 | 2273 | 63.18 | 2273 |
| person | thing | 1.47 | 355 | 1.48 | 358 | 1.48 | 358 | 1.47 | 355 | 1.47 | 355 |
| rider | thing | 2.33 | 53 | 2.41 | 55 | 2.41 | 55 | 2.33 | 53 | 2.33 | 53 |
| car | thing | 1.15 | 597 | 1.14 | 597 | 1.14 | 597 | 1.15 | 597 | 1.15 | 597 |
| truck | thing | 24.22 | 130 | 24.16 | 130 | 24.19 | 130 | 24.22 | 130 | 24.22 | 130 |
| bus | thing | 38.18 | 145 | 38.05 | 145 | 38.05 | 145 | 38.18 | 145 | 38.18 | 145 |
| train | thing | 35.83 | 71 | 35.59 | 71 | 35.71 | 71 | 35.83 | 71 | 35.83 | 71 |
| motorcycle | thing | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 |
| bicycle | thing | 2.50 | 145 | 2.53 | 147 | 2.53 | 147 | 2.50 | 145 | 2.52 | 146 |
