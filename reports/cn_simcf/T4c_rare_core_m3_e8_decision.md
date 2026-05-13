# T4c Decision: M3 Rare-Core, Erode 8

Date: 2026-04-27

## Run

T4c tested whether the T4b M3 mask was too strict only because the rare-core erosion was too large.

- Source masks: `rare_protected_masks_m3` (`agreement_required=3`).
- CN-SIMCF policy: `--protected_mask_policy rare_core`.
- Core extraction: `--rare_core_erode_px 8`, `--rare_core_boundary_px 8`.
- Output root: `/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_cn_simcf_rarecore_m3_e8`.
- Evaluation artifact: `reports/cn_simcf/T4c_rare_core_m3_e8_hungarian.json`.

## Aggregate Metrics

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore |
|---|---:|---:|---:|---:|---:|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 |
| T4b M3 rare-core e12 | 25.19 | 33.91 | 13.21 | 56.25 | 1.38 |
| T4c M3 rare-core e8 | 25.19 | 33.91 | 13.21 | 56.25 | 1.38 |

## Generation Metrics

| Metric | T4c |
|---|---:|
| Strict protected mask coverage | 0.86% |
| Rare-core coverage | 0.42% |
| Step B merges | 89 |
| Step B blocked candidates | 699 |
| Step C masked | 85,829,984 px / 1.38% |

## Rare-Class Metrics

| Class | T1 TP | T3 TP | T4 TP | T4b TP | T4c TP | T4c PQ |
|---|---:|---:|---:|---:|---:|---:|
| pole | 0 | 1 | 1 | 0 | 0 | 0.00 |
| traffic light | 0 | 0 | 0 | 0 | 0 | 0.00 |
| motorcycle | 0 | 0 | 0 | 0 | 0 | 0.00 |
| person | 355 | 358 | 358 | 355 | 355 | 1.47 |
| rider | 53 | 55 | 55 | 53 | 53 | 2.33 |
| bicycle | 145 | 147 | 147 | 145 | 146 | 2.52 |

## Decision

Reject T4c as the next Stage-2 root.

It is aggregate-neutral and clean, but it fails the actual rare-recovery objective. Relaxing erosion from 12 to 8 increased rare-core coverage from 0.24% to 0.42%, yet pole stayed dead and person/rider stayed at T1. This points to the M3 source mask itself being too selective; the T3/T4 rare positives likely depend on M2-only evidence.

Do not continue a plain M3 erosion sweep. The next branch should either tighten M2 instead of M3, or use M3 as seeds into M2 components:

1. T4d-M2-tight: M2 source mask, rare-core erosion 12, boundary 4 or 6.
2. T4d-hybrid: M3 seed pixels select connected M2 rare components; protect only seeded M2 components.

Promotion criterion remains rare-first: keep the T3/T4 pole/person/rider/bicycle TP gains while preserving substantially more Step B behavior than broad T3.
