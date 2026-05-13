# T4 Rare-Core Protection Decision

Date: 2026-04-27

## Verdict

Keep the implementation, but do not promote this exact T4 configuration as the Stage-2 root yet.

The sharper rare-core policy works mechanically: Step B is no longer fully disabled, and T4 recovers part of the T3 regression. The rare core is still too broad, though, so T4 remains slightly below T1.

## What Changed

Implemented `--protected_mask_policy rare_core` in `scripts/refine_cn_simcf.py`.

- Step A: no protection mask.
- Step B: normal same-cluster + similarity merge logic, except merge candidates are blocked when their merge corridor crosses a rare core.
- Step C: only rare-core pixels are protected from depth outlier masking.
- Legacy behavior remains available with `--protected_mask_policy all`.

## Metrics

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore | Decision |
|---|---:|---:|---:|---:|---:|---|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 | Reference |
| T3 broad protected v2 | 25.13 | 33.83 | 13.17 | 56.21 | 1.04 | Too blunt |
| T4 rare-core protected | 25.15 | 33.85 | 13.19 | 56.22 | 1.18 | Partial recovery |

## Generation Stats

| Run | Protected coverage | Rare-core coverage | Step B merges | Step B blocked | Step C masked |
|---|---:|---:|---:|---:|---:|
| T3 broad protected v2 | 31.57% | n/a | 0 | n/a | 65,042,279 px / 1.04% |
| T4 rare-core protected | 31.57% | 22.86% | 39 | 1,691 | 73,475,432 px / 1.18% |

## Rare-Class Signal

| Class | T1 TP | T3 v2 TP | T4 TP | T4 PQ | Note |
|---|---:|---:|---:|---:|---|
| pole | 0 | 1 | 1 | 0.02 | Rare signal preserved |
| traffic light | 0 | 0 | 0 | 0.00 | Still dead |
| motorcycle | 0 | 0 | 0 | 0.00 | Still dead |
| person | 355 | 358 | 358 | 1.48 | T3 rare gain preserved |
| rider | 53 | 55 | 55 | 2.41 | T3 rare gain preserved |
| bicycle | 145 | 147 | 147 | 2.53 | T3 rare gain preserved |

## Interpretation

T4 confirms the hypothesis direction: rare protection can preserve weak rare-class positives without catastrophically damaging the label map. But the current core extraction is not selective enough. A 22.86% rare-core mask still blocks 1,691 Step B merge candidates and leaves Step B at only 39 merges, far below T1's useful merge behavior.

## Next Configuration

Run a stricter T4b:

1. Rebuild rare masks with `--agreement_required 3`.
2. Keep `--protected_mask_policy rare_core`.
3. Increase `--rare_core_erode_px` from `8` to `12` or `16`.
4. Keep `--rare_core_boundary_px 8` initially, then lower it only if Step B remains too suppressed.
