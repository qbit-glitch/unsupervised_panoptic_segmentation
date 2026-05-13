# T3 Protected Mask Decision

Date: 2026-04-27

## Verdict

Reject the current T3 protected-mask ablation.

The corrected run loaded the rare protected masks successfully, but the mask is too broad for the current CN-SIMCF recipe. It blocks Step B entirely, which removes the instance-merge mechanism that T1 still needs.

## Metrics

| Run | PQ | PQ_stuff | PQ_things | mIoU | Ignore | Decision |
|---|---:|---:|---:|---:|---:|---|
| T1 CN-SIMCF | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 | Reference |
| T3 first protected run, Hungarian | 25.20 | 33.92 | 13.21 | 56.26 | 1.38 | Invalid/no-op, mask loader missed files |
| T3 corrected protected v2, Hungarian | 25.13 | 33.83 | 13.17 | 56.21 | 1.04 | Reject |

## Generation Stats

| Output | Step A changed | Step B merges | Step C masked | Notes |
|---|---:|---:|---:|---|
| `cups_pseudo_labels_dcfa_cn_simcf_protected` | 0 | 109 | 86,255,060 px / 1.38% | Invalid/no-op: protected masks were not loaded |
| `cups_pseudo_labels_dcfa_cn_simcf_protected_v2` | 0 | 0 | 65,042,279 px / 1.04% | Corrected loader; protection active |

## Rare-Class Signal

| Class | T1 PQ | T3 v2 PQ | T1 TP | T3 v2 TP | Note |
|---|---:|---:|---:|---:|---|
| pole | 0.00 | 0.02 | 0 | 1 | One recovered match, not enough to matter |
| traffic light | 0.00 | 0.00 | 0 | 0 | Still dead |
| motorcycle | 0.00 | 0.00 | 0 | 0 | Still dead |
| person | 1.47 | 1.48 | 355 | 358 | Tiny local gain |
| rider | 2.33 | 2.41 | 53 | 55 | Tiny local gain |
| bicycle | 2.50 | 2.53 | 145 | 147 | Tiny local gain |

## Notes

- The first `T3_protected.json` static-centroid eval collapsed to PQ=0.33 and should not be compared against T1, because T1/T2 were evaluated in Hungarian mode.
- The first `T3_protected_hungarian.json` exactly matched T1 because the protected-mask loader searched for `_leftImg8bit` mask stems while the builder saved base stems.
- The corrected loader produced the real ablation. It reduced ignored pixels from 1.38% to 1.04%, but total PQ regressed by 0.07.
- The current mask coverage is high: 31.57% of image pixels. That is probably too broad to apply to both Step B and Step C.

## Next Branch

The next ablation should keep T1 as the parent and make protection less destructive:

1. Rebuild masks with `--agreement_required 3`.
2. Apply rare protection only to Step C semantic relabel masking, not Step B instance merge eligibility.
3. Keep Step B at the T1 setting unless a merge crosses a high-confidence rare-class core.
