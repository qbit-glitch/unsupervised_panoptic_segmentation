# Statistics appendix — AnyUp vs bilinear class-agnostic instance eval

## Unit of analysis & sample
- Unit: one Cityscapes train frame, paired (same frame scored under both conditions against the same GT).
- N = 2975 frames present in both label dirs with GT; 2972 valid for paired per-image PQ/RQ
  (3 frames have zero GT thing instances → per-image ratio undefined, excluded from paired tests only,
  retained in global FP accounting).
- Metric direction: higher is better for PQ/SQ/RQ/precision/recall/boundary-F.

## Descriptive (global, pooled over all frames)
| metric | bilinear | anyup |
|---|---:|---:|
| PQ | 21.31 | 20.59 |
| SQ | 72.89 | 73.10 |
| RQ | 29.23 | 28.17 |
| precision | 33.31 | 35.84 |
| recall | 26.05 | 23.20 |
| boundary-F (2px) | 21.78 | 19.33 |
| TP / FP / FN | 13580 / 27194 / 38559 | 12098 / 21654 / 40041 |

Global PQ = ΣIoU_TP / (TP + 0.5FP + 0.5FN); SQ = ΣIoU_TP/TP; RQ = TP/(TP+0.5FP+0.5FN); PQ = SQ·RQ.

## Inferential test
- **Test:** Wilcoxon signed-rank on paired per-image differences (anyup − bilinear), two-sided.
- **Why non-parametric:** per-image PQ/RQ are bounded [0,1], right-skewed, with a mass of ties
  (median RQ Δ = 0; 7.0% exact RQ ties, 2.7% PQ ties). Paired t-test assumptions (normal diffs) fail;
  signed-rank is the correct paired test and handles ties via the standard correction (scipy default).

| contrast | n | mean Δ | median Δ | 95% CI (mean) | Wilcoxon p | frac anyup better |
|---|---:|---:|---:|---|---:|---:|
| per-image PQ | 2972 | −0.00829 | −0.00493 | [−0.01107, −0.00551] | 1.8e-12 | 0.434 |
| per-image RQ | 2972 | −0.01196 | 0.00000 | [−0.01609, −0.00784] | 9.5e-10 | 0.434 |

(Δ in [0,1] units; ×100 for PQ points: PQ −0.83 pt, RQ −1.20 pt.)

## Effect size
- Cohen's d_z (paired) = meanΔ / sd(Δ): **PQ d_z ≈ −0.107, RQ d_z ≈ −0.104** — *small* per-frame effect.
- Practical: anyup better on 43.4% of frames, worse-or-tie on 56.6%. Median RQ Δ = 0 → most frames
  unchanged; a left tail of losses moves the aggregate. Honest reading: **small but robust and
  directionally consistent**; significance comes from n, not from a large per-frame gap.
- Decision-relevant aggregate (pooled, what Stage-2 ingests): PQ −0.71, RQ −1.06, recall −2.84,
  boundary-F −2.45 — unambiguous and not effect-size-limited.

## Multiple comparisons
Two pre-specified paired contrasts (PQ, RQ). Both p < 1e-9; Bonferroni (×2) leaves both < 2e-9.
No correction changes any conclusion.

## Assumptions checked / blockers
- **GT convention:** thing GT = `instanceIds ≥ 1000`; crowd/group and ignore regions not specially
  handled (no iscrowd suppression). This inflates FP for *both* conditions equally; the paired delta is
  robust, absolute levels are not comparable to official 19-class PQ. Stated, not hidden.
- **No class term:** by design (isolates mask quality). Do not read these as panoptic PQ.
- **Pred min-area:** none applied here; generator already applied A_min=1000 upstream, identically for both.
- **Not blocked:** paired design, large n, identical GT → valid significance claims. No missing seeds
  (single deterministic pass, no stochastic component in the eval).
