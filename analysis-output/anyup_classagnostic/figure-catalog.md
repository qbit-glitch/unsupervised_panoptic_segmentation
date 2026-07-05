# Figure catalog — AnyUp vs bilinear class-agnostic instance eval

## figure-01-anyup-vs-bilinear (main comparison)
- **File:** `figures/figure-01-anyup-vs-bilinear.{pdf,png}`
- **Source:** `analysis-output/anyup_classagnostic/summary.json` (global) + per-image PQ arrays.
- **Panels:** (a) grouped bars of PQ/SQ/RQ/precision/recall/boundary-F, bilinear vs anyup;
  (b) histogram of per-image PQ Δ (anyup − bilinear), zero line + mean line.
- **Error bars:** none on (a) — global pooled scalars; the per-image dispersion lives in panel (b)
  and the CIs in `stats-appendix.md`.
- **Caption requirement:** state n=2972 paired frames, class-agnostic IoU>0.5 matching, and that these
  are NOT class-aware panoptic PQ.
- **Key observation:** every bar except SQ and precision is lower for anyup; the PQ Δ histogram is
  centered slightly left of zero (mean −0.83 pt) with most mass at 0 and a left tail.
- **Interpretation checklist:** (1) exists to test "did AnyUp help instance masks"; (2) reader should
  notice RQ↓, recall↓, boundary-F↓ with SQ flat; (3) changes belief → the AnyUp negative is an
  instance-mask defect, not a class-assignment artifact.
- **Caveat:** SQ flat (+0.21) is the trap metric — matched masks are equally good, but far fewer match.

## figure-02-count-decomposition (mechanism / supporting)
- **File:** `figures/figure-02-count-decomposition.{pdf,png}`
- **Source:** `summary.json` TP/FP/FN and precision/recall.
- **Panels:** (a) TP/FP/FN counts, bilinear vs anyup, with Δ annotations; (b) precision–recall point
  pair with an arrow bilinear→anyup.
- **Key observation:** anyup lowers BOTH TP (−1482) and FP (−5540) and raises FN (+1482); the P–R arrow
  points up-left (precision +2.5, recall −2.8).
- **Interpretation checklist:** (1) exists to distinguish *merging* from *fragmenting*; (2) reader should
  notice FP falls (fragmentation would raise it) → fewer, larger masks; (3) changes belief → correct the
  prior memory's "fragments" wording to "merges/drops," and explains the recall loss mechanistically.
- **Caveat:** counts pooled over all frames; per-frame variance not shown (not needed for the directional
  mechanism claim).
