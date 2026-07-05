# Class-agnostic instance-mask re-eval: AnyUp vs bilinear (Stage-1 pseudo-labels)

**Date:** 2026-07-05 · **Frames:** 2975 Cityscapes train (2972 valid paired) · **Eval:** `scripts/eval_anyup_vs_bilinear_classagnostic.py`

## Analysis question

The documented negative (`anyup_fulltrain_negative.md`) reported AnyUp-upsampled semantics HURT
whole-pipeline 19-class PQ (22.58 vs 23.17, damage in PQ_things). Objection raised: whole-pipeline
`PQ_things` confounds (a) instance-mask quality, (b) recognition, and (c) the `cluster→19-class`
majority-vote assignment — so the −1.42 might be a *class-assignment artifact*, not an instance
defect. **This eval isolates instance-mask quality by matching thing instances class-agnostically
(COCO-panoptic, IoU>0.5, no class term)** against `gtFine instanceIds`.

Decision rule fixed in advance: *AnyUp mask-RQ higher → the negative was a class artifact, reopen AnyUp.
AnyUp mask-RQ lower → fragmentation/merging confirmed at the mask level, negative double-locked.*

## Verdict: negative DOUBLE-LOCKED. AnyUp instance masks are worse, not better.

| metric (%)     | bilinear | anyup | Δ (any−bil) |
|----------------|---------:|------:|------------:|
| PQ (class-agn) | 21.31    | 20.59 | **−0.71**   |
| SQ             | 72.89    | 73.10 | +0.21       |
| **RQ**         | 29.23    | 28.17 | **−1.06**   |
| precision      | 33.31    | 35.84 | +2.54       |
| **recall**     | 26.05    | 23.20 | **−2.84**   |
| **boundary-F** | 21.78    | 19.33 | **−2.45**   |
| TP (count)     | 13580    | 12098 | −1482       |
| FP (count)     | 27194    | 21654 | −5540       |
| FN (count)     | 38559    | 40041 | +1482       |

Paired per-image (n=2972, Wilcoxon signed-rank): PQ meanΔ=−0.83 pt, 95% CI [−1.11, −0.55], p=1.8e-12;
RQ meanΔ=−1.20 pt, 95% CI [−1.61, −0.78], p=9.5e-10. AnyUp better on only 43.4% of frames.

## Three findings

1. **The RQ verdict kills the re-open hypothesis.** Mask-RQ is *lower* by 1.06 (global) / 1.20 (per-image,
   p<1e-9). Recall — the actual instance bottleneck — drops 2.84. The pre-registered flip condition
   (higher mask-RQ) is decisively false. Stripping the class-assignment confounder entirely, AnyUp's
   instances are still worse.

2. **The prior memory's *mechanism* was wrong; correct it.** The memory said AnyUp "fragments/shifts"
   masks. Fragmentation would raise FP (spurious extra masks). Instead **both TP and FP fall** (−1482,
   −5540) while FN rises (+1482) — the signature of **merging/dropping**, not fragmenting. Sharper
   *semantic* boundaries collapse adjacent same-class things into fewer, larger masks. The conclusion
   (things degrade) stands; the wording "fragments" does not. See `figure-02`.

3. **"Sharper" is false at the instance level — the whole premise for re-engineering AnyUp is dead.**
   The argument for tuning AnyUp was "sharper boundaries, just combine them better." But **boundary-F on
   matched masks is −2.45**, i.e. the surviving masks are *less* boundary-accurate, not more. Sharper
   semantic edges do not yield sharper instance masks; they yield merged ones. There is no sharpness
   signal downstream to preserve or recombine.

## What this changes

- **Belief:** the AnyUp negative is now proven at the instance-mask level, independent of semantics.
  Update `anyup_fulltrain_negative.md`: keep the RULE, replace "fragments" → "merges/drops (recall loss)",
  add the boundary-F fact.
- **The four objections, adjudicated by data:** (1) `PQ_things` unreliable → the class-agnostic mask
  metric *agrees* with `PQ_things`, so it was not misleading here; (2) trust SQ/RQ → correct, and RQ says
  no (SQ is flat +0.21, exactly why SQ alone would mislead); (3) combine to *preserve* instances → AnyUp
  destroys 1482 TP, nothing to preserve; (4) engineer *more* instances → AnyUp yields *fewer* (merging),
  the opposite direction, and the ones lost are the co-planar same-class neighbors already at the recall ceiling.
- **Decision:** bilinear cluster_probe stays production. AnyUp remains off the AAAI-2027 novelty list.
  The instance bottleneck is a *recall* problem (co-planar same-class split) that upsampling cannot touch —
  motion remains the only cue shown to move it (`motion_commonfate_probe`).

## Caveats / limitations

- Class-agnostic PQ ≠ official 19-class PQ (no class term, so absolute values sit above `PQ_things`=10.27).
  The GT convention (things = ids≥1000, crowd/ignore not specially handled) is *identical* for both
  conditions, so the any−bil delta is convention-robust; only absolute levels are affected.
- Per-frame effect size is small (Cohen's d_z ≈ −0.11) though highly significant at n=2972. The
  decision-relevant quantity is the pooled aggregate (what Stage-2 training consumes), which is
  unambiguous. Median RQ Δ = 0: most frames unchanged, a left-tail of losses drives the mean — reported honestly.
