# NeurIPS 2026 Review — Reviewer 3 (Empirical / Experimental Rigor)

**Paper**: Depth-Conditioned Pseudo-Labels for Scene-Centric Unsupervised Panoptic Segmentation
**Track**: NeurIPS 2026 — Main Conference

---

## Summary

The paper proposes a monocular replacement for the stereo+flow pseudo-label construction of CUPS (Hahn et al., CVPR 2025). The substitution is implemented through (a) a small (~225K-param) residual adapter, DCFA, that conditions a frozen unsupervised CAUSE-TR semantic code on monocular depth via sinusoidal positional encoding, and (b) SIMCF, a learning-free three-stage filter (per-region majority cleaning, DINOv3-based adjacent merge, depth-implausibility void rejection) over candidate panoptic labels. The Stage-1 generator feeds a CUPS-style Cascade Mask R-CNN (Stage-2 bootstrapping + 3 rounds of EMA self-training) on a frozen DINOv3 ViT-B/16 backbone.

Headline results on Cityscapes val: 35.83 PQ vs the published CUPS 27.80 PQ (+8.03), with a controlled monocular baseline at 32.76 PQ that isolates +3.07 PQ to the DCFA+SIMCF mechanisms. Cross-dataset zero-shot transfer is reported on KITTI (+9.4 PQ over CUPS), Waymo V2 (+17.1), and Mapillary v2 (no CUPS reference). The authors honestly admit (§6) that the 35.83 PQ is from a *single* Stage-3 run and that a same-backbone CUPS rerun was not executed (advisor-stated scope).

---

## Strengths

1. **Clean controlled comparison via the matched monocular baseline (32.76 PQ).** Reporting a same-backbone, same-trainer baseline that does *not* contain DCFA or SIMCF is exactly the right thing to do, and isolates +3.07 PQ to the proposed mechanisms. This is more empirically honest than many concurrent unsupervised-panoptic papers that conflate backbone, recipe, and method changes.

2. **Honest limitations section.** §6 explicitly enumerates: (i) six dead classes, (ii) COCO-Stuff-27 failure, (iii) co-planar pedestrian bottleneck, (iv) missing same-backbone CUPS rerun, (v) single Stage-3 seed, (vi) missing Stage-3 raw-K=80 control. This level of self-disclosure is rare and welcome.

3. **Per-class table is reported transparently (Table A.5).** Six vocabulary-dead classes (parking, guard rail, tunnel, polegroup, caravan, trailer) are shown at 0.0 PQ rather than hidden; motorcycle's anomalous SQ=100 / RQ=0.1 (single-instance match) is reported. This is exactly the failure-disclosure norm I want to see.

4. **DepthG-style loss-side baseline is included (Table 4).** The author actually retrained DepthG-style loss-side conditioning at matched parameter budget and preservation strength, finding it 1.39 PQ below DCFA. This is a non-trivial empirical isolation of the architectural conditioning axis.

5. **SIMCF threshold sensitivity sweep (Table B.1).** A ±12% sweep around the paper's chosen thresholds yields a 0.71 PQ band, smaller than the +1.31 PQ DCFA+SIMCF gain, supporting a "robust to threshold perturbation, not tuned-on-test" claim.

6. **Reproducibility scaffolding.** Seeds are fixed (DCFA=42, k-means=0, dataloader=42), Python/PyTorch/Detectron2/CUDA versions are pinned, hardware (2× GTX 1080 Ti, 11 GB) is reported, wall-clock is broken down per stage (~58 GPU-hours total), and a release commitment is made for code + Stage-1 pseudo-labels + Stage-2/3 checkpoints. This is much better than the modal NeurIPS submission.

7. **OOD honesty (§5.4 and Table C.1).** COCO-Stuff-27 failure (7.83 PQ) is reported in the main text, with vocabulary-mismatch failure-mode analysis in supp §A.5 (no attempt to bury or excuse it).

---

## Weaknesses

### W1. Single-seed Stage-3 result for the headline metric (CRITICAL)

§6 explicitly admits "the 35.83 PQ is from a single Stage-3 run; multi-seed runs are needed to report mean±std." For a headline comparison that crosses the prior SOTA (CUPS 27.80) by +8 PQ, this is a real concern. NeurIPS 2026 is a benchmark-driven venue and Stage-3 EMA self-training is well-known for high variance across rounds (CUPS itself reports ±0.5–1.0 PQ variance across self-training rounds in their supplementary). At minimum, the paper should report:

- Mean ± std over 3 seeds for the matched monocular baseline (32.76)
- Mean ± std over 3 seeds for DCFA+SIMCF (35.83)
- A statistical significance test (paired t-test or bootstrap CI on the +3.07 PQ delta)

**Mitigating factor**: The +3.07 PQ delta on the controlled comparison is large enough that, *if* the Stage-3 variance is in the typical CUPS range (±0.5–1.0 PQ), the mechanism contribution would survive multi-seed verification. The Stage-1 component-isolation table (Table 3) reports +1.31 PQ at the deterministic pseudo-label level (k-means seed fixed), which is a non-stochastic quantity — that part of the contribution is on solid ground. But the *amplification* from +1.31 PQ at Stage-1 to +3.07 PQ at Stage-3, which is the part that crosses CUPS, is the part most exposed to Stage-3 stochasticity.

**My ask**: Even one additional Stage-3 seed (total of 2) would provide a 2-point spread that, combined with CUPS's reported variance, would let me bound the claim.

### W2. Missing +DCFA-only Stage-3 checkpoint (acknowledged in supp §A.4)

The main ablation table (Table 3) toggles DCFA and SIMCF *at the Stage-1 pseudo-label level* — DCFA-only, SIMCF-only, and both. This is good. But the corresponding Stage-3 isolation (does +DCFA alone close the +3.07 PQ at the trained-network level? does +SIMCF alone?) is *not* reported. Supp §A.4 admits "the existing checkpoint set does not contain a Stage-3 model trained on DCFA-corrected pseudo-labels with SIMCF disabled, and we did not regenerate one for this submission."

This is a real ablation gap. Without it, the reader cannot tell whether the +3.07 PQ Stage-3 lift is dominated by DCFA, SIMCF, or genuinely requires both. The Stage-1 table (Table 3) shows DCFA-only and SIMCF-only at near-identical PQ (25.22 vs 25.27), but the *combined* Stage-1 lift is only +0.58 over either single component, while the Stage-3 lift of +3.07 PQ may come predominantly from one of the two if EMA self-training amplifies asymmetrically. This matters for downstream researchers who may want to adopt only one mechanism.

### W3. K-sweep is missing.

The paper claims k=80 over-clustering matters (§3.2: "collapsing to 27 classes at this stage would erase fine visual modes"; §5.3 cites supp Figure A.4 as visualization of fine-grained discrimination). However, **no k-sweep is reported**. CUPS Table 7b is cited as evidence that monotonic improvement holds (k=27→27.8, k=40→30.3, k=54→30.6 in CUPS), but those are CUPS numbers, not this paper's numbers. The author should report the matched DCFA+SIMCF Stage-3 PQ at k ∈ {27, 40, 54, 80} to:

- Show that k=80 is the sweet spot for *this* pipeline (not just CUPS's)
- Quantify how much of the +3.07 PQ contribution-isolated lift is attributable to k=80 vs. DCFA+SIMCF

If k=27 with DCFA+SIMCF still reaches >32 PQ, the over-clustering choice is not load-bearing. If it crashes to <28 PQ, then "DCFA+SIMCF + k=80" should be claimed as a coupled contribution, not factorized into "DCFA+SIMCF" alone.

### W4. Stage-3 raw-K=80 control is missing (acknowledged in §6).

The current Stage-3 controlled comparison runs DCFA+SIMCF Stage-1 → Stage-3 against monocular-baseline Stage-1 (no DCFA/SIMCF) → Stage-3. The natural counterfactual the paper does *not* run is: raw K=80 pseudo-labels (no DCFA, no SIMCF) → full CUPS Stage-2/3 recipe → final PQ. §6 (vi) flags this. Without this row, we cannot tell whether DCFA+SIMCF at Stage-1 are necessary at all, or whether Stage-2/3 on raw labels would converge to a similar asymptote with more rounds. This is more important than W1 (single-seed) for the *mechanism* claim.

### W5. Cross-dataset comparison fairness (Table 2).

The Mapillary number (39.19 PQ at 19-class, 27.79 PQ at 27-class) is reported with **no public baseline** for the same protocol — neither CUPS nor U2Seg report Mapillary numbers in their papers. The author should at least:

- Run the released CUPS checkpoint on Mapillary under the paper's "paper-defined remap" (footnote †) to provide a like-for-like CUPS Mapillary number, even if approximate.
- Or, more cautiously, drop the Mapillary column from the headline table and demote it to qualitative-only in the appendix, since the column is currently a single-method result and not a comparison.

The KITTI and Waymo numbers (+9.4 and +17.1 PQ over CUPS) are credible because CUPS reports them in its Table 2, so the comparison is at-least-protocol-matched. But the +17.1 PQ Waymo gap is suspiciously large for a method that gains only +3.07 PQ on the in-domain controlled setup. Some of this gap may reflect different Waymo-loader implementations between the two papers; the footnote ¶ hints at this ("Waymo via CUPS Waymo loader"). The author should clarify whether they ran CUPS's released checkpoint through their own evaluation harness on Waymo to confirm the +17.1 PQ delta is not partially an evaluation-protocol artifact.

### W6. Backbone confound is not closed (acknowledged in §6 as "advisor-stated scope").

The 35.83 vs 27.80 PQ gap is a joint effect of (i) DCFA+SIMCF, (ii) the monocular pseudo-label substitution, and (iii) the DINOv3 ViT-B/16 backbone replacing CUPS's DINO ResNet-50. The matched monocular baseline (32.76) closes the backbone confound *for the contribution-isolated +3.07 PQ claim*, which is fine. But the +5.0 PQ gap between the matched monocular baseline (32.76) and CUPS (27.80) is attributed in the abstract to "the pseudo-label-source substitution alone" — this is **not isolated**, because the backbone has also changed. The author's advisor told them they don't need to re-run CUPS with DINOv3, and that's a defensible scope for a single submission (DINOv3 inference on Cityscapes train + Cascade Mask R-CNN training with ResNet-50 swap is non-trivial), but the abstract claim that "the +5.0 PQ gap reflects the pseudo-label-source substitution alone" is over-stated — it reflects pseudo-label-source + backbone substitution jointly. This is a **wording fix**, not an additional experiment, but it should be made.

### W7. Hyperparameter "fixed before consulting annotations" claim is hard to verify.

§4.1 and supp §B both state "All operating hyperparameters were fixed before consulting Cityscapes annotations; the SIMCF sweep is a retrospective sensitivity analysis, not a model-selection procedure." This is a strong reproducibility claim. Without an audit trail (commit history, dated config files, or a development log), I have to take the author's word for it. The SIMCF sweep at Stage-1 train-set PQ (25.27–25.98) showing the chosen baseline (25.87) sits near the *middle* of the band, not at the maximum (25.98 is the "tighter" setting), is mild supporting evidence that the chosen settings were not retrospectively tuned to the maximum. But the author should consider, for camera-ready, releasing the dated configs.

### W8. Per-class motorcycle anomaly (SQ=100, RQ=0.1) deserves a sentence.

Table A.5 shows motorcycle at PQ=0.1 with SQ=100 and RQ=0.1. This means exactly one near-perfect mask was matched, and zero of the other ground-truth motorcycle instances were recovered. Discussion §A.7 mentions "motorcycle matches a single near-perfect instance" but does not explain why the recall is zero. Is motorcycle merged into bicycle? Into car? The failure mode is not analyzed.

---

## Soundness: 3 (good)

The contribution-isolation methodology (matched monocular baseline) is correct. The Stage-1 ablations are clean. The DepthG-style loss-side baseline (Table 4) is a real empirical isolation of the conditioning-axis claim. The OOD failure (COCO-Stuff-27) is honestly reported. Where soundness drops below 4: single-seed Stage-3 (W1), missing Stage-3 isolation of DCFA-only and SIMCF-only (W2), missing k-sweep (W3), missing Stage-3 raw-K=80 control (W4), and Mapillary baseline-free comparison (W5). The honest §6 limitations partially mitigate but do not close these gaps.

## Presentation: 3 (good)

The paper is well-organized: pipeline figure, three-stage method section, four ablation paragraphs (pipeline progression, component contribution, SIMCF sensitivity, conditioning mechanism), main+transfer+OOD tables. Some captions over-rely on cross-references — Table 2's footnotes (§, ¶, †) should be resolved in-table where possible. The "27 vs 27" ambiguity (CAUSE 27 concept prototypes vs Cityscapes 27 evaluation classes) is finally clarified in supp §B.6 ("The Over-Cluster-to-Concept Map φ Uses No Ground-Truth Labels"), but should be flagged earlier in the main text — at least one reviewer will get confused on a fast read. The abstract overclaims the +5.0 PQ baseline-vs-CUPS gap as "pseudo-label-source substitution alone" (W6); fix this wording.

## Contribution: 3 (good)

DCFA is a defensible contribution — to my knowledge, the first feature-level residual adapter applied to a frozen unsupervised concept-clusterbook code for monocular depth conditioning. The +1.31 PQ Stage-1 / +3.07 PQ Stage-3 deltas are not headline-grabbing in absolute terms, but the contribution-isolated framing is the right comparison. SIMCF is more engineering than novel — three sequential heuristic checks over candidate labels — but the diagnostic numbers (44→22 instances/image, 5,502→14,965 px median size, 50.7%→28.0% stuff contamination) demonstrate it does measurable work. The crossing of CUPS (27.80 → 35.83) is real and large, *if* the Stage-3 variance under multi-seed re-runs is in the typical range. The contribution would be elevated to 4 with W1, W2, W3, W4 closed.

## Overall: 6 (borderline accept, leaning weak accept)

The contribution-isolated +3.07 PQ on a matched-backbone, matched-trainer baseline is the right experimental framing for an unsupervised-panoptic paper, and the author's honesty in §6 is rare and should be rewarded. The +8 PQ headline crossing of CUPS is a large delta, even net of the DINOv3 backbone confound. The single-seed Stage-3 issue (W1) and missing Stage-3 ablation isolation (W2, W4) are real concerns that hold this back from a clear-accept (7), but for an unsupervised-panoptic paper at NeurIPS — where the modal submission has *neither* a contribution-isolated baseline *nor* an honest limitations section — this paper is materially above the median.

I would not auto-reject for single-seed given the magnitude of the delta, but I would expect the author to add at least one additional Stage-3 seed for the camera-ready and to either (a) regenerate the missing +DCFA-only Stage-3 checkpoint or (b) more aggressively scope the Stage-3 contribution claim to "+3.07 PQ from the joint application of DCFA+SIMCF, mechanism isolation deferred to future work."

Final recommendation: **6, weak accept**. Would move to 7 if W1 (multi-seed) and W2 (DCFA-only Stage-3 checkpoint) are addressed in the rebuttal.

## Confidence: 4 (confident, but not certain)

I am confident in the unsupervised-panoptic literature and in the CUPS baseline, and have read the paper carefully including the supplementary. I am not 100% certain that the +17.1 PQ Waymo gap is not partially a loader-protocol artifact (W5), and I have not independently re-run any of the reported numbers. My experimental-rigor concerns are clearly identified above and would be addressable in the rebuttal.

---

## Questions for Authors

1. **Multi-seed Stage-3**: Can you run at least one additional Stage-3 seed for both the matched monocular baseline (32.76 PQ) and DCFA+SIMCF (35.83 PQ) for the rebuttal? Even N=2 would let me bound the claim that the +3.07 PQ delta exceeds Stage-3 variance.

2. **Stage-3 DCFA-only**: Why was the +DCFA-only Stage-3 checkpoint not regenerated? Estimated cost is ~16 GPU-hours on your reported 2× GTX 1080 Ti hardware (one Stage-3 round), and would close the largest ablation gap. Will you regenerate it for the rebuttal?

3. **K-sweep**: At fixed DCFA+SIMCF, what is the Stage-3 PQ at k ∈ {27, 54, 80, 120}? Specifically, does k=27 + DCFA + SIMCF + Stage-3 still reach >32 PQ? This determines whether the over-clustering choice is load-bearing or substitutable.

4. **Stage-3 raw-K=80 control**: Per §6 (vi), what is the Stage-3 PQ if the raw K=80 pseudo-labels (no DCFA, no SIMCF) are fed through the full Stage-2/3 recipe? This is a different question from the matched monocular baseline (32.76), which is "Stage-1 raw K=80 → Stage-3", but the matched baseline IS this control. Please clarify whether the 32.76 PQ baseline IS the "raw K=80 → Stage-3" number, or whether the §6 (vi) limitation refers to a different missing experiment.

5. **Mapillary CUPS baseline**: Can you run the released CUPS checkpoint on Mapillary v2 under your remap to provide a like-for-like baseline? Without it, the Mapillary column is single-method.

6. **Waymo loader audit**: Did you re-run CUPS's released checkpoint through your own Waymo evaluation harness? The +17.1 PQ Waymo gap is suspiciously large given the +3.07 PQ in-domain controlled gain, and one possible explanation is loader/protocol mismatch.

7. **Abstract wording (W6)**: Will you revise the abstract claim that "the +5.0 PQ gap reflects the pseudo-label-source substitution alone"? The backbone has also changed (DINOv3 ViT-B/16 vs CUPS's DINO ResNet-50), so this gap is not pseudo-label-source-isolated.

8. **Motorcycle PQ=0.1 analysis (W8)**: Where do the un-recovered motorcycle ground-truth instances go? Bicycle? Car? Person? A one-sentence diagnosis would close out the per-class table honesty.

9. **Hyperparameter fixing claim**: Could you, for camera-ready, release dated commit hashes or config files demonstrating that operating hyperparameters were fixed before annotation consultation? This would transform a strong but unverifiable claim into a verifiable one.

10. **Stage-3 variance precedent**: What variance does CUPS report across its three EMA rounds? If their published variance is, e.g., ±0.4 PQ, then a single-seed +3.07 PQ delta is more credible than if their reported variance is ±2.0 PQ. Please report the CUPS variance number for context.
