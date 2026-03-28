# NeurIPS 2026 Review Audit

**Paper:** Monocular Pseudo-Label Generation for Unsupervised Panoptic Segmentation
**Review Date:** 2026-03-27
**Reviewer:** Simulated Brutal NeurIPS Reviewer

---

## 1. Summary

The paper proposes replacing CUPS's stereo-video pseudo-label pipeline with a monocular alternative composed of: (1) K-means overclustering (k=80) on CAUSE-TR 90-dim features, (2) SPIdepth monocular depth estimation, and (3) Sobel gradient thresholding + connected components for instance decomposition. The monocular pseudo-labels achieve PQ=26.74 (PQ_things=19.41) on Cityscapes val. When these pseudo-labels train a Cascade Mask R-CNN with a DINOv3 ViT-B/16 backbone and self-training, the full pipeline reaches PQ=32.76, surpassing CUPS (27.8) by 4.96 points. The paper also documents a 64-configuration hyperparameter sweep and extensive failed approaches.

---

## 2. Strengths

**S1.** The paper is remarkably honest. The metric confusion episode, the complete catalog of failed approaches, and the explicit acknowledgment that the backbone upgrade is "orthogonal engineering" rather than methodology are refreshing. Most papers would hide these. This transparency has scientific value.

**S2.** The instance method comparison (Table 6) is comprehensive — 8 methods compared under identical conditions. The depth-guided method's dominance (+8.21 PQ_things over the next best) is convincing for the driving domain.

**S3.** The overclustering-depth interaction (Tables 4-5) is a genuinely interesting empirical finding. The monotonic decrease of depth benefit from +2.51 (k=50) to +0.00 (k=300) is clean and informative.

**S4.** The practical contribution of eliminating stereo requirements is meaningful for real-world applicability.

**S5.** The codebase is extensive (~6000 lines of implementation) with verifiable eval files.

---

## 3. Weaknesses

### MAJOR

**W1. The core "novelty" is application of elementary image processing techniques, not a methodological contribution.**

The three claimed contributions decompose as follows:

- **K-means overclustering with larger k:** Running K-means with k=80 instead of k=27 on pre-existing CAUSE features is hyperparameter tuning, not a method. The CUPS paper itself (Table 7b) already showed that overclustering helps: k=27 -> PQ 27.8, k=40 -> 30.3, k=54 -> 30.6. The paper does not cite or discuss this CUPS result, making the overclustering appear more novel than it is.

- **Sobel + connected components on depth maps:** This is textbook image processing from Gonzalez & Woods (1992). Applying Gaussian smoothing, Sobel gradient computation, thresholding, and connected-component analysis to segment objects by depth discontinuities involves zero learned components and zero algorithmic novelty. The paper frames this as "depth-gradient instance decomposition" but it is thresholded edge detection followed by flood fill.

- **64-config hyperparameter sweep:** Observing that hyperparameters interact is not a contribution. Every practitioner running a grid search observes tradeoffs. The paper elevates this to "a previously undocumented tradeoff" and "itself a contribution" — this is overclaiming.

**W2. The headline result (PQ=32.76) is almost entirely attributable to the DINOv3 backbone, not the pseudo-labels.**

Table 2 is the most important table in the paper, and it undermines the claimed contribution:

| Config | PQ | Delta vs CUPS |
|---|---:|---:|
| Ours (RN50 backbone) | 24.68 | **-3.12** |
| Ours (DINOv3 backbone, Stage 2) | 27.87 | +0.07 |
| Ours (DINOv3 backbone, Stage 3) | 32.76 | +4.96 |

With the **same backbone as CUPS** (DINO RN50), the monocular pseudo-labels produce a model that is **3.12 points WORSE** than CUPS. The entire improvement — and then some — comes from swapping to DINOv3 ViT-B/16, which is acknowledged as "orthogonal engineering." The paper repeatedly claims the pseudo-labels are "the foundation" and the backbone is "an amplifier," but the data shows the opposite: the pseudo-labels are the bottleneck and the backbone is the rescue. **A fair comparison requires running CUPS with the same DINOv3 backbone**, which the paper conspicuously does not do.

If CUPS + DINOv3 achieves, say, PQ=35+, then the method's advantage disappears entirely. The absence of this ablation is the single most critical experimental gap in the paper.

**W3. The "monocular surpasses stereo" claim (PQ_things 19.41 vs 17.70) is cherry-picked.**

The paper's central narrative is that monocular pseudo-labels match or beat stereo. But:

- **Overall PQ**: Ours = 26.74, CUPS = 27.8. **CUPS wins by 1.06.**
- **PQ_stuff**: Ours = 32.08, CUPS = "comparable" (no number given for CUPS pseudo-label stuff quality in Table 1, making it impossible to verify).
- **PQ_things**: Ours = 19.41, CUPS = 17.70. Ours wins by 1.71.

The paper cherry-picks the one sub-metric where monocular wins while losing on the aggregate. Moreover, the CUPS PQ_things=17.70 in Table 1 appears to be **pseudo-label quality**, while in Table 2 it appears as the **trained model's** PQ_things. If CUPS's trained model PQ_things is also 17.70, that seems suspiciously low — typically trained models significantly exceed their pseudo-labels. This conflation needs clarification.

**W4. SPIdepth was trained on Cityscapes video sequences — the method is not truly "monocular."**

The paper states: "SPIdepth...trained solely on photometric consistency from Cityscapes video sequences." And in the Discussion: "The reliance on SPIdepth, which was trained on Cityscapes video sequences, introduces implicit domain dependence."

This is a critical flaw in the framing. SPIdepth requires **temporal video** from the same dataset during its training, just as CUPS requires stereo video. The distinction between "stereo video at pseudo-label time" (CUPS) and "monocular video at depth-model training time" (ours) is practically meaningful but theoretically thin — both methods exploit multi-frame geometric signals from Cityscapes. The paper's abstract states we "eliminate the stereo requirement" but we have merely **moved the multi-frame requirement** to a different stage of the pipeline. A truly monocular approach would use a depth model trained on a different dataset (e.g., MiDaS, ZoeDepth, Depth Anything) — and zero ablation on depth model choice is provided.

**W5. PQ_things = 34.13 at Stage 3 is not adequately explained and may indicate evaluation issues.**

The progression: pseudo-label PQ_things = 19.41 -> Stage 2 trained PQ_things = 23.17 -> Stage 3 PQ_things = **34.13**. Self-training nearly doubles thing-class quality. Meanwhile, PQ_stuff barely changes (30.63 -> 31.95). Having PQ_things > PQ_stuff is highly unusual for unsupervised panoptic methods and suggests either:
- The self-training massively improves instance recall (RQ goes from 36.29 to 40.75, a 4.46 increase, not enough to explain the 10.96 PQ_things jump)
- There is a bug or metric artifact in Stage 3 evaluation
- The DINOv3 model is memorizing training-domain-specific instance patterns

The paper provides no analysis of this anomalous jump. Table 3 reports only aggregate PQ for intermediate steps, hiding whether PQ_things grows steadily or jumps suddenly. This needs per-metric tracking at every checkpoint.

**W6. Single dataset, single seed, no error bars, no statistical significance.**

All results are single-run on Cityscapes val (500 images). No confidence intervals, no multi-seed runs, no significance tests. For a PQ improvement of 4.96 points to be credible at NeurIPS, it must be demonstrated that it is not a lucky seed or a favorable random initialization.

**W7. No COCO-Stuff evaluation despite claiming generalizability.**

Cityscapes-only evaluation is insufficient for a NeurIPS paper claiming methodological contribution. The cross-dataset "evaluation" (Mapillary PQ=2.2, KITTI PQ=0.0, COCONUT PQ=0.9) actually demonstrates that the method **catastrophically fails** outside Cityscapes, yet this is reported casually in one paragraph.

### MINOR

**W8. The paper structure is a project report, not a conference paper.** NeurIPS has a 9-page limit (main text). This report includes: engineering bug documentation (CUPS spatial misalignment bug, DINOv3 custom ViT bugs), DDP evaluation overestimation, TTA memory management, a metric confusion narrative, a mobile RepViT tangent, cross-dataset results that prove nothing. These belong in supplementary material or a technical report, not the main paper.

**W9. The "stuff-things classifier" (4th component) is mentioned exactly once and never ablated.** How does CLS token self-attention distinguish stuff from things? What accuracy does it achieve? What happens if you use ground-truth stuff/things labels?

**W10. No ablation on the depth model.** The entire pipeline depends on SPIdepth quality. What happens with ZoeDepth, MiDaS v3.1, Depth Anything v2, or Metric3D? If performance degrades substantially, the contribution is "SPIdepth is good at Cityscapes depth" not "monocular depth enables panoptic segmentation."

**W11. Table 1 is incomplete.** CUPS pseudo-label PQ, PQ_stuff are listed as "--". Without these numbers, the claim that monocular pseudo-labels are "comparable" overall cannot be verified.

**W12. Missing comparison with U2Seg** (Niu et al., NeurIPS 2023), which is cited in the references but never compared against in any table. This is a directly relevant baseline for unsupervised panoptic segmentation.

**W13. Table numbering is inconsistent.** Table 9 appears before Tables 8a/8b.

**W14. The "7 dead classes" claim needs verification.** Do you mean that specific CAUSE centroids collapse, or that the Hungarian matching maps no centroid to these classes? The distinction matters for understanding whether this is a CAUSE bug or an inherent property of overclustering.

**W15. Figures are ASCII art.** NeurIPS requires publication-quality figures. All four figures are text-based box diagrams that would not survive the camera-ready standard.

**W16. DINOv3 citation is to an unpublished technical report.** "Oquab et al., 2025" has no venue, no arXiv ID, no page numbers. If the backbone that provides the majority of the improvement comes from an unverifiable source, the reproducibility is compromised.

---

## 4. Questions for Authors

**Q1.** What is CUPS's PQ when trained with the same DINOv3 ViT-B/16 backbone and self-training? Without this control, the improvement cannot be attributed to pseudo-labels vs. backbone.

**Q2.** What happens with a different monocular depth model (e.g., Depth Anything v2) that was NOT trained on Cityscapes video?

**Q3.** Can you provide per-class PQ_things breakdown at Stage-3 step 8000? Specifically, what is person class PQ? Person PQ=4.2 is reported for pseudo-labels but never for the trained model.

**Q4.** Why are intermediate Stage-3 checkpoints (Table 3) missing PQ_things and PQ_stuff? What does the PQ_things trajectory look like?

**Q5.** Can you run the identical pipeline with 2 additional seeds and report mean +/- std?

**Q6.** CUPS Table 7b already shows k=54 -> PQ=30.6. How does your k=80 interact with CUPS's own overclustering results? Did you try k=54 with depth splitting?

---

## 5. Missing References

- Depth Anything (Yang et al., CVPR 2024) and Depth Anything v2 — directly relevant monocular depth alternatives
- ZoeDepth (Bhat et al., 2023) — monocular depth baseline
- MiDaS v3.1 (Ranftl et al., 2022) — monocular depth baseline
- U2Seg (Niu et al., NeurIPS 2023) — cited but not compared
- CUPS Table 7b overclustering results — should be discussed

---

## 6. Detailed Sentence-Level Issues

| Line(s) | Claim | Issue |
|---------|-------|-------|
| Abstract | "PQ_things of 19.41 surpassing CUPS's own stereo-derived thing pseudo-label quality of 17.70" | Cherry-picked sub-metric. Overall PQ is worse (26.74 vs 27.8). |
| Abstract | "full pipeline reaches PQ of 32.76, surpassing CUPS by 4.96 points" | Confounded by backbone upgrade. With same backbone: -3.12 vs CUPS. |
| Line 12 | "No prior work has demonstrated that monocular-only pseudo-labels can match the quality of stereo-based pseudo-labels" | They don't match overall. PQ 26.74 < 27.8. |
| Line 14 | "This purely geometric approach...outperforms every spectral and attention-based instance method we evaluated...by margins exceeding 8 PQ_things points" | Only in driving scenes. No evidence this holds elsewhere. |
| Line 14 | "Third, a systematic 64-configuration sweep...reveals a previously undocumented interaction" | This is a hyperparameter sweep observation, not a methodological insight. Tradeoffs between interacting hyperparameters are expected. |
| Line 16 | "it is the monocular pseudo-labels that provide the foundation, and the DINOv3 backbone that magnifies their signal" | Table 2 contradicts this: RN50 + your labels = 24.68 < 27.8. The labels are the weak link. |
| Line 23 | "No prior work has identified or remedied this failure mode through overclustering on the learned code space" | CUPS Table 7b already uses overclustering (k=40, k=54) and shows improvement. |
| Line 142 | "The thing-class quality of 19.41 surpasses CUPS's own stereo-derived pseudo-label thing quality of 17.70 by 1.71 points" | Confirmed, but margin is small (1.71) and could be within noise without error bars. |
| Line 189 | "essentially matching the CUPS baseline of 27.8" | 27.87 vs 27.8 = +0.07. Within noise. And this uses a much stronger backbone. |
| Line 262 | "the overclustering contributes approximately 1.74 points...and the depth-guided splitting contributes the remaining 1.90 points" | This decomposition assumes additivity. No evidence the contributions are independent. |
| Line 275 | "the backbone upgrade is thus an orthogonal amplifier" | If truly orthogonal, CUPS + DINOv3 should also improve by ~3 points, reaching ~31. This is not tested. |
| Line 292 | "PQ_things rising from 23.17 to 34.13" | Unexplained 10.96-point jump needs investigation. |
| Line 294 | "The improvement is overwhelmingly concentrated in thing-class segmentation" | But the paper's claimed contribution is the pseudo-labels, which are measured before thing improvement happens. |
| Line 405 | "never write custom ViT implementations" | Engineering advice, not scientific contribution. |
| Line 424 | "person PQ of 4.2, RQ of 8.8%" | The most safety-critical class in autonomous driving has 8.8% recall. This is a disqualifying failure for any practical application claim. |
| Lines 431-436 | Conclusion repeats all numbers from abstract/results verbatim | Redundant. |

---

## 7. Reproducibility Assessment

| Aspect | Status |
|--------|--------|
| Code available | Yes (extensive, ~6000 lines) |
| Pre-trained models | SPIdepth, DINOv3, CAUSE — all depend on third-party weights |
| Dataset | Cityscapes (requires registration + license) |
| Eval protocol | 27-class CAUSE + Hungarian matching — custom, not standard |
| Hardware | 2x GTX 1080 Ti (accessible) |
| Random seeds | Single seed, no variance reported |

Reproducibility is **moderate**. The custom 27-class eval protocol makes verification non-trivial. The dependency chain (CAUSE -> overclustering -> SPIdepth -> Sobel -> CUPS Detectron2 -> DINOv3 backbone) has many moving parts.

---

## 8. Rating

**Overall Score: 4/10 — Reject**

**Confidence: 4/5 (high)**

### Justification

The paper's primary methodological contribution — running K-means with a larger k and applying Sobel edge detection to depth maps — does not meet the novelty threshold for NeurIPS. The headline result (PQ=32.76) is confounded by the DINOv3 backbone upgrade, and the critical ablation (CUPS + DINOv3) is missing. The "monocular beats stereo" framing is undermined by (a) the overall PQ being worse, (b) the depth model being trained on Cityscapes video, and (c) the absence of alternative depth model comparisons. The single-dataset, single-seed evaluation without error bars is below the NeurIPS standard. The extensive documentation of failed approaches and engineering issues, while honest, indicates a project report rather than a focused research contribution. The paper has genuine empirical value — the instance method comparison and overclustering sweep are useful — but this value is better suited to a workshop paper, an arXiv technical report, or a journal submission with multi-dataset evaluation and the missing ablations.

### What Would Change My Mind
1. CUPS + DINOv3 result showing that pseudo-labels matter (not just backbone)
2. Ablation on 2+ depth models (including non-Cityscapes-trained ones)
3. COCO-Stuff-27 results demonstrating generalization
4. Multi-seed evaluation with error bars
5. Explanation of the PQ_things 23.17 -> 34.13 jump

---

**Meta-review note:** The paper's greatest strength — radical transparency — is also what enables this review to be so specific. The authors deserve credit for documenting failures, metric confusions, and engineering issues that most papers would never reveal. The authors are encouraged to strengthen the experimental controls and pursue publication at a venue with longer page limits where the full research narrative can be properly contextualized.
