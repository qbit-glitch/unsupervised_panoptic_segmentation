---
type: writing
title: "BMVC 2026 — Reviewer 1 (Unsupervised Segmentation Expert)"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, mock-review, reviewer-1]
reviewer_profile: Senior researcher, unsupervised segmentation
expertise: High
confidence: High
score: 5
verdict: Borderline Reject
updated: 2026-04-13
---

# Reviewer 1 — Unsupervised Segmentation Expert

**Score: 5/10 — Borderline Reject**
Expertise: High | Confidence: High

## Strengths

1. **Strong empirical result**: +5.0 PQ over the only prior method is meaningful. The +16.4 PQ_th gain on things is striking.
2. **Thorough ablation design**: Progressive investigation in Section 4.3 is well-structured. Table 2(b) combining depth model and splitting algorithm in one sub-table is effective.
3. **Honest limitations**: Pretraining caveat, single-seed concern, PQ_st gap all acknowledged. Builds trust.
4. **Good overclustering analysis**: Diagnosis that k=27 kills 7 classes via centroid collapse (Section 3.2) is a genuine insight.

## Weaknesses

### [Major] W1: Contribution is primarily engineering, not methodological

The entire training pipeline (Cascade Mask R-CNN + DropLoss + CopyPaste + Stage-3 EMA) is copied verbatim from CUPS. The "method" is:
1. Run K-means with k=80 instead of k=27 on existing CAUSE-TR features
2. Apply Sobel thresholding on an existing depth estimator

Neither is a novel algorithm. Overclustering is well-known (CUPS Table 7b already shows it helps). Sobel edge detection is textbook.

> **Core question**: What exactly is the novel contribution beyond combining existing tools?

### [Major] W2: DINOv3 backbone confound is unresolved

- CUPS used DINOv2 ResNet-50. This paper uses DINOv3 ViT-B/16 (strictly stronger, released after CUPS)
- Table 2(c): DINOv3 → +3.19 PQ over ResNet-50 at Stage-2
- Missing control: CUPS pseudo-labels on DINOv3
- Stage-2 margin: +0.07 PQ (27.87 vs 27.8) — within noise

> Without this control, **cannot determine how much of +5.0 PQ comes from better PLs vs. better backbone**.

### [Major] W3: Single seed, no variance, no significance testing

- All results from seed=42
- Stage-2 margin (+0.07 PQ) is almost certainly within standard deviation
- Self-training gain (+4.89 PQ) is large enough to likely survive variance, but not demonstrated

### [Minor] W4: "Monocular" framing is misleading

DAv3 was pretrained with multi-view geometry. The triple negation ("no stereo, no video, no flow") oversells the monocular story.

### [Minor] W5: Page limit violation

11 pages content vs BMVC 9-page limit. Desk-reject risk.

### [Minor] W6: Broken "Section ??" on page 6

## Questions for Authors

1. Can you provide CUPS PLs trained on DINOv3 backbone? Without this, contribution attribution is ambiguous.
2. What is the variance across seeds 42/123/456?
3. If you use the same ResNet-50 backbone as CUPS, what PQ after Stage-3? (Table 2d shows 25.93 — *below* CUPS 27.8)

## Our Response Strategy

| Concern | Response | Evidence |
|---------|----------|----------|
| Engineering contribution | Honest framing + ablation depth (15 methods, 4 depth models) | Table 2, Appendix B |
| DINOv3 confound | **E1 experiment in progress** — see [[Writing/BMVC2026-Reviewer-Response]] | Pending |
| Single seed | Acknowledge. Run 2 more seeds if time permits | Compute limited |
| Monocular framing | Already have pretraining caveat paragraph | Section 4.2 |
| Page limit | Move Fig 5 to supplementary, compress §3.5 | Saves ~2 pages |

## Links

- [[Writing/BMVC2026-Mock-Review-Summary]] — Overall review summary
- [[Writing/BMVC2026-Reviewer-Response]] — E1 control experiment status
