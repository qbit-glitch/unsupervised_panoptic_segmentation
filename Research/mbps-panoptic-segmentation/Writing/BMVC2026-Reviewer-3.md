---
type: writing
title: "BMVC 2026 — Reviewer 3 (Applied Panoptic Researcher)"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, mock-review, reviewer-3]
reviewer_profile: Applied panoptic segmentation and practical vision
expertise: Medium-High
confidence: High
score: 5
verdict: Borderline
updated: 2026-04-13
---

# Reviewer 3 — Applied Panoptic Segmentation

**Score: 5/10 — Borderline**
Expertise: Medium-High | Confidence: High

## Strengths

1. **Clear pipeline figure** (Figure 2): Three-panel pseudo-label generation diagram is well-designed and immediately communicable.
2. **Depth-split-ratio classifier** (Eq. 4): Simple, elegant solution for stuff/things separation without GT labels.
3. **Ablation table design is exemplary**: Table 2(b) "vary depth model" vs "vary splitting algorithm" in a single sub-table. Progressive investigation structure makes results genuinely readable.
4. **Reproducible**: All components are frozen public models (CAUSE-TR, DAv3, DINOv3), training follows CUPS, hyperparameters fully specified.

## Weaknesses

### [Major] W1: Only one training dataset — Cityscapes is narrow

- Paper trains only on Cityscapes
- k=80, tau=0.03, A_min=1000 all tuned on Cityscapes
- Cross-dataset evaluation (Table 3) is zero-shot transfer, NOT independent training
- Would these hyperparameters transfer to indoor scenes or aerial imagery?

### [Major] W2: Paper is over the 9-page BMVC limit

Content runs through page 11. **Desk-reject risk.**
- Fig 5 (qualitative, page 12) should move to supplementary
- Section 3.5 repeats content from setup paragraph — can compress

### [Major] W3: Person and car — two most common things — are weak

| Class | PQ | Practical Impact |
|-------|-----|-----------------|
| person | 20.75% | Most safety-critical class in driving |
| car | 17.51% | Most common thing class |

Co-planarity failure mode is acknowledged but fundamentally limits practical utility.

### [Minor] W4: Depth-split-ratio classifier lacks sensitivity analysis

Top-8-by-R_split rule for stuff/things assignment presented without analysis of:
- What happens when threshold changes?
- What if a stuff class is misclassified as thing or vice versa?
- How sensitive is final PQ to this classifier's accuracy?

### [Minor] W5: No computational cost comparison

Paper claims practical advantage (monocular only) but doesn't compare:
- Total pseudo-label generation time vs CUPS
- CUPS needs stereo+flow but may be faster in total
- DAv3 inference on 3000 images + K-means: what's the wall-clock time?

### [Minor] W6: Broken "Section ??" on page 6

## Questions for Authors

1. Stuff/things misclassification rate for depth-split-ratio classifier? How many classes are misassigned?
2. Total wall-clock time: pseudo-label generation (DAv3 + K-means + Sobel + assembly) vs CUPS?
3. With your pipeline trained on COCO instead of Cityscapes, what PQ?

## Our Response Strategy

| Concern | Response | Evidence |
|---------|----------|----------|
| Single dataset | Acknowledge honestly. Future: COCO + BDD training | COCO plan exists in `reports/coco_semantic_ablation_plan.md` |
| Page limit | Move Fig 5 to supplementary, compress §3.5 | Saves ~2 pages |
| Person/car weakness | Known co-planarity limitation. Depth cannot split co-planar objects. | Limitations section |
| Split-ratio sensitivity | Can run quick sensitivity sweep (change top-N from 8 to 6/10) | Straightforward experiment |
| Computational cost | Add comparison table: DAv3 (20min) + K-means (5min) + Sobel+CC (10min) vs CUPS stereo pipeline | Can estimate from logs |

## Key Insight from This Review

> R3 identifies **practical applicability** concerns: person (20.75%) and car (17.51%) are the most safety-critical classes, and they're among the weakest. This is a real limitation that the paper should not oversell.

> R3 also catches the **page limit violation** — this is a desk-reject risk that must be fixed before submission.

## Links

- [[Writing/BMVC2026-Mock-Review-Summary]] — Overall review summary
- [[Writing/BMVC2026-Supplementary]] — Appendix C per-class breakdown
