---
type: writing
title: "BMVC 2026 — Reviewer 2 (Foundation Models / SSL Expert)"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, mock-review, reviewer-2]
reviewer_profile: Foundation model and self-supervised learning expert
expertise: High
confidence: Medium-High
score: 6
verdict: Weak Accept
updated: 2026-04-13
---

# Reviewer 2 — Foundation Models / SSL Expert

**Score: 6/10 — Weak Accept**
Expertise: High | Confidence: Medium-High

## Strengths

1. **Practical simplification**: Removing stereo+flow requirement is genuinely useful for deployment. Even if DAv3's pretraining used multi-view data, single-RGB inference is a real practical advantage.
2. **Depth model = binding constraint insight (Table 2b) is the strongest contribution.** 15 alternative methods all fail to beat simple Sobel+CC. Useful finding for the community.
3. **Self-training scaling analysis** (Table 2d): 3.9x amplification from ResNet-50 to DINOv3 is interesting. Deserves more investigation.
4. **Writing quality is above average** for BMVC. Question-based ablation headers read well.

## Weaknesses

### [Major] W1: No analysis of WHY self-training scales super-linearly

The most interesting finding — EMA self-training yields +1.25 PQ for ResNet-50 but +4.89 PQ for DINOv3 (3.9x amplification) — is stated but never explained.

> What property of DINOv3 features enables this? Feature quality? Attention maps? Representation geometry? This is where the paper could make a **real scientific contribution** but stops at observation.

### [Major] W2: Cross-dataset evaluation is weak

| Dataset | PQ | Issue |
|---------|-----|-------|
| COCO-Stuff-27 | 8.1% | Effectively zero |
| Mapillary | 39.9% | Inflated by class-mapping artifacts (authors acknowledge) |
| MOTS | 63.4% | Simplified highway domain |

No meaningful evidence of generalization beyond Cityscapes-like driving scenes. For a paper emphasizing "frozen foundation models," this is a significant limitation.

### [Minor] W3: Stuff quality regression is hand-waved

PQ_st = 32.0% vs CUPS 35.1% attributed to "7 non-standard CAUSE classes." But this is a direct consequence of using frozen CAUSE-TR features. No path to recover these classes is proposed or discussed.

Oracle: +5.18 PQ if 7 classes reached moderate quality — the **single largest addressable gap**, yet treated as acceptable trade-off.

### [Minor] W4: Missing comparison with concurrent/recent work

Related work discusses U2Seg (CVPR 2024) and CUPS (CVPR 2025). Are there concurrent 2025/2026 methods? The field moves fast.

### [Minor] W5: Figure 5 (qualitative) is too small

7-row qualitative figure is difficult to parse at printed resolution. Consider 3-4 rows with larger images.

## Questions for Authors

1. Mechanistic explanation for super-linear self-training amplification? Even a hypothesis tested against per-class data would strengthen the paper.
2. Have you tried unfreezing CAUSE-TR features and fine-tuning on overclustered assignments? Could recover 7 dead classes.
3. Inference latency breakdown? 2-3 FPS cited but not decomposed into backbone / head / post-processing.

## Our Response Strategy

| Concern | Response | Evidence |
|---------|----------|----------|
| Self-training WHY | Add paragraph with hypothesis: DINOv3's richer feature manifold enables EMA teacher to generate more diverse/accurate pseudo-labels, creating a positive feedback loop | Per-class PQ progression data exists in Appendix D |
| Cross-dataset weakness | Acknowledge honestly. Our PLs are Cityscapes-tuned (k=80, tau=0.03). Not a generalization claim — a single-domain result. | Table 3 |
| Stuff regression | Discuss future: fine-tuning CAUSE-TR or training a semantic refinement head on overclustered features | Oracle analysis |
| Concurrent work | Do a literature search before submission | — |
| Figure 5 size | Move to supplementary (also fixes page limit) | — |

## Key Insight from This Review

> R2 identifies the **self-training scaling** as the paper's most scientifically interesting finding — and notes we under-analyze it. This is actionable: adding 1 paragraph with a mechanistic hypothesis could meaningfully strengthen the paper's contribution claim.

## Links

- [[Writing/BMVC2026-Mock-Review-Summary]] — Overall review summary
- [[Writing/BMVC2026-Experiments]] — Table 2d self-training scaling data
