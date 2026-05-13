---
type: writing
title: Paper Writing Discussion — 2026-04-06
project: mbps-panoptic-segmentation
status: active
tags: [paper-writing, bmvc-2026, accv-2026, discussion-log]
updated: 2026-04-08
---

# Paper Writing Discussion — 2026-04-06

> Running log of all paper writing discussions. Updated on request only.

## Context

Empirical pipeline paper for unsupervised panoptic segmentation. Best result: PQ=30.26% on Cityscapes (beats CUPS 27.8% by +2.5). Methodology §3.1-3.5 structured. Targeting A-grade venue (ACCV 2026 if COCO ready by July, otherwise BMVC 2026 or WACV 2027).

## Discussion Log

### 2026-04-04: Methodology Structure + Venue Assessment

**Proposed §3 structure:**
- §3.1 Overview — pipeline figure, two-phase story: pseudo-label generation → supervised-style training
- §3.2 Semantic Pseudo-Labels — CAUSE-TR 90-dim features, k=80 overclustering, Hungarian mapping to 19-class
- §3.3 Depth-Guided Instances — SPIdepth depth maps, Sobel gradients, CC on thing-class masks, τ=0.20, A_min=1000
- §3.4 Panoptic Assembly — instance-first merge, semantic majority vote per instance, fallback CC for uncovered things → PQ=26.74
- §3.5 Network Training — Stage-2 (DINOv3 ViT-B/16 + Cascade Mask R-CNN, PQ=27.865) + Stage-3 (EMA self-training, PQ≈30.78)

**Venue assessment:**

| Venue | Likelihood | Deadline |
|-------|-----------|---------|
| ACCV 2026 | Strong fit | ~July 2026 |
| WACV 2027 | Solid fallback | ~Aug 2026 |
| AAAI 2027 | Harder, needs broader framing | ~Aug 2026 |
| BMVC 2026 | Closest deadline, appropriate | ~May 2026 |

**A* (CVPR/ICCV/NeurIPS): Borderline/unlikely as-is** — reviewers will ask "what is the novel algorithmic contribution?" The combination of CAUSE-TR + SPIdepth + CUPS is well-engineered but each component is prior work.

**Why A-grade clears:** +3 PQ over CUPS (CVPR 2025 paper), strong ablation coverage (depth models, architecture, instance methods, Stage-2 vs Stage-3), "improved pipeline" papers regularly accepted at WACV/ACCV.

### 2026-04-04: Contribution Framing

**Overclustering (k=80) is NOT a novel contribution.** PiCIE, STEGO, HP and others have used overclustering. Giving it a name does not make it a contribution. Honest framing: it's an engineering choice that works well.

**What can actually be novel:**
- Option A: The analysis IS the contribution — diagnostic paper "What limits unsupervised panoptic segmentation?" (A-level, lower effort)
- Option B: Solve the co-planar problem — person PQ=4.2 is a genuine open problem (A*-viable, high effort)
- Option C: Self-training threshold discovery — self-training hurts below a pseudo-label quality threshold, helps above it (strengthens an A paper)
- Option D: Multi-dataset study — if COCO results (NeCo+NAMR) get to ~50%+, reframe as generalizable pipeline (most practical path to stronger venue)

**Decision: Target Option D (multi-dataset)** if COCO competitive; fall back to Option A (diagnostic) if not.

## Decisions

1. **Target A-grade venues, not A\*.** Results justify A-grade. Don't over-claim novelty.
2. **Central claim**: "compositing CAUSE-TR + SPIdepth produces pseudo-labels sufficient to train a panoptic network surpassing current SOTA in unsupervised panoptic segmentation on Cityscapes."
3. **CUPS credit**: Stage-2/3 training recipe is CUPS's. Must clearly state "we adopt the CUPS training protocol." Contribution is the pseudo-label pipeline.
4. **Overclustering is not framed as a contribution** — it's a design choice in §3.2.
5. **Person PQ=4.2 (co-planar bottleneck) goes in Limitations** — acknowledge explicitly, not hidden.
6. **Don't start full paper writing until COCO numbers are in hand** — they determine the framing.

### 2026-04-09: Architecture Diagrams Generated (AI Image Generation)

Generated 3 publication-quality architecture diagrams using an image generation model with detailed prompts + ASCII pipeline descriptions as input. All diagrams verified against actual implementation.

**Figure 1 — Pipeline Overview (CUPS Figure 1 style):** Top: 2x3 results grid (Ours vs CUPS, +2.5 PQ badges). Bottom: compact flow — RGB+Depth → Instance/Semantic Labeling → Panoptic PL → Loss → Network (Self-Train EMA loop) → Output.

**Figure 3 — Pseudo-Label Generation (CUPS Figure 3 style):** 1a: RGB → DAv3 → Depth → Sobel+CC → Instance PL. 1b: RGB → DINOv2+CAUSE-TR → K-Means k=80 → Semantic PL. 1c (dark bg): Stuff/Things Classifier → Align → Panoptic PL (PQ=26.74%).

**Figure 4 — Training Pipeline:** Stage-2: PLs → Cascade Mask R-CNN (DropLoss) → PQ=27.87%. Stage-3: Teacher (EMA+TTA) → Online PLs → Student → EMA update loop → PQ=30.26%.

### 2026-04-09: Visualization Notebooks Created

4 notebooks in `notebooks/visualizations/` with real DINOv3 inference and per-step figure saving. 01: instance PL pipeline (13 cells). 02: panoptic merge (10 cells). 03: Stage-2 training + inference (11 cells). 04: Stage-3 self-training + S2 vs S3 comparison (11 cells). All save individual sub-images to `figures_*_pipeline/` directories.

## Action Items

- [x] Generate Figure 1 (overview — results + pipeline)
- [x] Generate Figure 3 (pseudo-label generation detail)
- [x] Generate Figure 4 (Stage-2 + Stage-3 training detail)
- [x] Create visualization notebooks (01-04) with real inference
- [x] Update research_paper_draft.md with diagrams + embedded images
- [ ] Save final high-res AI-generated figures to `figures/` directory
- [ ] Wait for NeCo result → determines if COCO story is viable
- [ ] Run NAMR post-processing locally (quick win, ~1h)
- [ ] Generate COCO panoptic PQ numbers (depth-guided instances on COCO)
- [ ] Write class-recovery analysis (k=27 → 80 → 300 confusion matrices)
- [ ] Clarify: which depth model in final pipeline (SPIdepth τ=0.20 vs DA3 τ=0.03)?
- [ ] Write abstract and introduction
- [ ] Verify all citations programmatically

## Links

- [[00-Hub]]
- [[01-Plan]]
- [[Papers/CUPS-2025]]
- [[Papers/DINOv3-2025]]
- [[Knowledge/Key-Lessons]]
- [[Knowledge/COCO-Gap-Analysis]]
