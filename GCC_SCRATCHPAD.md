# GCC Scratchpad — NeurIPS Paper Writing Session
# Project: MBPS — Multi-modal Boundary-aware Panoptic Segmentation
# Author: Umesh (PhD, TCD)
# Assistant: Qbit
# Created: 2026-04-25
# Target Venue: NeurIPS 2026
# Deadline: May 6, 2026 (~11 days)

---

## 1. Paper Identity & Core Narrative

### Working Title (Draft)
**"From Depth to Panoptic: A Principled Decomposition for Unsupervised Panoptic Segmentation"**

Alternative titles:
- "Complementary Cues: Decomposing Unsupervised Panoptic Segmentation via Depth-Semantic Composition"
- "MBPS: Multi-modal Boundary-aware Panoptic Segmentation via Depth-Semantic Pseudo-Label Compositing"
- "Beyond Appearance: Geometry-Driven Instance Discovery for Unsupervised Panoptic Segmentation"

### One-Sentence Summary
We decompose unsupervised panoptic segmentation into complementary semantic and geometric signals, compose them via a learned boundary-aware merging strategy, and show that the resulting pseudo-labels enable network training that surpasses prior SOTA by +5.0 PQ.

### Three-Point Contribution (NeurIPS Standard)
1. **Decomposition Principle**: We formalize unsupervised panoptic segmentation as a composition of conditionally independent semantic and geometric observations, showing theoretically and empirically that their composition achieves tighter bounds on segmentation quality than either alone.
2. **Depth-Semantic Composition Pipeline**: We propose a pipeline that generates panoptic pseudo-labels by composing frozen foundation models (semantic features + monocular depth) with a learned depth-guided splitting mechanism, achieving PQ = 26.74 at pseudo-label level and PQ = 35.83 after network training.
3. **Self-Training Scaling Law**: We derive and empirically validate a scaling relationship between teacher backbone quality and self-training gains, showing that the improvement Δ_PQ scales super-linearly with teacher quality due to non-linear threshold effects in the PQ metric.

---

## 2. Results Summary (ALL NUMBERS)

### Best Results (DCFA + DepthPro + SIMCF-ABC)
| Metric | Value | Notes |
|--------|-------|-------|
| **PQ** | **35.83%** | Best result, DCFA+DepthPro(τ=0.20)+SIMCF-ABC |
| PQ_things | 22.93% | +8.56 over baseline |
| PQ_stuff | 40.89% | +0.25 over baseline |
| mIoU (pseudo-labels) | 62.2% | DCFA + SIMCF-ABC |

### Baseline Comparison
| Method | PQ | PQ_th | PQ_st | Source |
|--------|-----|-------|-------|--------|
| CUPS (CVPR 2025) | 27.8 | 17.7 | ~30.6 | CUPS paper |
| CUPS Stage-1 PLs | 26.5 | 17.7 | — | Pseudo-label quality |
| **Ours (SPIdepth, Stage 2)** | **27.9** | **23.2** | **30.6** | Old draft |
| **Ours (SPIdepth, Stage 3)** | **30.78** | **28.5** | **31.3** | Old draft |
| **Ours (DCFA+DepthPro+SIMCF-ABC)** | **35.83** | **22.93** | **40.89** | NEW |

### Ablation Results (DCFA+SIMCF-ABC Pipeline)
| Configuration | PQ | PQ_th | PQ_st | Δ vs Baseline |
|--------------|-----|-------|-------|---------------|
| Baseline (k=80, no DCFA, no SIMCF) | 31.62 | 14.37 | 40.64 | — |
| + DCFA (w/o SIMCF) | 32.56 | 14.80 | 41.51 | +0.94 |
| + SIMCF-ABC (w/o DCFA) | 34.52 | 20.86 | 40.89 | +2.90 |
| **+ DCFA + SIMCF-ABC** | **35.83** | **22.93** | **40.89** | **+4.21** |

### Near-Additivity Analysis
| Component | Gain | Additive Prediction | Actual | Deviation |
|-----------|------|---------------------|--------|-----------|
| Raw k=80 | 0.00 | 0.00 | 0.00 | — |
| + DCFA only | +0.94 | +0.94 | +0.94 | 0.00 |
| + SIMCF-ABC only | +2.90 | +2.90 | +2.90 | 0.00 |
| + Both | +4.21 | +3.84 | +4.21 | +0.37 (93% additive) |

### Self-Training Scaling
| Backbone | Stage 2 PQ | Stage 3 PQ | Δ | Notes |
|----------|-----------|-----------|-----|-------|
| DINOv2 ResNet-50 | 24.68 | 25.93 | +1.25 | Weak teacher |
| DINOv3 ViT-B/16 | 27.87 | 32.76 | +4.89 | Strong teacher |

### Cross-Dataset Transfer
| Target Dataset | In-Domain PQ | Transfer PQ | Δ | Notes |
|---------------|-------------|------------|---|-------|
| Cityscapes | 27.37 | — | — | Training domain |
| MOTS | — | 28.15 | +0.78 | Closely related |
| KITTI | — | 26.94 | -0.43 | Different camera, similar layout |
| Mapillary Vistas v2 | — | 29.86 | +2.49 | Different classes, same domain |
| COCO-Stuff-27 | — | 22.91 | -4.46 | Different domain |

---

## 3. Paper Structure (NeurIPS Template)

### Section Checklist

- [ ] **Abstract** (≤ 250 words) — Not started
- [ ] **Introduction** (~1.5 pages) — Not started
- [ ] **Related Work** (~1 page) — Partial draft exists
- [ ] **Method** (~3-4 pages) — Extensive draft exists (research_paper_draft.md)
- [ ] **Experiments** (~3-4 pages) — Extensive draft exists, needs DCFA+SIMCF-ABC updates
- [ ] **Discussion / Limitations** (~0.5 page) — Not started
- [ ] **Conclusion** (~0.3 page) — Not started
- [ ] **References** — Need verification
- [ ] **Appendix** — Need to decide scope

### Abstract Structure (Template)
```
[1] Problem: Unsupervised panoptic segmentation is hard because...
[2] Prior work limitation: Existing methods rely on X, but fail at Y...
[3] Our approach: We propose Z, which does A, B, and C...
[4] Key result: On Cityscapes, we achieve PQ=X, beating SOTA by Y...
[5] Broader impact: This enables Z applications / reveals W insight...
```

### Introduction Structure (Template)
```
§1.1 — Motivation & Problem
- Panoptic segmentation = semantic + instance
- Unsupervised = no labels
- Challenge: appearance alone fails for instances

§1.2 — Prior Work & Gap
- CUPS: stereo/video/flow required
- STEGO/PiCIE: semantic only, no instances
- HP: hand-crafted heuristics
- Gap: no principled way to get instances without multi-view

§1.3 — Our Approach (3-4 sentences)
- Decompose into semantic + geometric
- Compose via learned boundary-aware merging
- Self-training amplifies quality

§1.4 — Key Contributions (Bullet list)
- Decomposition principle (theory)
- Depth-semantic composition pipeline (method)
- Self-training scaling law (insight)

§1.5 — Results Preview (1 paragraph)
- PQ = 35.83, +5.0 over CUPS
- DCFA: +2.6 mIoU with 40K params
- SIMCF-ABC: +0.73 PQ pseudo-labels
- Near-additivity: 93%
```

---

## 4. Method Section — Current Status

### 3.1 Overview ✅ (Draft exists)
- Two-phase pipeline described
- Figure 1 placeholder defined
- Needs: update for DCFA+SIMCF-ABC

### 3.2 Semantic Pseudo-Label Generation ✅ (Draft exists)
- CAUSE-TR + k=80 overclustering
- L2 normalization justification
- Feature extraction protocol
- **NEEDS UPDATE**: Add DCFA description

### 3.3 Depth-Guided Instance Generation ✅ (Draft exists)
- SPIdepth/DAv3 depth maps
- Sobel gradient splitting
- τ and A_min selection
- **NEEDS UPDATE**: Add DepthPro (τ=0.20)

### 3.4 Panoptic Assembly ✅ (Draft exists)
- Instance-first merging
- Majority vote
- Fallback handling
- **NEEDS UPDATE**: Add SIMCF-ABC consistency filtering

### 3.5 DCFA: Depth-Conditioned Feature Adapter ⬜ (NEW SECTION)
- 40K parameter bottleneck adapter
- Depth-aware channel attention
- Cross-modal feature modulation
- Insert after 3.3 or as 3.3b

### 3.6 SIMCF-ABC: Semantic-Instance Mutual Consistency Filter ⬜ (NEW SECTION)
- Step A: Semantic filtering
- Step B: Instance filtering
- Step C: Cross-modal consistency
- Near-additivity analysis

### 3.7 Network Training (was 3.5) ⬜ (NEEDS UPDATE)
- Stage 2: Cascade Mask R-CNN
- Stage 3: EMA self-training
- DropLoss, Copy-Paste, etc.
- Update results to DCFA+SIMCF-ABC

---

## 5. Experiments Section — Current Status

### 4.1 Setup ✅ (Draft exists)
- Cityscapes validation
- 27-class CAUSE+Hungarian protocol
- Hardware specs

### 4.2 SOTA Comparison ⬜ (NEEDS UPDATE)
- Update with PQ=35.83
- Add DCFA+SIMCF-ABC row
- Verify CUPS numbers

### 4.3 Pseudo-Label Quality ⬜ (NEEDS UPDATE)
- Add DCFA row
- Add SIMCF-ABC row
- Add combined row

### 4.4 Depth Model Ablation ⬜ (NEEDS UPDATE)
- Add DepthPro row
- Add DCFA impact

### 4.5 DCFA Ablation ⬜ (NEW)
- w/o DCFA: PQ baseline
- w/ DCFA: +0.94 PQ
- mIoU improvement: +2.6
- Parameter count: 40K

### 4.6 SIMCF-ABC Ablation ⬜ (NEW)
- Step A, B, C individual gains
- Near-additivity: 93%
- Ceiling analysis

### 4.7 Semantic Backbone Ablation ⬜ (Draft exists, needs results)
- DINOv2 vs DINOv3
- ViT-S/B/L scaling

### 4.8 Training Backbone Ablation ⬜ (Draft exists, needs results)
- ResNet-50, ViT variants
- EUPE efficient variants

### 4.9 Self-Training Analysis ✅ (Draft exists)
- Scaling law derivation
- Weak vs strong teacher
- Threshold analysis

### 4.10 Qualitative Analysis ✅ (Draft exists)
- Figure 4 description
- Success/failure cases
- Per-class breakdown

### 4.11 Cross-Dataset Generalization ✅ (Draft exists, needs results)
- MOTS, KITTI, Mapillary, COCO

---

## 6. Theory Section — Draft Status

### Self-Training Scaling Theory ✅ (Draft exists)
- Location: `drafts/self_training_theory.md`
- Core claim: Δ_PQ ∝ γ(C_S) · (1-a_T) · |∂g/∂a|
- Status: Ready to integrate into paper

### Complementary Information Decomposition ⬜ (NOT STARTED)
- I(S; F_sem, F_geo) decomposition
- Conditional mutual information analysis
- Near-additivity formalization
- **PRIORITY**: This is the strongest theoretical claim

### Optimization Perspective on Overclustering ⬜ (NOT STARTED)
- k > C as slack variables
- Dead centroid detection via eigenvalue analysis
- Minimum k for class separability

---

## 7. Figures & Visualizations

### Figure 1: Pipeline Overview
- **Status**: AI-generated, needs update for DCFA+SIMCF-ABC
- **Content**: Full pipeline with results badges
- **Location**: `figures/pipeline_overview.pdf`

### Figure 2: Semantic Pseudo-Label Generation
- **Status**: Draft exists, needs DCFA update
- **Content**: DINOv2+CAUSE-TR → k-means → semantic PL
- **Location**: `figures/semantic_pseudolabel_architecture.pdf`

### Figure 3: Depth-Guided Instance Generation
- **Status**: Draft exists, needs DepthPro update
- **Content**: RGB → DAv3 → depth → Sobel → instances
- **Location**: `figures/instance_pseudolabel_architecture.pdf`

### Figure 4: Qualitative Comparison
- **Status**: Placeholder defined
- **Content**: 4 rows × 5 columns (CUPS vs Ours vs GT)
- **Location**: To be generated from notebook

### Figure 5: DCFA Architecture ⬜ (NEW)
- **Status**: Not created
- **Content**: Bottleneck adapter with depth-aware attention
- **Need**: Generate from notebook or draw

### Figure 6: SIMCF-ABC Process ⬜ (NEW)
- **Status**: Not created
- **Content**: Step A → Step B → Step C flow
- **Need**: Generate from notebook or draw

### Figure 7: Self-Training Scaling ⬜ (NEW)
- **Status**: Not created
- **Content**: Plot of Δ_PQ vs teacher PQ
- **Need**: Requires more backbone data points

---

## 8. Writing Tasks (Prioritized)

### P0 — Must Complete Before Submission
- [ ] Write Abstract
- [ ] Write Introduction
- [ ] Update Method section with DCFA + SIMCF-ABC
- [ ] Update Experiments section with new results
- [ ] Write Complementary Information Decomposition theory
- [ ] Verify all citations programmatically
- [ ] Generate missing figures (5, 6, 7)
- [ ] Write Limitations section
- [ ] Write Conclusion

### P1 — Strongly Recommended
- [ ] Run seed robustness experiments (seeds 43, 44)
- [ ] Add DINOv2 ViT-B/16 scaling point
- [ ] Oracle upper bound experiment
- [ ] Replace depth-split ratio heuristic with learned classifier
- [ ] Self-training with DCFA+SIMCF-ABC pseudo-labels

### P2 — Nice to Have
- [ ] BDD100K evaluation
- [ ] Deeper KITTI analysis
- [ ] More qualitative examples
- [ ] Video supplement

---

## 9. Key Decisions & Open Questions

### Decision: What is the main contribution?
**Options:**
1. DCFA + SIMCF-ABC pipeline (engineering)
2. Complementary cues theory (theoretical)
3. Self-training scaling law (empirical insight)
4. All three (risky for NeurIPS — might look unfocused)

**Current lean**: Frame as (2) with (1) as instantiation and (3) as additional insight.

### Decision: Venue
- **Primary target**: NeurIPS 2026 (deadline: May 6)
- **Backup**: ACCV 2026 (deadline: ~July)
- **Fallback**: WACV 2027 (deadline: ~August)

### Open Question: Is the theory strong enough?
- Information-theoretic decomposition is novel
- But needs formal derivation, not just intuition
- Risk: reviewers ask "where's the proof?"

### Open Question: Self-training with DCFA+SIMCF-ABC
- Does self-training on 35.83% PQ pseudo-labels produce >40% PQ?
- This would be a massive result
- Need to run this experiment

### Open Question: Person class bottleneck
- PQ_person = 4.2 across all configurations
- Co-planarity is geometric, not fixable with depth
- Options: (a) acknowledge as limitation, (b) add appearance-based splitting

---

## 10. Citation Verification Status

| Citation | Status | Notes |
|----------|--------|-------|
| CAUSE | ⬜ | Need full citation |
| DINOv2 | ⬜ | Oquab et al., TMLR 2024 |
| DINOv3 | ⬜ | Verify publication status |
| CUPS | ⬜ | CVPR 2025 |
| Depth Anything v3 | ⬜ | Verify reference |
| DepthPro | ⬜ | Apple, 2024 |
| SPIdepth | ⬜ | Self-supervised depth |
| Cascade Mask R-CNN | ⬜ | Cai & Vasconcelos |
| Cityscapes | ⬜ | Cordts et al., CVPR 2016 |
| PiCIE, HP, STEGO, DINOSAUR | ⬜ | Verify from originals |

---

## 11. Session Notes (Updated Live)

### 2026-04-25 Session
- Cloned 4 repos: GDA, STEGO, noah-research, svl_adapter
- Downloaded 19 papers to `Research/mbps-panoptic-segmentation/papers/`
- Created this scratchpad
- Next: Start writing Abstract + Introduction

---

## 12. Quick Links

- Draft paper: `research_paper_draft.md`
- Theory draft: `drafts/self_training_theory.md`
- DCFA+SIMCF-ABC report: `reports/dcfa_depthpro_simcf_abc_cvpr_report.md`
- Figures: `figures/` (pending generation)
- Experiments: `experiments/` on remote server
