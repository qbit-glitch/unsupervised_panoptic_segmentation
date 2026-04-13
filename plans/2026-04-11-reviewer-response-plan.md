# Plan: Address All BMVC 2026 Reviewer Criticisms

**Created**: 2026-04-11
**Deadline**: ~2 weeks (BMVC camera-ready / resubmission)
**Status**: PLANNING

---

## Part 1: Experiments to Run

### E1. CUPS + DINOv3 Control Experiment [BLOCKER — Priority 0]
**Addresses**: Fatal confound (all 3 reviewers), Q1
**Machine**: RTX A6000 (Anydesk) or 2x 1080 Ti (remote)
**Time**: ~3-4 days (Stage-2: 8K steps + Stage-3: 8K steps + validation)

**Steps**:
1. Clone `refs/cups/configs/train_cityscapes_dinov3_vitb_cups_official_mps.yaml` → GPU variant
2. Copy CUPS official pseudo-labels (`cups_pseudo_labels_official/`, 2552 files) to training machine
3. Run Stage-2: `python train.py --experiment_config_file <config> --disable_wandb`
4. Run Stage-3: Update `MODEL.CHECKPOINT` → best Stage-2, run `train_self.py`
5. Evaluate all checkpoints at 500-step intervals

**Decision tree**:
| CUPS+DINOv3 PQ | Implication | Paper strategy |
|---|---|---|
| < 30 | Our pseudo-labels contribute strongly | Claim is validated |
| 30-31 | Modest contribution | Reframe around monocular simplification |
| 32-33 | Marginal | Reframe as "matching with monocular only" |
| > 33 | Paper contribution collapses | Reframe entirely around monocular accessibility |

**Config notes**: Keep CUPS defaults (`IGNORE_UNKNOWN_THING_REGIONS: True`, `THING_STUFF_THRESHOLD: 0.08`)

### E2. Multi-Seed Runs [BLOCKER — Priority 1]
**Addresses**: Statistical rigor (all 3 reviewers)
**Machine**: Both available GPUs in parallel
**Time**: ~4-5 days (overlaps with E1)

Run 2 additional seeds (existing = seed 1996):
- Seed 42: `SYSTEM.SEED 42`
- Seed 2026: `SYSTEM.SEED 2026`
- Each needs Stage-2 (8K steps) + Stage-3 (8K steps)
- Report mean ± std for PQ, PQ_things, PQ_stuff

### E3. Hyperparameter Robustness Sweep [Priority 2]
**Addresses**: Tuning against GT concern (R2, R3), Q3, Q4
**Machine**: Local MacBook (pseudo-label eval only)
**Time**: ~1 day

1. **k sensitivity**: k ∈ {50, 60, 70, 80, 90, 100, 150, 200, 300} — pseudo-label PQ
2. **τ sensitivity**: τ ∈ {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20}
3. **A_min sensitivity**: A_min ∈ {250, 500, 1000, 2000, 4000}
4. **Thing/stuff split**: top-{6,7,8,9,10} classes as things
5. **k=80 vs k=300**: Show PQ (not mIoU) peaks around k=80-100

### E4. Self-Training Ablation [Priority 2]
**Addresses**: Missing self-training ablation (R2), Q5
**Machine**: 2x 1080 Ti (after E1/E2 finish)
**Time**: ~3-4 days

Conditions:
1. EMA decay: α ∈ {0.99, 0.999, 0.9999}
2. Confidence threshold step: ∈ {0.0, 0.05, 0.10}
3. Number of rounds × steps: {1×8K, 2×4K, 3×2.67K, 4×2K}
4. **Cross-teacher**: ResNet-50 teacher → DINOv3 student (answers Q5 directly)

### E5. Stuff Regression Analysis [Priority 2]
**Addresses**: -3.81 PQ_stuff drop (R1, R3)
**Machine**: Local (analysis only)
**Time**: ~4 hours

1. Extract per-class PQ for all 11 stuff classes (CUPS vs ours)
2. Identify which specific stuff classes regress (likely the 7 dead CAUSE classes)
3. Compute PQ_stuff excluding dead classes
4. Show trade-off: stuff loss comes from CAUSE dead classes, not from depth splitting

### E6. Person Class Failure Visualization [Priority 3]
**Addresses**: Safety concern (R2, R3)
**Machine**: Local
**Time**: ~4 hours

1. Generate qualitative failures for person class
2. Show: co-planar merge, small/distant miss, rider confusion
3. Note improvement: pseudo-label 6.36% → Stage-3 20.75%

### E7. Cross-Dataset Presentation Fix [Priority 3]
**Addresses**: Misleading Mapillary claim (R2)
**Time**: ~2 hours (reanalysis + writing only)

---

## Part 2: Analysis of Existing Data (No Compute)

### A1. Per-Class PQ Trajectory Through Self-Training
Extract from existing 7 Stage-3 eval logs (steps 600-8000). Show smooth PQ_things growth.

### A2. Full Attribution Decomposition Table
| Configuration | PQ | Δ vs baseline | Source |
|---|---|---|---|
| CUPS (RN50 + CUPS PLs) | 27.8 | — | Baseline |
| Ours (RN50 + our PLs, Stage-2) | 24.68 | -3.12 | Pseudo-labels HURT |
| Ours (DINOv3 + our PLs, Stage-2) | 27.87 | +0.07 | Backbone recovers |
| CUPS (DINOv3 + CUPS PLs, Stage-2) | ??? | ??? | **FROM E1** |
| Ours (DINOv3 + our PLs, Stage-3) | 32.76 | +4.96 | Self-training |
| CUPS (DINOv3 + CUPS PLs, Stage-3) | ??? | ??? | **FROM E1** |

### A3. Oracle Dead-Class Analysis
7 classes with PQ=0 → if each achieved PQ=20%, overall +5.19 PQ → 37.94%.
Present this as evidence that semantic clustering (not backbone) is the binding constraint.

---

## Part 3: Paper Rewriting

### Title
Consider: "Monocular Pseudo-Label Compositing for Unsupervised Panoptic Segmentation"

### Abstract
- Remove "+5.0 over CUPS" if E1 is unfavorable
- Emphasize monocular-only deployment advantage
- Qualify "unsupervised" → "requiring no panoptic annotations"
- Add error bars: "PQ = X ± Y"

### Introduction — Terminology Clarification
Add paragraph defining "unsupervised" = no panoptic annotations, acknowledging pretrained foundations.
Note: CUPS equally uses pretrained models (DINO RN50, LEAStereo, RAFT).

### Contributions (rewrite based on E1 result)
- C1: Monocular pseudo-label pipeline matching stereo quality without sequence data
- C2: Keep DINOv3 + overclustering (but attribute fairly)
- C3: Backbone-controlled comparison with honest attribution table
- C4: Self-training analysis with error bars and ablation

### Method (Section 3)
- §3.2: Add k robustness table, cite CUPS Table 7b
- §3.3: Be explicit — "Sobel edge detection + connected components on monocular depth"
- §3.4: Add thing/stuff split sensitivity from E3
- §3.5: Add self-training hyperparameter paragraph from E4

### Experiments (Section 4)
- **Table 1**: Add CUPS+DINOv3 row from E1
- **Table 2**: Add hyperparameter robustness, multi-seed stats, self-training ablation
- New subsection: "Backbone Attribution Analysis" (A2 table)
- Stuff regression explanation (E5)
- Person class safety paragraph (E6)

### Cross-Dataset (Table 5)
- Fix Mapillary footnote (E7)
- Reorder: in-domain first

### Limitations
Strengthen:
- "DINOv3/DAv3 use supervised pretraining. 'Unsupervised' = no panoptic labels."
- "Person PQ=20.75% insufficient for safety-critical deployment"
- "PQ = X ± Y across 3 seeds"

---

## Part 4: Killer Question Responses

**Q1** (CUPS+DINOv3 PQ): Report E1 directly. If favorable: "Our pseudo-labels contribute X beyond backbone." If unfavorable: "Monocular achieves parity without stereo."

**Q2** (RN50 + ours < CUPS): "Gap from 7 dead CAUSE stuff classes. Thing PQ comparable (19.1 vs 17.7). DINOv3 compensates stuff gap and amplifies thing advantage."

**Q3** (Why k=80 not k=300): "PQ peaks at k=80-100 due to Hungarian matching noise. mIoU improves but PQ does not."

**Q4** (Top 8 things tuned?): "Sensitivity analysis shows PQ stable for top-{6..10}. Clear gap between 8th/9th ranked classes."

**Q5** (Self-training: backbone vs teacher): Report E4 cross-teacher experiment. "Backbone quality dominant, teacher quality additive."

---

## Part 5: Timeline

### Week 1 (Days 1-7)
| Day | A6000 / 1080 Ti #1 | 1080 Ti #2 | Local |
|---|---|---|---|
| 1 | Start E1 Stage-2 | Start E2 seed 42 Stage-2 | E3 (hyperparam sweep), A1-A3 |
| 2-3 | E1 Stage-2 finishes | E2 seed 42 continues | E5, E6 |
| 3-4 | Start E1 Stage-3 | E2 seed 42 Stage-3 | Paper rewriting begins |
| 5-6 | E1 Stage-3 finishes | Start E2 seed 2026 | Continue rewriting |
| 7 | **E1 COMPLETE → DECISION POINT** | E2 seed 2026 continues | — |

### Week 2 (Days 8-14)
| Day | Machines | Local |
|---|---|---|
| 8-9 | E4 (self-training ablation) | Paper rewriting based on E1 |
| 10-11 | E2 completes | Multi-seed statistics, tables |
| 12-13 | E4 finishes | Finalize paper + supplementary |
| 14 | — | Final proofread, compile |

### GPU Hours
| Experiment | Hours | Parallelizable |
|---|---|---|
| E1 (CUPS+DINOv3) | ~100h | Yes (with E2) |
| E2 (2 extra seeds) | ~200h | Yes (with E1) |
| E4 (self-training ablation) | ~200h | After E1/E2 |
| **Total** | ~500h | ~10-11 days wall clock |

---

## Part 6: Contingency

### If CUPS+DINOv3 > 33 PQ
Reframe entirely around **monocular accessibility**:
- CUPS requires CityscapesStereoVideo (324GB sequence data) + RAFT + LEAStereo + SF2SE3
- Ours requires only standard left images (12GB) + DAv3 monocular depth
- "We show monocular pseudo-labels match stereo quality with a strong backbone"

### If multi-seed std > 2.0 PQ
Report honestly. Focus on PQ_things (should have lower variance). Acknowledge overlapping CIs.

### If E4 shows single-round self-training is equally effective
Simplify the method. Fewer moving parts = stronger paper.
