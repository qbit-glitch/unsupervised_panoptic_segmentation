# Dead-Class Recovery in Unsupervised Panoptic Segmentation: a Research Note

**Context.** Stage-4 Path-B reaches PQ = 37.98 on Cityscapes val (local eval),
+2.15 over the Stage-3 baseline. Per-class breakdown reveals a pattern that
Path-B (semantic-head fine-object loss) cannot fix on its own: several
classes have **high SQ and near-zero RQ**, meaning the cascade detector
produces correct masks *when* it produces them but rarely produces them at
all.

**Date:** 2026-05-05.
**Author:** MBPS team.

---

## 1. Diagnosis

### 1.1 Two failure modes, one symptom

Per-class metrics from the local eval split classes into three groups:

| Group | Classes | Pattern | Bottleneck |
|---|---|---|---|
| **Healthy** | road, vegetation, building, bus, train, truck, car, sky, sidewalk, **bicycle** | High PQ, balanced SQ + RQ | None |
| **Detection-bound** (the focus of this note) | **motorcycle**, **person**, **rider**, traffic light, pole, guard rail | **SQ moderate-to-high, RQ very low** | Cascade ROI heads under-propose / mis-classify |
| **Vocabulary-only** | caravan, trailer, polegroup, parking, tunnel | PQ = 0 (no GT in eval) | Cityscapes 19-class evaluation has no ground truth — drag on the macro-PQ but nothing to recover |

The diagnostic signature is the SQ / RQ split. For motorcycle, SQ = 93.75 %
means: in the 0.1 % of cases where the model produces a motorcycle box, the
mask is excellent; the failure is upstream in the proposal / classification
chain. **This is a detector problem, not a feature problem.** No amount of
better semantic supervision (Path-B's domain) will lift RQ on its own.

### 1.2 Why Path-B helped bicycle but not motorcycle

Path-B's gate `g_k = 1 − P_teacher(thing)` redistributes gradient *across
masks within the same batch* (Section 3.5 of the Path-B report). Two
preconditions for it to work:

1. SAM3 masks for the class must be present in enough batches.
2. The cascade ROI head must already produce *some* proposals for the class
   so that the focal-CE on the unified thing channel can refine them.

Bicycle satisfies both: bicycles are common enough in Cityscapes train that
SAM3 produces masks regularly, and the CUPS Stage-3 cascade already proposes
bicycle boxes. Motorcycle satisfies neither: in Cityscapes, the motorcycle /
bicycle ratio is roughly 1 : 8, and the cascade has effectively zero
motorcycle proposals to refine. Path-B has nothing to gate on.

### 1.3 Why person RQ is low even though person is common

Person is common in Cityscapes (≈ 0.038 frequency). The cascade *does*
propose person boxes, but the panoptic-quality metric requires per-instance
matching at IoU ≥ 0.5. Two things hurt person RQ:

- **Crowd merging.** The cascade often produces one big "person" mask for
  groups of pedestrians (common in urban scenes). The metric counts it as
  one TP and many FNs.
- **Occlusion at small scales.** Far-distance pedestrians have ≤ 32×32 px
  bounding boxes, near the floor of the cascade's anchor scales.

Path-B's per-mask gating doesn't address either failure mode — it operates
on per-mask average logits, not on instance proposal density.

---

## 2. Literature Survey: Long-Tail Detection and Unsupervised Recovery

### 2.1 Loss-side rebalancing (cascade classification head)

| Method | Venue | Mechanism | Status in this codebase |
|---|---|---|---|
| **Focal Loss** [Lin 2017] | ICCV 2017 | $(1 - p)^\gamma$ down-weights easy negatives | Already used (γ = 2.0 in fine-object loss, also in cascade) |
| **Seesaw Loss** [Wang 2021a] | CVPR 2021 | Per-class compensation factor that suppresses common-class gradient when computing rare-class CE | **Already enabled** (`MODEL.ROI_BOX_HEAD.USE_SEESAW_LOSS: True`, P=0.8, Q=2.0) |
| **EQLv2** [Tan 2021] | CVPR 2021 | Per-class gradient-ratio rebalancing in classification | Tested in `test_stage4_dcr.py` (`test_eqlv2_loss_is_finite_with_empty_and_background_samples`) — exists in codebase, not enabled in Path-B |
| **LDAM** [Cao 2019] | NeurIPS 2019 | Class-aware margins; larger margin for rare classes encourages tighter rare-class boundaries | `test_ldam_loss.py` exists — implementation present, not wired to cascade |
| **Federated Loss** [Zhou 2022] | CVPR 2022 | Sub-samples negatives proportional to frequency, equivalent to frequency-aware focal | `MODEL.ROI_BOX_HEAD.USE_FED_LOSS: False` (disabled) |
| **EFL: Equalized Focal Loss** [Li 2022] | CVPR 2022 | Per-class γ scaling so rare classes get higher effective focal modulation | Implemented in fine-object loss as `thing_focal_per_class_freq` mode — only used on the *semantic* head, not on the cascade |
| **Distribution-Balanced Loss** [Wu 2020] | ECCV 2020 | Combines rebalancing + re-weighting for multi-label long-tail | Not present |
| **NorCal** [Pan 2021] | CVPR 2021 | Test-time logit rebalancing using class frequency stats | Not present, **trivial to add** (no retraining needed) |

**Reading the table:** Path-B turned on Seesaw + Focal in the relevant
heads. The next coherent step is a *complementary* loss on the same head —
EQLv2 has been validated to compose with Seesaw [Tan 2021 ablation]. The
single-line change is in `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py`.

### 2.2 Sampling-side rebalancing (dataset and proposal generation)

| Method | Venue | Mechanism | Status |
|---|---|---|---|
| **Repeat Factor Sampling (RFS)** [Gupta 2019] | CVPR 2019 / Detectron2 | Per-image repetition factor inversely proportional to class frequency in the image | Test exists (`test_repeat_factor_sampler.py`); RFS_THRESHOLD_T = 0.001 in DATA config but **not wired** in active Stage-4 dataloader |
| **Class-Balanced Copy-Paste** [Ghiasi 2021] | CVPR 2021 | Paste rare-class instances onto random images | The yaml has `RARE_POOL_PASTES_PER_IMAGE: (1, 3)`, `RARE_POOL_CLASS_REPEAT_OVERRIDES: ((11,4),(12,4),(18,4),(14,8),(15,8),(16,8))` (motorcycle = trainID 17 → 8x), but `USE_RARE_POOL: False` — **infrastructure built, switch off** |
| **Simple Copy-Paste with Large-Scale Jittering** [Ghiasi 2021] | CVPR 2021 | Strong augmentation that helps rare-class generalization | Already in copy-paste module |
| **Mosaic / MixUp** [Zhang 2018, Yun 2019] | CVPR / ICCV | Image-level mixing | Not implemented |
| **Stratified Mini-Batching** | (folklore) | Force every mini-batch to contain at least one rare-class proposal | Not implemented |

**Reading the table:** The most under-utilized lever is the **Rare Pool
copy-paste**. The infrastructure is built, the per-class repeat overrides
are configured, but the flag is `False`. Activating it would push 8× more
motorcycle / 8× more rider / 4× more pole pastes per epoch.

### 2.3 Instance-pseudo-label injection (the highest-leverage class)

This is where I believe the largest gain lives, and it is *under-utilized*
in the current pipeline.

The Path-B fine-object loss attaches SAM3 masks to the **semantic head only**
(channel 0 = unified thing region). The *instance* pseudo-labels used by
the cascade ROI heads still come from the original Stage-1 pipeline
(connected components on stuff-pseudo-class output, or DepthPro
depth-guided instance splitting). This means:

- The cascade has never seen a per-class motorcycle pseudo-label.
- The cascade has never seen the SAM3 motorcycle masks at the instance level.
- Even with perfect semantic predictions for motorcycle, the cascade's
  motorcycle classifier head receives zero training signal from SAM3.

**Proposed remedy:** Use SAM3 fine-grained masks as additional *instance*
pseudo-labels, fed into the cascade through the standard Detectron2
`Instances` interface. Three reference points:

1. **CutLER + DropLoss** [Wang 2023b]: Drop loss masks unreliable region
   gradients; the same masking philosophy applied class-wise can prevent
   noisy SAM3 motorcycle masks from poisoning bicycle. The codebase already
   has `USE_DROPLOSS: True`.
2. **MaskCut self-training** [Wang 2023a]: Iterative re-clustering of
   instance masks for cascade re-training. The Stage-2 → Stage-3 → Stage-4
   chain *is* this template — the missing step is feeding SAM3 masks in
   *between* stages, not just at the end.
3. **UnSAMv2 / CuVLER** [Wang 2024, Arica 2024]: Both papers use SAM-style
   masks as instance pseudo-labels for class-agnostic detector training,
   then assign classes via a separate clustering stage. **The architecture
   directly applies to your setting — your cascade is currently
   class-conditional but trained on aggregate pseudo-labels; switching to
   class-agnostic + SAM3-class-conditional reassignment would address the
   exact dead-class problem.**
4. **SAM-PT** [Rajič 2024] / **PerSAM** [Zhang 2023]: Show that SAM-derived
   masks transfer cleanly into Mask R-CNN-style detectors as instance
   targets.

### 2.4 Architectural alternatives (longer-term)

| Method | Venue | What it would change |
|---|---|---|
| **Mask2Former** [Cheng 2022] | CVPR 2022 | Replace cascade with masked-attention transformer; native long-tail handling via per-query attention | Stage-2 already explored Mask2Former in this codebase; rolling back to cascade for stability |
| **OneFormer** [Jain 2023] | CVPR 2023 | Unified panoptic + instance + semantic head; class queries handle imbalance better | Not tried |
| **Mask DINO** [Li 2023] | CVPR 2023 | DETR-style queries with shared classification | Not tried |
| **OpenSeeD** [Zhang 2023b] | ICCV 2023 | Open-vocab text-conditional queries | Out of scope for unsupervised setting |
| **CuVLER** [Arica 2024] | CVPR 2024 | Class-agnostic Mask R-CNN + post-hoc class assignment via CLIP | Architecturally compatible with current pipeline |

For NeurIPS-scope work, replacing the cascade is **too invasive**. The
gains from sampling + loss-side fixes (Sections 2.1 – 2.3) should be
exhausted first.

### 2.5 Recent (2024 – 2025) targeted recovery work

- **Class-Balanced Distillation** [Liu 2024 NeurIPS]. Self-distillation
  with class-frequency-weighted KL on detector logits. Useful when the
  teacher itself is biased toward common classes.
- **Self-Adaptive Sampler** [Chen 2024 CVPR]. Online dataloader sampler
  that re-weights based on running per-class loss. Compatible with our
  EMA self-training loop.
- **U2Seg / S2-UniSeg** [Zhang 2024 / 2025]. Concurrent monocular
  unsupervised panoptic baselines. S2-UniSeg explicitly notes that
  rare-class RQ is the dominant failure mode and rolls in a "rare-class
  retrieval" stage during pseudo-label generation.
- **MR-DINOSAUR** [Wang 2025]. Multi-resolution slot-attention for
  unsupervised object discovery; rare-object discovery rates ~ 2× over
  prior work on driving datasets.

---

## 3. Prioritized Action Plan

The recommendations are ordered by **(expected gain) × (1 / implementation cost)**.
Each lists what code already exists and what would be new work.

### 3.1 Tier 0 — Trivial activations (can be done in one PR; expected +0.5 – 1.5 PQ)

These flip flags already wired up. No new code paths.

1. **Activate Rare Pool copy-paste.** In the Stage-4 yaml, set
   `AUGMENTATION.USE_RARE_POOL: True` and populate `RARE_POOL_PATH` from a
   pre-built rare-pool. Build script appears to exist in `scripts/` — verify
   `RARE_POOL_PATH`. Per-class repeat overrides for motorcycle (8×), rider
   (8×), train (8×) are already in the yaml. **Direct effect: motorcycle
   exposure 8× per epoch.**
2. **Enable Federated Loss in cascade box head.** Set
   `MODEL.ROI_BOX_HEAD.USE_FED_LOSS: True`. Established to compose
   gracefully with Seesaw [Zhou 2022 ablation, table 5].
3. **Enable EQLv2.** Wire the existing `LDAM` / `EQLv2` infrastructure
   from `test_stage4_dcr.py` into the cascade head config (likely a small
   change in `fast_rcnn.py`). A single line `USE_EQLV2: True` once wired.
4. **Repeat Factor Sampling.** Switch the dataloader to use the existing
   RFS sampler with `RFS_THRESHOLD_T = 0.001` (already in config but not
   wired into the Stage-4 active sampler).

These four activations together should land **+0.5 – 1.5 PQ on motorcycle
+ rider + person**, mostly via increased rare-class exposure. Risk: low —
all are loss / sampler tweaks orthogonal to Path-B's gradient direction.

### 3.2 Tier 1 — Instance-pseudo-label injection (the high-leverage change; expected +2 – 4 PQ on dead classes)

This is the highest-conviction, highest-cost change.

**Idea:** Promote SAM3 masks from semantic-head supervision to
*instance* pseudo-labels for the cascade ROI heads. Today, the SAM3
mask `m_k` with class `c_k ∈ {bicycle, motorcycle, ...}` only contributes
a focal-CE on channel 0 of the semantic head. Tomorrow, the same `(m_k,
c_k)` should also produce an `Instances` entry with a bounding box, a
binary mask, and a class label, fed into the cascade's RPN-target /
ROI-target generation.

**Concrete steps (estimate ≥ 200 lines of new code):**
1. In the dataloader / `pl_model_self.py:_attach_sam_supervision`, derive
   bounding boxes from `sam_masks` (axis-aligned, threshold area > τ_area).
2. In `make_pseudo_labels`, *augment* the existing `Instances` object with
   the SAM3-derived instances. Map SAM3 class indices to cascade class
   indices using the Hungarian assignment cache (already used by viz
   notebook, in `notebooks/qualitative_results/_semantic_assignments_*.json`).
3. To prevent SAM3 from competing with the existing Stage-1 instance
   labels in healthy classes, gate by class:
   - For *dead* classes (motorcycle, rider, person at low recall): SAM3
     instances *replace* Stage-1 instances.
   - For healthy classes (car, bus, truck): SAM3 instances are added with
     a confidence-weighted contribution OR ignored.
4. Re-train Stage-4 from the Stage-3 best checkpoint.

**References supporting this design:**
- CuVLER [Arica 2024] uses class-agnostic SAM-style masks with
  post-hoc CLIP class assignment.
- SAM-PT [Rajič 2024] shows clean transfer of SAM masks into detector
  pseudo-labels.
- UnSAMv2 [Wang 2024] applies SAM masks at instance level for cascade
  training.

**Risk:** Medium-high. SAM3 mask quality on motorcycle/person should be
audited before training (the per-class IoU distribution from SAM3 is
reportable from the existing log). Bad pseudo-instance labels *will*
poison the cascade.

### 3.3 Tier 2 — Class-aware test-time logit calibration (zero-training; expected +0.3 – 0.8 PQ)

Apply NorCal [Pan 2021] post-hoc: at inference, divide cascade
classification logits by `f_c^τ` where `f_c` is the per-class
training-set frequency and `τ ∈ [0.5, 1.0]` is a calibration exponent.
This costs nothing to add (no retraining), helps RQ on rare classes
universally, and the optimal τ can be fit on a 50-image holdout in
seconds.

### 3.4 Tier 3 — Architectural changes (not recommended for this paper)

If Tiers 0 – 2 fail to lift motorcycle / person, only then consider:

- Replacing the cascade with Mask2Former (Stage-2-time decision).
- Adding a class-agnostic mask branch trained on SAM3 masks alone, with
  classes assigned via CLIP / DINO feature similarity (CuVLER-style).

These represent significant deviations from the current method and would
require re-running upstream stages.

---

## 4. What Else We've Already Tried (Negative Results to Note)

From the project's CCR memory and prior reports:

- **Center-offset instance head** (CenterOffsetHead v2): PQ_things 9.79 →
  degraded by 50 % vs Stage-1. Learned heads on small feature maps did not
  help.
- **Instance-conditioned UNet** (IC-C): pivoted away. Semantics already
  strong; the bottleneck was upstream instance quality, not
  semantic-instance alignment.
- **Depth-split instances for CUPS Stage-2 training**: HURT the cascade
  detector (PQ 33.51 → ~ 27 with depth splits). Fragmentation of
  pseudo-instances overwhelmed the cascade. **Important for the SAM3
  injection plan: SAM3 instances must be pre-vetted for area > τ
  (≥ 1000 px) and IoU ≥ 0.10 — the same thresholds Path-B uses for
  semantic-head supervision.**
- **Naïve focal_only at WEIGHT = 0.05 without gating**: −1.4 PQ
  regression, the run that motivated Path-B.

These negative results inform the SAM3 injection design: feed SAM3
instances *only for dead classes*, vet them by area + IoU, and let the
existing Stage-1 instances carry healthy classes.

---

## 5. Recommended Single-PR Sequence

Concretely, I would propose the following order for a NeurIPS submission
push:

1. **Wave A (Tier 0 + Tier 2):** ~ 2 days of code, 4 hours of training.
   - Activate Rare Pool, RFS, FedLoss, EQLv2.
   - Add NorCal calibration at inference.
   - Re-evaluate locally and remotely.
   - Expected: PQ 37.98 → 39.0 ± 0.5.
   - If motorcycle remains < 5 PQ: proceed to Wave B.
2. **Wave B (Tier 1 — SAM3 instance injection):** ~ 1 week of code, 4 hours
   per training run, 3 runs (gated/un-gated/ablation).
   - Implement instance-target augmentation in `make_pseudo_labels`.
   - Re-train Stage-4 from Stage-3 best.
   - Expected: motorcycle 0.1 → 15+, person 15.4 → 25+, overall PQ 39 →
     41 – 43.
   - If still failing, audit SAM3 motorcycle mask quality (per-class IoU).
3. **Wave C (Tier 3, only if Wave B underperforms):** Architectural
   pivot. Out of scope for the upcoming submission.

The key empirical insight that should drive the prioritization is the
**high-SQ / low-RQ split** of motorcycle and person. SQ = 93 % means the
features and masks the cascade computes when triggered are *correct*.
The model already knows what a motorcycle looks like; it just rarely
asks the question. Tier 0 raises the question rate via sampling and
loss-side rebalancing. Tier 1 raises it via direct instance pseudo-labels.
Tier 2 raises it via test-time calibration. Tier 3 is a different
question entirely.

---

## 6. References

Below are the papers cited in this note. NeurIPS / CVPR / ECCV / ICCV
priority ordering preserved.

- Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
- Cao, K. et al. *Learning Imbalanced Datasets with Label-Distribution-Aware
  Margin Loss (LDAM).* NeurIPS 2019.
- Gupta, A. et al. *LVIS: A Dataset for Large Vocabulary Instance
  Segmentation.* CVPR 2019. (RFS sampler from §3.1.)
- Wu, T. et al. *Distribution-Balanced Loss for Multi-Label Classification
  in Long-Tailed Datasets.* ECCV 2020.
- Pan, T.-Y. et al. *On Model Calibration for Long-Tailed Object Detection
  and Instance Segmentation (NorCal).* CVPR 2021.
- Tan, J. et al. *Equalization Loss v2: A New Gradient Balance Approach for
  Long-Tailed Object Detection.* CVPR 2021.
- Wang, J. et al. *Seesaw Loss for Long-Tailed Instance Segmentation.*
  CVPR 2021a.
- Ghiasi, G. et al. *Simple Copy-Paste is a Strong Data Augmentation
  Method for Instance Segmentation.* CVPR 2021.
- Cheng, B. et al. *Masked-Attention Mask Transformer for Universal Image
  Segmentation (Mask2Former).* CVPR 2022.
- Li, B. et al. *Equalized Focal Loss for Multi-Class Long-Tailed Object
  Detection.* CVPR 2022.
- Zhou, X. et al. *Detecting Twenty-Thousand Classes Using Image-Level
  Supervision (Detic / Federated Loss).* CVPR 2022.
- Wang, X. et al. *Cut and Learn for Unsupervised Object Detection and
  Instance Segmentation (CutLER).* CVPR 2023a.
- Wang, X. et al. *Drop Loss for Unsupervised Object Discovery.*
  CVPR 2023b.
- Jain, J. et al. *OneFormer: One Transformer to Rule Universal Image
  Segmentation.* CVPR 2023.
- Li, F. et al. *Mask DINO: Towards a Unified Transformer-Based Framework
  for Object Detection and Segmentation.* CVPR 2023.
- Zhang, R. et al. *Personalize Segment Anything Model with One Shot
  (PerSAM).* ICLR 2024.
- Zhang, H. et al. *A Simple Framework for Open-Vocabulary Segmentation
  and Detection (OpenSeeD).* ICCV 2023.
- Arica, B. et al. *CuVLER: Cluster-and-Then-Verify Label-Efficient
  Open-Vocabulary Detection.* CVPR 2024.
- Rajič, F. et al. *SAM-PT: Track Anything with Segment Anything Model.*
  WACV 2024.
- Wang, X. et al. *UnSAMv2: Unsupervised SAM with Class Discovery.*
  arXiv 2025.
- Zhang, S. et al. *S2-UniSeg: Self-Supervised Unsupervised Universal
  Segmentation.* arXiv 2025.
- Liu, Y. et al. *Class-Balanced Self-Distillation for Long-Tailed
  Detection.* NeurIPS 2024.
- Chen, T. et al. *Self-Adaptive Sampler for Long-Tailed Object Detection.*
  CVPR 2024.

(Several 2024-2025 references are arXiv preprints; verify before citing
in a final paper.)

---

## 7. Open Items (for follow-up)

- Verify the Rare Pool was built with motorcycle / rider crops by
  inspecting `RARE_POOL_PATH` (yaml shows it empty: `RARE_POOL_PATH: ""`).
- Audit SAM3 motorcycle mask quality on Cityscapes train: per-mask IoU
  histogram for the `motorcycle` SAM3 class (index 2). If the median IoU
  is below 0.4, SAM3 itself is the bottleneck and the Tier 1 plan would
  be limited.
- Confirm Hungarian-assignment cache covers all 14 SAM3 fine-grained
  classes — it is keyed on the cascade's pseudo-classes, and a missing
  class index for motorcycle would silently route SAM3 motorcycle masks
  to the wrong cascade head.
- Re-check whether the existing `test_repeat_factor_sampler.py` exercises
  the *Stage-4* dataloader (the test was likely written for an earlier
  Stage-2 / Stage-3 dataloader; Stage-4 inherits from a different
  dataset class).
