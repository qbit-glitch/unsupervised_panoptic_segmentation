# Training Dynamics for Dead-Class Recovery: 5 Strategies

**Context:** DINOv3 ViT-B/16 + Cascade Mask R-CNN (PanopticFPN) trained in 3 stages on Cityscapes pseudo-labels. After Stage-4 (Seesaw Loss + CAT), 5 classes remain at 0% PQ: **guard rail, tunnel, polegroup, caravan, trailer**. Hardware: 2× GTX 1080 Ti (11 GB), DDP, batch size 1, grad accumulation 8.

**Goal:** Training-strategy interventions (no architecture changes) to ensure the model learns rare-class signal once it exists in pseudo-labels.

---

## 1. Rare-First Curriculum Learning (RFCL)

**Stage:** Stage-2 (bootstrap training)

**Core idea:** Invert standard curriculum learning. Instead of starting with easy frequent classes and progressing to hard rare ones, start with rare-class-heavy image subsets and decay to uniform sampling. Rare-class pixels get boosted cross-entropy loss early in training before frequent-class gradients dominate.

**Implementation:**
- Image-level rarity score = sum of rare-class pixel fractions.
- Sampling weight: `w_i ∝ rarity_score^α` where α decays from 2.0 → 0.0 over first 2000 steps.
- After 2000 steps, revert to uniform sampling.

**Why it works:** Standard training drowns rare classes in the first 1000 steps when the model is most plastic. RFCL gives rare classes a head start.

**Expected impact:** Moderate for all 5 dead classes; most effective when combined with improved pseudo-labels (A1–A4).

---

## 2. Class-Conditional OHEM (CC-OHEM)

**Stage:** Stage-2 & Stage-4

**Core idea:** Reserve 25% of the ROI/pixel gradient budget for rare-class hard negatives. Standard OHEM mines globally hardest examples — these are almost always frequent-class confusions. CC-OHEM enforces a per-class quota so rare-class hard negatives receive gradient.

**Implementation:**
- In `FastRCNNOutputLayers` and `CustomSemSegFPNHead`, sort losses per class independently.
- Keep top-k per class rather than global top-k.
- For rare classes with < k samples, keep all positives + mine hard negatives from confusing neighbor classes.

**Why it works:** Ensures rare classes always have gradient flow even when their positive count is tiny.

**Expected impact:** High for thing classes (caravan, trailer); moderate for stuff classes.

---

## 3. Rare-Class Spatial Mixup (RC-SMix)

**Stage:** Stage-2 & Stage-3

**Core idea:** Context-aware CutMix variant. Blend rare-class patches only into semantically compatible regions with feathered alpha blending. Random CutMix destroys scene geometry; RC-SMix respects it.

**Placement heuristics:**
- Guard rail → road edges
- Tunnel → building/road transition regions
- Caravan/trailer → road plane at matching car/truck depth

**Implementation:**
- Modify existing `CopyPasteAugmentation` to bias sampling toward rare-class prototypes.
- Check semantic compatibility of paste region (target region must be road/building, not sky/vegetation).
- Use alpha blending with Gaussian feathering at patch boundaries.

**Why it works:** Increases effective rare-class exposure without violating scene geometry.

**Expected impact:** High for all 5 classes when combined with A4 (prototype bank).

---

## 4. Asymmetric Multi-Round Self-Training (AMR-ST)

**Stage:** Stage-3

**Core idea:** EMA teacher uses class-dependent confidence thresholds and TTA logit sharpening for rare classes.

**Details:**
- `τ_c = τ_global × (freq_c / freq_max)^α` with α = 0.5
  - Frequent classes (road): τ ≈ 0.70
  - Rare classes (guard rail): τ ≈ 0.14
- TTA (multi-scale) inference for rare-class logit sharpening.
- Round-wise protocol: in round 2+, lower threshold further for classes still at 0% PQ.

**Why it works:** Current self-training uses a single threshold (0.5 or 0.95). Rare classes never exceed it. Asymmetric thresholds let more rare-class pixels through the pseudo-label filter.

**Expected impact:** High for all 5 classes. **Zero extra parameters. TTA predictions can be cached.**

**Recommended priority:** #1 — implement this first.

---

## 5. Boundary-Aware Pixel Contrastive Loss (BAPC)

**Stage:** Stage-2 & Stage-4

**Core idea:** Pixel-level InfoNCE on rare-class boundary pixels. Pull toward class interior, push away from confusing neighbor classes.

**Confusing neighbor pairs to target:**
- Guard rail ↔ fence
- Tunnel ↔ building
- Polegroup ↔ pole
- Caravan ↔ truck
- Trailer ↔ truck

**Implementation:**
- Sample boundary pixels: `dilated_mask XOR eroded_mask`
- Positives = interior pixels of same class
- Negatives = interior pixels of top-3 confused classes
- Loss = InfoNCE with temperature 0.1

**Why it works:** Boundary confusion kills PQ for thin classes. BAPC explicitly learns discriminative boundaries.

**Expected impact:** Moderate for stuff classes (guard rail, tunnel); low for thing classes where instance head dominates.

---

## Codebase Findings

- `CopyPasteAugmentation` randomly samples instances; **no rare-class bias exists**.
- `train_refine_net.py` already has focal loss support.
- `train_ttt_mamba2.py` has class-balanced CE — patterns can be reused.
- `TrainingCurriculum` and `SelfTrainer` classes provide natural hooks for Proposals 1 and 4.
- Dead classes span both stuff (guard rail, tunnel, polegroup) and things (caravan, trailer), so strategies must cover both pixel-level (stuff) and ROI-level (things) interventions.

---

## Recommended Execution Order

Given 2× GTX 1080 Ti constraints:

1. **AMR-ST (Proposal 4)** — zero extra parameters, highest immediate impact.
2. **RFCL (Proposal 1)** — re-run Stage-2 with rare-first sampling.
3. **CC-OHEM (Proposal 2)** — add to Stage-4 alongside Seesaw Loss.
4. **RC-SMix (Proposal 3)** — if rare-class patches are available from A4.
5. **BAPC (Proposal 5)** — final Stage-4 boundary polish.

---

*Report generated by Training Dynamics Research Agent for MBPS dead-class recovery track.*
