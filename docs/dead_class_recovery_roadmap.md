# Dead-Class Recovery: Synthesized Research Roadmap

**Context:** After Stage-4 (Seesaw Loss + CAT fine-tuning), 5 classes remain at 0% PQ: **guard rail, tunnel, polegroup, caravan, trailer**. Root cause: <0.02% pseudo-label frequency → zero training signal for Stage-2.

**Hypothesis:** Fix Stage-2 (and Stage-3) training + pseudo-label generation so rare classes enter the pipeline upstream, rather than trying to recover them in Stage-4 fine-tuning.

---

## Part A: Pseudo-Label Generation Fixes (Upstream — Biggest Leverage)

These fix the root cause: if pseudo-labels contain rare-class pixels, Stage-2 can learn them.

### A1. Frequency-Aware Overclustering [P1, Low Cost]
- **What:** Reserve 12 of 80 k-means centroids exclusively for low-density feature regions.
- **How:** Two-stage k-means — first fit 68 centroids on dense regions, then fit 12 on the bottom-15% density patches.
- **Targets:** guard rail, tunnel, polegroup
- **Expected ΔPQ:** +0.8–1.5
- **File:** `mbps_pytorch/generate_overclustered_semantics.py`

### A2. Depth-Edge-Aware Semantic Splitting [P1, Low Cost]
- **What:** Use depth gradients + geometric priors to re-label ambiguous boundary pixels as rare stuff classes.
- **How:** Guard rail = thin horizontal strip at road boundary + depth discontinuity. Tunnel = Hough-transform converging lines + planar depth gradient. Polegroup = merge nearby pole CCs in depth-smooth regions.
- **Targets:** guard rail, tunnel, polegroup
- **Expected ΔPQ:** +1.2–2.5
- **File:** New `mbps_pytorch/generate_depth_edge_semantic_split.py`

### A3. Transitive k-NN Label Propagation [P2, Medium Cost]
- **What:** Build FAISS index on DINOv3 patch features; propagate truck labels to caravan/trailer neighbors in feature space.
- **How:** High-confidence truck anchors → k-NN query → decayed confidence propagation. Lightweight attribute classifier (aspect ratio) distinguishes caravan vs. trailer.
- **Targets:** caravan, trailer
- **Expected ΔPQ:** +1.5–3.0
- **File:** New `mbps_pytorch/generate_propagated_pseudolabels.py`

### A4. Geometric Copy-Paste of Rare-Class Prototypes [P2, Medium Cost]
- **What:** Extract rare-class patches from weak CAUSE logits (even if not argmax), paste them onto training images with depth-scaled geometry.
- **How:** Build prototype bank from pixels where CAUSE logit > 0.10–0.15. Place guard rail at road edge, caravan/trailer on road plane at matching depth scale.
- **Targets:** caravan, trailer, guard rail
- **Expected ΔPQ:** +2.0–4.0
- **File:** New `mbps_pytorch/generate_copypaste_rare_prototypes.py`

**Combined upstream potential (k=80 only):** +2.5–5.0 PQ from pseudo-label improvements alone.

---

## Part B: Training Strategy Fixes (Stage-2 & Stage-3)

These ensure the model actually learns from rare-class signal once it exists in pseudo-labels.

### B1. Rare-First Curriculum Learning (RFCL) [Stage-2]
- **What:** Invert standard curriculum — start with rare-class-heavy image subsets, decay to uniform sampling.
- **Why:** Standard training drowns rare classes in the first 1000 steps. RFCL gives them a head start before frequent-class gradients dominate.
- **Implementation:** Image-level rarity score = sum of rare-class pixel fractions. Sample with weight `w_i ∝ rarity_score^α` where α decays from 2.0 → 0.0 over first 2000 steps.

### B2. Class-Conditional OHEM (CC-OHEM) [Stage-2 & Stage-4]
- **What:** Reserve 25% of ROI/pixel gradient budget for rare-class hard negatives.
- **Why:** Standard OHEM mines the globally hardest examples — these are almost always frequent-class confusions. CC-OHEM enforces a per-class quota so rare-class hard negatives get gradient.
- **Implementation:** In `FastRCNNOutputLayers` and `CustomSemSegFPNHead`, sort losses per class independently. Keep top-k per class rather than global top-k.

### B3. Rare-Class Spatial Mixup (RC-SMix) [Stage-2 & Stage-3]
- **What:** CutMix variant that blends rare-class patches only into semantically compatible regions with feathered alpha blending.
- **Why:** Random CutMix destroys scene geometry. RC-SMix places guard rail patches at road edges, caravan patches on road, respecting depth and semantic context.
- **Implementation:** Modify existing `CopyPasteAugmentation` to bias sampling toward rare-class prototypes and check semantic compatibility of paste region.

### B4. Asymmetric Multi-Round Self-Training (AMR-ST) [Stage-3]
- **What:** EMA teacher uses class-dependent thresholds (`τ_rare=0.35` vs `τ_freq=0.70`) + TTA logit sharpening for rare classes.
- **Why:** Current self-training uses a single threshold (0.5 or 0.95). Rare classes never exceed it. Asymmetric thresholds let more rare-class pixels through.
- **Implementation:** In `SelfSupervisedModel` EMA teacher, apply per-class threshold: `τ_c = τ_global × (freq_c / freq_max)^α` with α=0.5. Add TTA (multi-scale) for rare-class logit sharpening.

### B5. Boundary-Aware Pixel Contrastive Loss (BAPC) [Stage-2 & Stage-4]
- **What:** Pixel-level InfoNCE on rare-class boundary pixels — pull toward class interior, push away from confusing neighbors.
- **Why:** Guard rail vs. fence, tunnel vs. building — boundary confusion kills PQ. BAPC explicitly learns discriminative boundaries.
- **Implementation:** Sample boundary pixels (dilated mask XOR eroded mask). Positive = interior pixels of same class. Negatives = interior pixels of top-3 confused classes. Loss = InfoNCE with temperature 0.1.

---

## Part C: Model-Level Interventions (Stage-2 Architecture)

### C1. Rare-Class Anchor Boosting with Focal RPN (RCAB-FRPN)
- **What:** Custom RPN that adds elongated aspect-ratio anchors (3.5) for caravan/trailer and up-weights objectness loss for rare-class positive anchors by 5×.
- **Why:** Default RPN anchors never match elongated rare objects; the RPN never learns to propose them.
- **Targets:** caravan, trailer
- **VRAM Δ:** +~300 MB | **Speed Δ:** +5%
- **Files:** `refs/cups/cups/model/modeling/proposal_generator/rare_rpn.py`, config `MODEL.PROPOSAL_GENERATOR.NAME: "RareClassRPN"`

### C2. Equalization Loss v2 (EQL v2) in Cascade Box Heads
- **What:** Drop-in replacement for CE in `FastRCNNOutputLayers`. Maintains online EMA of pos/neg gradient magnitudes per class, rescales so tail classes get comparable gradient to head classes.
- **Why:** Seesaw depends on cumulative counts (near-zero for dead classes → unstable). EQL v2 works from gradient statistics regardless of count.
- **Targets:** all rare things
- **VRAM Δ:** 0 MB | **Speed Δ:** +2%
- **Files:** `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py`
- **Note:** Do NOT combine with Seesaw Loss — pick one.

### C3. Stage-Specific Rare-Class Sampling (SSRCS)
- **What:** After matcher/subsampling in each cascade stage, enforce at least 2 rare-class RoIs per image by relaxing IoU threshold to 0.35 for rare classes and promoting background proposals that overlap rare GT boxes.
- **Why:** Cascade thresholds (0.5→0.6→0.7) kill poorly-localized rare proposals by Stage 3. SSRCS guarantees gradient flow to all stages.
- **Targets:** caravan, trailer
- **VRAM Δ:** 0 MB | **Speed Δ:** +1%
- **Files:** `refs/cups/cups/model/modeling/roi_heads/custom_cascade_rcnn.py`

### C4. Boundary-Aware Class-Balanced CE + Temperature Scaling (BACC-TS)
- **What:** Semantic head uses (a) class-frequency weighting, (b) temperature-softened pseudo-labels (KL divergence instead of hard CE), (c) boundary pixel up-weighting (λ=2.0).
- **Why:** Thin stuff classes (guard rail, tunnel) are only a few pixels wide; boundary weighting forces crisp edges. Temperature scaling prevents overfitting to noisy pseudo-label argmax.
- **Targets:** guard rail, tunnel, polegroup
- **VRAM Δ:** 0 MB | **Speed Δ:** +0%
- **Files:** `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py`

### C5. Quality Focal Loss (QFL) for Box Calibration
- **What:** Generalized focal loss using matched IoU as continuous quality target [0,1]. High-IoU positives get full gradient; low-IoU/noisy positives down-weighted.
- **Why:** Rare-class pseudo-labels are often low-quality. QFL prevents the few precious rare samples from being drowned by low-quality frequent-class positives.
- **Targets:** all rare things
- **VRAM Δ:** 0 MB | **Speed Δ:** +0%
- **Files:** `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py`
- **Note:** Requires switching box head from softmax to sigmoid CE.

---

## Recommended Execution Order

### Phase 1: Pseudo-Label Rescue (Week 1)
1. **A1 + A2 in parallel** (low cost, high stuff-class impact)
   - Frequency-aware k=80 clustering
   - Depth-edge semantic splitting for guard rail/tunnel/polegroup
   - Test on val set: measure cluster-to-class mapping, pixel recall per dead class

### Phase 2: Thing-Class Recovery (Week 2)
2. **A3 + A4** (medium cost, high thing-class impact)
   - k-NN propagation: truck → caravan/trailer
   - Geometric copy-paste prototype bank
   - Ablate: propagation-only vs. copy-paste-only vs. both

### Phase 3: Training Strategy (Week 2–3)
3. **B4 (AMR-ST)** — zero extra params, immediate Stage-3 benefit
4. **B1 (RFCL)** — re-run Stage-2 with rare-first sampling
5. **B2 (CC-OHEM)** — integrate into Stage-4 alongside Seesaw Loss

### Phase 4: Polish (Week 3)
6. **B3 (RC-SMix)** — if rare-class patches are available
7. **B5 (BAPC)** — final boundary refinement in Stage-4

### Phase 5: Full Pipeline Run (Week 4)
9. Re-generate all pseudo-labels with A1–A4
10. Re-train Stage-2 (with B1, B3)
11. Re-run Stage-3 (with B4)
12. Fine-tune Stage-4 (with B2, B5, existing Seesaw)
13. Evaluate full 19-class PQ

---

## Key Insight

Your intuition is correct: **Stage-4 Seesaw Loss is too late in the pipeline.** Seesaw can only amplify signal that already exists. If pseudo-labels have 0 caravan pixels, no loss function can create them.

The real fix is **upstream pseudo-label generation** (Part A) + **asymmetric self-training thresholds** (B4). These together can push rare-class signal into Stage-2, where the detector actually has 4000 steps to learn representations.

If we recover just **guard rail** and **caravan** from 0% to even 5% PQ each, that's a ~+0.3–0.5 PQ boost. But more importantly, it proves the **unsupervised pipeline can recover structurally rare classes** — a strong narrative point for the paper.

---

*Generated from agent synthesis. Model-level proposals from Agent 1 to be merged when complete.*
